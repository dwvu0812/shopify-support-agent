"""Shop support agent — AI chatbot for an e-commerce / Shopify store.

What problem this solves (in a shop owner's words)
----------------------------------------------------
"Customers keep asking the same questions — where is my order, how long
does shipping take, can I return this — and my support inbox never empties.
I need something that answers from MY shop's own documents and can actually
look up a real order, not a chatbot that guesses."

What this is
------------
A FastAPI service that answers customer questions in two ways:

1. RAG over the shop's own knowledge base (``docs/``: product catalog,
   shipping and returns policies, FAQ). Answers carry real citations, and
   the model is told to say it does not know when the documents do not
   cover the question.
2. A deterministic order-lookup tool. When a question contains an order
   number (``where is my order #1002``), the answer comes straight from
   the shop's order data — never from the LLM, never invented. Order
   questions work even without an LLM key configured.

Design decisions (the parts clients check)
-------------------------------------------
- One LLM provider, chosen from the environment. Only one key is needed:
  ``OPENAI_API_KEY`` -> gpt-4o-mini (takes priority if both keys are set)
  ``GEMINI_API_KEY`` -> gemini-3.5-flash-lite via the Gemini REST API,
  called with ``urllib`` (no extra SDK). Free tier, no card needed.
- The LLM provider is resolved lazily inside ``/chat``, only when the
  question is NOT an order lookup. That is why order questions answer
  without a key, and why a blank question is rejected with 422 even
  when no key is set (FastAPI solves dependencies before validating
  the body, so a provider dependency would turn that 422 into a 500).
- Retrieval is optional and degrading, never fatal: if ``chroma_db/`` is
  missing or empty, ``/chat`` still answers, with ``citations: []``.
  A broken vector store must not take the whole API down.
- Shop data (``data/products.json``, ``data/orders.json``) is the source
  of truth for order status and stock. The LLM never invents an order
  status, a tracking number, or a delivery date.
- Everything that can fail at runtime is explicit: missing key -> 500
  with a clear message, bad input -> 400/422, unknown order -> a clear
  "not found" answer (not an error), LLM provider error -> 502,
  rate limit -> 429. Stack traces never leak to the client.

Run it
------
::

    cp .env.example .env   # add ONE key (needed for RAG answers only)
    pip install -r requirements.txt
    python ingest.py       # index docs/ into Chroma
    uvicorn main:app --reload

Then ``GET /health``, ``GET /orders/1002`` and
``POST /chat {"question": "Where is my order #1002?"}``.
The ``widget/`` folder holds the embeddable Next.js chat widget.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Logging: one JSON line per event, with the request id attached.
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("shop_support_agent")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]
logger.propagate = False


# ---------------------------------------------------------------------------
# Settings: every knob comes from the environment, with a safe default.
# A missing LLM key is not a startup error — the app still boots so
# /health works, order lookup works, and /chat explains the problem
# (500) instead of crashing.
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )
    chroma_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db")
    )
    chroma_collection: str = "docs"
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))
    docs_dir: str = field(default_factory=lambda: os.getenv("DOCS_DIR", "./docs"))
    admin_token: str | None = field(
        default_factory=lambda: os.getenv("ADMIN_TOKEN") or None
    )
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "3"))
    )
    max_question_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    )
    llm_max_retries: int = 3
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
            ).split(",")
            if o.strip()
        ]
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or self.gemini_api_key)

    @property
    def provider_name(self) -> str:
        if self.openai_api_key:
            return f"openai:{self.openai_model}"
        if self.gemini_api_key:
            return f"gemini:{self.gemini_model}"
        return "none"


settings = Settings()
logger.setLevel(settings.log_level.upper())
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{settings.gemini_model}:generateContent"
)

# ---------------------------------------------------------------------------
# API models: the contract. /chat always returns this shape, whether or
# not retrieval found anything — clients never have to guess.
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    source: str
    chunk: int
    text: str = ""


class OrderItem(BaseModel):
    product_id: str
    name: str
    qty: int
    price: float


class OrderInfo(BaseModel):
    id: str
    status: str  # processing | shipped | out_for_delivery | delivered | cancelled
    customer: str
    items: list[OrderItem]
    total: float
    tracking_number: str | None = None
    carrier: str | None = None
    placed_date: str
    estimated_delivery: str | None = None
    delivered_date: str | None = None


class ProductCard(BaseModel):
    id: str
    name: str
    price: float
    stock: int
    variants: list[str] = []


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    model: str
    retrieved: bool = False
    order: OrderInfo | None = None
    products: list[ProductCard] = []


class HealthResponse(BaseModel):
    status: str
    provider: str


class ReadyResponse(BaseModel):
    status: str  # "ok" or "degraded"
    provider: str
    retrieval_available: bool


class AdminDocRequest(BaseModel):
    filename: str
    content: str

    @field_validator("filename")
    @classmethod
    def filename_must_be_a_plain_markdown_file(cls, value: str) -> str:
        if (
            not value.endswith(".md")
            or "/" in value
            or "\\" in value
            or value.startswith(".")
        ):
            raise ValueError("filename must be a plain .md file name")
        return value


# ---------------------------------------------------------------------------
# LLM providers: one interface, two implementations. Retries only cover
# transient provider errors (429 / 5xx); a 400 from the provider is
# returned as-is, retrying it would never help.
# ---------------------------------------------------------------------------


class LLMProvider(Protocol):
    name: str

    def generate(self, system_prompt: str, question: str) -> str: ...


def _retryable(status: int | None) -> bool:
    return status in (429, 500, 502, 503, 504)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2**attempt, 8))


class OpenAIProvider:
    name = f"openai:{settings.openai_model}"

    def __init__(self) -> None:
        self._client = OpenAI(timeout=settings.llm_timeout_seconds)

    def generate(self, system_prompt: str, question: str) -> str:
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - mapped below
                last_error = exc
                status = getattr(exc, "status_code", None)
                logger.warning("openai error attempt=%d status=%s", attempt, status)
                if not _retryable(status):
                    break
                _backoff_sleep(attempt)
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {last_error}"
        )


class GeminiProvider:
    name = f"gemini:{settings.gemini_model}"

    def generate(self, system_prompt: str, question: str) -> str:
        body = json.dumps(
            {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\n{question}"}
                        ]
                    }
                ]
            }
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries):
            req = urllib.request.Request(GEMINI_URL, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("x-goog-api-key", settings.gemini_api_key or "")
            try:
                with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", errors="replace")
                logger.warning(
                    "gemini error attempt=%d status=%s", attempt, exc.code
                )
                if not _retryable(exc.code):
                    raise HTTPException(
                        status_code=502, detail=f"Gemini error {exc.code}: {detail}"
                    ) from exc
                _backoff_sleep(attempt)
                continue
            except Exception as exc:  # noqa: BLE001 - network error, retry
                last_error = exc
                logger.warning("gemini network error attempt=%d: %s", attempt, exc)
                _backoff_sleep(attempt)
                continue
            candidates = data.get("candidates", [])
            if not candidates:
                raise HTTPException(
                    status_code=502, detail=f"Gemini returned empty: {data}"
                )
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        raise HTTPException(status_code=502, detail=f"LLM provider error: {last_error}")


def get_llm_provider() -> LLMProvider:
    """Resolve the provider. Called lazily inside /chat, not as a
    dependency, so order-lookup questions work without a key."""
    if settings.openai_api_key:
        return OpenAIProvider()
    if settings.gemini_api_key:
        return GeminiProvider()
    raise HTTPException(
        status_code=500,
        detail="No OPENAI_API_KEY or GEMINI_API_KEY in .env",
    )


# ---------------------------------------------------------------------------
# Retrieval: Chroma, lazy and fault-tolerant. Imported lazily so the API
# boots and serves /health even on a machine without chromadb installed.
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    source: str
    chunk_index: int
    text: str


class ChromaRetriever:
    def __init__(self, settings_: Settings) -> None:
        self._settings = settings_
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        client = chromadb.PersistentClient(path=self._settings.chroma_path)
        self._collection = client.get_or_create_collection(
            self._settings.chroma_collection
        )
        return self._collection

    def available(self) -> bool:
        try:
            return self._get_collection().count() > 0
        except Exception:  # noqa: BLE001 - retrieval is best-effort
            return False

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return []
            query_embedding = embed_text(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self._settings.retrieval_top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001 - never fail /chat for retrieval
            logger.exception("retrieval failed, continuing without context")
            return []
        chunks: list[RetrievedChunk] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        for doc, meta in zip(documents, metadatas):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    source=str(meta.get("source", "docs")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                )
            )
        return chunks


def embed_text(text: str) -> list[float]:
    """Embed one text with the same provider priority as /chat.

    Gemini: gemini-embedding-001 via REST. OpenAI: text-embedding-3-small.
    """
    if settings.gemini_api_key:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_embedding_model}:embedContent"
        )
        body = json.dumps(
            {
                "model": f"models/{settings.gemini_embedding_model}",
                "content": {"parts": [{"text": text}]},
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("x-goog-api-key", settings.gemini_api_key)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini embedding error {exc.code}",
            ) from exc
        return [float(v) for v in data["embedding"]["values"]]
    if settings.openai_api_key:
        client = OpenAI(timeout=settings.llm_timeout_seconds)
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return [float(v) for v in resp.data[0].embedding]
    raise HTTPException(
        status_code=500,
        detail="No OPENAI_API_KEY or GEMINI_API_KEY in .env",
    )


_retriever: ChromaRetriever | None = None


def get_retriever() -> ChromaRetriever:
    """FastAPI dependency — tests override this with a fake retriever."""
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever(settings)
    return _retriever


# ---------------------------------------------------------------------------
# Shop data: the source of truth for orders and products. Loaded from
# data/*.json (in a real Shopify install these come from the Shopify API).
# ---------------------------------------------------------------------------

_shop_data: dict[str, Any] = {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_shop_data(data_dir: str | None = None) -> dict[str, Any]:
    """Load (and cache) the shop data. Tests reset the cache via
    ``main._shop_data.clear()`` after pointing settings at a temp dir."""
    global _shop_data
    if _shop_data:
        return _shop_data
    base = Path(data_dir or settings.data_dir)
    products = _load_json(base / "products.json")
    orders = _load_json(base / "orders.json")
    _shop_data = {
        "shop": products.get("shop", {}),
        "products": products.get("products", []),
        "orders": {o["id"]: o for o in orders.get("orders", [])},
    }
    return _shop_data


def get_order(order_id: str) -> OrderInfo | None:
    data = load_shop_data()
    raw = data["orders"].get(str(order_id))
    if raw is None:
        return None
    return OrderInfo(**raw)


def find_products(text: str) -> list[ProductCard]:
    """Products whose name appears in the given text (question or
    retrieved chunks). Used to attach product cards to a chat answer."""
    data = load_shop_data()
    haystack = text.lower()
    found: list[ProductCard] = []
    for p in data["products"]:
        name = p["name"].lower()
        # Match on the full name or its distinctive first words
        # ("trailhead hiking backpack", "ridgeLine waterproof jacket").
        short = " ".join(name.split()[:3])
        if name in haystack or short in haystack:
            found.append(ProductCard(**{k: p[k] for k in ("id", "name", "price", "stock", "variants")}))
    return found[:3]


_ORDER_PATTERNS = [
    re.compile(r"#(\d{3,})"),
    re.compile(r"\border\s*#?\s*(\d{3,})", re.IGNORECASE),
]


def extract_order_id(question: str) -> str | None:
    """Return the order id mentioned in the question, if any."""
    for pattern in _ORDER_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1)
    return None


def order_answer(order: OrderInfo) -> str:
    """Deterministic answer built from the order record only — the LLM
    is never asked to state an order status."""
    items = ", ".join(f"{i.qty}x {i.name}" for i in order.items)
    status_text = {
        "processing": "is being prepared",
        "shipped": "has shipped",
        "out_for_delivery": "is out for delivery today",
        "delivered": "was delivered",
        "cancelled": "was cancelled",
    }.get(order.status, f"is {order.status}")
    parts = [
        f"Order #{order.id} ({items}) {status_text}.",
    ]
    if order.status == "delivered" and order.delivered_date:
        parts.append(f"It was delivered on {order.delivered_date}.")
    if order.status in ("shipped", "out_for_delivery") and order.estimated_delivery:
        parts.append(f"Estimated delivery: {order.estimated_delivery}.")
    if order.status == "processing" and order.estimated_delivery:
        parts.append(f"Estimated delivery: {order.estimated_delivery}.")
    if order.tracking_number and order.carrier:
        parts.append(
            f"Tracking: {order.carrier} {order.tracking_number}."
        )
    if order.status == "cancelled":
        parts.append(
            "If this was a mistake, email support@northwind-outdoors.example "
            "and we will help you place the order again."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# App: request id, rate limit, CORS, error handling.
# ---------------------------------------------------------------------------

app = FastAPI(title="shopify-support-agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request_id_var.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


_rate_lock = threading.Lock()
_rate_hits: dict[str, list[float]] = {}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path != "/chat":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        hits = [t for t in _rate_hits.get(client_ip, []) if now - t < 60]
        if len(hits) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests, try again in a minute."},
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        _rate_hits[client_ip] = hits
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled error path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "request_id": request_id_var.get()},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness only: the process is up. No dependency is checked here."""
    return HealthResponse(status="ok", provider=settings.provider_name)


@app.get("/ready", response_model=ReadyResponse)
def ready(retriever: ChromaRetriever = Depends(get_retriever)):
    """Readiness: can this instance actually answer a question?"""
    retrieval_available = retriever.available()
    status = "ok" if settings.llm_configured else "degraded"
    return ReadyResponse(
        status=status,
        provider=settings.provider_name,
        retrieval_available=retrieval_available,
    )


@app.get("/orders/{order_id}", response_model=OrderInfo)
def get_order_endpoint(order_id: str):
    """Look up one order by id. 404 when the order does not exist."""
    order = get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order #{order_id} not found")
    return order


@app.get("/products", response_model=list[ProductCard])
def list_products():
    """The shop catalog (name, price, stock, variants)."""
    return [ProductCard(**{k: p[k] for k in ("id", "name", "price", "stock", "variants")}) for p in load_shop_data()["products"]]


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    retriever: ChromaRetriever = Depends(get_retriever),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")
    if len(question) > settings.max_question_length:
        raise HTTPException(
            status_code=400,
            detail=f"question longer than {settings.max_question_length} chars",
        )

    # 1) Order lookup first — deterministic, no LLM involved, works
    #    even when no API key is configured.
    order_id = extract_order_id(question)
    if order_id is not None:
        order = get_order(order_id)
        if order is None:
            return ChatResponse(
                answer=(
                    f"I couldn't find order #{order_id}. Please double-check "
                    "the order number from your confirmation email, or email "
                    "support@northwind-outdoors.example and we will look it up."
                ),
                citations=[],
                model="order-lookup",
                retrieved=False,
                order=None,
            )
        return ChatResponse(
            answer=order_answer(order),
            citations=[
                Citation(source=f"orders/{order.id}", chunk=0, text=order_answer(order))
            ],
            model="order-lookup",
            retrieved=True,
            order=order,
        )

    # 2) RAG path — needs an LLM key.
    provider = get_llm_provider()
    chunks = retriever.retrieve(question)
    retrieved = bool(chunks)
    if retrieved:
        context = "\n\n".join(
            f"[{c.source} chunk {c.chunk_index}]\n{c.text}" for c in chunks
        )
        system_prompt = (
            "You are the support assistant for Northwind Outdoors, an online "
            "outdoor gear shop. Answer briefly, in English, using ONLY the "
            "shop documents below. If the documents do not cover the question, "
            "say so plainly and point the customer to "
            "support@northwind-outdoors.example — never guess a price, a "
            "delivery date, or a stock level.\n\n"
            f"{context}"
        )
    else:
        logger.info("no retrieval context, answering from LLM only")
        system_prompt = (
            "You are the support assistant for Northwind Outdoors. "
            "Answer briefly, in English. Do not invent prices, delivery "
            "dates, or stock levels."
        )

    answer = provider.generate(system_prompt, question)
    search_text = question + "\n" + "\n".join(c.text for c in chunks)
    return ChatResponse(
        answer=answer,
        citations=[
            Citation(source=c.source, chunk=c.chunk_index, text=c.text)
            for c in chunks
        ],
        model=provider.name,
        retrieved=retrieved,
        products=find_products(search_text),
    )


@app.post("/admin/docs")
def admin_upload_doc(
    payload: AdminDocRequest,
    x_admin_token: str | None = Header(default=None),
):
    """Upload or replace one knowledge-base document and re-index it.

    Body: {"filename": "shipping.md", "content": "..."}.
    If ADMIN_TOKEN is set in the environment, the request must carry
    header ``X-Admin-Token`` with the same value, otherwise 401.
    Re-indexing needs an embedding key (500 without one).
    """
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / payload.filename
    target.write_text(payload.content, encoding="utf-8")

    # Re-index just this document: drop its old chunks, embed the new ones.
    from ingest import chunk_documents  # lazy: ingest imports main

    source = f"docs/{payload.filename}"
    chunks = chunk_documents([(source, payload.content)])
    collection = get_retriever()._get_collection()
    try:
        collection.delete(where={"source": source})
    except Exception:  # noqa: BLE001 - nothing to delete yet
        pass
    embeddings = [embed_text(text) for _, _, text in chunks]
    collection.add(
        ids=[f"{source}-{i}" for source, i, _ in chunks],
        documents=[text for _, _, text in chunks],
        metadatas=[{"source": source, "chunk_index": i} for source, i, _ in chunks],
        embeddings=embeddings,
    )
    logger.info("re-indexed %s (%d chunks)", source, len(chunks))
    return {"filename": payload.filename, "chunks": len(chunks), "source": source}
