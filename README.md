# Shopify Support Agent — AI support chatbot for online stores

> "Customers keep asking the same questions — where is my order, how long
> does shipping take, can I return this — and my support inbox never
> empties. I need something that answers from MY shop's own documents and
> can actually look up a real order, not a chatbot that guesses."

That is the problem this repo solves. It is a support chatbot for an
e-commerce / Shopify store that does two things a generic demo chatbot
cannot:

1. **Answers from your shop's own documents** — product catalog, shipping
   and returns policies, FAQ — with real citations, and says plainly when
   it does not know.
2. **Looks up real orders.** Ask "Where is my order #1002?" and the answer
   comes straight from the order data — status, tracking number, delivery
   date — never invented by the LLM. Order questions work even without an
   LLM key configured.

Built production-grade: FastAPI, a real LLM (Gemini or OpenAI), RAG with
citations, order-lookup tool, tests, eval, Docker and CI — not a notebook
demo.

## Live demo

```bash
git clone <this-repo>
cd shopify-support-agent
pip install -r requirements.txt
cp .env.example .env        # add ONE key: GEMINI_API_KEY (free) or OPENAI_API_KEY
python ingest.py            # index docs/ into Chroma (needs the key)
uvicorn main:app --reload
```

Then try it:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/orders/1002
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is my order #1002?"}'
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do I have to return something?"}'
```

Order questions answer without an API key. RAG questions need one key.

Docker:

```bash
docker compose up --build
```

The embeddable chat widget lives in `widget/` (Next.js). Deploy it
(e.g. Vercel), then add one tag to any store page:

```html
<script src="https://chat.example.com/embed.js"
        data-widget-url="https://chat.example.com"
        data-api-url="https://api.example.com"
        async></script>
```

## What it does

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness, always cheap, no dependency checked. |
| `GET /ready` | Readiness: is an LLM configured, is retrieval available? |
| `POST /chat {"question": "..."}` | Answers in English. Returns `answer`, `citations`, `model`, `retrieved`, plus `order` (when the question is about an order) and `products` (product cards when the answer cites a product). |
| `GET /orders/{order_id}` | One order by id: status, items, total, tracking, delivery dates. 404 when unknown. |
| `GET /products` | The catalog: name, price, stock, variants. |
| `POST /admin/docs` | Upload or replace a knowledge-base doc (`{"filename": "shipping.md", "content": "..."}`) and re-index it. Requires the `X-Admin-Token` header when `ADMIN_TOKEN` is set. |

Example `/chat` response for an order question:

```json
{
  "answer": "Order #1002 (1x Summit 2-Person Tent, 2x Aurora Headlamp 300lm) has shipped. Estimated delivery: 2026-11-28. Tracking: FedEx 784512369014.",
  "citations": [{ "source": "orders/1002", "chunk": 0, "text": "..." }],
  "model": "order-lookup",
  "retrieved": true,
  "order": { "id": "1002", "status": "shipped", "total": 208.98 },
  "products": []
}
```

## How it works

```
docs/*.md ──chunk (500 chars, 50 overlap)──► embed ──► Chroma
                                                          │ top-3
customer question ──► order id? ──yes──► order data (deterministic, no LLM)
                      └─no──► retrieve ──► grounded LLM answer + citations
```

- **RAG flow:** `docs/*.md` → chunk → embed (Gemini `gemini-embedding-001`,
  or OpenAI `text-embedding-3-small`) → Chroma → top-3 chunks per question
  → grounded answer with real citations. If the documents do not cover the
  question, the model is told to say it does not know instead of inventing
  an answer. If Chroma is missing or empty, `/chat` still answers from the
  LLM alone with `citations: []`.
- **Order tool:** a question containing an order number (`#1002`, `order
  1002`) is answered from `data/orders.json` only. The LLM is never asked
  to state an order status, a tracking number, or a delivery date. An
  unknown order gets a clear "couldn't find" answer, not an error.
- **Product cards:** when the question or the retrieved chunks mention a
  product, the response includes its name, price, stock and variants, so
  the widget can render a card.

## The demo shop (seed data)

The repo ships with a fictional demo shop, **Northwind Outdoors**, so the
whole thing runs end to end without a real store:

- 12 products in `data/products.json` (including 2 out of stock on purpose)
- 8 orders in `data/orders.json` (delivered, shipped, out for delivery,
  processing, cancelled, and one unknown id for the not-found case)
- Knowledge base in `docs/`: `products.md`, `shipping.md`, `returns.md`,
  `faq.md`

To use it with a real shop, replace `data/` with the shop's catalog and
orders (from a Shopify export or the Shopify API) and `docs/` with the
shop's own policies — no code changes needed for the standard case.

## Eval results

`scripts/eval.py` asks 20 realistic customer questions and scores each
one: the expected fact must appear in the answer AND the expected
citation must be present. Run it offline (no key, no server) or against
a live API with `--url http://localhost:8000`.

```
Eval mode: offline (20 questions)
Accuracy: 20/20 = 100%
```

| Group | Questions | Passed |
|---|---|---|
| Order status (incl. unknown order) | 5 | 5 |
| Shipping | 4 | 4 |
| Returns | 4 | 4 |
| Products (incl. out-of-stock) | 5 | 5 |
| FAQ / contact | 2 | 2 |

Method note: the offline run measures retrieval + order lookup, the parts
this repo owns — order questions go through the real `order_answer` code,
RAG questions through a keyword retriever over the same chunked docs with
an extractive answer. The live run (`--url`) measures the full stack
including the LLM. Run the live eval after deploying with a real key.

## Production features

- **Validation:** Pydantic request/response models, question length limits,
  whitespace-only questions rejected (422 even without a key).
- **Provider abstraction:** one `LLMProvider` interface, OpenAI
  (`gpt-4o-mini`) and Gemini (`gemini-3.5-flash-lite`, REST via `urllib`,
  no extra SDK) behind it. Only one key is needed; OpenAI wins if both
  are set.
- **Retries + timeouts:** transient provider errors (429 / 5xx) are
  retried with exponential backoff, up to 3 attempts, 60s timeout.
  A 400 from the provider is not retried.
- **Observability:** structured JSON logs, `X-Request-ID` on every
  response, `/health` + `/ready`.
- **Safety:** rate limit 60 req/min per IP on `/chat` (429 with
  `Retry-After`), CORS allow-list via env, generic 500 handler that
  never leaks a stack trace, `.env` and `chroma_db/` git-ignored.
  Set `ADMIN_TOKEN` in production so `/admin/docs` requires a token.
- **Tests + CI:** `pytest` suite (15 tests) with fake LLM provider and
  fake retriever (no key, no Chroma needed); GitHub Actions runs
  compile check + tests + the offline eval on every push.
- **Deploy:** `Dockerfile` (Python 3.12-slim, non-root user,
  healthcheck) and `docker-compose.yml` with a persistent Chroma volume.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (priority if both set) |
| `GEMINI_API_KEY` | — | Gemini key, free tier at Google AI Studio |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini chat model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `CHROMA_PATH` | `./chroma_db` | Chroma persistent directory |
| `DATA_DIR` | `./data` | Products + orders JSON |
| `DOCS_DIR` | `./docs` | Knowledge-base Markdown |
| `ADMIN_TOKEN` | — | If set, `/admin/docs` requires `X-Admin-Token` |
| `RETRIEVAL_TOP_K` | `3` | Chunks retrieved per question |
| `MAX_QUESTION_LENGTH` | `2000` | Max question length (chars) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP limit on `/chat` |
| `LLM_TIMEOUT_SECONDS` | `60` | LLM request timeout |
| `LOG_LEVEL` | `INFO` | Log level |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed frontend origins |

Notes: `gemini-2.5-flash-lite` is no longer available to new users (404);
`gemini-3.8-flash` was temporarily overloaded (503) when tested on
24 Nov 2026, so the default is `gemini-3.5-flash-lite`.

## Project structure

```
main.py            # FastAPI app: settings, providers, retrieval, order tool, routes
ingest.py          # chunk + embed docs/ into Chroma
data/products.json # demo catalog (12 products)
data/orders.json   # demo orders (8 orders)
docs/              # knowledge base: products, shipping, returns, faq
scripts/eval.py    # 20-question eval (offline or live)
scripts/eval_cases.json
tests/test_api.py  # 15 API tests, no key needed
widget/            # Next.js embeddable chat widget + one-tag embed.js
Dockerfile / docker-compose.yml
.github/workflows/ci.yml
```

## Roadmap

1. Shopify app: pull catalog + orders live from the Shopify API instead
   of seed JSON, and sync docs from the shop's own policy pages.
2. Multi-shop: one deployment serving many stores, keyed by shop domain.
3. Human handoff: escalate to a real agent when confidence is low, with
   the conversation attached.
4. Reranking + hybrid search (BM25 + vector), evals with a real LLM judge.
