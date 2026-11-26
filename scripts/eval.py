"""Eval: 20 realistic customer questions, scored for real.

Usage:
    python scripts/eval.py                 # offline mode (no key, no server)
    python scripts/eval.py --url http://localhost:8000   # live API mode

Scoring (per question, 1 point each):
- the expected fact appears in the answer, AND
- the expected citation is present (order questions cite orders/<id>,
  RAG questions cite the right doc; the unknown-order case expects no
  citation).

Offline mode runs the same code path as the API minus the LLM and the
vector store: order questions go through ``main.order_answer`` and RAG
questions through a keyword retriever over the same chunked docs plus
an extractive answer (the best-matching chunk text). That measures
retrieval + order lookup — the parts this repo owns — without an API
key. Live mode measures the full stack including the LLM.

Exit code is 0 only when every question passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def keyword_retrieve(chunks, question: str, top_k: int = 3):
    """Rank chunks by token overlap with the question (offline stand-in
    for Chroma retrieval)."""
    qtokens = _tokens(question)
    scored = []
    for i, (source, chunk_index, text) in enumerate(chunks):
        overlap = len(qtokens & _tokens(text))
        scored.append((overlap, -i, source, chunk_index, text))
    scored.sort(reverse=True)
    return [
        {"source": s, "chunk": c, "text": t}
        for overlap, _neg, s, c, t in scored[:top_k]
        if overlap > 0
    ]


def run_offline(cases):
    import main
    from ingest import chunk_documents, load_documents

    main.settings.data_dir = str(ROOT / "data")
    main._shop_data.clear()
    docs = load_documents(ROOT / "docs")
    chunks = chunk_documents(docs)
    results = []
    for case in cases:
        q = case["question"]
        order_id = main.extract_order_id(q)
        if order_id is not None:
            order = main.get_order(order_id)
            if order is None:
                answer = (
                    f"I couldn't find order #{order_id}. Please double-check "
                    "the order number from your confirmation email, or email "
                    "support@northwind-outdoors.example and we will look it up."
                )
                citations = []
            else:
                answer = main.order_answer(order)
                citations = [{"source": f"orders/{order.id}", "chunk": 0}]
        else:
            top = keyword_retrieve(chunks, q)
            # Answer from all retrieved chunks, like the API does
            # (RETRIEVAL_TOP_K=3) — the LLM sees the same top-3.
            answer = "\n".join(c["text"] for c in top)
            citations = top
        results.append({"case": case, "answer": answer, "citations": citations})
    return results


def run_live(cases, base_url: str):
    import httpx

    results = []
    with httpx.Client(base_url=base_url, timeout=60) as client:
        for case in cases:
            resp = client.post("/chat", json={"question": case["question"]})
            resp.raise_for_status()
            body = resp.json()
            results.append(
                {"case": case, "answer": body["answer"], "citations": body["citations"]}
            )
    return results


def score(results):
    rows = []
    correct = 0
    for r in results:
        case = r["case"]
        answer = r["answer"]
        sources = [c["source"] for c in r["citations"]]
        fact_ok = case["expected"].lower() in answer.lower()
        if case["citation"] is None:
            cite_ok = len(sources) == 0
        else:
            cite_ok = case["citation"] in sources
        passed = fact_ok and cite_ok
        correct += int(passed)
        rows.append(
            {
                "question": case["question"],
                "expected": case["expected"],
                "citation": case["citation"],
                "fact_ok": fact_ok,
                "cite_ok": cite_ok,
                "passed": passed,
            }
        )
    return rows, correct


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 20-question eval.")
    parser.add_argument("--url", default=None, help="Live API base URL (offline if omitted).")
    parser.add_argument("--cases", default=str(ROOT / "scripts" / "eval_cases.json"))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    mode = "live" if args.url else "offline"
    print(f"Eval mode: {mode} ({len(cases)} questions)")
    results = run_live(cases, args.url) if args.url else run_offline(cases)
    rows, correct = score(results)
    print(f"{'Q':<4}{'PASS':<6}{'Fact':<6}{'Cite':<6}Question")
    for i, row in enumerate(rows, 1):
        print(
            f"{i:<4}{'PASS' if row['passed'] else 'FAIL':<6}"
            f"{'ok' if row['fact_ok'] else 'MISS':<6}"
            f"{'ok' if row['cite_ok'] else 'MISS':<6}{row['question']}"
        )
        if not row["passed"]:
            print(f"      expected fact: {row['expected']!r} | citation: {row['citation']!r}")
    print(f"\nAccuracy: {correct}/{len(cases)} = {100 * correct / len(cases):.0f}%")
    return 0 if correct == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
