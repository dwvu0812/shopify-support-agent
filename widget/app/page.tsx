"use client";

import { useRef, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Product = {
  id: string;
  name: string;
  price: number;
  stock: number;
  variants: string[];
};

type Message = {
  role: "user" | "assistant";
  text: string;
  products?: Product[];
  orderStatus?: string | null;
};

const SUGGESTIONS = [
  "Where is my order #1002?",
  "How long does standard shipping take?",
  "How do I return something?",
  "Is the Alpine Merino Base Layer in stock?",
];

export default function ChatWidget() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Hi! I'm the Northwind Outdoors support assistant. Ask me about your order, shipping, returns, or any product.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }]);
    try {
      const resp = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: body.answer,
          products: body.products ?? [],
          orderStatus: body.order ? body.order.status : null,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "Sorry, I couldn't reach support right now. Please email support@northwind-outdoors.example.",
        },
      ]);
    } finally {
      setBusy(false);
      listRef.current?.scrollTo({ top: 99999 });
    }
  }

  return (
    <main style={{ maxWidth: 420, margin: "0 auto", padding: 16 }}>
      <h1 style={{ fontSize: 18, margin: "0 0 4px" }}>
        Northwind Outdoors — Support
      </h1>
      <p style={{ fontSize: 13, color: "#555", margin: "0 0 12px" }}>
        Answers come from the shop&apos;s own documents and real order
        data — never guessed.
      </p>

      <div
        ref={listRef}
        style={{
          border: "1px solid #ddd",
          borderRadius: 8,
          height: 420,
          overflowY: "auto",
          padding: 12,
        }}
      >
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <div
              style={{
                background: m.role === "user" ? "#e8f0fe" : "#f5f5f5",
                borderRadius: 8,
                padding: "8px 10px",
                whiteSpace: "pre-wrap",
                fontSize: 14,
              }}
            >
              {m.text}
            </div>
            {m.orderStatus && (
              <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>
                Order status: {m.orderStatus.replaceAll("_", " ")}
              </div>
            )}
            {(m.products ?? []).map((p) => (
              <div
                key={p.id}
                style={{
                  border: "1px solid #ddd",
                  borderRadius: 8,
                  padding: "8px 10px",
                  marginTop: 6,
                  fontSize: 13,
                }}
              >
                <strong>{p.name}</strong>
                <br />${p.price.toFixed(2)} ·{" "}
                {p.stock > 0 ? `${p.stock} in stock` : "Out of stock"}
                <br />
                <span style={{ color: "#555" }}>
                  {p.variants.join(", ")}
                </span>
              </div>
            ))}
          </div>
        ))}
        {busy && <div style={{ fontSize: 13, color: "#888" }}>Typing…</div>}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => send(s)}
            style={{
              fontSize: 12,
              padding: "4px 8px",
              borderRadius: 12,
              border: "1px solid #ccc",
              background: "#fff",
              cursor: "pointer",
            }}
          >
            {s}
          </button>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        style={{ display: "flex", gap: 8, marginTop: 10 }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your order or a product…"
          style={{
            flex: 1,
            padding: "8px 10px",
            borderRadius: 8,
            border: "1px solid #ccc",
            fontSize: 14,
          }}
        />
        <button
          type="submit"
          disabled={busy}
          style={{
            padding: "8px 14px",
            borderRadius: 8,
            border: "none",
            background: "#111",
            color: "#fff",
            cursor: "pointer",
            fontSize: 14,
          }}
        >
          Send
        </button>
      </form>
    </main>
  );
}
