# Support chat widget

A small Next.js app that talks to the shop support agent API and can be
embedded on any store page with one script tag.

## Run locally

```bash
cd widget
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# open http://localhost:3000
```

## Embed on a shop

```html
<script src="https://chat.example.com/embed.js"
        data-widget-url="https://chat.example.com"
        data-api-url="https://api.example.com"
        async></script>
```

`embed.js` renders a floating "Support" button bottom-right; clicking it
opens the widget in an iframe. The widget URL is where this Next.js app
is deployed (e.g. Vercel); the API URL is where the FastAPI backend runs.

## What it shows

- Chat answers from the API, with product cards whenever the answer
  cites a product (name, price, stock, variants).
- Order status line when the answer is about an order.
- Four one-tap suggestion chips for the most common questions.
- A graceful fallback message when the API is unreachable.

`NEXT_PUBLIC_API_URL` sets the API base URL at build time. For a
multi-shop setup, pass the API URL as a `?api=` query parameter on the
widget URL instead — `embed.js` appends it automatically.
