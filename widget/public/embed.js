/**
 * Embed the Northwind support chat on any shop page with one tag:
 *
 *   <script src="https://chat.example.com/embed.js"
 *           data-widget-url="https://chat.example.com"
 *           data-api-url="https://api.example.com"
 *           async></script>
 *
 * It renders a floating button bottom-right; clicking it opens the
 * widget in an iframe. No build step, no dependency on the host page.
 */
(function () {
  var script = document.currentScript;
  if (!script) return;
  var widgetUrl = script.getAttribute("data-widget-url") || "";
  if (!widgetUrl) return;
  var apiUrl = script.getAttribute("data-api-url") || "";

  var open = false;
  var btn = document.createElement("button");
  btn.textContent = "💬 Support";
  btn.setAttribute("aria-label", "Open support chat");
  btn.style.cssText =
    "position:fixed;bottom:20px;right:20px;z-index:9999;" +
    "padding:10px 16px;border-radius:24px;border:none;cursor:pointer;" +
    "background:#111;color:#fff;font-size:14px;" +
    "box-shadow:0 4px 16px rgba(0,0,0,.25);";

  var frame = document.createElement("iframe");
  frame.src = widgetUrl + (apiUrl ? "?api=" + encodeURIComponent(apiUrl) : "");
  frame.title = "Shop support chat";
  frame.style.cssText =
    "position:fixed;bottom:80px;right:20px;z-index:9999;" +
    "width:420px;max-width:calc(100vw - 40px);height:560px;max-height:calc(100vh - 110px);" +
    "border:1px solid #ddd;border-radius:12px;background:#fff;display:none;" +
    "box-shadow:0 8px 32px rgba(0,0,0,.25);";

  btn.addEventListener("click", function () {
    open = !open;
    frame.style.display = open ? "block" : "none";
    btn.textContent = open ? "✕ Close" : "💬 Support";
  });

  document.body.appendChild(btn);
  document.body.appendChild(frame);
})();
