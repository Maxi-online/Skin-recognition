// Guard against noisy/unrelated browser extension polyfills (часто polyfill.js),
// которые кидают "Uncaught (in promise) Error: Failed to parse invalid URL \"\""
// и могут ронять clientside callbacks в Dash.
(function () {
  function shouldSuppress(msg) {
    return typeof msg === "string" && msg.indexOf("Failed to parse invalid URL") !== -1;
  }

  try {
    // 1) Unhandled Promise rejection
    window.addEventListener(
      "unhandledrejection",
      function (e) {
        const r = e && e.reason;
        const msg = (r && (r.message || String(r))) || "";
        if (shouldSuppress(msg)) {
          e.preventDefault();
        }
      },
      true
    );
  } catch (_) {}

  try {
    // 2) Regular error event
    window.addEventListener(
      "error",
      function (e) {
        const msg = (e && e.message) ? String(e.message) : "";
        if (shouldSuppress(msg)) {
          e.preventDefault();
          return false;
        }
        return undefined;
      },
      true
    );
  } catch (_) {}
})();

