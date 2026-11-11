// Keep the left navigation in place when loading another documentation page.
document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.querySelector(".sidebar-scroll");
  if (!sidebar) return;

  // Separate documentation roots (including RTD versions) and browser tabs.
  const root = new URL(document.documentElement.dataset.content_root || "./", location.href);
  const key = `gdpx:sidebar-scroll:${root.pathname}`;

  try {
    const saved = sessionStorage.getItem(key);
    if (saved !== null && Number.isFinite(Number(saved))) {
      sidebar.scrollTo({ top: Number(saved), behavior: "instant" });
    }
  } catch {
    // Navigation must also work when browser storage is disabled.
  }

  window.addEventListener("pagehide", () => {
    try {
      sessionStorage.setItem(key, String(sidebar.scrollTop));
    } catch {
      // Storage is optional.
    }
  });
});
