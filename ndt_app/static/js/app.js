/* ==========================================================================
   App-wide navigation: each main tab has its own address (#individual,
   #combined, #dt-only), so refresh keeps the tab, Back/Forward move between
   tabs, and a tab can be linked to directly.
   ========================================================================== */
(function () {
  const ROUTES = { home: "home-tab", individual: "indiv-tab", combined: "combined-tab", "dt-only": "dtonly-tab" };
  const byTab = Object.fromEntries(Object.entries(ROUTES).map(([h, id]) => [id, h]));
  let fromHistory = false;

  function showTab(id) {
    const trigger = document.getElementById(id);
    if (!trigger) return;
    bootstrap.Tab.getOrCreateInstance(trigger).show();
    const dd = trigger.closest(".dropdown")?.querySelector(".dropdown-toggle");
    if (dd) bootstrap.Dropdown.getOrCreateInstance(dd).hide();
  }
  const routeFromHash = () => ROUTES[location.hash.slice(1)] || "home-tab";

  document.addEventListener("DOMContentLoaded", () => {
    // main navigation only (the result windows have their own tabs)
    Object.values(ROUTES).forEach(id => document.getElementById(id)?.addEventListener("shown.bs.tab", () => {
      const hash = "#" + byTab[id];
      if (!fromHistory && location.hash !== hash) {
        if (id === "home-tab" && !location.hash) history.replaceState(null, "", hash);
        else history.pushState(null, "", hash);
      }
      fromHistory = false;
      // mark the dropdown as active when one of its tabs is open
      document.querySelectorAll("#mainTabs .dropdown-toggle").forEach(t =>
        t.classList.toggle("active", !!t.closest(".dropdown").querySelector(".dropdown-item.active")));
    }));

    // Back / Forward: close any open result window, then show the tab
    window.addEventListener("popstate", () => {
      document.querySelectorAll(".modal.show").forEach(m => bootstrap.Modal.getInstance(m)?.hide());
      fromHistory = true;
      const id = routeFromHash();
      if (document.getElementById(id)?.classList.contains("active")) fromHistory = false;
      showTab(id);
      window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    });

    // Buttons and links that open a tab (Home tab cards, logo)
    document.querySelectorAll("[data-open-tab]").forEach(el => el.addEventListener("click", e => {
      e.preventDefault();
      showTab(el.dataset.openTab);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }));

    // Start on the tab in the address, at the top of the page
    if ("scrollRestoration" in history) history.scrollRestoration = "manual";
    const start = routeFromHash();
    if (start !== "home-tab") { fromHistory = true; showTab(start); }
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  });
})();
