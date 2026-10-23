// light / dark switch. without a saved choice the page follows the system setting
(() => {
  const root = document.documentElement;
  const button = document.querySelector(".theme-toggle");
  const system = matchMedia("(prefers-color-scheme: dark)");
  const hlLight = document.getElementById("hl-light");
  const hlDark = document.getElementById("hl-dark");

  const current = () => root.dataset.theme || (system.matches ? "dark" : "light");

  function apply() {
    const theme = current();
    // code highlighting: follow the chosen theme, or the system when nothing is chosen
    hlLight.media = root.dataset.theme ? (theme === "light" ? "all" : "not all") : "(prefers-color-scheme: light)";
    hlDark.media = root.dataset.theme ? (theme === "dark" ? "all" : "not all") : "(prefers-color-scheme: dark)";
    button.textContent = theme === "dark" ? "Light" : "Dark";
    // the canvases read their colours from CSS, so they redraw on this event
    window.dispatchEvent(new Event("themechange"));
  }

  button.addEventListener("click", () => {
    root.dataset.theme = current() === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("theme", root.dataset.theme);
    } catch (e) {}
    apply();
  });

  system.addEventListener("change", apply);
  apply();
})();
