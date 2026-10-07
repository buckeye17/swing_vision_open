/*
 * Keep every Plotly chart's text/axis colours in step with the Mantine colour scheme.
 * Figures use transparent backgrounds; only foreground colours need switching.
 */
(function () {
    "use strict";

    function colors() {
        const dark = document.documentElement.getAttribute("data-mantine-color-scheme") === "dark";
        return dark
            ? {text: "#e9edeb", axis: "#9ba5a0", grid: "rgba(233,237,235,0.07)"}
            : {text: "#1c2220", axis: "#5c6662", grid: "rgba(14,17,16,0.08)"};
    }

    const FONT = "Barlow, system-ui, -apple-system, Segoe UI, sans-serif";

    function apply(gd) {
        if (!window.Plotly || !gd || !gd.layout) return;
        const c = colors();
        // Idempotent: relayout fires plotly_afterplot, which calls apply() again.
        const font = gd.layout.font || {};
        const xaxis = gd.layout.xaxis || {};
        if (font.color === c.text && font.family === FONT && xaxis.color === c.axis) return;
        window.Plotly.relayout(gd, {
            "font.color": c.text,
            "font.family": FONT,
            "xaxis.color": c.axis, "yaxis.color": c.axis,
            "xaxis.gridcolor": c.grid, "yaxis.gridcolor": c.grid,
        });
    }

    function applyAll() {
        document.querySelectorAll(".js-plotly-plot").forEach((gd) => {
            if (!gd.dataset.svThemeBound && gd.on) {
                gd.dataset.svThemeBound = "1";
                // Dash re-renders figures (new data) → re-apply after each full plot.
                gd.on("plotly_afterplot", () => apply(gd));
            }
            apply(gd);
        });
    }

    new MutationObserver(applyAll).observe(document.documentElement, {attributes: true, attributeFilter: ["data-mantine-color-scheme"]});

    new MutationObserver(applyAll).observe(document.documentElement, {childList: true, subtree: true});
})();
