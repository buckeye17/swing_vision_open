/*
 * Forward shape edits (draw, move, resize, erase) and plot-area clicks of Plotly graphs
 * to dcc.Stores.
 *
 * Wrap the dcc.Graph in an element with data-shapes-store="<store id>". Every relayout
 * event that touches shapes is written to that store, so Dash sees it even when another
 * relayout (theme_sync.js re-colouring the axes) follows in the same tick and would
 * otherwise replace the graph's relayoutData before the callback runs.
 */
(function () {
    "use strict";

    function bind(host) {
        const gd = host.querySelector(".js-plotly-plot");
        if (!gd || !gd.on || gd.dataset.svShapesBound) return;
        gd.dataset.svShapesBound = "1";
        gd.on("plotly_relayout", (d) => {
            if (!d || !Object.keys(d).some((k) => k === "shapes" || k.startsWith("shapes["))) {
                return;
            }
            if (window.dash_clientside && window.dash_clientside.set_props) {
                window.dash_clientside.set_props(host.dataset.shapesStore, {
                    data: Object.assign({}, d, {_ts: Date.now()}),
                });
            }
        });
    }

    /*
     * Clicks anywhere on the plot area of a graph wrapped in data-click-store="<store id>"
     * are written to that store in data coordinates ({x, y, _ts}), not only clicks on points.
     */
    function bindClicks(host) {
        const gd = host.querySelector(".js-plotly-plot");
        if (!gd || gd.dataset.svClickBound) return;
        gd.dataset.svClickBound = "1";
        gd.addEventListener("click", (ev) => {
            const fl = gd._fullLayout;
            if (!fl || !fl.xaxis || !fl.yaxis) return;
            const box = gd.getBoundingClientRect();
            const px = ev.clientX - box.left - fl.xaxis._offset;
            const py = ev.clientY - box.top - fl.yaxis._offset;
            if (px < 0 || py < 0 || px > fl.xaxis._length || py > fl.yaxis._length) return;
            if (window.dash_clientside && window.dash_clientside.set_props) {
                window.dash_clientside.set_props(host.dataset.clickStore, {
                    data: {x: fl.xaxis.p2l(px), y: fl.yaxis.p2l(py), _ts: Date.now()},
                });
            }
        });
    }

    const scan = () => {
        document.querySelectorAll("[data-shapes-store]").forEach(bind);
        document.querySelectorAll("[data-click-store]").forEach(bindClicks);
    };
    new MutationObserver(scan).observe(document.documentElement, {childList: true, subtree: true});
    document.addEventListener("DOMContentLoaded", scan);
})();
