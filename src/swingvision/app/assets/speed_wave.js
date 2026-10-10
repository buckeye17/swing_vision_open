/*
 * Calibrate → Speed tab: a drag of a waveform's onset line (shapes[0]) goes to the spd-drag
 * store as {sound, x (ms), n}. Reading relayoutData instead would miss drags: the theme sync
 * (theme_sync.js) relayouts the chart right after, replacing relayoutData before the
 * callback reads it.
 */
(function () {
    "use strict";
    const GRAPHS = {"spd-wave-racket": "racket", "spd-wave-tape": "tape"};

    function bind() {
        Object.keys(GRAPHS).forEach((id) => {
            const host = document.getElementById(id);
            const gd = host && host.querySelector(".js-plotly-plot");
            if (!gd || !gd.on || gd.dataset.svDragBound) return;
            gd.dataset.svDragBound = "1";
            gd.on("plotly_relayout", (d) => {
                if (!d || d["shapes[0].x0"] === undefined) return;
                if (!window.dash_clientside || !window.dash_clientside.set_props) return;
                window.dash_clientside.set_props("spd-drag", {
                    data: {sound: GRAPHS[id], x: Number(d["shapes[0].x0"]), n: Date.now()},
                });
            });
        });
    }

    new MutationObserver(bind).observe(document.documentElement, {childList: true, subtree: true});
})();
