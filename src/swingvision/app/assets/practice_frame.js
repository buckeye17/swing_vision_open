/*
 * Practice page overlays, drawn by video_sync.js on every frame of the practice video
 * (data-frame="practice"): the targets for the side being hit to, and the shot label.
 *
 * Like review_frame.js, this reads the page's dcc.Stores and writes the DOM itself rather
 * than being a Dash clientside callback on a time store: each Dash update re-checks every
 * component on the page, and at 10 Hz that saturated the browser during playback.
 */
(function () {
    "use strict";
    const FULL = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;";

    function data(id) {
        const api = window.dash_component_api;
        const c = api && api.getLayout ? api.getLayout(id) : null;
        return c && c.props ? c.props.data : undefined;
    }

    function render(video) {
        const shots = data("pr-shots");
        const overlays = data("pr-overlays");
        const now = video.currentTime;
        // i: the shot playing now; last: the latest one started (its end keeps the targets on
        // the right half between shots).
        let i = -1, last = -1;
        if (shots) {
            for (let k = shots.t0.length - 1; k >= 0; k--) {
                if (shots.t0[k] <= now) { last = k; if (now <= shots.t1[k]) { i = k; } break; }
            }
        }
        const img = document.getElementById("pr-overlay");
        if (img) {
            const side = shots && shots.side.length ? shots.side[last >= 0 ? last : 0] : -1;
            const src = overlays ? (side === 1 ? overlays.far : overlays.near) : "";
            if (!src) {
                img.style.display = "none";
            } else {
                if (img.getAttribute("src") !== src) img.setAttribute("src", src);
                if (img.style.display !== "block") img.style.cssText = FULL + "display:block;";
            }
        }
        const label = document.getElementById("pr-label");
        if (!label) return;
        if (i < 0) { label.style.display = "none"; return; }
        const text = "Shot " + (i + 1) + " · " + shots.label[i];
        if (label.textContent !== text) label.textContent = text;
        label.style.cssText = "position:absolute;left:1%;top:1.5%;padding:2px 8px;" +
            "border-radius:4px;background:rgba(0,0,0,0.6);color:#fff;" +
            "font:600 14px system-ui,sans-serif;pointer-events:none;display:block;";
    }

    window.svFrame = window.svFrame || {};
    window.svFrame.practice = render;
})();
