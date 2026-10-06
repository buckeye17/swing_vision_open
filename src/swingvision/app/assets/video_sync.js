/*
 * Video ⇄ Dash sync for <video data-sv-player> elements.
 *
 * - Publishes {t, paused, duration} to the dcc.Store named in data-time-store
 *   (throttled to ~10 Hz while playing, immediately on seek/pause).
 * - Moves the cursor (shapes[0]) of the dcc.Graph named in data-timeline.
 * - Keyboard: Space play/pause, J/L ±5 s, K pause, ←/→ one frame, Shift+←/→ 1 s,
 *   N/P next/previous segment when the page put segment start times (a JSON list) in the
 *   video's data-segments attribute.
 */
(function () {
    "use strict";
    const PUBLISH_MS = 100;

    function timelineDiv(video) {
        const id = video.dataset.timeline;
        if (!id) return null;
        const host = document.getElementById(id);
        return host ? host.querySelector(".js-plotly-plot") : null;
    }

    function moveCursor(video) {
        const gd = timelineDiv(video);
        if (!gd || !window.Plotly || !gd.layout || !gd.layout.shapes) return;
        const t = video.currentTime;
        window.Plotly.relayout(gd, {"shapes[0].x0": t, "shapes[0].x1": t});
    }

    function bind(video) {
        if (video.dataset.svBound) return;
        video.dataset.svBound = "1";
        let last = 0;

        const publish = (force) => {
            const now = performance.now();
            if (!force && now - last < PUBLISH_MS) return;
            last = now;
            const store = video.dataset.timeStore;
            if (store && window.dash_clientside && window.dash_clientside.set_props) {
                window.dash_clientside.set_props(store, {
                    data: {t: video.currentTime, paused: video.paused, duration: video.duration},
                });
            }
            moveCursor(video);
        };

        const loop = () => {
            if (!document.body.contains(video) || video.paused) return;
            publish(false);
            requestAnimationFrame(loop);
        };

        video.addEventListener("play", () => requestAnimationFrame(loop));
        // rAF is throttled in hidden/background views; timeupdate (~4 Hz) keeps things moving.
        video.addEventListener("timeupdate", () => publish(false));
        ["seeked", "pause", "loadedmetadata"].forEach((ev) =>
            video.addEventListener(ev, () => publish(true)));
    }

    function activeVideo() {
        return document.querySelector("video[data-sv-player]");
    }

    document.addEventListener("keydown", (e) => {
        const tag = (e.target && e.target.tagName) || "";
        if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.target.isContentEditable) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const v = activeVideo();
        if (!v) return;
        const fps = parseFloat(v.dataset.fps) || 60;
        let handled = true;
        switch (e.key) {
            case " ":
                if (tag === "VIDEO") { handled = false; break; }  // native controls handle it
                v.paused ? v.play() : v.pause();
                break;
            case "k": case "K": v.pause(); break;
            case "j": case "J": v.currentTime = Math.max(0, v.currentTime - 5); break;
            case "l": case "L": v.currentTime = Math.min(v.duration || 1e9, v.currentTime + 5); break;
            case "ArrowLeft":
                v.pause();
                v.currentTime = Math.max(0, v.currentTime - (e.shiftKey ? 1 : 1 / fps));
                break;
            case "ArrowRight":
                v.pause();
                v.currentTime = Math.min(v.duration || 1e9, v.currentTime + (e.shiftKey ? 1 : 1 / fps));
                break;
            case "n": case "N": case "p": case "P": {
                let starts = [];
                try { starts = JSON.parse(v.dataset.segments || "[]"); } catch (err) { starts = []; }
                const now = v.currentTime;
                const next = e.key.toLowerCase() === "n";
                // "Previous" skips back past the segment we're in when we're well into it.
                const t = next ? starts.find((s) => s > now + 0.25)
                               : [...starts].reverse().find((s) => s < now - 1.0);
                if (t === undefined) { handled = false; break; }
                v.currentTime = t;
                v.play();
                break;
            }
            default: handled = false;
        }
        if (handled) e.preventDefault();
    });

    const scan = () => document.querySelectorAll("video[data-sv-player]").forEach(bind);
    new MutationObserver(scan).observe(document.documentElement, {childList: true, subtree: true});
    document.addEventListener("DOMContentLoaded", scan);
})();
