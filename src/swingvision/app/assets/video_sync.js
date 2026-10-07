/*
 * Playback sync for <video data-sv-player> elements: overlays, timeline cursor, keys.
 *
 * - On every frame, calls window.svFrame[name](video, force) for the renderer named in
 *   data-frame (the page's overlays, e.g. review_frame.js), and moves a cursor line over the
 *   dcc.Graph named in data-timeline. svFrameRefresh(videoId) redraws one now (for a page's
 *   toggles and late-arriving data while paused).
 *   Playback time deliberately never goes through a Dash store: each Dash update re-checks
 *   every component on the page, and at 10 Hz on a long session that stalled the video.
 * - Keyboard: Space play/pause, J/L ±5 s, K pause, ←/→ one frame, Shift+←/→ 1 s,
 *   N/P next/previous segment when the page put segment start times (a JSON list) in the
 *   video's data-segments attribute.
 */
(function () {
    "use strict";

    function timelineDiv(video) {
        const id = video.dataset.timeline;
        if (!id) return null;
        const host = document.getElementById(id);
        return host ? host.querySelector(".js-plotly-plot") : null;
    }

    // The cursor is a plain div over the plot, placed with the x axis's own mapping. A
    // Plotly.relayout of a shape redraws the whole timeline (~1 s on a long session), which
    // at 10 Hz starved the video until playback froze.
    function moveCursor(video) {
        const gd = timelineDiv(video);
        const fl = gd && gd._fullLayout;
        const xa = fl && fl.xaxis;
        if (!xa || !xa.l2p || !fl._size) return;
        gd._svVideo = video;
        if (!gd.dataset.svCursorBound && gd.on) {
            gd.dataset.svCursorBound = "1";
            // Zoom, pan and resize move the axis: put the cursor back where the video is.
            const follow = () => { if (gd._svVideo) moveCursor(gd._svVideo); };
            ["plotly_afterplot", "plotly_relayout", "plotly_relayouting"].forEach((ev) =>
                gd.on(ev, follow));
        }
        let line = gd.querySelector(":scope > .sv-cursor");
        if (!line) {
            if (getComputedStyle(gd).position === "static") gd.style.position = "relative";
            line = document.createElement("div");
            line.className = "sv-cursor";
            gd.appendChild(line);
        }
        const x = xa.l2p(video.currentTime);
        line.style.display = x >= 0 && x <= xa._length ? "block" : "none";
        line.style.left = `${xa._offset + x}px`;
        line.style.top = `${fl._size.t}px`;
        line.style.height = `${fl._size.h}px`;
    }

    function frame(video, force) {
        const name = video.dataset.frame;
        const render = name && window.svFrame && window.svFrame[name];
        if (render) render(video, force);
        moveCursor(video);
    }

    window.svFrameRefresh = (id) => {
        const video = document.getElementById(id);
        if (video) frame(video, true);
    };

    function bind(video) {
        if (video.dataset.svBound) return;
        video.dataset.svBound = "1";
        const tick = (force) => frame(video, force);

        const loop = () => {
            if (!document.body.contains(video) || video.paused) return;
            tick(false);
            requestAnimationFrame(loop);
        };

        video.addEventListener("play", () => requestAnimationFrame(loop));
        // rAF is throttled in hidden/background views; timeupdate (~4 Hz) keeps things moving.
        video.addEventListener("timeupdate", () => tick(false));
        ["seeked", "pause", "loadedmetadata"].forEach((ev) =>
            video.addEventListener(ev, () => tick(true)));
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
