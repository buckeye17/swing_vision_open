/*
 * Session review overlays, drawn by video_sync.js on every frame of the review video
 * (data-frame="review"): time readout, court overlay, player box, ball, shot path and label,
 * skeleton, and the minimap's trail, bounces and shot landing.
 *
 * These read the page's dcc.Stores (dash_component_api.getLayout) and write the DOM
 * themselves instead of being Dash clientside callbacks on a time store: every store write
 * costs ~30 Dash dispatches, and each dispatch re-checks every component on the page (some
 * 5,000 on a long session's review), so a 10 Hz time store stalled playback. Dash only hears
 * about the current shot (review-shot-id) and skeleton block (review-skel-key), when they
 * change. Settings come from the video's data-frame-config (JSON):
 * {linger, pathColor, skelBlock, edges}.
 */
(function () {
    "use strict";
    const MINIMAP_MS = 100;  // a minimap restyle costs ~20 ms: at most 10 a second
    const FULL = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;";

    const el = (id) => document.getElementById(id);
    const on = (id) => { const e = el(id); return !!(e && e.checked); };

    function data(id) {
        const api = window.dash_component_api;
        const c = api && api.getLayout ? api.getLayout(id) : null;
        return c && c.props ? c.props.data : undefined;
    }

    function setText(id, text) {
        const e = el(id);
        if (e && e.textContent !== text) e.textContent = text;
    }

    function setImg(id, src) {
        const e = el(id);
        if (!e) return;
        if (!src) { e.style.display = "none"; return; }
        if (e.getAttribute("src") !== src) e.setAttribute("src", src);
        if (e.style.display !== "block") e.style.cssText = FULL + "display:block;";
    }

    // Index of the last sample at or before now (-1: none).
    function lastAt(ts, now) {
        let lo = 0, hi = ts.length - 1, i = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (ts[mid] <= now + 1e-3) { i = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        return i;
    }

    function minimap() {
        const host = el("review-minimap");
        const gd = host ? host.querySelector(".js-plotly-plot") : null;
        return gd && window.Plotly && gd.data ? gd : null;
    }

    // Restyle minimap traces only when what they show changed (keyed per figure).
    function restyle(gd, slot, key, update, traces) {
        if (gd._svKeys === undefined || gd._svKeysData !== gd.data) {
            gd._svKeys = {};
            gd._svKeysData = gd.data;
        }
        if (gd._svKeys[slot] === key) return;
        gd._svKeys[slot] = key;
        window.Plotly.restyle(gd, update, traces);
    }

    function readout(video, now) {
        const fps = parseFloat(video.dataset.fps) || 60;
        const m = Math.floor(now / 60);
        const sec = (now - m * 60).toFixed(3).padStart(6, "0");
        setText("review-readout", `${String(m).padStart(2, "0")}:${sec} · frame ≈` +
            `${Math.round(now * fps)}` + (video.paused ? " · paused" : ""));
    }

    function courtOverlay(now) {
        const overlays = data("review-overlays");
        if (!overlays || !on("review-overlay-on")) { setImg("review-overlay", ""); return; }
        let src = overlays.main;
        for (const w of overlays.windows || []) {
            if (now >= w.t0 && now < w.t1) { src = w.src; break; }
        }
        setImg("review-overlay", src);
    }

    function player(now, gd) {
        const track = data("review-track");
        const box = el("review-box");
        let live = false, i = -1;
        if (track && track.t.length) {
            i = lastAt(track.t, now);
            live = i >= 0 && now - track.t[i] <= 0.15;
        }
        if (gd && gd.data.length >= 2) {
            const tx = [], ty = [];
            let px = [], py = [];
            if (live) {
                px = [track.x[i]]; py = [track.y[i]];
                for (let k = i; k >= 0 && track.t[k] >= now - 3 && track.run[k] === track.run[i];
                     k--) {
                    tx.push(track.x[k]); ty.push(track.y[k]);
                }
            }
            restyle(gd, "trail", live ? i : -1, {x: [tx, px], y: [ty, py]}, [0, 1]);
        }
        let label = "not tracked";
        if (live) {
            let v = 0;
            if (i > 0 && track.run[i - 1] === track.run[i]) {
                const dt = track.t[i] - track.t[i - 1];
                v = Math.hypot(track.x[i] - track.x[i - 1], track.y[i] - track.y[i - 1]) / dt;
            }
            label = "x " + track.x[i].toFixed(1) + " y " + track.y[i].toFixed(1) + " m · " +
                    (v * 3.6).toFixed(1) + " km/h" + (track.interp[i] ? " · bridged" : "");
        }
        if (track && track.t.length) setText("review-pos", label);
        if (!box) return;
        if (!live || !on("review-box-on")) { box.style.display = "none"; return; }
        const b = track.b[i];
        box.style.cssText = "position:absolute;pointer-events:none;box-sizing:border-box;" +
            `left:${b[0]}%;top:${b[1]}%;width:${b[2] - b[0]}%;height:${b[3] - b[1]}%;` +
            `border:2px ${track.interp[i] ? "dashed" : "solid"} #ffd43b;border-radius:3px;`;
    }

    function ball(now, gd) {
        const events = data("review-events");
        if (events && gd && gd.data.length >= 3) {
            const bx = [], by = [], ks = [];
            for (let k = 0; k < events.t.length; k++) {
                if (events.kind[k] === "bounce" && events.cx[k] !== null &&
                    events.t[k] <= now && events.t[k] >= now - 3) {
                    bx.push(events.cx[k]); by.push(events.cy[k]); ks.push(k);
                }
            }
            restyle(gd, "bounces", ks.join(","), {x: [bx], y: [by]}, [2]);
        }
        const track = data("review-ball-store");
        if (!on("review-ball-on") || !track || !track.t.length) { setImg("review-ball", ""); return; }
        const ts = track.t;
        const i = lastAt(ts, now);
        if (i < 0 || now - ts[i] > 0.1) { setImg("review-ball", ""); return; }
        const pts = [];
        for (let k = i; k >= 0 && ts[k] >= now - 0.4; k--) {
            if (k < i && ts[k + 1] - ts[k] > 0.1) { break; }
            pts.push(track.x[k] + "," + track.y[k]);
        }
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" ' +
            'preserveAspectRatio="none"><polyline points="' + pts.join(" ") +
            '" fill="none" stroke="#d8f5a2" stroke-opacity="0.7" stroke-width="2" ' +
            'vector-effect="non-scaling-stroke"/><circle cx="' + track.x[i] + '" cy="' +
            track.y[i] + '" r="0.6" fill="none" stroke="#e8590c" stroke-width="2" ' +
            'vector-effect="non-scaling-stroke"/></svg>';
        setImg("review-ball", "data:image/svg+xml;utf8," + encodeURIComponent(svg));
    }

    function shot(now, gd, cfg, state) {
        const shots = data("review-shots");
        let i = -1;
        if (shots) {
            for (let k = shots.t0.length - 1; k >= 0; k--) {
                if (shots.t0[k] <= now + 0.05) {
                    if (now <= shots.t1[k] + cfg.linger) { i = k; }
                    break;
                }
            }
        }
        const id = i >= 0 ? shots.id[i] : null;
        if (gd && gd.data.length >= 4) {
            const lx = [], ly = [], txt = [];
            if (i >= 0 && shots.cx[i] !== null) {
                lx.push(shots.cx[i]); ly.push(shots.cy[i]);
                txt.push(shots.v[i] !== null ? shots.v[i] + " km/h" : "");
            }
            restyle(gd, "landing", id, {x: [lx], y: [ly], text: [txt]}, [3]);
        }
        if (id !== state.shotId) {
            state.shotId = id;  // the server fills in the shot's detail card and side view
            window.dash_clientside.set_props("review-shot-id", {data: id});
        }
        const label = el("review-shot-label");
        if (i < 0) {
            setImg("review-shotpath", "");
            if (label) label.style.display = "none";
            return;
        }
        if (label) {
            setText("review-shot-label", (shots.v[i] !== null ? shots.v[i] +
                (shots.e[i] !== null ? " ± " + shots.e[i] : "") + " km/h" : "speed ?") +
                " · " + shots.o[i].replace("_", " "));
            label.style.cssText = "position:absolute;left:1%;top:1.5%;padding:2px 8px;" +
                "border-radius:4px;background:rgba(0,0,0,0.6);color:#fff;" +
                "font:600 14px system-ui,sans-serif;pointer-events:none;display:block;";
        }
        const path = shots.path[i];
        if (!on("review-shot-on") || !path.length) { setImg("review-shotpath", ""); return; }
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" ' +
            'preserveAspectRatio="none"><polyline points="' +
            path.map((p) => p[0] + "," + p[1]).join(" ") + '" fill="none" stroke="' +
            cfg.pathColor + '" stroke-opacity="0.9" stroke-width="2" stroke-dasharray="6 3" ' +
            'vector-effect="non-scaling-stroke"/></svg>';
        setImg("review-shotpath", "data:image/svg+xml;utf8," + encodeURIComponent(svg));
    }

    // The server sends the pose of the block (cfg.skelBlock seconds) playback is in.
    function skeleton(now, cfg, state) {
        const show = on("review-skel-on");
        const key = show ? Math.floor(now / cfg.skelBlock) : null;
        if (key !== state.skelKey) {
            state.skelKey = key;
            window.dash_clientside.set_props("review-skel-key", {data: key});
        }
        setImg("review-skel-img", show ? window.svSkeletonSrc(data("review-skel"), now, cfg.edges)
                                       : "");
    }

    function render(video, force) {
        if (!window.dash_clientside || !window.dash_clientside.set_props) return;
        let state = video._svReview;
        if (!state) {
            let cfg = {};
            try { cfg = JSON.parse(video.dataset.frameConfig || "{}"); } catch (err) { cfg = {}; }
            state = video._svReview = {cfg, shotId: null, skelKey: null, minimapAt: 0};
        }
        const cfg = state.cfg;
        const now = video.currentTime;
        readout(video, now);
        courtOverlay(now);
        const t = performance.now();
        const gd = force || t - state.minimapAt >= MINIMAP_MS ? minimap() : null;
        if (gd) state.minimapAt = t;
        player(now, gd);
        ball(now, gd);
        shot(now, gd, cfg, state);
        skeleton(now, cfg, state);
    }

    window.svFrame = window.svFrame || {};
    window.svFrame.review = render;
})();
