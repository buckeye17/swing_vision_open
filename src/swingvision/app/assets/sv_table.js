/*
 * Read-only tables rendered in the browser from one Dash component: <div data-sv-table='…'>.
 *
 * A long table built from Dash components costs one component per cell, and every Dash
 * update re-checks every component on the page (a session's 600 shots made each update take
 * ~0.4 s). The JSON spec:
 *   {head: [{label, align?}], rows: [{t?, cells: [text | {badge, color}]}], video?}
 * Clicking a row that has `t` seeks the video with id `video` there and plays it.
 */
(function () {
    "use strict";

    function cell(tag, value, align) {
        const td = document.createElement(tag);
        if (align) td.style.textAlign = align;
        if (value && typeof value === "object") {
            const b = document.createElement("span");
            b.className = "sv-badge";
            b.textContent = value.badge;
            b.style.color = value.color;
            b.style.background = value.color + "26";  // ~15% alpha, like a light badge
            td.appendChild(b);
        } else {
            td.textContent = value === null || value === undefined ? "" : String(value);
        }
        return td;
    }

    function render(host) {
        let spec;
        try { spec = JSON.parse(host.dataset.svTable); } catch (err) { return; }
        host.dataset.svRendered = host.dataset.svTable.length;
        const table = document.createElement("table");
        table.className = "sv-table";
        const head = table.createTHead().insertRow();
        const aligns = spec.head.map((h) => h.align || "");
        spec.head.forEach((h, k) => head.appendChild(cell("th", h.label, aligns[k])));
        const body = table.createTBody();
        for (const r of spec.rows) {
            const tr = body.insertRow();
            if (r.t !== undefined && r.t !== null) {
                tr.dataset.t = r.t;
                tr.className = "sv-clickable";
            }
            r.cells.forEach((c, k) => tr.appendChild(cell("td", c, aligns[k])));
        }
        host.replaceChildren(table);
        if (spec.video) host.dataset.svVideo = spec.video;
    }

    function scan() {
        document.querySelectorAll("[data-sv-table]").forEach((host) => {
            // Re-render when Dash swaps the spec (its length is a cheap change marker).
            if (host.dataset.svRendered !== String(host.dataset.svTable.length)) render(host);
        });
    }

    document.addEventListener("click", (e) => {
        const tr = e.target.closest && e.target.closest("[data-sv-table] tr[data-t]");
        if (!tr) return;
        const host = tr.closest("[data-sv-table]");
        const video = host.dataset.svVideo && document.getElementById(host.dataset.svVideo);
        if (!video) return;
        video.currentTime = Math.max(0, parseFloat(tr.dataset.t));
        video.play();
    });

    new MutationObserver(scan).observe(document.documentElement, {
        childList: true, subtree: true, attributes: true, attributeFilter: ["data-sv-table"],
    });
    document.addEventListener("DOMContentLoaded", scan);
})();
