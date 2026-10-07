/*
 * Swings page overlay, drawn by video_sync.js on every frame of the swings video
 * (data-frame="swings"): the 2D skeleton. Reads the page's sw-skel store and writes the
 * image itself rather than going through a Dash time store (see review_frame.js for why).
 * The video's data-frame-config holds {edges}.
 */
(function () {
    "use strict";
    const FULL = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;";

    function render(video) {
        const img = document.getElementById("sw-skel-img");
        if (!img) return;
        if (!video._svEdges) {
            try { video._svEdges = JSON.parse(video.dataset.frameConfig || "{}").edges || []; }
            catch (err) { video._svEdges = []; }
        }
        const toggle = document.getElementById("sw-skel-on");
        const api = window.dash_component_api;
        const c = api && api.getLayout ? api.getLayout("sw-skel") : null;
        const src = toggle && !toggle.checked ? ""
            : window.svSkeletonSrc(c && c.props ? c.props.data : null, video.currentTime,
                                   video._svEdges);
        if (!src) { img.style.display = "none"; return; }
        if (img.getAttribute("src") !== src) img.setAttribute("src", src);
        if (img.style.display !== "block") img.style.cssText = FULL + "display:block;";
    }

    window.svFrame = window.svFrame || {};
    window.svFrame.swings = render;
})();
