/*
 * Skeleton overlay drawing, shared by the session review's and the swings page's per-frame
 * overlays (review_frame.js, swings_frame.js).
 *
 * svSkeletonSrc(skel, now, edges): an SVG data URI of the pose at time `now`, or "" when
 * there's none. skel: {t, kp} with kp[i] a flat [x, y, score, ...] in % of the frame.
 */
(function () {
    "use strict";

    window.svSkeletonSrc = function (skel, now, edges) {
        if (!skel || !skel.t || !skel.t.length) return "";
        const ts = skel.t;
        let lo = 0, hi = ts.length - 1, i = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (ts[mid] <= now + 1e-3) { i = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        if (i < 0 || now - ts[i] > 0.05) return "";
        const kp = skel.kp[i];
        let d = "";
        for (const [a, b] of edges) {
            if (kp[3 * a + 2] < 0.3 || kp[3 * b + 2] < 0.3) { continue; }
            d += "M" + kp[3 * a] + "," + kp[3 * a + 1] + "L" + kp[3 * b] + "," + kp[3 * b + 1];
        }
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" ' +
            'preserveAspectRatio="none"><path d="' + d + '" fill="none" stroke="#74c0fc" ' +
            'stroke-width="2.5" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>';
        return "data:image/svg+xml;utf8," + encodeURIComponent(svg);
    };
})();
