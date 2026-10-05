/*
 * Keyboard shortcuts for the Labeling page: each key clicks a button by id.
 *   ←/→ previous/next frame (Shift: ±5)   Enter accept prediction   N no ball
 *   O occluded   Delete clear   I interpolate   H hit   B bounce
 */
(function () {
    "use strict";
    const KEYS = {
        ArrowLeft: "lab-prev", ArrowRight: "lab-next", Enter: "lab-accept",
        n: "lab-none", o: "lab-occ", Delete: "lab-clear", Backspace: "lab-clear",
        i: "lab-interp", h: "lab-hit", b: "lab-bounce",
    };
    const SHIFT = {ArrowLeft: "lab-prev5", ArrowRight: "lab-next5"};
    document.addEventListener("keydown", (e) => {
        if (!window.location.pathname.startsWith("/labeling")) return;
        const tag = (e.target && e.target.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        const id = (e.shiftKey && SHIFT[e.key]) || KEYS[e.key] || KEYS[e.key.toLowerCase()];
        if (!id) return;
        const btn = document.getElementById(id);
        if (!btn || btn.disabled) return;
        e.preventDefault();
        btn.click();
    });
})();
