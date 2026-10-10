"""M7c spike: net-tape serves in existing footage, the tape sound's SNR, and how repeatable
its onset is (docs/spikes/m7c-tape-sound.md).

    uv run python scripts/spikes/tape_sound_spike.py <session-id> [<session-id> ...]

For each session (processed through ``swings``): every near-end serve measured as a
reference (``ball.speed_refs.measure``), the candidates, the audio/video offset, and how
often a serve that clears the tape by > 30 cm still has a ≥ 15 / 20 dB transient in its tape
window (the false-alarm rate the SNR gate alone would have). Then the repeatability test:
each candidate's tape click (and the first one's racket crack) is cut out and pasted into
the tape windows of the clear serves, at its own level and at ½ and ¼, and its onset is
measured again.
"""

from __future__ import annotations

import sys

import numpy as np

from swingvision import services
from swingvision.ball import speed_refs as sr
from swingvision.court import calibration as calib
from swingvision.pipeline.stages.speed import serve_inputs
from swingvision.settings import load_settings


def measure_session(settings, sid: str):
    session = services.session_by_id(settings, sid)
    cfg = session.load_config()
    p = sr.RefParams()
    audio = sr.AudioClip.from_file(session.audio_path)
    cont = (cfg.video.audio_start_time_s or 0.0) - cfg.video.start_time_s
    serves = serve_inputs(session, calib.load(session.calibration_path))
    av, n_av = sr.estimate_av_offset(serves, audio, cont, cfg.air_temp_c, p)
    refs = [
        sr.measure(
            s, audio, container_offset_s=cont, av_offset_s=av, temp_c=cfg.air_temp_c,
            fps=cfg.video.fps_avg, rotation_cw=cfg.video.rotation_cw, p=p,
        )
        for s in serves
    ]  # fmt: skip
    return audio, refs, av, n_av


def false_alarms(refs) -> tuple[int, int, int]:
    clear = [r for r in refs if r.clearance_m is not None and r.clearance_m > 0.3]
    snr = [r.snr_tape_db for r in clear if r.snr_tape_db is not None]
    return len(snr), sum(v >= 15 for v in snr), sum(v >= 20 for v in snr)


def paste_test(audio, onset_t: float, backgrounds: list[np.ndarray], scale: float, p):
    lead, sr_ = 0.002, audio.sr
    snippet, _ = audio.window(onset_t - lead, onset_t + 0.03)
    errs, wrong = [], 0
    rng = np.random.default_rng(0)
    for y in backgrounds:
        y = y.copy()
        off = rng.uniform(-0.015, 0.015)
        i0 = round((0.1 + off - lead) * sr_)
        seg = snippet[: len(y) - i0] * scale
        y[i0 : i0 + len(seg)] += seg
        o = sr.onset_near(sr.AudioClip.from_array(y, sr_), 0.1, p.tape_window_s, p)
        if o is None:
            continue
        e = o.t - (0.1 + off)
        if abs(e) > 0.005:
            wrong += 1  # another transient in the window was stronger
        else:
            errs.append(e)
    e = np.array(errs) * 1000
    return len(e), wrong, float(np.median(e)), float(np.std(e))


def main(ids: list[str]) -> None:
    settings = load_settings()
    p = sr.RefParams()
    for sid in ids:
        audio, refs, av, n_av = measure_session(settings, sid)
        cands = [r for r in refs if r.is_candidate]
        measured = [r for r in refs if r.ratio is not None]
        n, fa15, fa20 = false_alarms(refs)
        print(
            f"\n{sid}: {len(refs)} near serves with a flight, {len(measured)} measured, "
            f"{len(cands)} candidates; audio {1000 * av:.1f} ms behind the video ({n_av} serves)"
        )
        print(
            f"  clear serves (> 30 cm over the tape) with a tape-window transient: "
            f"≥ 15 dB {fa15}/{n}, ≥ 20 dB {fa20}/{n}"
        )
        for r in cands:
            print(
                f"  candidate shot {r.shot_id} at {r.t_contact:.2f} s: {r.end_kind} "
                f"{' '.join(r.flags)}; clearance {100 * r.clearance_m:+.0f} cm, SNR racket "
                f"{r.snr_racket_db:.0f} dB, tape {r.snr_tape_db:.0f} dB; Δt "
                f"{1000 * r.dt_s:.1f} ms; ratio {r.ratio:.4f} ± {r.ratio_sigma:.4f}"
            )
        backgrounds = []
        for r in refs:
            if r.clearance_m is not None and r.clearance_m > 0.3 and r.t_tape_pred_audio:
                y, _ = audio.window(r.t_tape_pred_audio - 0.1, r.t_tape_pred_audio + 0.1)
                backgrounds.append(np.asarray(y))
        sounds = [(f"tape, shot {r.shot_id}", r.t_tape_audio) for r in cands]
        if cands:
            sounds.append((f"racket, shot {cands[0].shot_id}", cands[0].t_racket_audio))
        for name, t_on in sounds:
            for scale in (1.0, 0.5, 0.25):
                n_ok, wrong, bias, sd = paste_test(audio, t_on, backgrounds, scale, p)
                print(
                    f"  paste {name} ×{scale}: {n_ok} timed (bias {bias:+.2f} ms, SD "
                    f"{sd:.2f} ms), {wrong} lost to another transient"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
