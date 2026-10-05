"""M0 stages: ingest (probe + thumbnail + audio), proxy, audio_onsets."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from swingvision.io.audio import OnsetParams, audio_duration_s, detect_onsets, extract_audio
from swingvision.io.ffmpeg import FFmpegCancelled, thumbnail_jpeg, thumbnail_time
from swingvision.io.probe import fast_hash, probe_video
from swingvision.io.proxy import make_proxy
from swingvision.pipeline.stage import Cancelled, NeedsUserAction, Stage, StageContext, chunk_ranges
from swingvision.storage import tables
from swingvision.storage.fsutil import atomic_write
from swingvision.storage.schemas import AUDIO_ONSETS


def _source_path(ctx: StageContext) -> Path:
    src = Path(ctx.config.source.path)
    if not src.exists():
        raise NeedsUserAction("relink", f"Source video not found: {src}. Relink it in the Library.")
    return src


class IngestStage(Stage):
    name = "ingest"
    title = "Ingest"
    version = 1
    weight = 1.0

    def config(self, session, config, settings):
        return {"source_hash": config.source.fast_hash}

    def outputs(self, session):
        video = session.load_config().video
        return [session.thumb_path] + ([session.audio_path] if video and video.has_audio else [])

    def run(self, ctx: StageContext):
        src = _source_path(ctx)
        ctx.progress(0.0, "Checking source file")
        if fast_hash(src) != ctx.config.source.fast_hash:
            raise NeedsUserAction(
                "relink", f"{src.name} no longer matches the file this session was created from."
            )
        ffmpeg = ctx.settings.ffmpeg()
        info = probe_video(ctx.settings.ffprobe(), src)
        config = ctx.session.load_config()
        config.video = info
        ctx.session.save_config(config)
        ctx.progress(0.05, "Thumbnail")

        jpeg = thumbnail_jpeg(ffmpeg, src, thumbnail_time(info.duration_s), width=640)
        atomic_write(ctx.session.thumb_path, lambda tmp: tmp.write_bytes(jpeg), suffix=".jpg")

        if info.has_audio:
            ctx.progress(0.1, "Extracting audio")
            try:
                extract_audio(
                    ffmpeg,
                    src,
                    ctx.session.audio_path,
                    info.duration_s,
                    on_progress=lambda f: ctx.progress(0.1 + 0.9 * f, "Extracting audio"),
                    is_cancelled=ctx.is_cancelled,
                )
            except FFmpegCancelled as exc:
                raise Cancelled() from exc
        else:
            ctx.session.audio_path.unlink(missing_ok=True)
        ctx.progress(1.0)
        return {
            "duration_s": info.duration_s,
            "fps_avg": info.fps_avg,
            "is_vfr": info.is_vfr,
            "rotation_cw": info.rotation_cw,
            "has_audio": info.has_audio,
        }


class ProxyStage(Stage):
    name = "proxy"
    title = "Playback proxy"
    version = 1
    depends_on = ("ingest",)
    uses_gpu = True
    weight = 6.0

    def config(self, session, config, settings):
        return {"height": settings.processing.proxy_height, "gop": settings.processing.proxy_gop}

    def outputs(self, session):
        return [session.proxy_path]

    def run(self, ctx: StageContext):
        src = _source_path(ctx)
        info = ctx.config.video
        assert info is not None, "ingest must run first"
        ctx.progress(0.0, "Encoding 720p proxy")
        try:
            path_used = make_proxy(
                ctx.settings.ffmpeg(),
                src,
                ctx.session.proxy_path,
                info,
                height=ctx.settings.processing.proxy_height,
                gop=ctx.settings.processing.proxy_gop,
                on_progress=lambda f: ctx.progress(f, "Encoding 720p proxy"),
                is_cancelled=ctx.is_cancelled,
            )
        except FFmpegCancelled as exc:
            raise Cancelled() from exc
        return {"encode_path": path_used}


class AudioOnsetsStage(Stage):
    name = "audio_onsets"
    title = "Audio onsets"
    version = 2  # v2: time-domain onset refinement
    depends_on = ("ingest",)
    weight = 1.0

    def config(self, session, config, settings):
        return {"params": OnsetParams().as_config(), "chunk_s": settings.processing.chunk_seconds}

    def outputs(self, session):
        return [session.audio_onsets_path]

    def run(self, ctx: StageContext):
        info = ctx.config.video
        assert info is not None, "ingest must run first"
        out = ctx.session.audio_onsets_path
        if not info.has_audio:
            tables.write_table(AUDIO_ONSETS.empty_table(), out, AUDIO_ONSETS)
            return {"n_onsets": 0, "note": "no audio stream"}

        params = OnsetParams()
        # Put onsets on the video timeline (t=0 at the first video frame).
        offset = (info.audio_start_time_s or 0.0) - info.start_time_s
        ranges = chunk_ranges(
            audio_duration_s(ctx.session.audio_path), ctx.settings.processing.chunk_seconds
        )
        tracker = ctx.chunks(len(ranges))
        for i, (a, b) in enumerate(ranges):
            ctx.check_cancel()
            if tracker.is_done(i):
                continue
            table: pa.Table = detect_onsets(ctx.session.audio_path, a, b, params, offset)
            tables.write_part(table, tracker.parts_dir, i, AUDIO_ONSETS)
            tracker.mark_done(i)
            ctx.progress((i + 1) / len(ranges), f"Chunk {i + 1}/{len(ranges)}")
        result = tables.consolidate_parts(tracker.parts_dir, out, AUDIO_ONSETS, sort_by="t_s")
        tracker.cleanup()
        return {"n_onsets": result.num_rows, "time_offset_s": offset}
