# anime-short-generator — SPEC v1

Source: `specs/v0_09292026.md` (original design), revised after model/service review on 2026-09-29.
This file is the contract between modules. If code and spec disagree, fix one of them explicitly.

## 0. Goal

`topic / spec.yaml → final.mp4`: a 30–60 s vertical (9:16) anime Short with Vietnamese narration,
burned-in subtitles and optional background music. One Python package, one process, FFmpeg for media,
local GPU for image/video models. Deterministic stage pipeline (DAG with persisted state), no agents.

MVP scope: 1 narrator, ≤ 2 recurring characters, 30–60 s, 2.5–5 s per shot.

## 1. Environment facts (do not assume otherwise)

- GPU: 1× H100 80 GB **shared** with other services (vLLM etc. hold ~50 GB). Only ~29 GB is free.
  **Never stop or disturb other GPU processes.** Always measure free VRAM (`torch.cuda.mem_get_info`)
  before loading a model.
- Local LLM: OpenAI-compatible vLLM at `http://localhost:8093/v1`, model `qwen3-vl-8b`, max context 8192.
- No ElevenLabs API key is present on this machine. ElevenLabs code must work when a key is set, but
  local end-to-end runs use the local TTS provider.
- FFmpeg: the `ffmpeg` on PATH (miniconda 6.1.1) lacks libx264 and libass. Use the full static
  build via `.env`: `ANIME_SHORT_FFMPEG` / `ANIME_SHORT_FFPROBE` (here `/home/ding/ffmpeg-7.0.2-amd64-static/`).
  All code must go through `utils.media` (`run_ffmpeg`, `ffmpeg_bin()`, `ffprobe`), never a literal "ffmpeg".
- Python via `uv` (project venv `.venv`, Python 3.12).
- Disk is ~90 % full (~390 GB free). Download only the model files actually needed.

## 2. Pipeline

```
spec.yaml / --topic
   │
   ▼
[script]     LLM → hook + scenes (narration vi, visual description en, emotion)
   ▼
[plan]       LLM → per scene: 1–3 shots (structured visual fields, motion, camera)
   ▼
[tts]        per scene audio → real duration (+ word timestamps if provider supports)
   ▼
[timeline]   distribute scene duration over its shots; re-plan a scene if a shot > max_shot_seconds
   ▼
[images]     per shot keyframe (T2I provider, 9:16)
   ▼
[videos]     per shot clip: AnimeGen-I2V if VRAM allows, else "still_motion" (animated keyframe)
   ▼
[subtitles]  ASS file (word-level if timestamps exist, else phrase chunks)
   ▼
[render]     FFmpeg: concat clips + voice + music + subtitles + loudnorm → output/final.mp4
```

Key change vs v0: **scene = one narration/audio segment; shot = one visual.** A scene has 1..n shots.
TTS is never re-run to fix pacing; long scenes get more shots instead.

Stages are batched (all images, then unload, then all videos) — never interleave models on the GPU.

## 3. Data model (`src/anime_short/schemas.py`, Pydantic v2)

`workspace/<project_id>/project.json` is the single source of truth. Every stage reads it, enriches it,
and saves it after **each item** (atomic write: temp file + rename), so `resume` continues at item level.

The authoritative definitions live in `schemas.py`. Summary:

- `ProjectSpec`: id, title, language (`vi`), width 1080, height 1920, fps 24, style name, voice config,
  characters (dict of `Character`), `content` (topic, duration_seconds, tone, audience), scenes, stages
  (dict stage → `done|failed|pending`).
- `Scene`: id (`scene_01`), narration (vi), visual_description (en), emotion, importance,
  `audio: AudioInfo | None` (path, duration, words: list[`WordTiming`]), shots: list[`Shot`].
- `Shot`: id (`scene_01_a`), `visual: VisualSpec` (structured fields, see §6), motion_prompt (en),
  camera (`CameraMove` enum), characters (ids), start, duration (filled by timeline),
  `image: ImageInfo | None` (path, seed, provider, prompt, negative_prompt),
  `video: VideoInfo | None` (path, mode `i2v|still_motion`, provider).
- `CameraMove`: `static, push_in, pull_out, pan_left, pan_right, tilt_up, tilt_down`.
  The planner must choose one (no default) and is told the previous shot's move, since each scene
  is planned in its own call; a `static` shot right after another `static` becomes `push_in`.

All paths stored **relative to the project dir**.

## 4. Configuration

- `configs/default.yaml`: global defaults (llm, tts, image, video, subtitle, render, gpu).
- `configs/styles/<name>.yaml`: style bible (positive/negative text in both `tags` and `natural` form,
  camera preferences).
- Input `spec.yaml` (per video) overrides defaults — format as in v0 §17 plus optional `characters:`.
- Secrets only from env (`.env` supported): `ELEVENLABS_API_KEY`, `LLM_API_KEY`.
- `config.py` loads + deep-merges these into a typed `AppConfig`.

## 5. Script + plan (LLM)

- Client: OpenAI-compatible chat completions (`openai` SDK, configurable `base_url`, `model`).
  Structured output via `response_format` JSON schema derived from Pydantic models; validate with
  Pydantic; on invalid JSON retry up to 3 times with the validation error appended.
- `script`: produces hook, scenes (narration in project language, visual_description in English),
  ending. Rules: narration and visual must not be literal restatements (v0 §4); total narration length
  targets `duration_seconds` at ~4.2 Vietnamese words/second (configurable; measured VieNeu pace),
  and re-asks the LLM (up to 2 more times) when the narration is outside 85–120% of that word target
  (keeping the closest attempt). Scene count is derived from the word target (~13–20 words/scene),
  since the LLM writes ~15 words per scene regardless of the requested length.
- `plan`: per scene, 1–3 shots. Each shot has a `VisualSpec` (single frame, English), a
  `motion_prompt` (only gentle subject/environment/camera motion, no appearance words; screens,
  writing and graphics never change, because the i2v model turns that into gibberish), `camera`, and an
  estimated weight (relative duration). Characters are referenced by id; appearance text is **never**
  written by the LLM — the prompt builder injects it from the character bible.
- `replan_scene(scene, actual_duration)`: re-split one scene into more shots after TTS.
- Prompts in `templates/*.txt`. Keep context small (8k model limit): plan one scene per call.

## 6. Images

`ImageGenerator` protocol (`image/base.py`): `load()`, `unload()`,
`generate(prompt, negative_prompt, width, height, seed, output_path) -> ImageResult`.

Providers (all via Diffusers, bf16):

| provider   | weights                                                        | prompt style | defaults                  | license note |
|------------|----------------------------------------------------------------|--------------|---------------------------|--------------|
| `anima`    | circlestone-labs Anima (Turbo, CFG 1, 8–12 steps)              | tags         | 10 steps, CFG 1.0         | non-commercial — dev only |
| `netayume` | `duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40` (Lumina2Pipeline) | tags   | 40 steps, CFG 5           | Apache-2.0 |
| `zanime`   | `SeeSee21/Z-Anime` distill-8-step (ZImagePipeline)              | natural      | 8 steps, CFG 1.0          | Apache-2.0 |

Default provider: `zanime` (Apache 2.0, chosen by the user from the benchmark contact sheet for its
character consistency across shots). `anima` stays available for non-commercial dev work only.
Benchmark (`scripts/bench_images.py`, 4 prompts, 864×1536, H100): anima 2.6 s/img, 10.6 GB peak;
netayume 14.7 s/img, 12.9 GB; zanime 7.5 s/img, 22.3 GB.

Keyframe size: 864×1536 (exact 9:16, multiple of 16) unless the provider needs otherwise.

`VisualSpec` fields: `subject`, `action`, `setting`, `lighting`, `shot_type`, `mood`, `extra`, `text`.
`text` holds the exact words of any writing in the frame, in English, max 40 chars, usually empty:
image models render short English text legibly but invent Chinese or gibberish for other
languages or for writing that is only described. The planner must always emit it (required field).
`prompt_builder.build(shot, characters, style, prompt_style) -> (prompt, negative)`:
- `tags` style: comma-separated lowercase tags, quality prefix, character `tags`, style tags.
- `natural` style: fluent English sentences, character `description`, style natural text.
Character bible entries have both `tags` and `description`.

Seeds: deterministic per shot (hash of project id + shot id) unless set.

## 7. Videos

`VideoGenerator` protocol (`video/base.py`): `load()`, `unload()`,
`generate(image, prompt, camera, duration, output_path) -> VideoResult`. Output clips are always normalised to
**width×height of the project, project fps, exact shot duration, no audio** (`video/postprocess.py`).

1. `animegen` (`aidealab/AnimeGen-I2V`, WanImageToVideoPipeline + Wan2.2 Lightning 4-step LoRA,
   CFG 1, 16 fps, frames = 4n+1 ≥ duration·16, max 81). Resolution 480×832 (default) or 720×1280.
   Use fp8 layerwise casting + model CPU offload to fit in shared VRAM.
   Prompt: `Japanese anime style, <camera phrase>, <motion_prompt>, any screens, writing and
   graphics stay exactly the same and do not change` (CFG 1 ignores the negative prompt).
   **VRAM gate:** before loading, if free VRAM < `video.animegen.min_free_vram_gb`, do not load;
   the stage falls back to `still_motion` for all shots and logs why.
   Postprocess: trim to duration, 16→project fps, lanczos upscale to 1080×1920.
2. `still_motion` (fallback, CPU only, must look alive, not a slideshow):
   - smooth sub-pixel camera move driven by `shot.camera` with ease-in-out (no zoompan jitter:
     render from an oversampled source or float crops),
   - subtle secondary motion: gentle "breathing" scale oscillation, slight drift/parallax,
     optional light film grain / vignette per style,
   - `static` still gets a very slow push-in so nothing is fully frozen.

The `videos` stage records `video.mode` per shot so render and reports know what happened.

## 8. TTS

`TTSProvider` protocol (`tts/base.py`): `synthesize(text, output_path) -> AudioResult`
(path, duration via ffprobe, words: list[WordTiming] possibly empty).

- `elevenlabs`: `eleven_v3`, `/v1/text-to-speech/{voice_id}/with-timestamps`, convert character
  alignment → word timings. HTTP via `httpx`, retries on 429/5xx.
- `vieneu`: local VieNeu-TTS (Apache-2.0, Vietnamese). No timestamps. Used for local E2E.
- Audio normalised to 48 kHz mono WAV. Leading/trailing silence trimmed to ≤ 0.15 s.

## 9. Timeline

`pipeline/timeline.py`, pure functions, fully unit tested:
- scene start = cumulative sum of scene durations (+ `scene_gap_seconds`, default 0.15).
- split scene duration over shots by weight; snap cut points to nearest word boundary when words exist.
- if any shot > `max_shot_seconds` (5.0) → scene needs re-plan (the timeline stage calls `script.stage.replan_scene` once, so it needs the LLM).
- shots < `min_shot_seconds` (1.5) are merged into neighbour.

## 10. Subtitles

`subtitle/ass.py`: ASS with a "shorts" style: bold Vietnamese-capable font (Be Vietnam Pro, OFL,
vendored in `assets/fonts/`), large, white with thick dark outline, centred at ~70 % height, safe
margins for Shorts UI. Chunks of ≤ 5 words / ≤ 28 chars.
- With word timings: chunk timing from words; current word highlighted (karaoke `\k` or colour).
- Without: split narration into chunks, time proportionally by character count within scene audio.

## 11. Render

`render/` builds FFmpeg commands in Python (no MoviePy):
1. concat shot clips (concat demuxer, all clips already normalised);
2. voice track: scene audios placed at scene start (adelay/amix or concat with silences);
3. optional music: loop/trim, `volume_db`, side-chain ducking under voice, fade in/out;
4. burn ASS subtitles (`fontsdir` = `assets/fonts`);
5. loudness normalise to −14 LUFS, true peak −1 dB;
6. H.264 High, yuv420p, CRF 18, AAC 192k 48 kHz, `+faststart` → `output/final.mp4`.
Short fade-in/out (0.3 s) at start/end. Cuts between shots (Shorts pacing).

## 12. GPU manager

`utils/gpu.py`: `free_vram_gb()`, `with gpu_session(generator):` load → yield → unload +
`gc.collect()` + `torch.cuda.empty_cache()`. Refuses to load when free VRAM < required and raises
`InsufficientVRAM` (callers decide fallback).

## 13. CLI (`anime-short`, Typer)

```
anime-short generate spec.yaml | --topic "..." --duration 45 --style cinematic_anime
anime-short script|plan|tts|timeline|images|videos|subtitles|render <project_id> [--force]
anime-short resume <project_id>        # run every stage not marked done, item-level skip
anime-short status <project_id>        # table of stages and per-shot artifacts/modes
```
Each stage skips items whose artifact already exists unless `--force`.

## 14. Quality bar

- `ruff check` and `ruff format --check` clean; `pytest` green.
- Unit tests for: schemas round-trip, config merge, timeline, prompt builder, ASS generation,
  FFmpeg command building, still_motion (short clip, correct size/fps/duration via ffprobe).
- GPU / network tests marked `@pytest.mark.gpu` / `@pytest.mark.network`, skipped by default.
- Definition of done for MVP: `anime-short generate examples/tin_nhan_10_nam.yaml` produces a
  playable 1080×1920 final.mp4 with Vietnamese voice + subtitles, verified by ffprobe and by looking at
  extracted frames.

## 15. Later (not MVP)

Character reference / LoRA, Qwen-Image-Edit for identity consistency, music generation, SFX,
thumbnails, quality scoring, prompt retry, multi-GPU, web UI.
