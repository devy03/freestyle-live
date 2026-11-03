# Earlier versions

Every version of the script, oldest first. They're kept to show how the project got here; for actually running it, use `freestyle_live.py` in the main folder. The story of each one is in [docs/ITERATIONS.md](../docs/ITERATIONS.md).

| File | What changed |
|---|---|
| `v00_one_word_test.py` | One word, one clip (Llama 3.1 8B + ACE-Step) |
| `v01_nonstop_chunks.py` | Back-to-back chunks with crossfades |
| `v04_continuation_gemma.py` | Each chunk continues the last one; switched to Gemma 3 4B |
| `v05_voice_anchor.py` | First clip used as an anchor so the voice doesn't fade |
| `v06_speed_dedupe_cleanup.py` | Faster settings, no repeated lines, automatic cleanup, timing log |
| `v07_song_mode_hooks.py` | Polished song style with verses and hooks |
| `v08_fresh_pieces_voice_reference.py` | Fresh clips with a voice reference instead of chaining |
| `v09_steady_beat_demucs.py` | One looping beat; vocals separated with Demucs and lined up on the bar |
| `v10_anti_cliche_lyrics.py` | Cliché filter, style example, cutting at gaps between words |
| `v12_qwen3_experiments.py` | Qwen3 lyric models and GPU-layer tuning |
| `v14_first_web_ui.py` | First browser interface (pairs with `ui_v13_local_player.html`; audio on the host's speakers) |
| `v15_vram_fixes.py` | Unloading stale models, filtering planning notes, back to Gemma |

The comment at the top of some later files still says "v10"; the file names are the accurate version numbers. Versions 2, 3, 11, and 13 were edited in place and not saved separately.
