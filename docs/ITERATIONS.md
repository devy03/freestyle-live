# How Freestyle Live got built

The goal from the start: something like Suno, running on my own PC, that raps continuously and works in whatever word I give it, like a freestyler would. Here's every version, what went wrong, and what fixed it. The code for each one is in [`../history/`](../history/).

Hardware the whole way through: Windows 11, RTX 4070 Ti (12 GB VRAM), 32 GB RAM.

## Phase 1: Is local music generation even fast enough?

**Setup.** ACE-Step 1.5 (Windows portable package) through its browser interface.

**First test.** A 30-second rap clip about "Chicago" took **36 seconds**, but almost all of that (34 s) was ACE-Step's built-in planning model. The actual music took under 2 seconds.

**Fix.** Turned the planner off. The same clip took **1.6 seconds** and sounded just as good. That one number is what made the whole project possible.

Caption testing settled the sound: a raw cypher style ("freestyle rap, off the top, raw cypher, conversational flow, boom bap") won over trap and lo-fi versions.

## v0: One word, one clip

[`v00_one_word_test.py`](../history/v00_one_word_test.py)

Type a word, Llama 3.1 8B (in Ollama) writes 8 bars, ACE-Step raps them, the clip plays. It worked, and it pronounced the words clearly, but each word was a separate song.

## v1: Nonstop chunks

[`v01_nonstop_chunks.py`](../history/v01_nonstop_chunks.py)

Made 20-second chunks back to back, writing the next one while the current one played, with a short crossfade between them.

**Problems.** The music kept pausing, the seams didn't line up, and the lyrics were corny ("I'm on the mic, got my flow on a roll, pizza").

## v2–v3: Continue the song instead of restarting it

Used ACE-Step's "repaint" feature to generate each chunk as a continuation of the last one.

**Problem 1.** A diagnostic showed ACE-Step treats a 10-second request as a whole song and wraps it up early: the rapping stopped at about **7 seconds**, then continuing that silence produced static.
**Fix.** Ask for twice as much song as needed and keep only the first part, so the model never thinks it's at the end.

**Problem 2.** Everything froze after a few words. Task Manager showed **11.0 / 12.0 GB** of VRAM used, with another 1.1 GB spilling into system RAM. Llama 3.1 8B and ACE-Step didn't fit together.

## v4: A smaller lyric model

[`v04_continuation_gemma.py`](../history/v04_continuation_gemma.py)

Switched to Gemma 3 4B, added timeouts so nothing hangs forever, and added a `PERSONA` setting.

**Bug found.** The model kept copying the example lines from the prompt back into the song ("Mama worked doubles, we split one pizza four ways" showed up again and again). Removed the example and told it never to repeat a line.

## v5: Keeping the voice from fading

[`v05_voice_anchor.py`](../history/v05_voice_anchor.py)

**Problem.** After a few chunks, the vocals faded out until only the beat was left. Each continuation was built on the previous continuation, so small losses stacked up.
**Fix.** Keep the first clip, which has strong vocals, as an "anchor" and include it every time.

## v6: Speed and cleanup

[`v06_speed_dedupe_cleanup.py`](../history/v06_speed_dedupe_cleanup.py)

Continuations had slowed to **8–18 seconds** each, longer than the chunks they made. Turned off ACE-Step's CPU offload and shortened the throwaway part. Also: drop any line the model already rapped, delete the engine's saved clips automatically (about **1 GB** had piled up), and write a timing log.

## v7: Song mode

[`v07_song_mode_hooks.py`](../history/v07_song_mode_hooks.py)

Instead of fighting to make it sound like a raw freestyle, aimed for a polished song that keeps going: a studio-rap caption, songwriter-style prompts, and a real song structure of verse → **hook** → verse → verse → hook. The hook says the word at least twice and repeats like a real chorus.

**Problem.** It started mumbling after the first few bars.

## v8: Fresh clips with a voice reference

[`v08_fresh_pieces_voice_reference.py`](../history/v08_fresh_pieces_voice_reference.py)

The log made the cause clear. The first clip, made fresh, came out clear and loud (loudness **0.147**). Every continuation got quieter and mumblier (down to **0.020**) and took 8–21 seconds.

**Fix.** Stop continuing entirely. Every clip is made fresh, the same way as the first one, and gets the first clip as a **voice reference** (ACE-Step's reference-audio feature) to keep the same rapper. Clips became 8 bars, cut on the bar line.

**Result.** Clear vocals, about 4 seconds per clip. The remaining issue was that every clip had its own slightly different beat.

## v9: One steady beat

[`v09_steady_beat_demucs.py`](../history/v09_steady_beat_demucs.py)

Added Demucs to split each clip into vocals and beat.

- The first clip's beat becomes a loop that plays the whole time and never changes.
- Each new clip's vocals get lined up with the loop by comparing drum hits, then dropped in on the next bar line.
- If the next vocals aren't ready, the beat keeps playing instead of going silent.

Vocal separation took about **0.4 seconds** per clip. This was the first version that sounded like one continuous song.

## v10: Fixing the corny lyrics

[`v10_anti_cliche_lyrics.py`](../history/v10_anti_cliche_lyrics.py)

The log showed the same clichés over and over: "bittersweet", "a soulful plea", "lost in the rhythm", "bathed in light", "can't be beat", plus a grandma in almost every verse. Added:

- A prompt that asks for punchlines, wordplay, and multi-syllable rhymes, with an original style example (and a check that rejects copies of it)
- A filter that drops any line containing about 35 overused phrases
- A "smart cut" that ends each clip at the quietest gap between words, so the last word doesn't get chopped

## v11–v12: Trying bigger lyric models

[`v12_qwen3_experiments.py`](../history/v12_qwen3_experiments.py)

**Qwen3 8B** wrote much better bars ("Hurricane in a handkerchief", "Denial got a face, but it don't show its teeth"), but it squeezed ACE-Step so hard that music took **11–14 seconds** per clip instead of 4. Splitting Qwen3 between GPU and CPU made it worse. **Qwen3 4B** fit, but kept writing its planning notes into the song ("Punchline idea: …", "*scribbles mental notes*").

## v13: First interface

![First interface](screenshots/history-v13-first-ui.png)

[`ui_v13_local_player.html`](../history/ui_v13_local_player.html)

A page in the browser with a word box, Spotify-style lyrics, and a visualizer. The audio still played through the PC's speakers.

## v14–v15: Out of GPU memory again

[`v14_first_web_ui.py`](../history/v14_first_web_ui.py) · [`v15_vram_fixes.py`](../history/v15_vram_fixes.py)

ACE-Step's log showed **0.00 GB** of free VRAM, so it fell back to decoding audio on the CPU, which took minutes. Old lyric models from earlier tests were still sitting in GPU memory.

**Fixes.** Unload every other Ollama model on startup, turn ACE-Step's CPU offload back on (cheap now that there's no chaining), have Demucs hand its memory back after each split, go back to Gemma 3 4B, and filter out any line that looks like a note.

## v16: The website

The current [`freestyle_live.py`](../freestyle_live.py) and [`ui.html`](../ui.html).

- **Music plays in the browser,** not on the host's speakers. The server keeps one song clock, and each browser schedules the beat loop and verse clips on it with the Web Audio API, so everyone hears the same song.
- **Built for sharing.** A queue of up to 5 words from anyone listening, one word per visitor every 8 seconds, a blocklist, a host-only stop button, and no generating when nobody's listening.
- **Public link** through a free Cloudflare tunnel.
- **A redesigned interface:** set in Big Shoulders (a typeface made for Chicago), with an amber "on air" light, a lyrics panel that changes color per word with a bar counter, and a scrolling track showing vocals over the beat with a line at every bar.

![The website](screenshots/02-live-verse.png)

## What I'd try next

- Run on a bigger GPU (Qwen3 8B or larger for lyrics, ACE-Step XL for sound) through UIC's research cluster or a free notebook GPU, to see how much better it gets
- Train an ACE-Step LoRA on freestyle recordings so it sounds less like a studio track and more like a live cypher
- A permanent public address with a named Cloudflare tunnel
