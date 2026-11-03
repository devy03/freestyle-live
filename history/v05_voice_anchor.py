# Freestyle LIVE v5
# - Seamless: every new piece is generated as a CONTINUATION of the last one
#   (ACE-Step "repaint" outpainting), so the beat and voice carry straight through.
# - Short 4-bar pieces, so a new word lands within a few seconds.
# - Lyrics aim for meaning, not just word-stuffing.
# Needs: start_engine.bat running, and Ollama running (llama icon in the tray).

import io
import time
import json
import threading
import requests
import numpy as np
import soundfile as sf
import sounddevice as sd

# ---------------- Settings (change these freely) ----------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"   # smaller = leaves GPU room for the music model
PERSONA = "You grew up in Chicago."   # who the rapper is; change to anything
ACE_URL = "http://127.0.0.1:8001"

CAPTION = ("freestyle rap, off the top, raw cypher, conversational flow, male rapper, "
           "boom bap drums, minimal beat, dry vocals, rap vocals throughout, 90 bpm")
BPM = 90
KEY = "A Minor"
SEED = 1234
BARS_PER_CHUNK = 4      # 4 bars = 4 lines per piece (about 10.7s at 90 bpm)
CONTEXT_SECONDS = 16    # how much past audio the model hears to keep the flow going
SEAM_FADE_SECONDS = 0.08
SAMPLE_RATE = 48000
# ----------------------------------------------------------------

BAR_SECONDS = 4 * 60 / BPM
CHUNK_SECONDS = BARS_PER_CHUNK * BAR_SECONDS
CHUNK_SAMPLES = int(round(CHUNK_SECONDS * SAMPLE_RATE))
FADE = int(SEAM_FADE_SECONDS * SAMPLE_RATE)
CONTEXT_SAMPLES = int(CONTEXT_SECONDS * SAMPLE_RATE)
CONTEXT_LINES = int(CONTEXT_SECONDS / BAR_SECONDS)
TOTAL_LINES = BARS_PER_CHUNK * 2   # 4 lines we keep + 4 "future" lines so the song never thinks it's ending

lock = threading.Lock()
running = True
pending_word = None     # word typed but not used yet
topic = None            # what we're rapping about right now
queue = []              # pieces ready to play
playing = None          # piece playing right now
play_pos = 0            # how far into it we are (samples)
gen_times = []          # how long recent pieces took to make
anchor = None           # first piece - keeps the voice consistent


# ======================= LYRICS (Ollama) =======================

def ask_llama(prompt):
    reply = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1h",
        "options": {"temperature": 0.9, "num_predict": 250, "num_ctx": 2048},
    }, timeout=30)
    text = reply.json()["response"]
    lines = []
    for line in text.split("\n"):
        line = line.strip().strip('"').strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        if line == "" or line.endswith(":") or line.startswith("["):
            continue
        if line.lower().startswith("here") or "verse" in line.lower():
            continue
        lines.append(line)
    return lines[:TOTAL_LINES]


def write_bars(word, previous_lines):
    n = str(TOTAL_LINES)
    if previous_lines:
        said = "What you just rapped:\n" + "\n".join(previous_lines)
    else:
        said = "You're just starting the freestyle."

    if word:
        task = ("Someone in the crowd just shouted the word \"" + word + "\". "
                "Switch your topic to it smoothly and write the next " + n + " lines. "
                "Make the lines actually be ABOUT " + word + " - a story, a memory, a joke, a point you're making. "
                "Say \"" + word + "\" once or twice wherever it fits naturally in the sentence.")
    elif topic:
        task = ("You're freestyling about \"" + topic + "\". Write the next " + n + " lines, "
                "continuing your thought - add a new detail, take the story further, or drift to something related. "
                "You don't have to say the word again.")
    else:
        task = "Write the next " + n + " lines about whatever is on your mind tonight."

    prompt = (
        "You're a real rapper in a cypher, freestyling off the top of your head. "
        "You rap like a person talking - real thoughts, real feelings, a point you're making. " + PERSONA + "\n\n"
        + said + "\n\n" + task + "\n\n"
        "How to sound human:\n"
        "- Every line means something and connects to the line before it.\n"
        "- Specific details (places, people, moments, numbers) beat generic boasts.\n"
        "- Talk the way people actually talk. Casual, confident, a little funny.\n"
        "- Slant rhymes and internal rhymes are fine. Don't force rhymes.\n"
        "- NEVER use: \"on the mic\", \"my flow\", \"rhymes so tight\", \"haters\", \"spittin' fire\", "
        "\"the game\", \"on a roll\", \"in the zone\".\n"
        "- Each line is one bar: 8 to 12 syllables.\n\n"
        "- Never repeat a line you already rapped. Always say something new.\n\n"
        "Output ONLY the " + n + " lines. No title, no numbers, no notes."
    )

    lines = ask_llama(prompt)
    if word:
        found = False
        for line in lines[:BARS_PER_CHUNK]:
            if word.lower() in line.lower():
                found = True
        if not found:
            lines = ask_llama(prompt)
    return lines


# ======================= MUSIC (ACE-Step) =======================

def fit(audio, sr, length):
    # make audio 2 channels, 48 kHz, exactly `length` samples
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)
    if sr != SAMPLE_RATE:
        old_t = np.arange(len(audio)) / sr
        new_t = np.arange(int(len(audio) * SAMPLE_RATE / sr)) / SAMPLE_RATE
        left = np.interp(new_t, old_t, audio[:, 0])
        right = np.interp(new_t, old_t, audio[:, 1])
        audio = np.stack([left, right], axis=1).astype("float32")
    if len(audio) < length:
        pad = np.zeros((length - len(audio), 2), dtype="float32")
        audio = np.concatenate([audio, pad])
    return audio


def wait_for_result(task_id):
    started = time.time()
    while True:
        if time.time() - started > 30:
            raise Exception("music engine took over 30s - GPU may be out of memory")
        time.sleep(0.2)
        result = requests.post(ACE_URL + "/query_result",
                               json={"task_id_list": [task_id]}, timeout=60).json()
        item = result["data"][0]
        if item["status"] == 1:
            file_url = json.loads(item["result"])[0]["file"]
            data = requests.get(ACE_URL + file_url, timeout=60).content
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return audio, sr
        if item["status"] == 2:
            raise Exception("ACE-Step failed: " + str(item.get("result")))


def base_fields(lyrics):
    return {
        "prompt": CAPTION,
        "lyrics": lyrics,
        "thinking": "false",
        "use_cot_caption": "false",
        "use_cot_language": "false",
        "vocal_language": "en",
        "bpm": str(BPM),
        "key_scale": KEY,
        "time_signature": "4",
        "batch_size": "1",
        "inference_steps": "8",
        "use_random_seed": "false",
        "seed": str(SEED),
        "audio_format": "wav",
    }


def make_first(lines, seed):
    # make 2 pieces' worth so the model doesn't wrap the song up early; keep the first piece
    fields = base_fields("[Verse]\n" + "\n".join(lines))
    fields["seed"] = str(seed)
    fields["audio_duration"] = str(CHUNK_SECONDS * 2)
    task = requests.post(ACE_URL + "/release_task", data=fields, timeout=60).json()
    audio, sr = wait_for_result(task["data"]["task_id"])
    audio = fit(audio, sr, CHUNK_SAMPLES)
    return audio[:CHUNK_SAMPLES], None


def make_continuation(base, lines, seed):
    # Context = the FIRST piece (a clean "anchor" with strong vocals) + the piece that just played.
    # The anchor reminds the model every time what the rapper sounds like, so the voice
    # doesn't slowly fade into just the beat. The model continues right after the last piece.
    if base is anchor:
        context = base["audio"]
        ctx_lines = base["lines"]
    else:
        context = np.concatenate([anchor["audio"], base["audio"]])
        ctx_lines = anchor["lines"] + base["lines"]
    ctx_seconds = len(context) / SAMPLE_RATE
    buf = io.BytesIO()
    sf.write(buf, context, SAMPLE_RATE, format="WAV", subtype="PCM_16")

    fields = base_fields("[Verse]\n" + "\n".join(ctx_lines + lines))
    fields["seed"] = str(seed)
    fields["task_type"] = "repaint"
    fields["repainting_start"] = str(ctx_seconds)
    fields["repainting_end"] = str(ctx_seconds + CHUNK_SECONDS * 2)
    fields["audio_duration"] = str(ctx_seconds + CHUNK_SECONDS * 2)

    files = {"src_audio": ("context.wav", buf.getvalue(), "audio/wav")}
    task = requests.post(ACE_URL + "/release_task", data=fields, files=files, timeout=60).json()
    audio, sr = wait_for_result(task["data"]["task_id"])

    ctx_len = len(context)
    audio = fit(audio, sr, ctx_len + CHUNK_SAMPLES)
    new_part = audio[ctx_len:ctx_len + CHUNK_SAMPLES]
    lead_in = audio[ctx_len - FADE:ctx_len]
    return new_part, lead_in


def loudness(audio):
    return float(np.sqrt((audio ** 2).mean()))


def build_piece(base, word):
    start = time.time()
    previous = base["history_lines"] if base else []
    lines = write_bars(word, previous)
    t_lyrics = time.time() - start

    seed = SEED
    for attempt in range(3):
        if base is None:
            audio, lead_in = make_first(lines, seed)
        else:
            audio, lead_in = make_continuation(base, lines, seed)
        if loudness(audio) > 0.02:
            break
        print("   (piece came out quiet, retrying...)")
        seed = seed + 1

    keep = lines[:BARS_PER_CHUNK]
    if base is None:
        history_lines = keep
    else:
        history_lines = (base["history_lines"] + keep)[-CONTEXT_LINES:]

    total = time.time() - start
    gen_times.append(total)
    if len(gen_times) > 5:
        gen_times.pop(0)
    print("   (made next piece: lyrics %.1fs + music %.1fs)" % (t_lyrics, total - t_lyrics))
    piece = {"audio": audio, "lead_in": lead_in, "lines": keep, "word": word,
            "history_lines": history_lines}
    global anchor
    if anchor is None:
        anchor = piece
    return piece


def expected_gen_time():
    if len(gen_times) == 0:
        return 6.0
    return sum(gen_times) / len(gen_times)


def generator():
    global pending_word, topic
    while running:
        job = None
        word = None
        with lock:
            if pending_word is not None:
                word = pending_word
                pending_word = None
                topic = word
                job = "word"
                # throw away a waiting filler piece if there's time, so the word comes in sooner
                if playing is not None and len(queue) > 0:
                    time_left = (CHUNK_SAMPLES - play_pos) / SAMPLE_RATE
                    if time_left > expected_gen_time() + 1.0:
                        queue.clear()
            elif len(queue) == 0:
                job = "filler"
            if job is not None:
                if len(queue) > 0:
                    base = queue[-1]
                else:
                    base = playing
        if job is None:
            time.sleep(0.05)
            continue
        try:
            piece = build_piece(base, word)
            with lock:
                queue.append(piece)
        except Exception as e:
            print("   [error]", e)
            if word is not None:
                with lock:
                    if pending_word is None:
                        pending_word = word
            time.sleep(1)


# ======================= INPUT + PLAYBACK =======================

def read_words():
    global pending_word, running
    while running:
        w = input().strip()
        if w.lower() == "q":
            running = False
            break
        if w != "":
            with lock:
                pending_word = w
            print(">> got it: " + w)


def take_piece():
    global playing, play_pos
    with lock:
        if len(queue) == 0:
            return None
        piece = queue.pop(0)
        playing = piece
        play_pos = 0
        return piece


def play_forever():
    global play_pos
    print("Warming up...")
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype="float32")
    stream.start()
    fade_in = np.linspace(0, 1, FADE, dtype="float32")[:, None]
    fade_out = np.linspace(1, 0, FADE, dtype="float32")[:, None]
    block = int(0.05 * SAMPLE_RATE)
    held = None   # last few ms of the previous piece, blended into the next one

    while running:
        piece = take_piece()
        if piece is None:
            if held is not None:
                stream.write(held * fade_out)
                held = None
                print("   (waiting for the next piece...)")
            while piece is None and running:
                time.sleep(0.01)
                piece = take_piece()
            if piece is None:
                break

        if held is not None:
            if piece["lead_in"] is not None:
                stream.write(held * fade_out + piece["lead_in"] * fade_in)
            else:
                stream.write(held * fade_out)
            held = None

        if piece["word"]:
            print("\n=== " + piece["word"] + " ===")
        for line in piece["lines"]:
            print("  " + line)

        audio = piece["audio"]
        end = len(audio) - FADE
        pos = 0
        while pos < end and running:
            stop = min(pos + block, end)
            stream.write(audio[pos:stop])
            pos = stop
            with lock:
                play_pos = pos
        held = audio[end:]

    stream.stop()
    stream.close()


print("=== Freestyle LIVE v5 ===")
print("Type a word anytime and press Enter. Type q to quit.")
first = ""
while first == "":
    first = input("First word: ").strip()
pending_word = first

threading.Thread(target=generator, daemon=True).start()
threading.Thread(target=read_words, daemon=True).start()
play_forever()
print("Peace out.")
