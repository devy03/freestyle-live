# Freestyle LIVE v8 (song mode)
# Every piece is a FRESH, clean generation (no chaining = no mumbling build-up).
# The first piece is used as a "voice reference" so the rapper and beat stay the same.
# Pieces are 8 bars and cut exactly on the bar line so the beat keeps time.
# Needs: start_engine.bat running, and Ollama running (llama icon in the tray).

import io
import os
import time
import json
import threading
import urllib.parse
import requests
import numpy as np
import soundfile as sf
import sounddevice as sd

# ---------------- Settings (change these freely) ----------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
PERSONA = "You grew up in Chicago."   # who the rapper is; change to anything
ACE_URL = "http://127.0.0.1:8001"

CAPTION = ("hip hop, polished studio rap song, male rapper, confident smooth flow, clear upfront vocals, "
           "punchy drums, deep bass, soulful piano loop, professionally mixed and mastered, "
           "rap vocals throughout, 90 bpm")
BPM = 90
KEY = "A Minor"
SEED = 1234
BARS_PER_PIECE = 8          # 8 bars = 8 lines, about 21s at 90 bpm
EXTRA_BARS = 4              # made and thrown away so the song never wraps up early
VERSES_BETWEEN_HOOKS = 2    # after a new word: verse, HOOK, verse, verse, HOOK ...
USE_VOICE_REFERENCE = True  # keep the same rapper/beat by referencing the first piece
SAMPLE_RATE = 48000
EDGE_FADE_SECONDS = 0.01    # tiny fade at each cut to stop clicks
# ----------------------------------------------------------------

BAR_SECONDS = 4 * 60 / BPM
PIECE_SECONDS = BARS_PER_PIECE * BAR_SECONDS
EXTRA_SECONDS = EXTRA_BARS * BAR_SECONDS
PIECE_SAMPLES = int(round(PIECE_SECONDS * SAMPLE_RATE))
TOTAL_LINES = BARS_PER_PIECE + EXTRA_BARS
EDGE = int(EDGE_FADE_SECONDS * SAMPLE_RATE)
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_AUDIO_FOLDER = os.path.join(HERE, "ACE-Step-1.5", ".cache", "acestep", "tmp", "api_audio")
LOG_PATH = os.path.join(HERE, "freestyle_log.txt")

lock = threading.Lock()
running = True
pending_word = None
topic = None
queue = []
playing = None
play_pos = 0
gen_times = []
anchor = None       # first piece = voice reference
recent_lines = []   # last lines rapped, so the lyrics keep flowing
hooks = {}          # topic -> hook lines (real songs repeat the hook)


# ======================= small helpers =======================

def write_log(text):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + text + "\n")
    except Exception:
        pass


def clean_engine_folder():
    if os.path.isdir(ENGINE_AUDIO_FOLDER):
        for name in os.listdir(ENGINE_AUDIO_FOLDER):
            try:
                os.remove(os.path.join(ENGINE_AUDIO_FOLDER, name))
            except Exception:
                pass


def delete_engine_file(file_url):
    try:
        query = urllib.parse.urlparse(file_url).query
        path = urllib.parse.parse_qs(query)["path"][0]
        os.remove(path)
        json_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(json_path):
            os.remove(json_path)
    except Exception:
        pass


def simple(text):
    out = ""
    for ch in text.lower():
        if ch.isalpha():
            out = out + ch
    return out


def loudness(audio):
    return float(np.sqrt((audio ** 2).mean()))


# ======================= LYRICS (Ollama) =======================

def ask_llama(prompt):
    reply = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1h",
        "options": {"temperature": 0.9, "num_predict": 350, "num_ctx": 2048},
    }, timeout=30)
    text = reply.json()["response"]
    lines = []
    for line in text.split("\n"):
        line = line.strip().strip('"').strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        if line == "" or line.endswith(":") or line.startswith("["):
            continue
        if line.lower().startswith("here") or "verse" in line.lower() or "chorus" in line.lower():
            continue
        if len(line.split()) > 13:     # very long lines get mumbled - skip them
            continue
        lines.append(line)
    return lines


def remove_repeats(lines):
    seen = []
    for line in recent_lines:
        seen.append(simple(line))
    kept = []
    for line in lines:
        s = simple(line)
        if s != "" and s not in seen:
            kept.append(line)
            seen.append(s)
    return kept


def write_verse(word):
    n = str(TOTAL_LINES)
    if recent_lines:
        said = "What you just rapped:\n" + "\n".join(recent_lines[-8:])
    else:
        said = "You're just starting the song."

    if word:
        task = ("The song's new topic is \"" + word + "\". Move to it smoothly and write the next " + n +
                " lines. Make the lines actually be ABOUT " + word +
                " - a story, a memory, a point you're making. Say \"" + word +
                "\" once or twice where it fits naturally.")
    else:
        task = ("This part of the song is about \"" + str(topic) + "\". Write the next " + n +
                " lines, taking the story further with new details.")

    prompt = (
        "You're a skilled rapper writing a polished hip hop song verse. " + PERSONA + "\n\n"
        + said + "\n\n" + task + "\n\n"
        "Rules:\n"
        "- SHORT lines: 7 to 10 syllables each, easy to rap clearly. No long sentences.\n"
        "- Every line means something and connects to the line before it.\n"
        "- Specific details (places, people, moments) beat generic boasts.\n"
        "- Rhyme in couplets (lines 1+2, 3+4, ...).\n"
        "- Never use: \"on the mic\", \"my flow\", \"haters\", \"the game\", \"concrete jungle\".\n"
        "- Never repeat a line you already rapped.\n\n"
        "Output ONLY the " + n + " lines. No title, no labels, no notes."
    )

    lines = remove_repeats(ask_llama(prompt))
    if word and not any(word.lower() in l.lower() for l in lines[:BARS_PER_PIECE]):
        retry = remove_repeats(ask_llama(prompt))
        if len(retry) >= BARS_PER_PIECE:
            lines = retry
    if len(lines) < BARS_PER_PIECE:
        lines = lines + remove_repeats(ask_llama(prompt))
    return lines[:TOTAL_LINES]


def get_hook(about):
    if about in hooks:
        return hooks[about]
    prompt = ("You're a hit songwriter. Write a catchy 4-line rap song HOOK (chorus) about \"" + str(about) +
              "\". " + PERSONA + "\n"
              "- Say \"" + str(about) + "\" at least twice.\n"
              "- Short, simple, chantable lines, 5 to 8 syllables each.\n"
              "- It should feel like the heart of the song.\n"
              "Output ONLY the 4 lines. No title, no labels, no notes.")
    lines = ask_llama(prompt)[:4]
    while len(lines) < 4:
        lines.append(str(about) + ", yeah, " + str(about))
    hooks[about] = lines
    return lines


# ======================= MUSIC (ACE-Step) =======================

def fit(audio, sr, length):
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)
    if sr != SAMPLE_RATE:
        old_t = np.arange(len(audio)) / sr
        new_t = np.arange(int(len(audio) * SAMPLE_RATE / sr)) / SAMPLE_RATE
        audio = np.stack([np.interp(new_t, old_t, audio[:, 0]),
                          np.interp(new_t, old_t, audio[:, 1])], axis=1).astype("float32")
    if len(audio) < length:
        audio = np.concatenate([audio, np.zeros((length - len(audio), 2), dtype="float32")])
    return audio[:length]


def wait_for_result(task_id):
    started = time.time()
    while True:
        if time.time() - started > 40:
            raise Exception("music engine took over 40s")
        time.sleep(0.2)
        result = requests.post(ACE_URL + "/query_result",
                               json={"task_id_list": [task_id]}, timeout=60).json()
        item = result["data"][0]
        if item["status"] == 1:
            file_url = json.loads(item["result"])[0]["file"]
            data = requests.get(ACE_URL + file_url, timeout=60).content
            delete_engine_file(file_url)
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return audio, sr
        if item["status"] == 2:
            raise Exception("ACE-Step failed: " + str(item.get("result")))


def make_music(lyrics, seed):
    fields = {
        "prompt": CAPTION, "lyrics": lyrics, "thinking": "false",
        "use_cot_caption": "false", "use_cot_language": "false",
        "vocal_language": "en", "bpm": str(BPM), "key_scale": KEY, "time_signature": "4",
        "audio_duration": str(PIECE_SECONDS + EXTRA_SECONDS),
        "batch_size": "1", "inference_steps": "8",
        "use_random_seed": "false", "seed": str(seed), "audio_format": "wav",
    }
    files = None
    if USE_VOICE_REFERENCE and anchor is not None:
        buf = io.BytesIO()
        sf.write(buf, anchor["audio"], SAMPLE_RATE, format="WAV", subtype="PCM_16")
        files = {"ref_audio": ("reference.wav", buf.getvalue(), "audio/wav")}
    task = requests.post(ACE_URL + "/release_task", data=fields, files=files, timeout=60).json()
    audio, sr = wait_for_result(task["data"]["task_id"])
    return fit(audio, sr, PIECE_SAMPLES)


def build_piece(previous, word):
    global anchor, recent_lines
    start = time.time()

    if word or previous is None:
        section = "Verse"
        verses = VERSES_BETWEEN_HOOKS if word else 1   # new word -> hook comes next
    elif previous["verses"] >= VERSES_BETWEEN_HOOKS:
        section = "Chorus"
        verses = 0
    else:
        section = "Verse"
        verses = previous["verses"] + 1

    if section == "Chorus":
        hook = get_hook(topic)
        keep = hook + hook                      # 4-line hook sung twice = 8 bars
        lyrics = "[Chorus]\n" + "\n".join(keep) + "\n[Verse]\n" + "\n".join(recent_lines[-EXTRA_BARS:])
    else:
        lines = write_verse(word)
        keep = lines[:BARS_PER_PIECE]
        extra = lines[BARS_PER_PIECE:]
        lyrics = "[Verse]\n" + "\n".join(keep)
        if extra:
            lyrics = lyrics + "\n[Chorus]\n" + "\n".join(extra)
    t_lyrics = time.time() - start

    seed = SEED
    for attempt in range(3):
        audio = make_music(lyrics, seed)
        if loudness(audio) > 0.03:
            break
        print("   (piece came out quiet, retrying...)")
        seed = seed + 1

    # tiny fades at both cut points so there's no click
    ramp = np.linspace(0, 1, EDGE, dtype="float32")[:, None]
    audio[:EDGE] = audio[:EDGE] * ramp
    audio[-EDGE:] = audio[-EDGE:] * ramp[::-1]

    if section == "Verse":
        recent_lines = (recent_lines + keep)[-12:]
    total = time.time() - start
    gen_times.append(total)
    if len(gen_times) > 5:
        gen_times.pop(0)

    print("   (made next piece: lyrics %.1fs + music %.1fs)" % (t_lyrics, total - t_lyrics))
    write_log("piece %s word=%s lyrics=%.1fs music=%.1fs tries=%d loud=%.3f" %
              (section, word, t_lyrics, total - t_lyrics, attempt + 1, loudness(audio)))
    for l in keep:
        write_log("    " + l)

    piece = {"audio": audio, "lines": keep, "word": word, "section": section, "verses": verses}
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
                # drop a waiting piece if there's time, so the new word comes in sooner
                if playing is not None and len(queue) > 0:
                    time_left = (PIECE_SAMPLES - play_pos) / SAMPLE_RATE
                    if time_left > expected_gen_time() + 1.0:
                        queue.clear()
            elif len(queue) == 0:
                job = "filler"
            if job is not None:
                previous = queue[-1] if len(queue) > 0 else playing
        if job is None:
            time.sleep(0.05)
            continue
        try:
            piece = build_piece(previous, word)
            with lock:
                queue.append(piece)
        except Exception as e:
            print("   [error]", e)
            write_log("ERROR " + str(e))
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
    block = int(0.05 * SAMPLE_RATE)
    while running:
        piece = take_piece()
        if piece is None:
            print("   (waiting for the next piece...)")
            write_log("WAITING (music stopped)")
            while piece is None and running:
                time.sleep(0.01)
                piece = take_piece()
            if piece is None:
                break
        if piece["word"]:
            print("\n=== " + piece["word"] + " ===")
        if piece["section"] == "Chorus":
            print("  [HOOK]")
        for line in piece["lines"]:
            print("  " + line)
        audio = piece["audio"]
        pos = 0
        while pos < len(audio) and running:
            stop = min(pos + block, len(audio))
            stream.write(audio[pos:stop])
            pos = stop
            with lock:
                play_pos = pos
    stream.stop()
    stream.close()


print("=== Freestyle LIVE v8 (song mode) ===")
print("Type a word anytime and press Enter. Type q to quit.")
clean_engine_folder()
open(LOG_PATH, "w").close()
first = ""
while first == "":
    first = input("First word: ").strip()
pending_word = first

threading.Thread(target=generator, daemon=True).start()
threading.Thread(target=read_words, daemon=True).start()
play_forever()
clean_engine_folder()
write_log("quit")
print("Peace out.")
