# Freestyle LIVE (Phase 3)
# Nonstop freestyle: type a word anytime, it works it in, then keeps freestyling.
# Needs: start_engine.bat running, and Ollama running (llama icon in the tray).

import os
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
OLLAMA_MODEL = "llama3.1:8b"
ACE_URL = "http://127.0.0.1:8001"

CAPTION = ("freestyle rap, off the top, raw cypher, conversational flow, male rapper, "
           "boom bap drums, minimal beat, dry vocals, 90 bpm")
BPM = 90
KEY = "A Minor"         # same key every chunk so the beat stays consistent
SEED = 1234             # same seed every chunk so the voice/beat stays similar
CHUNK_SECONDS = 20      # length of each generated piece
NUM_BARS = 6            # lyric lines per chunk
LEAD_SECONDS = 10       # start making the next chunk when this much is left
CROSSFADE_SECONDS = 0.5
MIN_PLAY_SECONDS = 4    # play at least this much before cutting to a new word
# ----------------------------------------------------------------

BAR_SECONDS = 4 * 60 / BPM

lock = threading.Lock()
need_next = threading.Event()
ready = []              # finished chunks waiting to play
current_word = None
word_is_new = False
running = True
previous_bars = []


def ask_llama(prompt):
    reply = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.95},
    }, timeout=120)
    text = reply.json()["response"]
    lines = []
    for line in text.split("\n"):
        line = line.strip().strip('"')
        if line == "" or line.endswith(":") or line.lower().startswith("here"):
            continue
        lines.append(line)
    return lines[:NUM_BARS]


def write_bars(word):
    if previous_bars:
        context = "Your last bars were:\n" + "\n".join(previous_bars) + "\nKeep the flow going from there."
    else:
        context = "This is the start of the freestyle."

    if word:
        task = ("Now switch it up and write exactly " + str(NUM_BARS) + " new lines built around the word \"" + word + "\".\n"
                "Rules:\n"
                "- Use \"" + word + "\" naturally inside the sentences, at least 3 times. Never tack it on after a comma.\n"
                "- End at least 2 lines with \"" + word + "\" so it lands on the beat.\n"
                "- Lines rhyme in pairs (AABB), using rhymes that work with \"" + word + "\".\n")
    else:
        task = ("Now keep freestyling: write exactly " + str(NUM_BARS) + " new lines about anything - "
                "your skills, the city, the crowd, the vibe. You can call back to earlier topics.\n"
                "Rules:\n"
                "- Lines rhyme in pairs (AABB).\n")

    prompt = ("You are a battle rapper freestyling off the top of your head.\n" + context + "\n\n" + task +
              "- Each line is short and punchy, 8 to 12 syllables.\n"
              "- Confident, clever, street style. Wordplay is good.\n"
              "- Output ONLY the lines. No title, no numbering, no explanations.")

    lines = ask_llama(prompt)
    if word:
        count = 0
        for line in lines:
            count = count + line.lower().count(word.lower())
        if count < 2:
            lines = ask_llama(prompt)
    return lines


def make_audio(lines):
    lyrics = "[Verse]\n" + "\n".join(lines)
    task = requests.post(ACE_URL + "/release_task", json={
        "prompt": CAPTION,
        "lyrics": lyrics,
        "thinking": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "vocal_language": "en",
        "bpm": BPM,
        "key_scale": KEY,
        "time_signature": "4",
        "audio_duration": CHUNK_SECONDS,
        "batch_size": 1,
        "inference_steps": 8,
        "use_random_seed": False,
        "seed": SEED,
        "audio_format": "wav",
    }, timeout=60).json()
    task_id = task["data"]["task_id"]

    while True:
        time.sleep(0.3)
        result = requests.post(ACE_URL + "/query_result",
                               json={"task_id_list": [task_id]}, timeout=60).json()
        item = result["data"][0]
        if item["status"] == 1:
            file_url = json.loads(item["result"])[0]["file"]
            break
        if item["status"] == 2:
            raise Exception("ACE-Step failed: " + str(item.get("result")))

    data = requests.get(ACE_URL + file_url, timeout=60).content
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio, sr


def generator():
    # makes the next chunk whenever the player (or a new word) asks for one
    global word_is_new, previous_bars
    while running:
        need_next.wait()
        need_next.clear()
        if not running:
            break
        with lock:
            word = current_word if word_is_new else None
            word_is_new = False
        try:
            start = time.time()
            lines = write_bars(word)
            audio, sr = make_audio(lines)
            previous_bars = lines[-4:]
            with lock:
                ready.append({"audio": audio, "sr": sr, "lines": lines, "word": word})
            print("   (next chunk ready in %.1fs)" % (time.time() - start))
        except Exception as e:
            print("   [error]", e)
            time.sleep(1)
            need_next.set()


def read_words():
    # lets you type a new word anytime
    global current_word, word_is_new, running
    while running:
        w = input().strip()
        if w.lower() == "q":
            running = False
            need_next.set()
            break
        if w != "":
            with lock:
                current_word = w
                word_is_new = True
            need_next.set()
            print(">> got it: " + w)


def take_next_chunk():
    # prefer the newest chunk that has a new word, otherwise the oldest filler
    with lock:
        if len(ready) == 0:
            return None
        pick = None
        for c in ready:
            if c["word"]:
                pick = c
        if pick is None:
            pick = ready[0]
        ready.clear()
        return pick


def word_chunk_waiting():
    with lock:
        for c in ready:
            if c["word"]:
                return True
    return False


def play_forever():
    print("Warming up...")
    chunk = None
    while chunk is None and running:
        time.sleep(0.1)
        chunk = take_next_chunk()
    if chunk is None:
        return

    sr = chunk["sr"]
    channels = chunk["audio"].shape[1]
    stream = sd.OutputStream(samplerate=sr, channels=channels, dtype="float32")
    stream.start()

    fade = int(CROSSFADE_SECONDS * sr)
    fade_in = np.linspace(0, 1, fade, dtype="float32")[:, None]
    fade_out = np.linspace(1, 0, fade, dtype="float32")[:, None]
    block = int(0.1 * sr)
    bar = int(BAR_SECONDS * sr)
    carry = None

    while running:
        audio = chunk["audio"].copy()
        if carry is not None:
            audio[:fade] = audio[:fade] * fade_in + carry

        print("\n" + ("=== " + chunk["word"] + " ===" if chunk["word"] else "=== freestyle ==="))
        for line in chunk["lines"]:
            print("  " + line)
        print("(type a new word anytime, q to quit)")

        end = len(audio) - fade
        pos = 0
        asked = False
        while pos < end and running:
            stop = min(pos + block, end)
            stream.write(audio[pos:stop])
            old_bar = pos // bar
            pos = stop

            if not asked and (end - pos) / sr < LEAD_SECONDS:
                need_next.set()
                asked = True

            # a new word is ready: cut over on the next bar line
            crossed_bar = (pos // bar) != old_bar
            if crossed_bar and pos / sr > MIN_PLAY_SECONDS and word_chunk_waiting():
                break

        carry = audio[pos:pos + fade] * fade_out[:len(audio[pos:pos + fade])]
        if len(carry) < fade:
            carry = None

        nxt = take_next_chunk()
        if nxt is None:
            if carry is not None:
                stream.write(carry)
                carry = None
            need_next.set()
            print("   (thinking...)")
            while nxt is None and running:
                time.sleep(0.05)
                nxt = take_next_chunk()
        if nxt is None:
            break
        chunk = nxt

    stream.stop()
    stream.close()


print("=== Freestyle LIVE ===")
first = ""
while first == "":
    first = input("First word: ").strip()
current_word = first
word_is_new = True
need_next.set()

threading.Thread(target=generator, daemon=True).start()
threading.Thread(target=read_words, daemon=True).start()
play_forever()
print("Peace out.")
