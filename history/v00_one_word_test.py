# Freestyle test (Phase 2)
# Type a word -> Llama writes freestyle bars with it -> ACE-Step raps it -> it plays.
# Needs: start_engine.bat running, and Ollama running (llama icon in the tray).

import os
import time
import json
import requests

# ---------------- Settings (change these freely) ----------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
ACE_URL = "http://127.0.0.1:8001"

CAPTION = ("freestyle rap, off the top, raw cypher, conversational flow, male rapper, "
           "boom bap drums, minimal beat, dry vocals, 90 bpm")
BPM = 90
SECONDS = 30
NUM_BARS = 8

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# ----------------------------------------------------------------

previous_bars = []  # last few lines, so the next verse keeps the flow going


def write_bars(word):
    if previous_bars:
        context = "Your last bars were:\n" + "\n".join(previous_bars) + "\nKeep the same energy and flow."
    else:
        context = "This is the start of the freestyle."

    prompt = (
        "You are a battle rapper freestyling off the top of your head.\n"
        + context + "\n\n"
        + "Now write exactly " + str(NUM_BARS) + " new lines of raw freestyle rap built around the word \"" + word + "\".\n"
        + "Rules:\n"
        + "- Use the word \"" + word + "\" naturally inside the sentences, at least 3 times. "
        + "Never just tack it on after a comma.\n"
        + "- Put the word at the end of a line at least twice so it lands on the beat.\n"
        + "- Lines rhyme in pairs (AABB). Use rhymes that work with \"" + word + "\".\n"
        + "- Each line is short and punchy, 8 to 12 syllables.\n"
        + "- Confident, clever, street style. Wordplay is good.\n"
        + "- Output ONLY the " + str(NUM_BARS) + " lines. No title, no numbering, no explanations."
    )

    reply = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.9},
    }, timeout=120)
    text = reply.json()["response"]

    # clean up: keep real lyric lines only
    lines = []
    for line in text.split("\n"):
        line = line.strip().strip('"')
        if line == "":
            continue
        if line.endswith(":"):
            continue
        if line.lower().startswith("here"):
            continue
        lines.append(line)
    return lines[:NUM_BARS]


def count_word(lines, word):
    total = 0
    for line in lines:
        total = total + line.lower().count(word.lower())
    return total


def make_song(lines):
    lyrics = "[Verse]\n" + "\n".join(lines)
    task = requests.post(ACE_URL + "/release_task", json={
        "prompt": CAPTION,
        "lyrics": lyrics,
        "thinking": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "vocal_language": "en",
        "bpm": BPM,
        "time_signature": "4",
        "audio_duration": SECONDS,
        "batch_size": 1,
        "inference_steps": 8,
        "audio_format": "mp3",
    }, timeout=60).json()
    task_id = task["data"]["task_id"]

    # wait until it's done
    while True:
        time.sleep(0.5)
        result = requests.post(ACE_URL + "/query_result",
                               json={"task_id_list": [task_id]}, timeout=60).json()
        item = result["data"][0]
        if item["status"] == 1:
            songs = json.loads(item["result"])
            file_url = songs[0]["file"]
            break
        if item["status"] == 2:
            print("ACE-Step failed:", item.get("result"))
            return None

    audio = requests.get(ACE_URL + file_url, timeout=60)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, "freestyle_" + str(int(time.time())) + ".mp3")
    with open(path, "wb") as f:
        f.write(audio.content)
    return path


print("=== Freestyle Test ===")
print("Type a word and press Enter. Type q to quit.\n")

while True:
    word = input("Word: ").strip()
    if word.lower() == "q":
        break
    if word == "":
        continue

    start = time.time()
    lines = write_bars(word)
    if count_word(lines, word) < 2:
        print("(word missing, rewriting...)")
        lines = write_bars(word)
    lyric_time = time.time() - start

    print("\n--- Bars ---")
    for line in lines:
        print(line)
    print("------------")

    start = time.time()
    path = make_song(lines)
    song_time = time.time() - start

    print("Lyrics took %.1fs, song took %.1fs" % (lyric_time, song_time))
    if path:
        print("Saved:", path)
        os.startfile(path)  # plays in your default music player
    previous_bars = lines[-4:]
    print()
