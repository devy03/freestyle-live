# Freestyle AI - website version
# The browser plays the music. This program writes the lyrics, makes the music,
# and serves the website at http://127.0.0.1:5050 (share it online with share_online.bat).
# Needs: start_engine.bat running, and Ollama running (llama icon in the tray).

import io
import os
import re
import time
import json
import uuid
import threading
import urllib.parse
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import requests
import numpy as np
import soundfile as sf
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
import julius
from demucs.pretrained import get_model
from demucs.apply import apply_model

# ---------------- Settings (change these freely) ----------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"    # options: "gemma3:4b" (reliable), "qwen3:8b" (best lyrics but too big for 12GB with everything else)
LYRIC_GPU_LAYERS = 99         # 99 = whole lyric model on the GPU (fastest). Lower it only if the music gets slow
PERSONA = "You're from Chicago, but only mention it when it really fits."   # who the rapper is
ACE_URL = "http://127.0.0.1:8001"

CAPTION = ("hip hop, polished studio rap song, male rapper, confident smooth flow, clear upfront vocals, "
           "punchy drums, deep bass, soulful piano loop, professionally mixed and mastered, "
           "rap vocals throughout, 90 bpm")
BPM = 90
KEY = "A Minor"
SEED = 1234
BARS_PER_PIECE = 8          # 8 bars = 8 lines; also the length of the looping beat
EXTRA_BARS = 4              # made and thrown away so the song never wraps up early
VERSES_BETWEEN_HOOKS = 2    # after a new word: verse, HOOK, verse, verse, HOOK ...
USE_VOICE_REFERENCE = True  # keep the same rapper by referencing the first piece
VOCAL_VOLUME = 1.0
BEAT_VOLUME = 0.9
SAMPLE_RATE = 48000
# ----------------------------------------------------------------

BAR_SECONDS = 4 * 60 / BPM
BAR_SAMPLES = int(round(BAR_SECONDS * SAMPLE_RATE))
PIECE_SECONDS = BARS_PER_PIECE * BAR_SECONDS
EXTRA_SECONDS = EXTRA_BARS * BAR_SECONDS
PIECE_SAMPLES = BARS_PER_PIECE * BAR_SAMPLES
TOTAL_LINES = BARS_PER_PIECE + EXTRA_BARS
BEAT_SAMPLES = int(round(60 / BPM * SAMPLE_RATE))
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_AUDIO_FOLDER = os.path.join(HERE, "ACE-Step-1.5", ".cache", "acestep", "tmp", "api_audio")
LOG_PATH = os.path.join(HERE, "freestyle_log.txt")

lock = threading.Lock()
running = True
pending_words = []  # words waiting their turn (from everyone listening)
topic = None
timeline = []       # pieces scheduled on the shared song clock
t0 = None           # server time when the beat started
gen_times = []
anchor = None       # first piece = voice reference
beat_loop = None    # the nonstop beat
beat_wav = None
recent_lines = []
hooks = {}
listeners = {}      # listener id -> last time they checked in
last_word_by_ip = {}
ui = {"making": None, "timing": "", "words": []}

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
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt + ("\n/no_think" if "qwen3" in OLLAMA_MODEL else ""),
        "stream": False,
        "keep_alive": "1h",
        "options": {"num_gpu": LYRIC_GPU_LAYERS, "temperature": 1.0, "top_p": 0.95, "repeat_penalty": 1.15, "num_predict": 350, "num_ctx": 2048},
    }
    if "qwen3" in OLLAMA_MODEL:
        body["think"] = False      # qwen3 "thinks" out loud by default - turn that off
    reply = requests.post(OLLAMA_URL, json=body, timeout=30)
    text = reply.json()["response"]
    if "</think>" in text:          # just in case thinking text sneaks through
        text = text.split("</think>")[-1]
    lines = []
    bad_starts = ("-", "*", "#", "(", "•")
    for line in text.split("\n"):
        line = line.strip().strip('"').strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        if line == "" or line.endswith(":") or line.startswith("[") or line.startswith(bad_starts):
            continue
        low = line.lower()
        head = line.split(":")[0]
        if ":" in line and len(head) < 28 and '"' not in head and len(head.split()) <= 3:
            continue   # looks like a note ("Punchline idea: ..."), not a lyric
        if "syllable" in low or "example" in low or "mental note" in low or "checks " in low:
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


BANNED = ["bittersweet", "plea", "lost in the", "bathed in", "just you and me", "just me and",
          "you see", "can't be beat", "cant be beat", "feel the heat", "feelin' the heat", "shining bright",
          "shinin' bright", "wild and free", "familiar", "concrete jungle", "on the mic", "my flow",
          "the game", "haters", "golden light", "fading light", "city light", "soulful", "sheen",
          "embrace", "divine", "remember", "grandma", "grandpa", "heart and soul", "the rhythm of",
          "a hungry beast", "delight", "so free", "eternally", "honestly", "bright and bold"]

STYLE_EXAMPLE = [
    "Alarm clock screamin' like I owe it some rent",
    "Hit snooze twice, that's the only time I spent",
    "Sun in my window like it's lookin' for a fight",
    "I ain't a mornin' person, I'm a mornin'-after-night",
]


def clean_lines(lines):
    kept = []
    for line in lines:
        low = line.lower()
        bad = False
        for phrase in BANNED:
            if phrase in low and phrase not in str(topic).lower():
                bad = True
        for ex in STYLE_EXAMPLE:
            if simple(ex) == simple(line):
                bad = True
        if not bad:
            kept.append(line)
    return remove_repeats(kept)


def write_verse(word):
    n = str(TOTAL_LINES)
    if recent_lines:
        said = "Your last bars:\n" + "\n".join(recent_lines[-6:])
    else:
        said = "This is the first verse."

    if word:
        task = ("New topic: \"" + word + "\". Flip to it with a clever transition and write the next " + n +
                " bars ABOUT " + word + ". Say \"" + word + "\" once or twice, anywhere in a line.")
    else:
        task = ("Topic: \"" + str(topic) + "\". Write the next " + n +
                " bars - a new angle on it: a punchline, a twist, a brag, a story beat.")

    prompt = (
        "You are an elite rapper writing original bars. Rap, not poetry. " + PERSONA + "\n\n"
        + said + "\n\n" + task + "\n\n"
        "What makes it hard:\n"
        "- Punchlines, wordplay, double meanings, attitude, humor.\n"
        "- Multi-syllable and internal rhymes. Rhyme in couplets.\n"
        "- Concrete, surprising images - say something nobody has heard a thousand times.\n"
        "- Talk like a confident person, not a greeting card. No flowery filler words.\n"
        "- Short bars: 7 to 11 syllables.\n"
        "- Don't start bars with \"Remember\", \"Used to\", \"Yo\" or \"Then\".\n"
        "- Keep it fit for a public stream: no slurs, no hate, no sexual content.\n\n"
        "Style example (topic: alarm clock) - copy the STYLE, never these lines:\n"
        + "\n".join(STYLE_EXAMPLE) + "\n\n"
        "Output ONLY the " + n + " bars. No title, no labels, no notes."
    )

    lines = clean_lines(ask_llama(prompt))
    if word and not any(word.lower() in l.lower() for l in lines[:BARS_PER_PIECE]):
        retry = clean_lines(ask_llama(prompt))
        if any(word.lower() in l.lower() for l in retry[:BARS_PER_PIECE]):
            lines = retry
    tries = 0
    while len(lines) < TOTAL_LINES and tries < 2:
        lines = lines + clean_lines(ask_llama(prompt))
        tries = tries + 1
    return lines[:TOTAL_LINES]


def get_hook(about):
    if about in hooks:
        return hooks[about]
    prompt = ("You write hit rap hooks. Write a 4-line hook about \"" + str(about) + "\".\n"
              "- Say \"" + str(about) + "\" at least twice.\n"
              "- Punchy, chantable, with attitude - something a crowd would yell back.\n"
              "- 5 to 8 syllables per line. One clever idea, not a list of adjectives.\n"
              "- No slurs, no hate, no sexual content.\n"
              "- Don't mention any city. No \"can't be beat\", \"feel the heat\", \"shining bright\".\n"
              "Output ONLY the 4 lines. No title, no labels, no notes.")
    lines = clean_lines(ask_llama(prompt))[:4]
    if len(lines) < 4:
        lines = (lines + clean_lines(ask_llama(prompt)))[:4]
    fallback = [str(about) + " on my mind, yeah", "Say it loud: " + str(about),
                "Can't shake " + str(about) + " tonight", "That's " + str(about) + ", all right"]
    while len(lines) < 4:
        lines.append(fallback[len(lines)])
    hooks[about] = lines
    return lines


# ======================= VOCAL SEPARATION (Demucs) =======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading vocal separator (first time downloads ~80MB)...")
separator = get_model("htdemucs")
separator.to(DEVICE)
separator.eval()
VOCALS_INDEX = separator.sources.index("vocals")


def separate(audio):
    # returns (vocals, beat) at 48 kHz
    wav = torch.from_numpy(audio.T.copy()).float()
    wav = julius.resample_frac(wav, SAMPLE_RATE, separator.samplerate)
    with torch.no_grad():
        stems = apply_model(separator, wav[None], device=DEVICE, split=True, overlap=0.1, progress=False)[0]
    vocals = stems[VOCALS_INDEX]
    beat = stems.sum(0) - vocals
    del stems
    vocals = julius.resample_frac(vocals.cpu(), separator.samplerate, SAMPLE_RATE).numpy().T
    beat = julius.resample_frac(beat.cpu(), separator.samplerate, SAMPLE_RATE).numpy().T
    n = len(audio)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()   # hand the GPU memory back so the music engine has room
    return fit(vocals.astype("float32"), SAMPLE_RATE, n), fit(beat.astype("float32"), SAMPLE_RATE, n)


def smart_cut(vocals):
    # end the vocals in a gap between words near the 8-bar mark, instead of mid-word
    hop = 960   # 20ms
    frames = len(vocals) // hop
    energy = np.sqrt((vocals[:frames * hop].mean(axis=1).reshape(frames, hop) ** 2).mean(axis=1))
    smooth = np.convolve(energy, np.ones(10) / 10, mode="same")
    lo = int((PIECE_SAMPLES - BAR_SAMPLES * 0.75) / hop)
    hi = min(int((PIECE_SAMPLES + BAR_SAMPLES * 1.0) / hop), frames - 1)
    cut_frame = lo + int(np.argmin(smooth[lo:hi]))
    cut = cut_frame * hop
    out = vocals[:cut].copy()
    fade = min(2400, len(out))
    out[-fade:] = out[-fade:] * np.linspace(1, 0, fade, dtype="float32")[:, None]
    return np.ascontiguousarray(out)


def hit_curve(audio):
    # how "hitty" each 10ms is - used to line pieces up with the beat
    mono = audio.mean(axis=1)
    hop = 480
    frames = len(mono) // hop
    energy = np.sqrt((mono[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    rise = np.maximum(np.diff(energy, prepend=energy[0]), 0)
    return rise - rise.mean()


def find_offset(piece_beat):
    # how many samples late this piece's beat is compared to the loop (within +-1 beat)
    a = hit_curve(beat_loop)
    b = hit_curve(piece_beat)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    max_lag = BEAT_SAMPLES // 480
    best_lag = 0
    best_score = -1e9
    for lag in range(-max_lag, max_lag + 1):
        score = float((a * np.roll(b, -lag)).sum())
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag * 480


def shift(audio, samples):
    # positive = move earlier
    out = np.zeros_like(audio)
    if samples > 0:
        out[:len(audio) - samples] = audio[samples:]
    elif samples < 0:
        out[-samples:] = audio[:len(audio) + samples]
    else:
        out = audio
    return out


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
    return np.ascontiguousarray(audio[:length], dtype="float32")


def wait_for_result(task_id):
    started = time.time()
    while True:
        if time.time() - started > 90:
            raise Exception("music engine took over 90s - GPU memory is probably full (see engine window)")
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
        sf.write(buf, anchor["mix"], SAMPLE_RATE, format="WAV", subtype="PCM_16")
        files = {"ref_audio": ("reference.wav", buf.getvalue(), "audio/wav")}
    task = requests.post(ACE_URL + "/release_task", data=fields, files=files, timeout=60).json()
    audio, sr = wait_for_result(task["data"]["task_id"])
    return fit(audio, sr, PIECE_SAMPLES + 2 * BAR_SAMPLES)


def build_piece(previous, word):
    global anchor, recent_lines, beat_loop
    start = time.time()

    if word or previous is None:
        section = "Verse"
        verses = VERSES_BETWEEN_HOOKS if word else 1
    elif previous["verses"] >= VERSES_BETWEEN_HOOKS:
        section = "Chorus"
        verses = 0
    else:
        section = "Verse"
        verses = previous["verses"] + 1

    if section == "Chorus":
        hook = get_hook(topic)
        keep = hook + hook
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
        mix = make_music(lyrics, seed)
        t_music = time.time() - start - t_lyrics
        vocals, beat = separate(mix)
        if loudness(vocals) > 0.02:
            break
        print("   (vocals came out quiet, retrying...)")
        seed = seed + 1

    beat = beat[:PIECE_SAMPLES]
    offset = 0
    if beat_loop is None:
        beat_loop = beat          # the first piece's beat becomes the nonstop beat
    else:
        offset = find_offset(beat)
        vocals = shift(vocals, offset)
    vocals = smart_cut(vocals)

    if section == "Verse":
        recent_lines = (recent_lines + keep)[-12:]
    total = time.time() - start
    gen_times.append(total)
    if len(gen_times) > 5:
        gen_times.pop(0)

    print("   (made next piece: lyrics %.1fs + music %.1fs + vocals %.1fs)" %
          (t_lyrics, t_music, total - t_lyrics - t_music))
    ui["timing"] = "Last verse took %.0f s to make" % total
    write_log("piece %s word=%s lyrics=%.1fs music=%.1fs sep=%.1fs tries=%d vocal_loud=%.3f offset_ms=%d" %
              (section, word, t_lyrics, t_music, total - t_lyrics - t_music, attempt + 1,
               loudness(vocals), int(offset * 1000 / SAMPLE_RATE)))
    for l in keep:
        write_log("    " + l)

    piece = {"vocals": vocals, "mix": mix, "lines": keep, "word": word, "section": section, "verses": verses}
    if anchor is None:
        anchor = piece
    return piece


def expected_gen_time():
    if len(gen_times) == 0:
        return 8.0
    return sum(gen_times) / len(gen_times)


def to_wav(audio, mono=False):
    if mono:
        audio = audio.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def next_bar(t):
    bars = int(np.ceil((t - t0) / BAR_SECONDS - 1e-6))
    return t0 + max(bars, 0) * BAR_SECONDS


def listening_now():
    now = time.time()
    for k in list(listeners.keys()):
        if now - listeners[k] > 20:
            del listeners[k]
    return len(listeners)


def generator():
    global topic, t0, beat_wav
    while running:
        job = None
        word = None
        with lock:
            now = time.time()
            people = listening_now()
            latest = timeline[-1] if timeline else None
            if pending_words and (people > 0 or beat_loop is None):
                word = pending_words.pop(0)
                topic = word
                job = "word"
            elif people > 0 and topic is not None and beat_loop is not None:
                ends = latest["start"] + latest["dur"] if latest else now
                if ends - now < expected_gen_time() + 5:
                    job = "filler"
        if job is None:
            time.sleep(0.1)
            continue
        try:
            ui["making"] = word if word else ""
            piece = build_piece(latest, word)
            vocal_wav = to_wav(piece["vocals"], mono=True)
            with lock:
                now = time.time()
                if t0 is None:
                    t0 = now + 2.0                     # give browsers a moment to load the beat
                    beat_wav = to_wav(beat_loop)
                if word:
                    # a new word replaces any normal piece that hasn't started yet
                    keep = []
                    for p in timeline:
                        if p["start"] > now + 1.0 and not p["word"]:
                            continue
                        keep.append(p)
                    timeline[:] = keep
                last = timeline[-1] if timeline else None
                earliest = now + 1.5                   # time for browsers to fetch it
                if last:
                    earliest = max(earliest, last["start"] + last["dur"])
                piece["id"] = uuid.uuid4().hex[:12]
                piece["start"] = next_bar(max(earliest, t0))
                piece["dur"] = len(piece["vocals"]) / SAMPLE_RATE
                piece["wav"] = vocal_wav
                del piece["vocals"]
                timeline.append(piece)
                timeline[:] = [p for p in timeline if p["start"] + p["dur"] > now - 60 or p is anchor]
            ui["making"] = None
        except Exception as e:
            ui["making"] = None
            print("   [error]", e)
            write_log("ERROR " + str(e))
            if word is not None:
                with lock:
                    pending_words.insert(0, word)
            time.sleep(2)


# ======================= WEBSITE =======================

UI_PORT = 5050
UI_FILE = os.path.join(HERE, "ui.html")
BLOCKED_FILE = os.path.join(HERE, "blocked_words.txt")
WORD_OK = re.compile(r"^[A-Za-z0-9 '\-\.!?&]{1,40}$")


def blocked_words():
    words = []
    try:
        with open(BLOCKED_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    words.append(line)
    except Exception:
        pass
    return words


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def visitor_ip(self):
        return self.headers.get("Cf-Connecting-Ip") or self.headers.get("X-Forwarded-For") or self.client_address[0]

    def is_host(self):
        # only someone sitting at this PC (not coming through the online tunnel)
        return self.client_address[0] == "127.0.0.1" and not self.headers.get("Cf-Connecting-Ip") \
            and not self.headers.get("X-Forwarded-For")

    def send(self, code, body, kind, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, data):
        self.send(code, json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path)
        if path.path in ("/", "/index.html"):
            with open(UI_FILE, "rb") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        elif path.path == "/api/state":
            q = urllib.parse.parse_qs(path.query)
            who = (q.get("id") or [""])[0][:40]
            with lock:
                if who:
                    listeners[who] = time.time()
                now = time.time()
                pieces = []
                for p in timeline:
                    if p["start"] + p["dur"] > now - 2:
                        pieces.append({"id": p["id"], "start": p["start"], "dur": p["dur"],
                                       "lines": p["lines"], "section": p["section"], "word": p["word"]})
                state = {
                    "now": now, "t0": t0, "bar": BAR_SECONDS, "loop": len(beat_loop) / SAMPLE_RATE if beat_loop is not None else 0,
                    "beat": "/audio/beat.wav" if beat_wav else None,
                    "pieces": pieces, "queue": list(pending_words), "making": ui["making"],
                    "topic": topic, "words": ui["words"][-10:], "listeners": listening_now(),
                    "timing": ui["timing"], "host": self.is_host(),
                }
            self.send_json(200, state)
        elif path.path in ("/fonts/BigShouldersDisplay.ttf", "/fonts/BigShouldersText.ttf"):
            with open(os.path.join(HERE, "fonts", os.path.basename(path.path)), "rb") as f:
                self.send(200, f.read(), "font/ttf", cache=True)
        elif path.path == "/audio/beat.wav" and beat_wav:
            self.send(200, beat_wav, "audio/wav", cache=True)
        elif path.path.startswith("/audio/") and path.path.endswith(".wav"):
            pid = path.path[len("/audio/"):-4]
            with lock:
                found = None
                for p in timeline:
                    if p["id"] == pid:
                        found = p["wav"]
            if found:
                self.send(200, found, "audio/wav", cache=True)
            else:
                self.send_json(404, {"error": "That verse has already played."})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        global running
        length = min(int(self.headers.get("Content-Length", 0) or 0), 2000)
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}
        if self.path.startswith("/api/word"):
            w = " ".join(str(data.get("word", "")).split())
            if not WORD_OK.match(w):
                self.send_json(400, {"error": "Use letters and numbers, up to 40 characters."})
                return
            low = w.lower()
            for bad in blocked_words():
                if bad in low:
                    self.send_json(400, {"error": "That word isn't allowed here. Try another."})
                    return
            ip = self.visitor_ip()
            now = time.time()
            with lock:
                if not self.is_host() and now - last_word_by_ip.get(ip, 0) < 8:
                    self.send_json(429, {"error": "One word every 8 seconds. Hang tight."})
                    return
                if len(pending_words) >= 5:
                    self.send_json(429, {"error": "The queue is full. Try again in a moment."})
                    return
                last_word_by_ip[ip] = now
                pending_words.append(w)
                ui["words"] = (ui["words"] + [w])[-20:]
            print(">> word: " + w)
            self.send_json(200, {"ok": True, "position": len(pending_words)})
        elif self.path.startswith("/api/quit") and self.is_host():
            running = False
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "Not found"})


def unload_other_ollama_models():
    try:
        loaded = requests.get(OLLAMA_URL.replace("/api/generate", "/api/ps"), timeout=5).json().get("models", [])
        for m in loaded:
            name = m.get("name", "")
            if name and name != OLLAMA_MODEL and name.split(":")[0] + ":" + name.split(":")[-1] != OLLAMA_MODEL:
                requests.post(OLLAMA_URL, json={"model": name, "keep_alive": 0}, timeout=30)
                print("Unloaded old lyric model from GPU: " + name)
    except Exception as e:
        print("(couldn't check Ollama models: " + str(e) + ")")


print("=== Freestyle AI (website) - lyrics by " + OLLAMA_MODEL + " ===")
unload_other_ollama_models()
clean_engine_folder()
open(LOG_PATH, "w").close()
if not os.path.exists(BLOCKED_FILE):
    with open(BLOCKED_FILE, "w", encoding="utf-8") as f:
        f.write("# One word or phrase per line. Any submitted word containing one of these is rejected.\n")

server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler)
threading.Thread(target=generator, daemon=True).start()
url = "http://127.0.0.1:" + str(UI_PORT)
print("Website running at " + url)
print("To share it online, double-click share_online.bat")
print("Close this window to stop.")
webbrowser.open(url)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    while running:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
running = False
server.shutdown()
clean_engine_folder()
write_log("quit")
print("Peace out.")
