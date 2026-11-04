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
import unicodedata
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
OLLAMA_MODEL = "gemma3:4b"    # default lyric model; change it live from the website's Studio settings
LYRIC_GPU_LAYERS = 99         # 99 = whole lyric model on the GPU (fastest). Lower it only if the music gets slow
DEFAULT_PERSONA = "You're from Chicago, but only mention it when it really fits."   # used when the persona box is empty
ACE_URL = "http://127.0.0.1:8001"

BPM = 90                    # starting tempo; the Studio settings can change it with a new beat
KEY = "A Minor"
SEED = 1234
BARS_PER_PIECE = 4          # each chunk is 4 bars = 4 lines (about 10.7 s at 90 BPM)
EXTRA_BARS = 2              # 2 more lines are sung and thrown away so the song never wraps up early
LOOP_BARS = 8               # the nonstop beat loop is 8 bars, taken from the first clip
STORY_BATCH = 8             # lines written per lyric request (two chunks' worth)
USE_VOICE_REFERENCE = True  # keep the same rapper by referencing the first piece
VOCAL_VOLUME = 1.0
BEAT_VOLUME = 0.9
SAMPLE_RATE = 48000
# ----------------------------------------------------------------

TOTAL_LINES = BARS_PER_PIECE + EXTRA_BARS


def set_bpm(bpm):
    # tempo-dependent sizes; changed only when a new beat starts
    global BPM, BAR_SECONDS, BAR_SAMPLES, PIECE_SECONDS, EXTRA_SECONDS, PIECE_SAMPLES, BEAT_SAMPLES
    BPM = int(bpm)
    BAR_SECONDS = 4 * 60 / BPM
    BAR_SAMPLES = int(round(BAR_SECONDS * SAMPLE_RATE))
    PIECE_SECONDS = BARS_PER_PIECE * BAR_SECONDS
    EXTRA_SECONDS = EXTRA_BARS * BAR_SECONDS
    PIECE_SAMPLES = BARS_PER_PIECE * BAR_SAMPLES
    BEAT_SAMPLES = int(round(60 / BPM * SAMPLE_RATE))
    global LOOP_SAMPLES
    LOOP_SAMPLES = LOOP_BARS * BAR_SAMPLES


set_bpm(BPM)
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
reactions = []      # recent crowd reactions: {"id", "e", "t"}
last_react_by_ip = {}
beat_version = 0
reset_requested = False


# ======================= STUDIO SETTINGS (editable on the website by the host) =======================

STUDIO_FILE = os.path.join(HERE, "studio_settings.json")
GENRES = {
    "polished": ("Polished rap", "polished studio hip hop, punchy drums, deep bass, soulful piano loop"),
    "boombap": ("Boom bap", "boom bap hip hop, dusty drums, vinyl samples, jazzy chops"),
    "trap": ("Trap", "trap, booming 808 bass, crisp rolling hi-hats"),
    "lofi": ("Lo-fi", "lo-fi hip hop, mellow keys, soft drums, vinyl crackle"),
    "drill": ("Drill", "drill, sliding 808s, dark piano, sharp hi-hats"),
    "rnb": ("R&B", "r&b, smooth sung vocals, warm keys, laid back groove"),
}
LANGUAGES = {"en": "English", "es": "Spanish", "ja": "Japanese", "zh": "Chinese (Mandarin)",
             "hi-latn": "Hindi in English letters (din raat)", "hi": "Hindi in Devanagari",
             "ko": "Korean", "fr": "French", "pt": "Portuguese", "de": "German"}
TONES = {
    "mixed": "a mix of punchlines, a little story, and confident flexes",
    "story": "a short story with a setup, a turn, and a payoff in the last two bars",
    "brag": "clever, specific flexes - confident, never generic",
    "funny": "jokes and playful wordplay that land on the last word of the line",
    "deep": "honest, specific feelings, like telling a friend the truth",
}
RHYMES = {
    "loose": "Loose, conversational rhymes. Slant rhymes are fine.",
    "couplets": "Lines 1+2 rhyme, 3+4 rhyme, and so on.",
    "multis": "Lines rhyme in pairs, using two- or three-syllable rhymes where they come naturally (\"sold 'em / hold 'em\").",
}
WORD_USE = {
    "subtle": "Say \"{w}\" once, where it hits hardest.",
    "natural": "Say \"{w}\" once or twice, anywhere in a line.",
    "heavy": "Say \"{w}\" at least three times, and end at least one bar on it.",
}
DEFAULT_STUDIO = {
    "model": OLLAMA_MODEL, "persona": "",
    "genre": "polished", "voice": "male", "energy": 60, "mood": 45, "flow": 50, "polish": 85,
    "bpm": BPM, "extra_tags": "",
    "creativity": 100, "word_use": "natural", "rhyme": "couplets", "tone": "mixed", "language": "en",
    "verses_between_hooks": 2, "keep_rapping": True,
    "cooldown": 8, "queue_max": 5,
}
STYLE_KEYS = ("genre", "voice", "energy", "mood", "flow", "polish", "bpm", "extra_tags")


def clamp(v, lo, hi, default):
    try:
        v = int(float(v))
    except Exception:
        return default
    return max(lo, min(hi, v))


def clean_studio(data, base):
    out = dict(base)
    if "model" in data and isinstance(data["model"], str) and re.match(r"^[\w.\-:/]{1,80}$", data["model"]):
        out["model"] = data["model"]
    if "persona" in data:
        out["persona"] = " ".join(str(data["persona"]).split())[:300]
    if data.get("genre") in GENRES:
        out["genre"] = data["genre"]
    if data.get("voice") in ("male", "female"):
        out["voice"] = data["voice"]
    for k in ("energy", "mood", "flow", "polish"):
        if k in data:
            out[k] = clamp(data[k], 0, 100, base[k])
    if "bpm" in data:
        out["bpm"] = clamp(data["bpm"], 70, 150, base["bpm"])
    if "extra_tags" in data:
        out["extra_tags"] = re.sub(r"[^A-Za-z0-9 ,&'\-]", "", str(data["extra_tags"]))[:120]
    if "creativity" in data:
        out["creativity"] = clamp(data["creativity"], 50, 140, base["creativity"])
    if data.get("word_use") in WORD_USE:
        out["word_use"] = data["word_use"]
    if data.get("rhyme") in RHYMES:
        out["rhyme"] = data["rhyme"]
    if data.get("tone") in TONES:
        out["tone"] = data["tone"]
    if data.get("language") in LANGUAGES:
        out["language"] = data["language"]
    if "verses_between_hooks" in data:
        out["verses_between_hooks"] = clamp(data["verses_between_hooks"], 1, 4, base["verses_between_hooks"])
    if "keep_rapping" in data:
        out["keep_rapping"] = bool(data["keep_rapping"])
    if "cooldown" in data:
        out["cooldown"] = clamp(data["cooldown"], 0, 60, base["cooldown"])
    if "queue_max" in data:
        out["queue_max"] = clamp(data["queue_max"], 1, 10, base["queue_max"])
    return out


def load_studio():
    try:
        with open(STUDIO_FILE, encoding="utf-8") as f:
            return clean_studio(json.load(f), DEFAULT_STUDIO)
    except Exception:
        return dict(DEFAULT_STUDIO)


def save_studio():
    try:
        with open(STUDIO_FILE, "w", encoding="utf-8") as f:
            json.dump(studio, f, indent=2)
    except Exception as e:
        print("(couldn't save studio settings: " + str(e) + ")")


studio = load_studio()
active_style = {k: studio[k] for k in STYLE_KEYS}   # the sound the current beat was made with


def persona():
    return studio["persona"].strip() or DEFAULT_PERSONA


def build_caption(s):
    parts = [GENRES[s["genre"]][1]]
    sung = s["genre"] == "rnb"
    parts.append(("female " if s["voice"] == "female" else "male ") + ("singer" if sung else "rapper"))
    e, m, f, p = s["energy"], s["mood"], s["flow"], s["polish"]
    parts.append("laid back, relaxed" if e < 35 else ("high energy, hard-hitting, aggressive" if e > 65 else "confident"))
    if m < 35:
        parts.append("dark, moody")
    elif m > 65:
        parts.append("uplifting, bright, feel-good")
    parts.append("smooth melodic flow" if f < 35 else ("fast rapid-fire flow" if f > 65 else "steady flow"))
    parts.append("raw, gritty, underground" if p < 35 else ("professionally mixed and mastered" if p > 65 else "clean mix"))
    if s["extra_tags"].strip():
        parts.append(s["extra_tags"].strip())
    parts.append("clear upfront vocals, " + ("vocals" if sung else "rap vocals") + " throughout, " + str(s["bpm"]) + " bpm")
    return ", ".join(parts)


def key_for(s):
    return "C Major" if s["mood"] > 65 else ("E Minor" if s["mood"] < 35 else "A Minor")


def style_summary(s):
    words = [GENRES[s["genre"]][0]]
    words.append("chill" if s["energy"] < 35 else ("hype" if s["energy"] > 65 else "steady"))
    if s["mood"] < 35:
        words.append("dark")
    elif s["mood"] > 65:
        words.append("bright")
    words.append(str(s["bpm"]) + " BPM")
    return " · ".join(words)


_models_cache = {"t": 0, "list": []}


def ollama_models():
    if time.time() - _models_cache["t"] < 15:
        return _models_cache["list"]
    names = []
    try:
        tags = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=5).json()
        for m in tags.get("models", []):
            names.append(m.get("name", ""))
    except Exception:
        pass
    names = sorted(n for n in names if n)
    _models_cache.update({"t": time.time(), "list": names})
    return names

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

def llm(system, user, temperature=None, max_tokens=400):
    # one chat turn with the lyric model; returns raw text
    model = studio["model"]
    if temperature is None:
        temperature = 0.85 * studio["creativity"] / 100      # 100 on the slider = 0.85
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user + ("\n/no_think" if "qwen3" in model else "")}],
        "stream": False,
        "keep_alive": "1h",
        "options": {"num_gpu": LYRIC_GPU_LAYERS, "temperature": temperature, "top_p": 0.9,
                    "repeat_penalty": 1.1, "num_predict": max_tokens, "num_ctx": 2048},
    }
    if "qwen3" in model:
        body["think"] = False
    reply = requests.post(OLLAMA_URL.replace("/api/generate", "/api/chat"), json=body, timeout=40)
    text = reply.json()["message"]["content"]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text


def to_lines(text):
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
        if low.startswith("here") or "verse" in low or "chorus" in low or low.startswith("hook"):
            continue
        if len(line.split()) > 14:     # very long lines get mumbled - skip them
            continue
        lines.append(line)
    return lines


def remove_repeats(lines):
    seen = []
    for line in story["history"][-16:] + [q["text"] for q in story["queue"]]:
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

EXAMPLE_VERSE = [
    "Sunday got me sortin' whites from the colors",
    "Same way I sort real friends from the others",
    "Quarters in my pocket, that's the laundromat budget",
    "Dryer ate a sock and I ain't even gon' judge it",
    "Fold it like a poker hand, I know when to hold 'em",
    "Fresh shirts stacked so high you'd think somebody sold 'em",
    "Mama said separate, I took it as a lesson",
    "Clean clothes, clean name, now that's a whole confession",
]
EXAMPLE_HOOK = [
    "First of the month, here we go again",
    "Landlord knockin' like he's my best friend",
    "First of the month, I ain't scared of it",
    "Every bill paid and I'm proud of it",
]
HINDI_EXAMPLE_VERSE = [
    "Subah ki chai, adrak wali, garam garam",
    "Pehle ghoont mein hi mit gaye saare gham",
    "Maa kehti thodi cheeni, main kehta zyada",
    "Meetha bolna seekha, wahi mera iraada",
    "Tapri pe dost, kulhad mein kahaani",
    "Do rupaye ki chai, par yaadein purani",
    "Log peete coffee, karte badi badi baat",
    "Humne chai pe suljhaye saare halaat",
]
HINDI_EXAMPLE_HOOK = [
    "Ek cup chai, aur sab theek hai",
    "Thakaan ho kitni, phir bhi jeet hai",
    "Ek cup chai, bas itna chahiye",
    "Baaki sab baad mein, pehle chai pilaiye",
]
STYLE_EXAMPLE = EXAMPLE_VERSE + EXAMPLE_HOOK + HINDI_EXAMPLE_VERSE + HINDI_EXAMPLE_HOOK


LATIN_LANGS = ("en", "es", "fr", "pt", "de", "hi-latn")


HINDI_WORDS = set("""hai hain ho hoon hu tha thi mein main mera meri mere tera teri tere tu tum aap hum humne
ka ki ke ko se ne pe par aur bhi nahi nahin na kya kyun kyon jo woh yeh ye wo ab jab tab phir sab kuch dil raat din
pyaar pyar yaar jaan duniya zindagi baat apna apni apne kar karo karte karta karti raha rahi rahe gaya gayi gaye
hua hui kahan yahan wahan mujhe tujhe usko unko bas toh hi koi kabhi sapne sapna wala wali wale ek do teen
chal chalo aaja jaana jaane dekh dekho bol bolo sun suno lagta lagti laga tujhse mujhse unse isse""".split())
ENGLISH_WORDS = set("""the and you your my is are was were i me we they it of in on with that this for but not be
have has had just all so what when where how can don't i'm it's you're can't won't""".split())


def is_roman_hindi(line):
    words = re.findall(r"[a-z']+", line.lower().replace("\u2019", "'"))
    hindi = sum(1 for w in words if w in HINDI_WORDS)
    english = sum(1 for w in words if w in ENGLISH_WORDS)
    return hindi >= 1 and hindi >= 2 * english + 1


def right_script(line):
    # drop lines written in the wrong language's script (e.g. Hindi lines in an English song)
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if ord(ch) < 0x250)
    share = latin / len(letters)
    if studio["language"] == "hi-latn":
        return share > 0.8 and is_roman_hindi(line)
    if studio["language"] in LATIN_LANGS:
        return share > 0.8
    return share < 0.5


def clean_lines(lines, dedupe=True, check_script=True):
    kept = []
    for line in lines:
        low = line.lower().replace("\u2019", "'").replace("\u2018", "'")
        bad = check_script and not right_script(line)
        for phrase in BANNED:
            if phrase in low and phrase not in str(topic).lower():
                bad = True
        for ex in STYLE_EXAMPLE:
            if simple(ex) == simple(line):
                bad = True
        if not bad:
            kept.append(line)
    return remove_repeats(kept) if dedupe else kept


ROMAN_HINDI = ("Hindi spelled with English letters, the way people text in Hindi (like 'din raat', "
               "'dil se', 'tu hi meri jaan'). Use simple sound-it-out spelling: 'aa' for long a, 'ee' for long i, "
               "no apostrophes inside words. Never use Devanagari, and don't switch to English sentences")


def ace_language():
    # what the music engine should sing in
    return "hi" if studio["language"] == "hi-latn" else studio["language"]


def language_rule():
    if studio["language"] == "hi-latn":
        return "- Write every bar in " + ROMAN_HINDI + ". An English word here and there is fine, like real Hindi rap.\n"
    if studio["language"] == "en":
        return ("- Write every bar in English, even if the topic word is from another language "
                "(keep that word as it is and explain it with English bars).\n")
    return ("- Write every bar in " + LANGUAGES[studio["language"]] + ", in its native script (no romanization), "
            "and never switch to English. If the topic word is in another language, keep it as written.\n")


WRITER_RULES = (
    "You are a sharp, witty rapper and a careful songwriter. {persona}\n"
    "How you write:\n"
    "- Meaning first. Every line makes sense on its own and moves the idea forward. "
    "A plain, clever line beats a fancy confusing one.\n"
    "- Use everyday words and concrete, real-life details. No sci-fi, no tech jargon, "
    "no 'matrix', 'algorithm', 'cosmic' or 'interstellar' unless the topic is about that.\n"
    "- Land a punchline every two lines: a double meaning, a twist, or a joke that pays off on the last word.\n"
    "- {rhyme}\n"
    "- Each line is one bar: 8 to 12 syllables.\n"
    "- Never write filler like 'yeah', 'let it ring', 'feel the vibe', or 'can't be beat'.\n"
    "- Keep it fit for a public stream: no slurs, no hate, no sexual content.\n"
    "{language}"
    "{example}"
    "Never reuse those lines."
)
EXAMPLE_TEXT_EN = ("Here is the level you aim for. This verse is about laundry - notice that every line is about the topic, "
                   "it's easy to follow, and the wordplay (sort friends, poker hand, clean name) is the point:\n"
                   + "\n".join(EXAMPLE_VERSE) + "\n")
EXAMPLE_TEXT_HI = ("Here is the level and the exact writing style you aim for: Hindi in English letters. This verse is about "
                   "chai - every line is about the topic, easy to follow, and it rhymes in pairs (garam/gham, zyada/iraada, "
                   "baat/halaat):\n" + "\n".join(HINDI_EXAMPLE_VERSE) + "\n")
# the running story: everything is one continuous song
story = {"premise": None, "queue": [], "history": [], "since_hook": 0}
# queue: lines written but not sung yet, each {"text", "word"}; history: lines already scheduled to play


def parse_story(text):
    premise = None
    body = []
    for line in text.split("\n"):
        s = line.strip().strip("*").strip()
        if s.upper().startswith("STORY") and ":" in s:
            premise = s.split(":", 1)[1].strip()
        else:
            body.append(line)
    return premise, "\n".join(body)


def system_prompt():
    example = EXAMPLE_TEXT_HI if studio["language"] == "hi-latn" else EXAMPLE_TEXT_EN
    return WRITER_RULES.format(persona=persona(), rhyme=RHYMES[studio["rhyme"]], language=language_rule(), example=example)


SCRIPTS = {"hi": "Hindi in Devanagari script", "ja": "Japanese (kana and kanji)", "ko": "Korean in Hangul",
           "zh": "Mandarin Chinese in Chinese characters", "es": "Spanish", "fr": "French", "pt": "Portuguese", "de": "German"}


def language_task():
    # repeated at the end of each request, because the English examples pull small models toward English
    lang = studio["language"]
    if lang == "en":
        return ""
    if lang == "hi-latn":
        return "\nWrite every line in " + ROMAN_HINDI + "."
    return "\nWrite every line in " + SCRIPTS.get(lang, LANGUAGES[lang]) + ". Not a single line in English."


def write_story(new_word=None):
    # writes the next STORY_BATCH lines of one continuous story; a new word gets woven in right away
    n = str(STORY_BATCH)
    if story["premise"] is None:
        subject = new_word or str(topic)
        user = ("Start a story-driven rap song. The first topic is \"" + subject + "\". "
                "If it has several meanings, use the one most people would mean.\n"
                "First line: STORY: <one sentence - who this is about and what's happening>.\n"
                "Then write the first " + n + " bars of the story. The topic shows up in the first two bars. "
                + WORD_USE[studio["word_use"]].replace("{w}", subject) +
                " Each bar is one line. No titles or notes." + language_task())
    else:
        sofar = "\n".join(story["history"][-6:]) or "(nothing yet)"
        user = ("The story so far: " + story["premise"] + "\n"
                "The last bars that were rapped:\n" + sofar + "\n\n")
        if new_word:
            user += ("Someone just threw a new word into the song: \"" + new_word + "\". "
                     "Bring it into the story in the next two bars - it should change what happens or drive it, "
                     "not just get mentioned. " + WORD_USE[studio["word_use"]].replace("{w}", new_word) + "\n"
                     "If the story turns, first write a new line STORY: <one sentence>.\n")
        else:
            user += "Keep the story moving forward - something new happens, with a punchline every two bars.\n"
        user += ("Write the next " + n + " bars, continuing straight on from the last bar (same flow, no recap). "
                 "Each bar is one line. No titles or notes." + language_task())
    best = []
    best_premise = None
    raw_last = []
    for attempt in range(3):
        premise, body = parse_story(llm(system_prompt(), user))
        raw = to_lines(body)
        raw_last = raw
        lines = clean_lines(raw)
        has_word = (not new_word) or any(new_word.lower() in l.lower() for l in lines[:4])
        if len(lines) >= 4 and has_word:
            best, best_premise = lines, premise
            break
        if len(lines) > len(best):
            best, best_premise = lines, premise
    if len(best) < 4:
        best = clean_lines(raw_last, check_script=False)
    if best_premise:
        story["premise"] = best_premise[:200]
    elif story["premise"] is None:
        story["premise"] = "A rapper's story about " + str(new_word or topic)
    tag = new_word
    for line in best[:STORY_BATCH]:
        story["queue"].append({"text": line, "word": tag})
        tag = None
    return len(best)


def get_hook(about):
    key = (str(about).lower(), studio["language"], studio["tone"], studio["model"])
    if key in hooks:
        return hooks[key]
    user = ("Write a 4-line rap hook about \"" + str(about) + "\".\n"
            "How a great hook works:\n"
            "- Line 1 is a short, catchy title phrase that includes \"" + str(about) + "\".\n"
            "- Line 2 pays it off with a clever twist.\n"
            "- Line 3 repeats the title phrase (or a small variation).\n"
            "- Line 4 is the punchline people remember.\n"
            "- 6 to 9 syllables per line, simple words, easy to chant back.\n"
            + ("Example (about chai):\n" + "\n".join(HINDI_EXAMPLE_HOOK) if studio["language"] == "hi-latn"
               else "Example (about rent):\n" + "\n".join(EXAMPLE_HOOK)) + "\n"
            "Never reuse the example. Output ONLY the 4 lines." + language_task())
    best = []
    for attempt in range(3):
        lines = clean_lines(to_lines(llm(system_prompt(), user)), dedupe=False)[:4]
        mentions = sum(1 for l in lines if str(about).lower() in l.lower())
        if len(lines) == 4 and mentions >= 2:
            best = lines
            break
        if len(lines) > len(best):
            best = lines
    if len(best) == 0:
        best = [str(about) + ", " + str(about)]
    if len(best) < 4:
        # repeat what we have in the usual hook order (A B A B) instead of padding with filler
        base = best[:2] if len(best) >= 2 else best * 2
        best = (base + base)[:4]
    hooks[key] = best
    return best


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


def find_phrases(vocals):
    # where each sung line is: runs of voice separated by short breaths
    hop = 960   # 20 ms
    mono = vocals.mean(axis=1)
    frames = len(mono) // hop
    if frames < 10:
        return []
    energy = np.sqrt((mono[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    smooth = np.convolve(energy, np.ones(3) / 3, mode="same")
    loud = np.percentile(smooth, 95)
    if loud <= 1e-5:
        return []
    voiced = smooth > loud * 0.14
    phrases = []
    start = None
    quiet = 0
    for i in range(frames):
        if voiced[i]:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= 7:                      # 140 ms of quiet ends a line
                end = i - quiet + 1
                if end - start >= 12:           # ignore blips under 240 ms
                    phrases.append((start * hop / SAMPLE_RATE, end * hop / SAMPLE_RATE))
                start = None
                quiet = 0
    if start is not None and frames - start >= 12:
        phrases.append((start * hop / SAMPLE_RATE, frames * hop / SAMPLE_RATE))
    return phrases


def cut_chunk(vocals, lines_given):
    # decide where this chunk ends and how many lines actually got sung, so no line is lost
    want = min(BARS_PER_PIECE, lines_given)
    bar = BAR_SECONDS
    phrases = find_phrases(vocals)
    sung = None
    cut = None
    times = None
    if len(phrases) >= want and phrases[want - 1][1] <= (BARS_PER_PIECE + 1.25) * bar:
        sung = want
        # take one more line if it's already done before the chunk would end anyway
        if len(phrases) > want and lines_given > want and phrases[want][1] <= (BARS_PER_PIECE + 0.25) * bar:
            sung = want + 1
        cut = phrases[sung - 1][1] + 0.08
        times = [round(p[0], 2) for p in phrases[:sung]]
    else:
        # couldn't tell the lines apart: cut in the quietest spot near the 4-bar mark
        hop = 960
        mono = vocals.mean(axis=1)
        frames = len(mono) // hop
        energy = np.sqrt((mono[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
        smooth = np.convolve(energy, np.ones(10) / 10, mode="same")
        lo = int((BARS_PER_PIECE - 0.5) * bar * SAMPLE_RATE / hop)
        hi = min(int((BARS_PER_PIECE + 1.0) * bar * SAMPLE_RATE / hop), frames - 1)
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * hop / SAMPLE_RATE
        done = [p for p in phrases if p[1] <= cut]
        sung = len(done) if 1 <= len(done) <= want else want
        if len(done) == sung and sung > 0:
            times = [round(p[0], 2) for p in done]
    n = min(int(cut * SAMPLE_RATE), len(vocals))
    out = vocals[:n].copy()
    fade = min(2400, len(out))
    out[-fade:] = out[-fade:] * np.linspace(1, 0, fade, dtype="float32")[:, None]
    return np.ascontiguousarray(out), sung, times


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


def make_music(lyrics, seed, bars):
    fields = {
        "prompt": build_caption(active_style), "lyrics": lyrics, "thinking": "false",
        "use_cot_caption": "false", "use_cot_language": "false",
        "vocal_language": ace_language(), "bpm": str(BPM), "key_scale": key_for(active_style), "time_signature": "4",
        "audio_duration": str(bars * BAR_SECONDS),
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
    return fit(audio, sr, bars * BAR_SAMPLES)


def build_piece(previous, word):
    global anchor, beat_loop
    start = time.time()
    first = beat_loop is None

    # a hook every few verses, but never right after a new word (the word gets its verse first)
    hook_every = 2 * studio["verses_between_hooks"]
    if not first and not word and story["since_hook"] >= hook_every and topic:
        section = "Chorus"
        hook = get_hook(topic)
        lines = hook[:4]
        extra = hook[:EXTRA_BARS]
    else:
        section = "Verse"
        if word:
            story["queue"].clear()                 # the old plan is out; the new word changes the story
            story["since_hook"] = min(story["since_hook"], max(hook_every - 2, 0))   # the word gets 8 bars before a hook
            write_story(word)
        need = (2 * BARS_PER_PIECE if first else BARS_PER_PIECE) + EXTRA_BARS
        tries = 0
        while len(story["queue"]) < need and tries < 2:
            write_story(None)
            tries += 1
        take = 2 * BARS_PER_PIECE if first else BARS_PER_PIECE
        lines = [q["text"] for q in story["queue"][:take]]
        extra = [q["text"] for q in story["queue"][take:take + EXTRA_BARS]]
        if not lines:
            raise Exception("the lyric model didn't write any usable lines")
    t_lyrics = time.time() - start

    bars = len(lines) + len(extra)
    if first:
        bars = max(bars, LOOP_BARS + EXTRA_BARS)
    lyrics = ("[Chorus]\n" if section == "Chorus" else "[Verse]\n") + "\n".join(lines + extra)
    seed = SEED
    for attempt in range(3):
        mix = make_music(lyrics, seed, bars)
        t_music = time.time() - start - t_lyrics
        vocals, beat = separate(mix)
        if loudness(vocals) > 0.02:
            break
        print("   (vocals came out quiet, retrying...)")
        seed = seed + 1

    offset = 0
    if first:
        beat_loop = beat[:LOOP_SAMPLES]           # the first clip's beat becomes the nonstop 8-bar loop
    else:
        offset = find_offset(beat[:PIECE_SAMPLES])
        vocals = shift(vocals, offset)
    vocals, sung, times = cut_chunk(vocals, len(lines))
    sung_lines = lines[:sung]

    if section == "Verse":
        word_here = None
        for q in story["queue"][:sung]:
            if q["word"]:
                word_here = q["word"]
        del story["queue"][:sung]                  # anything not sung stays at the front for the next chunk
        story["since_hook"] += 1
    else:
        word_here = None
        story["since_hook"] = 0

    total = time.time() - start
    gen_times.append(total)
    if len(gen_times) > 5:
        gen_times.pop(0)
    print("   (made next chunk: lyrics %.1fs + music %.1fs + vocals %.1fs, %d of %d lines sung)" %
          (t_lyrics, t_music, total - t_lyrics - t_music, sung, len(lines)))
    ui["timing"] = "Last chunk took %.0f s to make" % total
    write_log("chunk %s word=%s lyrics=%.1fs music=%.1fs sep=%.1fs tries=%d sung=%d/%d timed=%s offset_ms=%d" %
              (section, word, t_lyrics, t_music, total - t_lyrics - t_music, attempt + 1, sung, len(lines),
               "yes" if times else "no", int(offset * 1000 / SAMPLE_RATE)))
    for l in sung_lines:
        write_log("    " + l)

    piece = {"vocals": vocals, "mix": mix, "lines": sung_lines, "times": times, "word": word or word_here,
             "section": section, "verses": 0}
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


def do_reset():
    # start over with a new beat in the current Studio sound
    global beat_loop, anchor, beat_wav, t0, beat_version, reset_requested, active_style, recent_lines
    with lock:
        beat_loop = None
        anchor = None
        beat_wav = None
        t0 = None
        timeline.clear()
        gen_times.clear()
        hooks.clear()
        story.update({"premise": None, "queue": [], "history": [], "since_hook": 0})
        active_style = {k: studio[k] for k in STYLE_KEYS}
        set_bpm(active_style["bpm"])
        beat_version += 1
        reset_requested = False
        if topic and not pending_words:
            pending_words.insert(0, topic)
    print(">> new beat: " + style_summary(active_style))
    write_log("NEW BEAT " + build_caption(active_style))


def generator():
    global topic, t0, beat_wav
    while running:
        if reset_requested:
            do_reset()
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
            elif people > 0 and topic is not None and beat_loop is not None and studio["keep_rapping"]:
                ends = latest["start"] + latest["dur"] if latest else now
                if ends - now < expected_gen_time() + 3:
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
                    # a new word replaces any normal chunk that hasn't started yet
                    keep = []
                    for p in timeline:
                        if p["start"] > now + 1.0 and not p["word"] and p is not anchor:
                            for line in p["lines"]:        # those lines were never heard
                                if story["history"] and story["history"][-1] == line:
                                    story["history"].pop()
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
                story["history"] = (story["history"] + piece["lines"])[-24:]
                timeline[:] = [p for p in timeline if p["start"] + p["dur"] > now - 60 or p is anchor]
            ui["making"] = None
        except Exception as e:
            ui["making"] = None
            print("   [error]", e)
            write_log("ERROR " + str(e))
            if "music engine" in str(e):
                # the GPU is too full for the music engine: free the lyric model's memory so it can catch up
                try:
                    requests.post(OLLAMA_URL, json={"model": studio["model"], "keep_alive": 0}, timeout=20)
                    print("   (freed the lyric model's GPU memory so the music engine can finish)")
                except Exception:
                    pass
            if word is not None:
                with lock:
                    pending_words.insert(0, word)
            time.sleep(2)


# ======================= WEBSITE =======================

UI_PORT = 5050
UI_FILE = os.path.join(HERE, "ui.html")
BLOCKED_FILE = os.path.join(HERE, "blocked_words.txt")
def word_ok(w):
    # letters from any language (and their accent marks), digits, spaces, and a little punctuation
    if not 1 <= len(w) <= 40:
        return False
    for ch in w:
        if not (ch.isalnum() or ch in " '-.!?&" or unicodedata.category(ch).startswith("M")):
            return False
    return True


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
                        pieces.append({"id": p["id"], "start": p["start"], "dur": p["dur"], "times": p.get("times"),
                                       "lines": p["lines"], "section": p["section"], "word": p["word"]})
                state = {
                    "now": now, "t0": t0, "bar": BAR_SECONDS, "loop": len(beat_loop) / SAMPLE_RATE if beat_loop is not None else 0,
                    "beat": ("/audio/beat.wav?v=" + str(beat_version)) if beat_wav else None,
                    "pieces": pieces, "queue": list(pending_words), "making": ui["making"],
                    "topic": topic, "words": ui["words"][-10:], "listeners": listening_now(),
                    "timing": ui["timing"], "host": self.is_host(),
                    "beat_version": beat_version, "style": style_summary(active_style), "story": story["premise"],
                    "reactions": [r for r in reactions if now - r["t"] < 8],
                    "keep_rapping": studio["keep_rapping"], "cooldown": studio["cooldown"],
                }
            self.send_json(200, state)
        elif path.path in ("/fonts/BigShouldersDisplay.ttf", "/fonts/BigShouldersText.ttf"):
            with open(os.path.join(HERE, "fonts", os.path.basename(path.path)), "rb") as f:
                self.send(200, f.read(), "font/ttf", cache=True)
        elif path.path == "/api/config":
            self.send_json(200, self.config_payload())
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

    def config_payload(self):
        with lock:
            pending = any(studio[k] != active_style[k] for k in STYLE_KEYS)
            return {
                "host": self.is_host(), "studio": studio, "default_persona": DEFAULT_PERSONA,
                "models": ollama_models() if self.is_host() else [studio["model"]],
                "genres": {k: v[0] for k, v in GENRES.items()}, "languages": LANGUAGES,
                "caption_preview": build_caption(studio), "caption_active": build_caption(active_style),
                "style_preview": style_summary(studio), "style_active": style_summary(active_style),
                "style_pending": pending,
            }

    def do_POST(self):
        global running, reset_requested
        length = min(int(self.headers.get("Content-Length", 0) or 0), 2000)
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}
        if self.path.startswith("/api/word"):
            w = " ".join(str(data.get("word", "")).split())
            if not word_ok(w):
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
                wait = studio["cooldown"]
                if not self.is_host() and now - last_word_by_ip.get(ip, 0) < wait:
                    self.send_json(429, {"error": "One word every " + str(wait) + " seconds. Hang tight."})
                    return
                if len(pending_words) >= studio["queue_max"]:
                    self.send_json(429, {"error": "The queue is full. Try again in a moment."})
                    return
                last_word_by_ip[ip] = now
                pending_words.append(w)
                ui["words"] = (ui["words"] + [w])[-20:]
            print(">> word: " + w)
            self.send_json(200, {"ok": True, "position": len(pending_words)})
        elif self.path.startswith("/api/react"):
            kind = str(data.get("e", ""))
            if kind not in ("fire", "clap", "laugh", "hundred"):
                self.send_json(400, {"error": "Unknown reaction."})
                return
            ip = self.visitor_ip()
            now = time.time()
            with lock:
                if now - last_react_by_ip.get(ip, 0) < 0.6:
                    self.send_json(429, {"error": "Slow down a little."})
                    return
                last_react_by_ip[ip] = now
                reactions.append({"id": uuid.uuid4().hex[:8], "e": kind, "t": now})
                reactions[:] = [r for r in reactions if now - r["t"] < 30][-80:]
            self.send_json(200, {"ok": True})
        elif self.path.startswith("/api/config"):
            if not self.is_host():
                self.send_json(403, {"error": "Only the host can change the studio settings."})
                return
            with lock:
                old_model = studio["model"]
                updated = clean_studio(data, studio)
                studio.clear()
                studio.update(updated)
                save_studio()
            if studio["model"] != old_model:
                print(">> lyric model: " + studio["model"])
                threading.Thread(target=unload_other_ollama_models, daemon=True).start()
            self.send_json(200, self.config_payload())
        elif self.path.startswith("/api/newbeat"):
            if not self.is_host():
                self.send_json(403, {"error": "Only the host can start a new beat."})
                return
            reset_requested = True
            self.send_json(200, {"ok": True})
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
            if name and name != studio["model"]:
                requests.post(OLLAMA_URL, json={"model": name, "keep_alive": 0}, timeout=30)
                print("Unloaded old lyric model from GPU: " + name)
    except Exception as e:
        print("(couldn't check Ollama models: " + str(e) + ")")


print("=== Freestyle AI (website) - lyrics by " + studio["model"] + " ===")
set_bpm(active_style["bpm"])
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
