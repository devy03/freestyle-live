# Diagnostic: makes 1 normal piece + 1 continuation and saves everything,
# so we can see exactly where the static / cut-off comes from.
import io, json, time, os, requests
import numpy as np, soundfile as sf

ACE_URL = "http://127.0.0.1:8001"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "diag_log.txt"), "w", encoding="utf-8")

def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    log.write(msg + "\n")
    log.flush()

CAPTION = ("freestyle rap, off the top, raw cypher, conversational flow, male rapper, "
           "boom bap drums, minimal beat, dry vocals, 90 bpm")
LINES1 = ["Mama worked doubles, we split one pizza four ways",
          "Now I tip the delivery man like it's his birthday",
          "Cold slice at midnight, that's a Chicago thing",
          "Still taste the struggle every time I take a bite"]
LINES2 = ["Grandma made dinner out of nothing but rice",
          "Told me be humble, never pay the full price",
          "Now I'm at the table where they used to say no",
          "Pull up a chair, let me tell you how it goes"]

def fields(lyrics):
    return {"prompt": CAPTION, "lyrics": lyrics, "thinking": "false",
            "use_cot_caption": "false", "use_cot_language": "false",
            "vocal_language": "en", "bpm": "90", "key_scale": "A Minor",
            "time_signature": "4", "batch_size": "1", "inference_steps": "8",
            "use_random_seed": "false", "seed": "1234", "audio_format": "wav"}

def wait(task_json, name):
    say(name, "release_task reply:", json.dumps(task_json)[:500])
    tid = task_json["data"]["task_id"]
    while True:
        time.sleep(0.3)
        r = requests.post(ACE_URL + "/query_result", json={"task_id_list": [tid]}).json()
        item = r["data"][0]
        if item["status"] != 0:
            break
    say(name, "query_result:", json.dumps(item)[:1500])
    url = json.loads(item["result"])[0]["file"]
    data = requests.get(ACE_URL + url).content
    with open(os.path.join(OUT, name + "_raw.wav"), "wb") as f:
        f.write(data)
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    say(name, "sr=%d samples=%d seconds=%.2f channels=%d peak=%.3f rms=%.3f" %
        (sr, len(audio), len(audio) / sr, audio.shape[1], np.abs(audio).max(), np.sqrt((audio ** 2).mean())))
    return audio, sr

# ---- Test 1: normal piece, 10.67s ----
f = fields("[Verse]\n" + "\n".join(LINES1))
f["audio_duration"] = "10.67"
t = time.time()
a1, sr1 = wait(requests.post(ACE_URL + "/release_task", data=f).json(), "test1_first")
say("test1 took %.1fs" % (time.time() - t))

# ---- Test 2: continuation of test 1 ----
buf = io.BytesIO()
sf.write(buf, a1, sr1, format="WAV", subtype="PCM_16")
ctx = len(a1) / sr1
f = fields("[Verse]\n" + "\n".join(LINES1 + LINES2))
f["task_type"] = "repaint"
f["repainting_start"] = str(ctx)
f["repainting_end"] = str(ctx + 10.67)
f["audio_duration"] = str(ctx + 10.67)
t = time.time()
a2, sr2 = wait(requests.post(ACE_URL + "/release_task", data=f,
               files={"src_audio": ("context.wav", buf.getvalue(), "audio/wav")}).json(), "test2_continue")
say("test2 took %.1fs" % (time.time() - t))

say("DONE - files are in", OUT)
log.close()
