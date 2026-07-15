import csv, os
import soundfile as sf
import numpy as np

# Hindi eot pauses only, with preceding context, to check finisher-word intuition
rows = list(csv.DictReader(open("eot_data/hindi/labels.csv")))
by_turn = {}
for r in rows:
    by_turn.setdefault(r["turn_id"], []).append(r)

os.makedirs("clips_hindi", exist_ok=True)
count = 0
for turn_id, turn_rows in sorted(by_turn.items())[:12]:
    turn_rows.sort(key=lambda r: int(r["pause_index"]))
    path = os.path.join("eot_data/hindi", turn_rows[0]["audio_file"])
    x, sr = sf.read(path)
    for r in turn_rows:
        if r["label"] != "eot":
            continue
        ps = float(r["pause_start"])
        start = max(0, ps - 2.0)
        s, e = int(start * sr), int(min(ps + 0.3, len(x)/sr) * sr)
        name = f"clips_hindi/eot_{turn_id}_p{r['pause_index']}.wav"
        sf.write(name, x[s:e], sr)
        print(f"wrote {name}  (last 2s before eot pause)")
        count += 1
print(f"total: {count} clips")
