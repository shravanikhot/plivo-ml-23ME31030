"""
Find worst misclassifications on the held-out split, print details,
and list exact wav files + timestamps to go listen to.
"""
import csv, os, pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from features import load_wav, extract_features
from train_final import load_data

data_dirs = ["eot_data/english", "eot_data/hindi"]
cache = {}
Xs_list, ys_list, groups_list, keys_list, meta_list = [], [], [], [], []

for d in data_dirs:
    rows = list(csv.DictReader(open(os.path.join(d, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    for turn_id, turn_rows in by_turn.items():
        turn_rows.sort(key=lambda r: int(r["pause_index"]))
        prev_durs = []
        for r in turn_rows:
            path = os.path.join(d, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi = int(r["pause_index"]); ps = float(r["pause_start"])
            Xs_list.append(extract_features(x, sr, ps, pi, list(prev_durs)))
            ys_list.append(1 if r["label"] == "eot" else 0)
            groups_list.append(f"{d}:{turn_id}")
            keys_list.append((turn_id, r["pause_index"]))
            meta_list.append({"data_dir": d, "audio_file": r["audio_file"],
                               "pause_start": ps, "pause_end": float(r["pause_end"]),
                               "label": r["label"]})
            prev_durs.append(float(r["pause_end"]) - ps)

X = np.array(Xs_list); y = np.array(ys_list)

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["scaler"]
Xs = scaler.transform(X)

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(Xs, y, groups_list))
clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=0)
clf.fit(Xs[tr], y[tr])
p_te = clf.predict_proba(Xs[te])[:, 1]
y_te = y[te]

results = []
for i, idx in enumerate(te):
    results.append({
        "turn_id": keys_list[idx][0], "pause_index": keys_list[idx][1],
        "true_label": meta_list[idx]["label"], "p_eot": p_te[i],
        "audio_file": meta_list[idx]["audio_file"], "data_dir": meta_list[idx]["data_dir"],
        "pause_start": meta_list[idx]["pause_start"], "pause_end": meta_list[idx]["pause_end"],
    })

# false positives: true hold, but high p_eot (model wrongly confident it's over)
fp = sorted([r for r in results if r["true_label"] == "hold"], key=lambda r: -r["p_eot"])[:8]
# false negatives: true eot, but low p_eot (model wrongly confident it's not over)
fn = sorted([r for r in results if r["true_label"] == "eot"], key=lambda r: r["p_eot"])[:8]

print("="*70)
print("FALSE POSITIVES (true=hold, model says high p_eot -> would cut user off)")
print("="*70)
for r in fp:
    print(f"  {r['data_dir']}/{r['audio_file']}  pause@{r['pause_start']:.2f}s-{r['pause_end']:.2f}s"
          f"  turn={r['turn_id']} idx={r['pause_index']}  p_eot={r['p_eot']:.3f}")

print()
print("="*70)
print("FALSE NEGATIVES (true=eot, model says low p_eot -> would keep waiting)")
print("="*70)
for r in fn:
    print(f"  {r['data_dir']}/{r['audio_file']}  pause@{r['pause_start']:.2f}s-{r['pause_end']:.2f}s"
          f"  turn={r['turn_id']} idx={r['pause_index']}  p_eot={r['p_eot']:.3f}")

print()
print(f"Held-out: {len(te)} pauses, {sum(y_te)} true EOT, {len(y_te)-sum(y_te)} true HOLD")
