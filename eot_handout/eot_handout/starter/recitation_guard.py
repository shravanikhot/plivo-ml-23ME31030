"""
Post-processing heuristic layered on top of the ML model's raw p_eot.

Idea (from manual error analysis): number/digit recitation produces short,
evenly-spaced pauses in a row. When we detect that rhythm in the pauses
ALREADY COMPLETED before this one (fully causal), we discount the current
p_eot - requiring stronger model confidence before we let it fire - since
a mid-recitation breath is easily confused with a true ending.

    python recitation_guard.py
"""
import csv, os, pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from features import load_wav, extract_features
from train_final import load_data

data_dirs = ["eot_data/english", "eot_data/hindi"]
cache = {}
X, y, groups, keys, prev_durs_list = [], [], [], [], []

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
            X.append(extract_features(x, sr, ps, pi, list(prev_durs)))
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(f"{d}:{turn_id}")
            keys.append((turn_id, r["pause_index"]))
            prev_durs_list.append(list(prev_durs))
            prev_durs.append(float(r["pause_end"]) - ps)

X, y = np.array(X), np.array(y)

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["scaler"]
Xs = scaler.transform(X)

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(Xs, y, groups))
clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=0)
clf.fit(Xs[tr], y[tr])
p_raw = clf.predict_proba(Xs[te])[:, 1]


def is_listing_rhythm(prev_durs, short_thresh=0.6, regularity_thresh=0.35):
    """True if the last 2-3 completed pauses look like chunked recitation:
    short and roughly evenly spaced. Purely causal - only uses already-
    completed pause durations."""
    if len(prev_durs) < 2:
        return False
    recent = prev_durs[-3:]
    if not all(d < short_thresh for d in recent):
        return False
    return float(np.std(recent)) < regularity_thresh


DISCOUNT = 0.6  # how much to shrink p_eot when listing rhythm detected

p_adjusted = []
n_guarded = 0
for i, idx in enumerate(te):
    p = p_raw[i]
    if is_listing_rhythm(prev_durs_list[idx]):
        p = p * DISCOUNT
        n_guarded += 1
    p_adjusted.append(p)
p_adjusted = np.array(p_adjusted)

print(f"guard triggered on {n_guarded}/{len(te)} held-out pauses")

te_keys = [keys[i] for i in te]
te_turns = set(tid for tid, _ in te_keys)

for name, p in [("raw", p_raw), ("guarded", p_adjusted)]:
    fname = f"predictions_{name}_holdout.csv"
    with open(fname, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pidx), prob in zip(te_keys, p):
            w.writerow([tid, pidx, f"{prob:.4f}"])

os.makedirs("holdout_eval_guard", exist_ok=True)
with open("holdout_eval_guard/labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["turn_id","audio_file","pause_index","pause_start","pause_end","label"])
    for d in data_dirs:
        for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
            if r["turn_id"] in te_turns:
                w.writerow([r["turn_id"], r["audio_file"], r["pause_index"],
                            r["pause_start"], r["pause_end"], r["label"]])

from score import score as score_fn
r_raw = score_fn("holdout_eval_guard/labels.csv", "predictions_raw_holdout.csv")
r_guard = score_fn("holdout_eval_guard/labels.csv", "predictions_guarded_holdout.csv")
print(f"RAW      -> AUC={r_raw['auc']:.3f} delay={r_raw['latency']*1000:.0f}ms cutoff={r_raw['cutoff']*100:.1f}%")
print(f"GUARDED  -> AUC={r_guard['auc']:.3f} delay={r_guard['latency']*1000:.0f}ms cutoff={r_guard['cutoff']*100:.1f}%")
