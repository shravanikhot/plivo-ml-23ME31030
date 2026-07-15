import csv, os, pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from features import load_wav, extract_features_v2, FEATURE_NAMES_V2

data_dirs = ["eot_data/english", "eot_data/hindi"]
cache = {}
X, y, groups, keys = [], [], [], []

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
            X.append(extract_features_v2(x, sr, ps, pi, list(prev_durs)))
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(f"{d}:{turn_id}")
            keys.append((turn_id, r["pause_index"]))
            prev_durs.append(float(r["pause_end"]) - ps)

X, y = np.array(X), np.array(y)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(Xs, y, groups))
clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=0)
clf.fit(Xs[tr], y[tr])

print("held-out acc:", clf.score(Xs[te], y[te]))
for name, imp in sorted(zip(FEATURE_NAMES_V2, clf.feature_importances_), key=lambda t: -t[1])[:6]:
    print(f"  {name:24s} {imp:.3f}")

p_te = clf.predict_proba(Xs[te])[:, 1]
te_keys = [keys[i] for i in te]
te_turns = set(tid for tid, _ in te_keys)
with open("predictions_v2_holdout.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["turn_id", "pause_index", "p_eot"])
    for (tid, pidx), prob in zip(te_keys, p_te):
        w.writerow([tid, pidx, f"{prob:.4f}"])

os.makedirs("holdout_eval_v2", exist_ok=True)
with open("holdout_eval_v2/labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["turn_id","audio_file","pause_index","pause_start","pause_end","label"])
    for d in data_dirs:
        for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
            if r["turn_id"] in te_turns:
                w.writerow([r["turn_id"], r["audio_file"], r["pause_index"],
                            r["pause_start"], r["pause_end"], r["label"]])

# also save full model for potential submission
clf.fit(Xs, y)
with open("model_v2.pkl", "wb") as f:
    pickle.dump({"clf": clf, "scaler": scaler}, f)
