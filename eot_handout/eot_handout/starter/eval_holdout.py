import csv, os, pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from features import load_wav, extract_features
from train_final import load_data

data_dirs = ["eot_data/english", "eot_data/hindi"]
cache = {}
Xs_list, ys_list, groups_list, keys_list = [], [], [], []
for d in data_dirs:
    X, y, g, k = load_data(d, cache)
    Xs_list.append(X); ys_list.append(y); groups_list += g; keys_list += k
X = np.concatenate(Xs_list); y = np.concatenate(ys_list)

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["scaler"]

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups_list))

from sklearn.ensemble import GradientBoostingClassifier
Xs = scaler.transform(X)
clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=0)
clf.fit(Xs[tr], y[tr])
p_te = clf.predict_proba(Xs[te])[:, 1]

# write predictions + a matching labels.csv containing ONLY held-out turns
te_keys = [keys_list[i] for i in te]
te_turns = set(tid for tid, _ in te_keys)
with open("predictions_holdout.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["turn_id", "pause_index", "p_eot"])
    for (tid, pidx), prob in zip(te_keys, p_te):
        w.writerow([tid, pidx, f"{prob:.4f}"])

os.makedirs("holdout_eval", exist_ok=True)
with open("holdout_eval/labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["turn_id","audio_file","pause_index","pause_start","pause_end","label"])
    for d in data_dirs:
        for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
            if r["turn_id"] in te_turns:
                w.writerow([r["turn_id"], r["audio_file"], r["pause_index"],
                            r["pause_start"], r["pause_end"], r["label"]])
print(f"held-out: {len(te_turns)} turns, {len(te_keys)} pauses")

# --- try simpler model on same split for comparison ---
for n_est, depth, lr in [(40, 2, 0.08), (60, 1, 0.1), (25, 2, 0.1)]:
    clf2 = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=0)
    clf2.fit(Xs[tr], y[tr])
    p2 = clf2.predict_proba(Xs[te])[:, 1]
    with open("predictions_holdout.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pidx), prob in zip(te_keys, p2):
            w.writerow([tid, pidx, f"{prob:.4f}"])
    from score import score as score_fn
    r = score_fn("holdout_eval/labels.csv", "predictions_holdout.csv")
    print(f"n_est={n_est} depth={depth} lr={lr} -> AUC={r['auc']:.3f} delay={r['latency']*1000:.0f}ms cutoff={r['cutoff']*100:.1f}%")
