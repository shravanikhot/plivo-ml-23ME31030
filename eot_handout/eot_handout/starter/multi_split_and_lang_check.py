import csv, os, pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from features import load_wav, extract_features
from score import score as score_fn

def load_lang(d, cache):
    rows = list(csv.DictReader(open(os.path.join(d, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    X, y, groups, keys = [], [], [], []
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
            prev_durs.append(float(r["pause_end"]) - ps)
    return np.array(X), np.array(y), groups, keys

cache = {}
X_en, y_en, g_en, k_en = load_lang("eot_data/english", cache)
X_hi, y_hi, g_hi, k_hi = load_lang("eot_data/hindi", cache)
X_all = np.concatenate([X_en, X_hi]); y_all = np.concatenate([y_en, y_hi])
g_all = g_en + g_hi

def run_split(X, y, groups, seed, tag):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(Xs, y, groups))
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=0)
    clf.fit(Xs[tr], y[tr])
    p = clf.predict_proba(Xs[te])[:, 1]
    te_turns = set(groups[i].split(":")[-1] for i in te)
    return te, p, te_turns

print("="*60)
print("MULTI-SPLIT STABILITY CHECK (combined en+hi, 5 seeds)")
print("="*60)
for seed in range(5):
    te, p, te_turns = run_split(X_all, y_all, g_all, seed, f"seed{seed}")
    keys_all = k_en + k_hi
    te_keys = [keys_all[i] for i in te]
    with open("tmp_pred.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["turn_id","pause_index","p_eot"])
        for (tid,pidx), prob in zip(te_keys, p):
            w.writerow([tid, pidx, f"{prob:.4f}"])
    with open("tmp_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id","audio_file","pause_index","pause_start","pause_end","label"])
        for d in ["eot_data/english", "eot_data/hindi"]:
            for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
                if r["turn_id"] in te_turns:
                    w.writerow([r["turn_id"], r["audio_file"], r["pause_index"],
                                r["pause_start"], r["pause_end"], r["label"]])
    r = score_fn("tmp_labels.csv", "tmp_pred.csv")
    print(f"  seed={seed}: AUC={r['auc']:.3f} delay={r['latency']*1000:.0f}ms cutoff={r['cutoff']*100:.1f}%")

print()
print("="*60)
print("HINDI-ONLY vs COMBINED MODEL, evaluated on HINDI held-out only")
print("="*60)

# combined-trained model, scored only on hindi portion of its held-out set
te, p, te_turns = run_split(X_all, y_all, g_all, 0, "combined")
te_keys = [ (k_en+k_hi)[i] for i in te ]
is_hindi = [g_all[i].startswith("eot_data/hindi") for i in te]
hi_idx = [i for i,flag in enumerate(is_hindi) if flag]
with open("tmp_pred.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["turn_id","pause_index","p_eot"])
    for i in hi_idx:
        tid,pidx = te_keys[i]; w.writerow([tid,pidx,f"{p[i]:.4f}"])
hi_te_turns = set(te_keys[i][0] for i in hi_idx)
with open("tmp_labels.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["turn_id","audio_file","pause_index","pause_start","pause_end","label"])
    for r in csv.DictReader(open("eot_data/hindi/labels.csv")):
        if r["turn_id"] in hi_te_turns:
            w.writerow([r["turn_id"], r["audio_file"], r["pause_index"], r["pause_start"], r["pause_end"], r["label"]])
r_combined = score_fn("tmp_labels.csv", "tmp_pred.csv")
print(f"  combined model on Hindi-only held-out: AUC={r_combined['auc']:.3f} delay={r_combined['latency']*1000:.0f}ms cutoff={r_combined['cutoff']*100:.1f}%")

# hindi-only trained model
te_hi, p_hi, te_turns_hi = run_split(X_hi, y_hi, g_hi, 0, "hindi_only")
te_keys_hi = [k_hi[i] for i in te_hi]
with open("tmp_pred.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["turn_id","pause_index","p_eot"])
    for (tid,pidx),prob in zip(te_keys_hi, p_hi):
        w.writerow([tid,pidx,f"{prob:.4f}"])
with open("tmp_labels.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["turn_id","audio_file","pause_index","pause_start","pause_end","label"])
    for r in csv.DictReader(open("eot_data/hindi/labels.csv")):
        if r["turn_id"] in te_turns_hi:
            w.writerow([r["turn_id"], r["audio_file"], r["pause_index"], r["pause_start"], r["pause_end"], r["label"]])
r_hindi_only = score_fn("tmp_labels.csv", "tmp_pred.csv")
print(f"  Hindi-ONLY-trained model on Hindi held-out:   AUC={r_hindi_only['auc']:.3f} delay={r_hindi_only['latency']*1000:.0f}ms cutoff={r_hindi_only['cutoff']*100:.1f}%")
