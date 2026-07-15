"""
Train on causal prosodic features (features.py), save fitted model+scaler.

    python train_final.py --data_dirs eot_data/english eot_data/hindi --model_out model.pkl
"""
import argparse, csv, os, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

from features import load_wav, extract_features, FEATURE_NAMES


def load_data(data_dir, cache):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)

    X, y, groups, keys = [], [], [], []
    for turn_id, turn_rows in by_turn.items():
        turn_rows.sort(key=lambda r: int(r["pause_index"]))
        prev_durs = []
        for r in turn_rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi = int(r["pause_index"])
            ps = float(r["pause_start"])
            X.append(extract_features(x, sr, ps, pi, list(prev_durs)))
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(f"{data_dir}:{turn_id}")
            keys.append((turn_id, r["pause_index"]))
            prev_durs.append(float(r["pause_end"]) - ps)
    return np.array(X), np.array(y), groups, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--model_out", default="model.pkl")
    args = ap.parse_args()

    cache = {}
    Xs_list, ys_list, groups = [], [], []
    for d in args.data_dirs:
        X, y, g, _ = load_data(d, cache)
        Xs_list.append(X); ys_list.append(y); groups += g
    X = np.concatenate(Xs_list, axis=0)
    y = np.concatenate(ys_list, axis=0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(Xs, y, groups))

    clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=0)
    clf.fit(Xs[tr], y[tr])
    acc = clf.score(Xs[te], y[te])
    print(f"held-out turn accuracy: {acc:.3f} (chance ~ {max(np.mean(y), 1 - np.mean(y)):.3f})")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:22s} {imp:.3f}")

    clf.fit(Xs, y)
    with open(args.model_out, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    print(f"saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
