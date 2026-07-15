"""
    python predict.py --data_dir <folder> --out predictions.csv

Loads a SAVED model - does not refit - works on unseen data_dirs.
"""
import argparse, csv, os, pickle
import numpy as np

from features import load_wav, extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.pkl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    clf, scaler = bundle["clf"], bundle["scaler"]

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)

    cache = {}
    keys, X = [], []
    for turn_id, turn_rows in by_turn.items():
        turn_rows.sort(key=lambda r: int(r["pause_index"]))
        prev_durs = []
        for r in turn_rows:
            path = os.path.join(args.data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi = int(r["pause_index"])
            ps = float(r["pause_start"])
            X.append(extract_features(x, sr, ps, pi, list(prev_durs)))
            keys.append((turn_id, r["pause_index"]))
            prev_durs.append(float(r["pause_end"]) - ps)

    X = np.array(X)
    Xs = scaler.transform(X)
    p = clf.predict_proba(Xs)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pidx), prob in zip(keys, p):
            w.writerow([tid, pidx, f"{prob:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
