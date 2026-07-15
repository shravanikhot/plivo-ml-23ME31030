# Run Log

## Run 1: Baseline (silence-only)
- Command: `python baseline.py --data_dir eot_data/english --out baseline_en.csv`
- Score: AUC=0.514, mean delay=1600ms @ 0% interrupted turns
- This is the naive VAD-timeout baseline every candidate must beat.

## Run 2: First model (GradientBoosting, n_estimators=150, depth=2, lr=0.08)
- Features: causal prosodic features (f0 slope/final pitch relative to
  speaker median, final-syllable-length relative to turn average, energy
  decay, voicing ratio, trailing silence, prior-pause statistics, speaking
  rate, pause position) — see features.py:extract_features.
- Trained on English+Hindi combined (248+248 pauses), refit on all data
  after an internal held-out-turn sanity check (0.641 acc vs 0.597 chance).
- Scored predict.py output on the SAME data it was trained on:
  AUC=0.930 (en) / 0.946 (hi), delay=612ms / 600ms.
- CAUGHT ISSUE: this is not a fair estimate — the model saw these exact
  turns during the final refit. Re-evaluated on a proper held-out split
  (25% of turns, GroupShuffleSplit by turn_id, never seen during fit):
  AUC=0.681, mean delay=1024ms @ 4.0% interrupted turns.
- This 1024ms (vs 1600ms baseline) is the number I trust: a 36% delay
  reduction at a lower interruption rate than the 5% budget.

## Run 3: Hyperparameter sweep on same held-out split
| config                        | AUC   | delay   | cutoff |
|--------------------------------|-------|---------|--------|
| n_est=150, depth=2, lr=0.08 (orig) | 0.681 | 1024ms  | 4.0%   |
| n_est=40,  depth=2, lr=0.08    | 0.681 | 1060ms  | 4.0%   |
| n_est=60,  depth=1, lr=0.1     | 0.693 | 1132ms  | 4.0%   |
| n_est=25,  depth=2, lr=0.1     | 0.689 | 1141ms  | 4.0%   |

- Conclusion: the original config still wins on delay despite similar AUC —
  AUC and the operational delay metric don't always agree, since delay
  depends on the specific threshold/delay operating point chosen at the
  5% budget. Kept n_estimators=150, depth=2, lr=0.08 as final.
- Final model.pkl trained on ALL data (both languages) with this config,
  used for the submitted predictions_english.csv / predictions_hindi.csv.
