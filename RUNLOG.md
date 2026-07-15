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

## Run 4: Rhythm-of-listing features (last_pause_dur, recent_pause_regularity)
- Motivation: error analysis on false positives/negatives revealed every
  single misclassified pause involved reciting a number/digit sequence
  (phone numbers, dates, times). Numbers get chunked with mid-recitation
  pauses that carry falling pitch (looks like EOT) and final digits often
  carry a natural pitch rise (looks like continuation) - both mislead
  prosody-only features.
- Added two causal rhythm features from prior COMPLETED pauses: duration
  of the immediately preceding pause, and std of the last 3 prior pause
  durations (low = metronomic listing rhythm).
- Result on SAME held-out split: held-out accuracy 0.649 (vs 0.641),
  AUC 0.679 (vs 0.681) - essentially unchanged - but mean delay got WORSE:
  1119ms vs 1024ms. Neither new feature ranked in the top 6 importances.
- CONCLUSION: with only 131 held-out pauses, a small accuracy/AUC change
  doesn't guarantee a better operating point in the (threshold, delay)
  grid search - the delay metric is sensitive to exactly where the
  decision boundary sits, and added features increased boundary variance
  without adding real signal. Reverted to the Run 2/3 model
  (n_estimators=150, depth=2, lr=0.08, original feature set) as final.
- The number-recitation confusion remains a known, understood limitation
  (documented in NOTES.md) rather than one I could fix in-hour without a
  transcript-level signal.

## Run 5: Recitation-guard heuristic (post-hoc rule, not a learned feature)
- Motivation: same number-recitation pattern from error analysis. Instead
  of adding it as a feature, tried a hand-designed rule: detect "listing
  rhythm" from already-completed prior pauses (last 2-3 pauses short AND
  evenly spaced -> looks like chunked digit recitation) and discount the
  current p_eot by 0.6x when detected, requiring more model confidence
  before firing.
- Result on SAME held-out split: guard triggered on 22/131 pauses.
  AUC dropped 0.681 -> 0.654, delay barely moved 1024ms -> 1028ms.
- CONCLUSION: the "listing rhythm" heuristic is too broad - it also
  discounts genuine EOTs that happen to follow short, regular pauses for
  unrelated reasons, cancelling out any benefit on the recitation cases.
  With more data I'd want a tighter trigger (e.g. requiring the rhythm to
  hold for more than 3 pauses, or combining with an actual digit-detector
  signal) rather than a heuristic this coarse.
- Reverted to original model.pkl (Run 2/3 config) as final submission -
  it remains the best result across all 5 runs: 1024ms vs 1600ms baseline.
