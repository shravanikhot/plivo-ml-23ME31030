My model uses causal prosodic features computed only on audio before each
pause: final pitch relative to the speaker's median F0, final-syllable
lengthening relative to the turn's average voiced-run duration, short-term
energy decay, voicing ratio, trailing silence fraction, speaking rate, and
statistics of the turn's prior (already-completed) pause durations. These
are fed into a GradientBoostingClassifier (150 trees, depth 2) trained on
English+Hindi combined. On a proper held-out split of turns (never seen
during training), it reaches AUC 0.681 and a mean response delay of 1024ms
at a 4% interruption rate, versus the silence-only baseline's 1600ms at 0% —
a genuine ~36% latency reduction. Held-out accuracy (0.641) is only modestly
above chance (0.597), so similarity plateaued because 250-ish pauses per
language is a small dataset for 12 features, and prosodic cues alone
(without lexical/semantic content) can't fully disambiguate short holds from
true ends. With one more day I'd add lexical completion cues (e.g. from a
lightweight causal language model over the transcript-so-far) and more
training turns to reduce variance.
