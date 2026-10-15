# SkySentry baseline evaluation dataset

## What is included

- `labelled_manifest.csv`: all 5,491 one-second audio segments, with binary label (`drone=1`, `background=0`), provided interval-disjoint split, source group, available source timestamps, features, and RF pairing status.
- `predictions.csv`: per-segment model probabilities for audit and error analysis.
- `metrics.json`: validation and held-out test results, including confusion counts, balanced accuracy, sensitivity, specificity, false-positive rate, and an explicitly assumption-based false positives/hour estimate.
- `logistic_regression.json` and `gaussian_naive_bayes.json`: fitted baseline parameters and feature scaling.
- `../train_baselines.js`: dependency-free feature extraction, leakage checks, training, threshold selection, and evaluation.

## Data and labels

The manifest uses `Dataset/itu-aris-dataset`, version 1.0.0. Segment labels come from the corpus's hand annotations and folder labels. The original 1 s mono PCM files are retained in place and are not duplicated. Drone timestamps are reconstructed from the annotated source interval start plus the known 0.5 s hop. Background source timestamps are absent from the supplied metadata and remain blank. No times are fabricated.

The corpus is one continuous 3.12-hour recording session, not a set of independent flight/session recordings. Therefore a true flight/session-held-out evaluation cannot be done with the supplied data. The strongest available protocol is the corpus-provided `split_interval_disjoint.json`: segments from each annotated drone interval and each contiguous background block stay in one split. The training script checks this invariant and stops if a group crosses splits. This prevents overlapping audio windows from appearing across train, validation, and test, but it does not measure generalization to a new recording session or flight.

The `Dataset/DroneRF` directory contains RAR files from a separate RF corpus. There are no shared recording IDs, timestamps, or paired audio/RF observations linking them to the ITU audio. They are consequently not joined into this manifest or used as synchronized fusion examples. `rf_pair_status` records that no paired RF recording is available for each audio segment. These data do not support a synchronized multimodal model.

## Features and models

Thirteen deterministic features are computed from each waveform: RMS, log RMS, mean absolute amplitude, peak amplitude, zero-crossing rate, and eight log mean-squared first-difference features at power-of-two sample strides. They are simple signal baselines, not learned spectrogram or CNN features. Logistic regression is trained by deterministic batch gradient descent; Gaussian Naive Bayes uses training-only per-class means and variances. Feature scaling is fit on training rows only. Model thresholds are selected by F1 on validation rows only, then held fixed for test evaluation. No test rows affect fitting or threshold choice.

## Results

See `metrics.json` for exact counts and metrics. On the 824 test windows (674 drone, 150 background), logistic regression at validation-selected threshold 0.59 achieved 97.69% accuracy, 94.70% balanced accuracy, 99.41% recall, and a 10.00% false-positive rate (15 false positives among 150 background windows). Gaussian Naive Bayes at threshold 0.28 achieved 92.48% accuracy, 84.78% balanced accuracy, 96.88% recall, and a 27.33% false-positive rate (41/150).

The false-positives-per-hour field assumes each background window represents one unique second. Treat it only as a rough window-rate conversion: the available metadata does not provide exact background source timestamps, so it cannot establish unique continuous negative recording time or an operational alarm rate. The background-window false-positive rate and counts are the defensible measures here.

## Reproduce

Run from the project root with Node.js:

```text
node train_baselines.js
```

The script writes the five generated data/model files to `evaluation/`. It uses only Node.js built-ins. The seed is recorded for reproducibility; this implementation is deterministic and has no randomized fitting step.

## Probability visuals

From the project root, run:

```text
node generate_probability_visuals.mjs
```

This reads the saved held-out test predictions and writes four SVG images plus `index.html` under `output/visuals/`: a feature scatter colored by each model's drone probability, and a probability distribution by true class, for each model. Open `output/visuals/index.html` in a browser to view them together. The scatter uses log RMS and zero-crossing rate only as a two-dimensional view; the models still predict from all 13 features, so the plot is not a two-feature decision boundary.

## Limitations

- There are no paired simultaneous acoustic and RF recordings in the supplied folders, so synchronization and sensor fusion could not be evaluated.
- The held-out split separates source intervals/blocks but comes from one continuous session; it is not a flight/session split.
- Background window times are not included in the source metadata, limiting time-based false-alarm rates.
- The models and compact waveform features are baselines for this corpus, not evidence of field performance or calibrated confidence.
- The acoustic corpus is CC BY 4.0; see its source `LICENSE` and metadata for attribution.
