# CSE382 — Hotel Booking Demand Prediction

## 1. Project Overview

This project develops an end-to-end machine learning system using the City Hotel (H2) booking dataset. The project explores hotel booking data and applies machine learning to regression, classification, and clustering tasks.

### Objectives

- **Regression:** Predict the Average Daily Rate (ADR) of a booking.
- **Classification:** Predict whether a booking will be canceled (`IsCanceled`).
- **Clustering:** Discover groups of similar hotel bookings.
- Compare machine learning models and evaluate their performance.
- Apply preprocessing, feature selection, and dimensionality reduction.
- Develop a usable application for making predictions and viewing results.

## 2. Dataset

- **Dataset:** Hotel Booking Demand — City Hotel (H2)
- **File:** `H2.csv`
- **Number of bookings:** 79,330
- **Number of source variables:** 31

Download the dataset from:

https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv

The downloaded file is inside the `data/` directory.

## 3. Technologies

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Git and GitHub

Additional libraries may be added as the project develops.



## 4. Current Progress — Milestone 1

- [ ] GitHub repository and project structure established.
- [ ] Dataset loaded and verified.
- [ ] Exploratory data analysis completed.
- [ ] Data cleaning and preprocessing implemented.
- [ ] Regression baseline evaluated.
- [x] Classification baseline implemented; reproducible training and validation outputs in `outputs/classification/`.
- [ ] Clustering baseline evaluated.
- [ ] Preliminary leakage audit documented.
- [ ] Milestone 1 report completed.


## 5. Reproducibility

The project will use documented preprocessing procedures, fixed random seeds where applicable, and version-controlled source code. The repository will contain instructions for reproducing the experiments.

## 6. Classification — Milestone 1

Omar Momen Ahmed (24p0036): one untuned Logistic Regression baseline for
`IsCanceled`, with a prior-probability dummy comparison. This contribution covers
classification only; the checklist above remains a team responsibility.

Run these commands **from the repository root** with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.classification train
python -m src.classification predict --input examples/booking.csv
python -m unittest discover -s tests -v
```

`requirements.txt` pins the direct project dependencies. For the complete versions
used in the recorded run, install `requirements-lock.txt` instead (Python 3.12).

The example is a manually constructed booking for demonstrating the input schema,
not a reserved test row. Prediction loads the fitted model without retraining.
All 21 selected columns are required; missing cell values and unseen categories
are supported. Extra columns, including the target, are ignored.

- [Experiment report, metrics and leakage audit](outputs/classification/report.md)
- [Arabic walkthrough](docs/classification_explained_ar.md)
- [Validation plots](outputs/classification/validation_plots.png)
- Training entry point: `src/classification.py`.
- Reusable fitted pipeline: `artifacts/classification/logistic_regression.joblib`.
- Locked split membership: `artifacts/classification/split_manifest.csv` (zero-based
  row positions; valid only for the dataset SHA-256 in `metadata.json`).

Split: approximately 60/20/20 by identical-input groups, seed 42; three-fold grouped
CV on **training only**. The **final test is reserved and never evaluated in Milestone 1**.
The model and all learned imputation/scaling/encoding parameters use training rows
only. Model parameters and the 0.50 threshold are not tuned. See the report for
booking-time assumptions, conservative feature exclusions and limitations.

Training regenerates `outputs/classification/` and `artifacts/classification/`.
Keep the dataset, seed and split manifest unchanged for Milestone 2. Model artifacts
are intentionally gitignored and should be excluded from the submission ZIP;
recreate them with the training command. Only load trusted joblib artifacts.
Existing regression/clustering scripts are outside this contribution's validation.
