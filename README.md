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
- [x] Classification baseline evaluated; metrics and confusion matrix in `outputs/`.
- [ ] Clustering baseline evaluated.
- [ ] Preliminary leakage audit documented.
- [ ] Milestone 1 report completed.


## 5. Reproducibility

The project will use documented preprocessing procedures, fixed random seeds where applicable, and version-controlled source code. The repository will contain instructions for reproducing the experiments.

## 6. Classification — Milestone 1

Omar Momen Ahmed (24p0036): untuned Logistic Regression to predict `IsCanceled`
(1 = canceled). `src/classification.py` reuses `make_preprocessor` from the team's
existing `src/PreProcessing.py`: median imputation + RobustScaler for numeric
inputs, most-frequent imputation + one-hot encoding for categories, and imputation
for the repeat-guest binary flag. Fitted preprocessing uses training rows only.

From the repository root (Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/classification.py
```

- Results: [metrics](outputs/classification_metrics.csv) and
  [confusion matrix](outputs/classification_confusion_matrix.png).
- Validation accuracy is 78.74%, F1 is 72.22%, and recall is 67.05%: the untuned
  baseline identifies about two-thirds of cancellations at the fixed threshold.
- Approximately 60/20/20 split by identical-input groups, seeds 42/43, with three-fold
  grouped CV inside training only. Original split membership is preserved; the
  final test is never evaluated in Milestone 1. Threshold is fixed at 0.50, with
  no tuning. Average precision summarizes the precision–recall curve.
  Split keys retain the original `Undefined` channel label to preserve the reserved
  rows; model inputs use the shared loader's missing-value representation.
- Prediction point: initial reservation. Exclude the target, ReservationStatus,
  ReservationStatusDate, AssignedRoomType, BookingChanges and DaysInWaitingList
  because they reveal the outcome or later events. Conservatively exclude ADR,
  DepositType, RequiredCarParkingSpaces and TotalOfSpecialRequests because their
  initial-booking values are not verified. Retained reservation details are assumed
  available initially; H2 is not a versioned snapshot. The split is not chronological.
- Agent/Company are categorical IDs. Raw loading and cleanup directly reuse the
  team's `load_and_clean_data` in `src/clustring.py`; its strings are trimmed and
  NULL/Undefined/nan markers become missing. No separate cleanup is implemented.
  `dataPreparation.py` remains the team's standalone inspection script.
- Fitted pipeline, dataset fingerprint and split manifest are saved locally under
  `artifacts/classification/`, excluded from Git/submission ZIP. The pipeline expects
  the cleaned inputs returned by `load_and_clean_data(path)`. Re-run the script to regenerate.
- Milestone 2: compare additional classifiers, tune on development data, assess
  PCA/LDA and integrate inference before evaluating the reserved test.
