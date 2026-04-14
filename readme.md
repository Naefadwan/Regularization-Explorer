# Regularization Explorer

Coefficient-path visualization for telecom churn using logistic regression across a range of regularization strengths.

## Project Goal

Show how model coefficients change as regularization varies, and compare:

- L1-style regularization (sparse, feature selection behavior)
- L2-style regularization (smooth shrinkage behavior)

The script trains models at 20 log-spaced values of `C` from `0.001` to `100`, records coefficients, and produces a side-by-side regularization-path plot.

## Repository Contents

- `regularization_explorer.py` - end-to-end script (load, preprocess, fit, plot, interpret)
- `data/telecom_churn.csv` - dataset
- `artifacts/regularization_path.png` - generated L1 vs L2 coefficient-path plot
- `artifacts/interpretation.md` - generated one-paragraph interpretation
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python regularization_explorer.py
```

Expected output:

- `artifacts/regularization_path.png`
- `artifacts/interpretation.md`

## Method Summary

1. Load telecom churn data.
2. Use `churned` as the target.
3. Drop identifier column (`customer_id`) from features.
4. Preprocess with:
   - `StandardScaler` for numeric columns
   - `OneHotEncoder` for categorical columns
5. Train logistic regression with `solver="saga"` at each `C` value.
6. Repeat for L1-like (`l1_ratio=1.0`) and L2-like (`l1_ratio=0.0`) settings.
7. Plot all feature trajectories against `C` (log scale), one subplot per regularization type.
8. Annotate which features are zeroed first under L1.

## Key Outcome

- L1 drives many coefficients exactly to zero as regularization gets stronger.
- L2 keeps coefficients dense and mainly shrinks magnitudes.
- This makes the trade-off clear:
  - choose L1 for sparsity and feature selection
  - choose L2 for stability and retaining distributed signal

## Notes

- Small `C` means strong regularization.
- Large `C` means weak regularization.
- A small numerical tolerance is used to treat near-zero coefficients as zero when identifying eliminated features.
