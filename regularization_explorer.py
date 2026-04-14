from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Input/output paths for this mini-project.
DATA_PATH = Path("data/telecom_churn.csv")
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "regularization_path.png"
INTERPRETATION_PATH = ARTIFACTS_DIR / "interpretation.md"

# Modeling constants.
TARGET = "churned"
C_VALUES = np.logspace(-3, 2, 20)
ZERO_TOL = 1e-6


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing to scale numeric and one-hot encode categorical columns."""
    # Detect feature types automatically from the dataframe.
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features.select_dtypes(exclude=["number"]).columns.tolist()

    # Apply the proper transform per column type before modeling.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def fit_path(
    x: pd.DataFrame, y: pd.Series, penalty: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit LogisticRegression across all C values and return coefficient paths."""
    # Fit preprocessing once and reuse the transformed matrix for all C values.
    preprocessor = build_preprocessor(x)
    transformed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Store one coefficient vector per C value.
    coefs = []
    # New sklearn versions prefer l1_ratio instead of penalty.
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    for c_value in C_VALUES:
        # Train with saga so both L1-like and L2-like variants are supported.
        model = LogisticRegression(
            solver="saga",
            l1_ratio=l1_ratio,
            C=c_value,
            max_iter=5000,
            random_state=42,
        )
        model.fit(transformed, y)
        coefs.append(model.coef_[0])

    # Shape is (num_c_values, num_features).
    return C_VALUES, np.vstack(coefs), feature_names


def first_l1_zero_points(
    c_values: np.ndarray, coefs: np.ndarray, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Find the first C value where each feature reaches ~zero under L1."""
    first_zeroed: list[tuple[str, float]] = []

    # Move from weak -> strong regularization (high C to low C).
    reverse_cs = c_values[::-1]
    reverse_coefs = coefs[::-1]

    for feat_idx, feat_name in enumerate(feature_names):
        # Track the first point where a feature coefficient becomes effectively zero.
        abs_path = np.abs(reverse_coefs[:, feat_idx])
        zero_indices = np.where(abs_path <= ZERO_TOL)[0]
        if len(zero_indices) > 0:
            first_zeroed.append((feat_name, reverse_cs[zero_indices[0]]))

    # Earlier elimination corresponds to larger C in the weak->strong traversal.
    first_zeroed.sort(key=lambda item: item[1], reverse=True)
    return first_zeroed


def make_plot(
    c_values: np.ndarray,
    l1_coefs: np.ndarray,
    l2_coefs: np.ndarray,
    feature_names: list[str],
    zeroed_features: list[tuple[str, float]],
) -> None:
    """Create side-by-side L1 and L2 coefficient trajectory plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Draw one line per transformed feature on each subplot.
    for feat_idx, feat_name in enumerate(feature_names):
        axes[0].plot(c_values, l1_coefs[:, feat_idx], alpha=0.55, linewidth=1)
        axes[1].plot(c_values, l2_coefs[:, feat_idx], alpha=0.55, linewidth=1)

    axes[0].set_title("L1 (Lasso) Regularization Path")
    axes[1].set_title("L2 (Ridge) Regularization Path")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("C (inverse regularization strength, log scale)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Coefficient value")

    # Add compact annotation listing the earliest zeroed L1 features.
    annotation_lines = ["First zeroed under L1 (weak->strong):"]
    top_zeroed = zeroed_features[:5]
    if top_zeroed:
        annotation_lines.extend([f"- {name} @ C={c_val:.3g}" for name, c_val in top_zeroed])
    else:
        annotation_lines.append("- No exact zeros found")

    axes[0].text(
        0.02,
        0.02,
        "\n".join(annotation_lines),
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    fig.suptitle("Telecom Churn: Coefficient Trajectories Across Regularization Strengths")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(
    l1_coefs: np.ndarray, l2_coefs: np.ndarray, feature_names: list[str], zeroed: list[tuple[str, float]]
) -> str:
    """Generate a one-paragraph interpretation based on summary path statistics."""
    # C_VALUES is ascending, so index 0 is strongest regularization (smallest C).
    strongest_idx = 0
    # Index -1 corresponds to weakest regularization (largest C).
    weakest_idx = -1

    # Count exact/near-zero coefficients at strongest regularization for each penalty.
    l1_zero_at_strong = int(np.sum(np.abs(l1_coefs[strongest_idx]) <= ZERO_TOL))
    l2_zero_at_strong = int(np.sum(np.abs(l2_coefs[strongest_idx]) <= ZERO_TOL))

    # Identify top coefficients under weak regularization as "stable" high-impact features.
    top_stable_idx = np.argsort(np.abs(l2_coefs[weakest_idx]))[::-1][:3]
    top_stable_features = ", ".join(feature_names[i] for i in top_stable_idx)
    early_zeroed = ", ".join(f"{name} (C={c_val:.3g})" for name, c_val in zeroed[:3]) or "none"

    return (
        "Across the regularization path, L1 drives many coefficients exactly to zero as C decreases, "
        f"with {l1_zero_at_strong} features eliminated at the strongest regularization setting, while L2 "
        f"keeps coefficients dense (only {l2_zero_at_strong} exact zeros) and mainly shrinks magnitudes. "
        f"The largest-magnitude predictors under weak regularization are {top_stable_features}, suggesting "
        "they remain influential even as penalties change, whereas weaker signals are pruned first under L1 "
        f"(earliest eliminations: {early_zeroed}). For this dataset, L2 is the safer default when preserving "
        "all signal is important, while L1 is preferable when feature selection and model sparsity are priorities."
    )


def main() -> None:
    """Run the full workflow: load data, fit models, plot, and write interpretation."""
    # Fail early if the dataset is missing.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # Ensure output directory exists.
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Read dataset and split into features/target.
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET, "customer_id"], errors="ignore")
    y = df[TARGET]

    # Fit both regularization variants across the same C grid.
    c_values, l1_coefs, feature_names = fit_path(x, y, penalty="l1")
    _, l2_coefs, _ = fit_path(x, y, penalty="l2")

    # Build outputs required by the assignment.
    zeroed_features = first_l1_zero_points(c_values, l1_coefs, feature_names)
    make_plot(c_values, l1_coefs, l2_coefs, feature_names, zeroed_features)

    interpretation = build_interpretation(l1_coefs, l2_coefs, feature_names, zeroed_features)
    INTERPRETATION_PATH.write_text(interpretation + "\n", encoding="utf-8")

    print(f"Saved plot to: {PLOT_PATH}")
    print(f"Saved interpretation to: {INTERPRETATION_PATH}")


if __name__ == "__main__":
    # Entry point for script execution.
    main()
