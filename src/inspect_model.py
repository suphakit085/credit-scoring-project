"""Print model metadata and the most important selected features."""

from __future__ import annotations

import pandas as pd

try:
    from .inference import DEFAULT_BUNDLE_PATH, load_bundle
except ImportError:
    from inference import DEFAULT_BUNDLE_PATH, load_bundle


def main() -> None:
    bundle = load_bundle(DEFAULT_BUNDLE_PATH)
    model = bundle["model"]
    importance = pd.DataFrame(
        {
            "feature": bundle["selected_feature_names"],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(f"Bundle version: {bundle['bundle_version']}")
    print(f"Created at: {bundle['created_at']}")
    print(f"Decision threshold: {bundle['decision_threshold']:.6f}")
    print(f"Validation metrics: {bundle.get('metrics', {})}")
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
