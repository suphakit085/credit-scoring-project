"""Compatibility helper for the retired standalone scaler script."""

try:
    from .feature_generation import generate_features as create_features
except ImportError:
    from feature_generation import generate_features as create_features


def main() -> None:
    print("Standalone scaler artifacts are retired.")
    print("Run: python src/model_training.py")


if __name__ == "__main__":
    main()
