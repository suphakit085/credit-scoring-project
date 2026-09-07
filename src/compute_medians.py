"""Export human-readable defaults from the current model bundle."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .inference import DEFAULT_BUNDLE_PATH, load_bundle
except ImportError:
    from inference import DEFAULT_BUNDLE_PATH, load_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bundle = load_bundle(DEFAULT_BUNDLE_PATH)
    path = PROJECT_ROOT / "data" / "processed" / "input_defaults.json"
    path.write_text(
        json.dumps(bundle["input_defaults"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(bundle['input_defaults'])} defaults to {path}")


if __name__ == "__main__":
    main()
