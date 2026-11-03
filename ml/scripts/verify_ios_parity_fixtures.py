"""Verify frozen Python artifacts used as numerical references by the iOS port."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ML_ROOT / "fixtures" / "ios_parity_v1.json"


def canonical_digest(document: dict, keys: list[str]) -> str:
    selected = {key: document[key] for key in keys}
    payload = json.dumps(
        selected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported iOS parity fixture schema")
    results: list[dict] = []
    for case in manifest.get("cases", []):
        artifact_path = ML_ROOT / case["artifact"]
        with artifact_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        actual_digest = canonical_digest(document, case["digest_keys"])
        if actual_digest != case["canonical_sha256"]:
            raise ValueError(f"{case['id']} canonical digest changed")
        for key, expected in case.get("expected_counts", {}).items():
            if len(document[key]) != expected:
                raise ValueError(f"{case['id']} {key} count changed")
        frames = case.get("expected_frames", {})
        frame_keys = {
            "launch": "launch_frame",
            "termination": "termination_frame",
            "observation_end": "observation_end_frame",
        }
        for fixture_key, expected in frames.items():
            if document.get(frame_keys[fixture_key]) != expected:
                raise ValueError(f"{case['id']} {fixture_key} frame changed")
        results.append({"id": case["id"], "status": "verified", "artifact": str(artifact_path)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    results = verify_manifest(args.manifest)
    for result in results:
        print(f"verified {result['id']}: {result['artifact']}")


if __name__ == "__main__":
    main()
