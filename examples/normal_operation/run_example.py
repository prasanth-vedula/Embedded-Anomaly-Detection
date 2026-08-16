"""Run the normal-operation TinyML anomaly-detection demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.evaluation.evaluate import evaluate


def main() -> int:
    result = evaluate()
    scenario = result["scenarios"]["normal"]

    output = {
        "scenario": "normal_operation",
        "model": result["architecture"],
        "threshold": result["threshold"],
        "window_count": scenario["window_count"],
        "mean_error": scenario["mean_error"],
        "p95_error": scenario["p95_error"],
        "p99_error": scenario["p99_error"],
        "max_error": scenario["max_error"],
        "anomaly_rate": scenario["anomaly_rate"],
        "expected_behavior": "predominantly normal operation",
    }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())