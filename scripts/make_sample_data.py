#!/usr/bin/env python3
"""Create a small synthetic sample dataset for trying boxtrot: 30 georeferenced 4-band scenes
in two "areas", with known objects and a simulated detector's output (misses, false alarms,
scores, two sources).

    python scripts/make_sample_data.py ~/boxtrot-sample
    boxtrot      # New project -> imagery: ~/boxtrot-sample/images,
                   #                detections: ~/boxtrot-sample/detections

The imagery is synthetic (dark "water", a vegetated land strip, bright elongated objects), so
the colours are deliberately plain; use it to learn the workflow, not to judge detectors.
"""
import argparse
import json
from pathlib import Path

from boxtrot.testing import make_detections, make_scene_set


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="folder to create")
    ap.add_argument("-n", type=int, default=30, help="number of scenes (default 30)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    truth = make_scene_set(a.out / "images", n=a.n, seed=a.seed)
    n_det = make_detections(a.out / "detections", a.out / "images", truth, seed=a.seed + 1)
    (a.out / "truth.json").write_text(json.dumps({k: v[0] for k, v in truth.items()}, indent=1))
    print(f"{len(truth)} scenes, {sum(len(v[0]) for v in truth.values())} true objects, {n_det} detections -> {a.out}")
    print(f"truth boxes (pixel coordinates per scene) are in {a.out / 'truth.json'}")


if __name__ == "__main__":
    main()
