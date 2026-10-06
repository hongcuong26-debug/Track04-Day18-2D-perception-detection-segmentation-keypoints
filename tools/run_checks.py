"""Run the notebook's original checks on CPU without models or datasets.

Usage (PowerShell): .venv\Scripts\python.exe tools\run_checks.py
"""

import ast
import importlib.util
import json
import linecache
import math
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".uv-cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".uv-cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").mkdir(exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import torch
import yaml


def main():
    path = ROOT / "lab_2d_perception_student.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    cells = notebook["cells"]
    namespace = dict(math=math, np=np, torch=torch, cv2=cv2, plt=plt,
                     yaml=yaml, Path=Path)

    def run_cell(index):
        source = "".join(cells[index]["source"])
        filename = f"{path}:cell_{index}"
        # inspect.getsource() in the original checker must see real source.
        linecache.cache[filename] = (len(source), None,
                                    source.splitlines(keepends=True), filename)
        exec(compile(source, filename, "exec"), namespace)

    # Validate Python syntax without running downloads, inference or training.
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if not source.lstrip().startswith("%pip"):
                ast.parse(source, filename=f"cell_{index}")

    run_cell(5)
    for index in (16, 18, 20, 32, 40, 42, 59, 66):
        source = "".join(cells[index]["source"])
        assert not any("..." in line and "🧩" in line
                       for line in source.splitlines()), f"Unfilled cell {index}"
        run_cell(index)

    # Use the actual installed package YAML, without downloading tiger images.
    spec = importlib.util.find_spec("ultralytics")
    dataset_yaml = Path(spec.origin).parent / "cfg" / "datasets" / "tiger-pose.yaml"
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    namespace["TIGER_KPTS"] = cfg["kpt_names"][0]
    run_cell(79)
    names, flip = namespace["TIGER_KPTS"], namespace["FLIP_IDX"]
    mirror = namespace["_mirror_name"]
    assert flip == [names.index(mirror(name)) for name in names]
    assert all(flip[flip[i]] == i for i in range(len(flip)))
    print("Tiger keypoints:", dict(enumerate(names)))
    print("Anatomical FLIP_IDX:", flip)

    expected = ("box_iou", "nms", "batched_nms", "average_precision",
                "mask_iou", "polygon_to_mask", "mask_to_yolo_seg", "oks",
                "joint_angle", "FLIP_IDX")
    progress = namespace["PROGRESS"]
    failed = [name for name in expected if progress.get(name) != "ok"]
    assert not failed, f"Failed checks: {failed}"

    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("Q"):
            run = compile(source, "answer", "exec")
            exec(run, namespace)
    for index in range(1, 13):
        answer = namespace[f"Q{index}"]
        assert len(answer.strip()) >= 30
        assert namespace["PLACEHOLDER"] not in answer

    # Edge cases not all covered by the original suite.
    boxes = torch.empty((0, 4))
    assert namespace["box_iou"](boxes, torch.ones((2, 4))).shape == (0, 2)
    assert namespace["nms"](boxes, torch.empty(0)).numel() == 0
    assert namespace["mask_to_yolo_seg"](np.zeros((20, 30), bool), 0) == ""
    assert namespace["average_precision"](np.array([]), np.array([]), 0) == 0
    assert namespace["oks"](np.zeros((17, 2)), np.ones((17, 2)), 100,
                             vis=np.zeros(17, bool)) == 0
    for section in ("1B", "2B", "3B", "3C", "4A"):
        namespace["gate"](section)

    print(f"PASS: {len(expected)}/{len(expected)} original checks, edge cases, "
          "5 gates, notebook schema and Python syntax.")
    print("NOT RUN: weights/inference, latency, SAM autolabel, GPU training, "
          "mAP, Q11 observations, final_report(). Run these on Colab T4.")
    print("Q1-Q12 contain answer drafts; replace <<ĐIỀN SỐ TỪ OUTPUT>> "
          "before submitting. No submission metrics were generated locally.")


if __name__ == "__main__":
    main()
