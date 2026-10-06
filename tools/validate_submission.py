"""Audit real Colab deliverables before pushing the submission to GitHub."""

import ast
import json
import sys
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    notebook = nbformat.read(ROOT / "lab_2d_perception_student.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 94, "Expected original 94 cells"
    results = json.loads((ROOT / "submission/ket_qua.json").read_text(encoding="utf-8"))
    progress = results["progress"]
    required = ("box_iou", "nms", "batched_nms", "mask_iou", "polygon_to_mask",
                "mask_to_yolo_seg", "oks", "joint_angle", "FLIP_IDX", "autolabel")
    assert all(progress.get(name) == "ok" for name in required), progress
    assert "lifeline" not in progress.values()
    assert len(results["latency"]) == 4
    assert results["env"]["device"] == "cuda", results["env"]
    trained = results["result_4b"]
    assert trained["epochs"] == 40 and trained["imgsz"] == 640, trained
    assert 0 <= trained["Pose mAP50-95"] <= trained["Pose mAP50"] <= 1

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 53
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells
                   for output in cell.outputs), "Notebook contains error outputs"
    assert sum(bool(cell.outputs) for cell in code_cells) >= 30, "Missing output evidence"
    for cell in code_cells:
        if cell.source.startswith("Q"):
            statement = ast.parse(cell.source).body[0]
            name = statement.targets[0].id
            answer = ast.literal_eval(statement.value).strip()
            assert len(answer) >= 30 and "ĐIỀN SỐ" not in answer
            assert "viết câu trả lời của bạn" not in answer
            assert results["answers"][name] == answer, f"Stale answer in JSON: {name}"
    assert len(results["answers"]) == 12
    train_text = "".join(output.get("text", "") for output in notebook.cells[82].outputs)
    assert "40 epochs completed" in train_text and "Tesla T4" in train_text
    assert "imgsz=640" in train_text

    label_path = ROOT / "submission/autolabel/bus.txt"
    labels = label_path.read_text(encoding="utf-8").splitlines()
    assert len(labels) == 5, f"Expected 5 SAM instances, found {len(labels)}"
    for line in labels:
        tokens = line.split()
        assert int(tokens[0]) in (0, 5)
        coords = np.asarray(tokens[1:], dtype=float)
        assert len(coords) >= 6 and len(coords) % 2 == 0
        assert np.isfinite(coords).all() and ((coords >= 0) & (coords <= 1)).all()
    assert "Link notebook đã chạy:" in (ROOT / "submission/report.md").read_text(encoding="utf-8")
    print("PASS: original notebook structure, executed cells, no errors, all mandatory checks,")
    print("      CUDA/Tesla T4 40 epochs at 640, four latency rows, 12 synchronized answers,")
    print("      five valid YOLO-seg labels and notebook report link.")
    print(json.dumps(trained, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
