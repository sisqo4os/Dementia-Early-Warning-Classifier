#!/usr/bin/env python3
# =========================================================
# Model   27: Dementia Early Warning Classifier
# Domain : Health & Medicine
# File   : dementia_early_warning_classifier.onnx
# Output : dementia_early_score
# =========================================================
# Flags early cognitive decline signals that may indicate pre-dementia states.
#
# Input features (shape [1, 5]):
#   [0] memory_complaints              — Self or carer memory complaints 0–1
#   [1] adl_difficulty                 — Difficulty with daily tasks 0–1
#   [2] disorientation                 — Occasional disorientation 0–1
#   [3] age_norm                       — Age / 90
#   [4] family_history_dementia        — First-degree relative 0 or 1
#
# Score < 0.5 → NO WARNING SIGNS ✅
# Score ≥ 0.5 → EARLY WARNING ⚠️
#
# Run : py run_dementia_early_warning_classifier.py
# Need: pip install onnxruntime numpy
# =========================================================

import numpy as np
import onnxruntime as rt
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dementia_early_warning_classifier.onnx")

print(f"\nLoading dementia_early_warning_classifier.onnx ...")
session = rt.InferenceSession(MODEL_PATH)
print("Model ready!\n")

def predict(values: list) -> dict:
    """Run inference. Pass a list of 5 floats."""
    x = np.array([values], dtype=np.float32)
    score = float(session.run(None, {"features": x})[0][0][0])
    label = "EARLY WARNING ⚠️" if score >= 0.5 else "NO WARNING SIGNS ✅"
    conf  = score if score >= 0.5 else 1 - score
    bar   = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    return {"score": score, "label": label, "confidence": conf, "bar": bar}

def show(result, values, label=""):
    if label: print(f"  Scenario   : {label}")
    print(f"  Input      : {values}")
    print(f"  Result     : {result['label']}")
    print(f"  Confidence : [{result['bar']}] {result['confidence']*100:.1f}%")
    print(f"  Raw score  : {result['score']:.4f}")
    print()

# ── Demo ──────────────────────────────────────────────────
print("=" * 58)
print(f"  Dementia Early Warning Classifier — Demo")
print("=" * 58 + "\n")

samples = [
    {"label": "Young no symptoms", "values": [0.0, 0.0, 0.0, 0.2, 0.0], "expected": "NO WARNING"},
    {"label": "Elder with memory loss", "values": [0.8, 0.7, 0.8, 0.9, 1.0], "expected": "EARLY WARNING"},
    {"label": "Middle-aged mild complaints", "values": [0.3, 0.2, 0.2, 0.4, 0.0], "expected": "EARLY WARNING"},
]
for s in samples:
    show(predict(s["values"]), s["values"], s["label"])

# ── Interactive ───────────────────────────────────────────
print("✏️  Type 5 comma-separated values (or 'quit'):")
print(f"   Features: memory_complaints, adl_difficulty, disorientation, age_norm, family_history_dementia\n")
while True:
    raw = input("   > ").strip()
    if raw.lower() in ("quit","exit","q"): break
    if not raw: continue
    try:
        vals = [float(x) for x in raw.split(",")]
        if len(vals) != 5: print(f"   Need exactly 5 values\n"); continue
        show(predict(vals), vals)
    except ValueError:
        print("   Numbers only, please\n")
