# codes to make visualization of your weights.
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mynn as nn
import numpy as np

_CODES_DIR = Path(__file__).resolve().parent
_PJ1_DIR = _CODES_DIR.parent

_MODEL_CANDIDATES = [
    _CODES_DIR / "saved_models" / "best_model_1.pickle",
    _CODES_DIR / "best_models_mlp" / "best_model.pickle",
]


def _find_model_path():
    for p in _MODEL_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No MLP weights found. Tried:\n  "
        + "\n  ".join(str(p) for p in _MODEL_CANDIDATES)
    )


model = nn.models.Model_MLP()
model.load_model(str(_find_model_path()))

mats = []
mats.append(model.layers[0].params["W"])
mats.append(model.layers[2].params["W"])

fig, ax = plt.subplots(figsize=(8, 6))
ax.matshow(mats[1])
ax.set_xticks([])
ax.set_yticks([])

out_path = _PJ1_DIR / "Figure_5.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
