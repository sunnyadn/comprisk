"""SEER breast competing-risks loader for the conformal-CR paper empirics.

Consumes the cleaned cohort produced by ``validation/gen_seer_breast.py`` (which
requires the user's own SEER export). Public + reproducible-when-available; not
present on every box, so this raises a clear, actionable error rather than failing
obscurely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLEAN_PARQUET = Path("/tmp/seer_breast_clean.parquet")

# Clinically meaningful horizons (months, the SEER survival-time unit).
HORIZONS = {"36m": 36.0, "60m": 60.0}


def load_seer(src: Path | str = CLEAN_PARQUET):
    """Return (X, time, event, feature_names) from the cleaned SEER parquet.

    Columns x0..xK + time + status, status 0=censored / 1=cancer death /
    2=other death (per gen_seer_breast.py).
    """
    import pandas as pd

    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(
            f"SEER cohort not found at {src}. Build it first:\n"
            "  python validation/gen_seer_breast.py --src ~/data/seer/export.csv\n"
            "(requires your own SEER Research Data access; see "
            "validation/comparisons/SEER_README.md)."
        )
    df = pd.read_parquet(src)
    feat_cols = [c for c in df.columns if c not in ("time", "status")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    time = df["time"].to_numpy(dtype=np.float64)
    event = df["status"].to_numpy(dtype=np.int64)
    return X, time, event, feat_cols
