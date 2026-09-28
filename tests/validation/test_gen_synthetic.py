from pathlib import Path

import pandas as pd
from validation.gen_synthetic import generate_synthetic


def test_vendored_parquet_exists_and_matches():
    pq = Path(__file__).resolve().parents[2] / "validation" / "data" / "synthetic.parquet"
    assert pq.exists(), "run `python -m validation.gen_synthetic` to materialize"
    df = pd.read_parquet(pq)
    expected = generate_synthetic()
    pd.testing.assert_frame_equal(df, expected)
