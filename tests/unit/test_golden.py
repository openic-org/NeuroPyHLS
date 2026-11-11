from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
EXCERPTS = sorted(GOLDEN.glob("*/quiroga_*.npz"))


@pytest.mark.parametrize("path", EXCERPTS, ids=lambda p: p.name)
def test_excerpt_is_attributed_and_consistent(path: Path) -> None:
    with np.load(path, allow_pickle=False) as z:
        assert "Quian Quiroga" in str(z["source"])
        assert str(z["license"]).startswith("CC-BY-4.0")
        assert str(z["changes"])
        x, xq, lsb = z["x"], z["x_q"], float(z["lsb"])
        assert xq.dtype == np.int16 and xq.shape == x.shape
        assert np.max(np.abs(xq * lsb - x)) <= lsb / 2 + 1e-12
        st = z["spike_times"]
        assert st.size == z["spike_class"].size > 0
        assert st.min() >= 0 and st.max() < x.size
    assert path.stat().st_size < 1 << 20


def test_excerpts_exist() -> None:
    assert EXCERPTS
