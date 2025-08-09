from argparse import Namespace

from trade_platform import presets as preset_mod


def test_builtin_preset_load_and_apply():
    pr = preset_mod.get_preset("momentum_1h")
    assert pr is not None
    ns = Namespace(rsi_min=None, fee=None, stop_pct=None, rsi_max=None)
    out = preset_mod.apply_preset(ns, pr, keys=["rsi_min", "fee", "stop_pct", "rsi_max"])
    # preset sets rsi_min/fee/stop_pct; leaves rsi_max untouched
    assert out.rsi_min == pr["rsi_min"]
    assert out.fee == pr["fee"]
    assert out.stop_pct == pr["stop_pct"]
    assert out.rsi_max is None

    # Explicit args override preset (apply only if None)
    ns2 = Namespace(rsi_min=10, fee=None, stop_pct=None)
    out2 = preset_mod.apply_preset(ns2, pr, keys=["rsi_min", "fee", "stop_pct"])
    assert out2.rsi_min == 10  # unchanged
    assert out2.fee == pr["fee"]

