from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional


_BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    # Momentum on 1h with modest ATR floor and TP/SL
    "momentum_1h": {
        "rsi_min": 55,
        "min_atr_pct": 0.004,
        "fee": 0.0005,
        "stop_pct": 0.02,
        "tp_pct": 0.04,
    },
    # Mean-reversion on 1h with ATR ceiling
    "meanrev_1h": {
        "rsi_max": 45,
        "max_atr_pct": 0.02,
        "fee": 0.0005,
        "stop_pct": 0.02,
        "tp_pct": 0.03,
    },
    # MTF momentum with HTF breakout and min run
    "mtf_momentum_4h_1d": {
        "rsi_min": 55,
        "min_atr_pct": 0.004,
        "fee": 0.0005,
        "stop_pct": 0.02,
        "tp_pct": 0.04,
        "require_htf_breakout": True,
        "min_htf_run": 3,
    },
    # MTF mean-reversion (weaker HTF requirement)
    "mtf_meanrev_4h_1d": {
        "rsi_max": 45,
        "max_atr_pct": 0.02,
        "fee": 0.0005,
        "stop_pct": 0.02,
        "tp_pct": 0.03,
        "min_htf_run": 2,
    },
}


def load_presets_from_file(path: str) -> Dict[str, Dict[str, Any]]:
    """Load presets from JSON file. If fails, returns empty dict.

    Note: JSON chosen to avoid external deps; YAML may be supported externally.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
    except Exception:
        pass
    return {}


def get_preset(name: str, *, file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a preset by name from file or builtin.

    Explicit file overrides builtin registry.
    """
    if file:
        ext = os.path.splitext(file)[1].lower()
        if ext in (".json", ""):
            presets = load_presets_from_file(file)
        else:
            # Best-effort: attempt JSON anyway
            presets = load_presets_from_file(file)
        return presets.get(name)
    return _BUILTIN_PRESETS.get(name)


def apply_preset(args_ns, preset: Dict[str, Any], *, keys: Optional[list[str]] = None):
    """Apply preset values to argparse Namespace only for missing attributes.

    keys: optional allowlist; if None, apply all keys present in preset.
    """
    if not preset:
        return args_ns
    for k, v in preset.items():
        if keys and k not in keys:
            continue
        if not hasattr(args_ns, k) or getattr(args_ns, k) is None:
            setattr(args_ns, k, v)
    return args_ns

