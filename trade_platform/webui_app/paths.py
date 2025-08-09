from __future__ import annotations

import os


BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "runs", "webui"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TRADES_DIR = os.path.join(BASE_DIR, "trades")
STATS_DIR = os.path.join(BASE_DIR, "stats")


def ensure_dirs():
    for d in (BASE_DIR, UPLOAD_DIR, PLOTS_DIR, TRADES_DIR, STATS_DIR):
        os.makedirs(d, exist_ok=True)

