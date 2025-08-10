from __future__ import annotations

import argparse
import sys


def run_gen():
    from .generate_synthetic import main as gen
    gen()


def run_backtest():
    from .backtest_example import main as bt
    bt()


def run_mtf():
    from .mtf_example import main as mtf
    mtf()


def run_plot():
    from .plot_example import main as plot
    plot()


def run_all():
    run_gen()
    run_backtest()
    run_mtf()
    run_plot()


def run_webui(host: str = "127.0.0.1", port: int = 8000):
    from trade_platform.webui_app.server import main as webui_main
    print(f"Starting WebUI at http://{host}:{port} (Ctrl+C to stop)")
    webui_main(["--host", host, "--port", str(port)])


CHOICES = {
    "gen": ("Generate synthetic data", run_gen),
    "backtest": ("Run single-timeframe backtest", run_backtest),
    "mtf": ("Run MTF alignment + backtest", run_mtf),
    "plot": ("Render PNG plot with signals/trades", run_plot),
    "all": ("Run all examples in sequence", run_all),
    "webui": ("Start WebUI server", run_webui),
}


def interactive_menu():
    print("Examples runner:\n")
    keys = list(CHOICES.keys())
    for i, k in enumerate(keys, start=1):
        print(f" {i}) {k:8s} - {CHOICES[k][0]}")
    print()
    try:
        s = input(f"Select [1-{len(keys)}] (default 1): ").strip()
    except EOFError:
        s = ""
    if not s:
        s = "1"
    try:
        idx = int(s)
    except Exception:
        print("Invalid selection")
        return 1
    if not (1 <= idx <= len(keys)):
        print("Out of range")
        return 1
    key = keys[idx - 1]
    label, fn = CHOICES[key]
    if key == "webui":
        run_webui()
    else:
        fn()
    return 0


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run examples: generate, backtest, mtf, plot, webui")
    parser.add_argument("--list", action="store_true", help="List available example targets")
    parser.add_argument("--run", choices=list(CHOICES.keys()), default=None, help="Run a specific target")
    parser.add_argument("--host", default="127.0.0.1", help="WebUI host (with --run webui)")
    parser.add_argument("--port", type=int, default=8000, help="WebUI port (with --run webui)")
    args = parser.parse_args(argv)

    if args.list:
        for k, (desc, _) in CHOICES.items():
            print(f"{k:8s} - {desc}")
        return 0
    if args.run:
        label, fn = CHOICES[args.run]
        if args.run == "webui":
            run_webui(args.host, args.port)
        else:
            fn()
        return 0
    return interactive_menu()


if __name__ == "__main__":
    sys.exit(main())

