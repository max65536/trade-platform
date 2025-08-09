from __future__ import annotations

import html
from typing import Optional


def html_page(title: str, body: str) -> bytes:
    doc = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{html.escape(title)}</title>
      <style>
      body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'; margin: 24px; line-height: 1.4; }}
      input, select {{ padding: 6px; margin: 4px 0; }}
      .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
      .col {{ flex: 1 1 300px; min-width: 280px; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
      .btn {{ padding: 8px 12px; background: #166ff6; color: #fff; border: none; border-radius: 6px; cursor: pointer; }}
      .btn:disabled {{ background: #aaa; }}
      table {{ border-collapse: collapse; }}
      td, th {{ border: 1px solid #ddd; padding: 6px 8px; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
      a {{ color: #166ff6; text-decoration: none; }}
      </style>
    </head>
    <body>
      <div style="display:flex;align-items:center;gap:16px;">
        <h2 style="margin:0;">{html.escape(title)}</h2>
        <div style="margin-left:auto;font-size:14px;">
          <a href="/">Backtest</a> · <a href="/echarts">Chart</a> · <a href="/echarts-mtf">MTF</a>
        </div>
      </div>
      {body}
    </body>
    </html>
    """
    return doc.encode("utf-8")


def backtest_index(error: Optional[str] = None) -> bytes:
    from . import snippets

    err_html = f"<p style='color:#c00'>{html.escape(error)}</p>" if error else ""
    body = f"""
    {err_html}
    {snippets.BACKTEST_FORM}
    """
    return html_page("Trade Platform WebUI", body)


def chart_page() -> bytes:
    from . import snippets

    body = snippets.CHART_FORM
    return html_page("Trade Platform WebUI · Chart", body)


def echarts_page() -> bytes:
    from . import snippets

    body = snippets.ECHARTS_SINGLE
    return html_page("Trade Platform WebUI · ECharts", body)


def echarts_mtf_page() -> bytes:
    from . import snippets

    body = snippets.ECHARTS_MTF
    return html_page("Trade Platform WebUI · MTF", body)

