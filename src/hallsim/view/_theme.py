"""Design tokens, and the stylesheet beside this module that reads them."""

from __future__ import annotations

from pathlib import Path

TOKENS = {
    "font-heading": '"Barlow Condensed", "Arial Narrow", sans-serif',
    "font-body": 'Barlow, "Helvetica Neue", Arial, sans-serif',
    "font-mono": '"IBM Plex Mono", ui-monospace, Menlo, monospace',
    "color-text": "#1d1f20",
    "color-bg": "#fafafa",
    "color-neutral-400": "#b7b7ba",
    "color-neutral-700": "#5d5d60",
    "color-divider": "rgba(29,31,32,0.16)",
    "color-accent": "#5980a6",
    "color-accent-900": "#1d2d3d",
    "color-pulse": "#de8f05",
    "color-blue": "#123d63",
    "color-green": "#2f7d5b",
    "color-red": "#b3402f",
}
GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700"
    "&family=Barlow:ital,wght@0,400;0,500;1,400"
    "&family=IBM+Plex+Mono:wght@400;500&display=swap"
)
#: Row colours, one process after another.
PALETTE = (
    "color-text",
    "color-accent",
    "color-blue",
    "color-green",
    "color-pulse",
    "color-red",
    "color-accent-900",
)
CURVE_WIDTH = 1.8
REFERENCE_LINE = dict(
    color=TOKENS["color-neutral-700"], width=1.4, dash="1px,3px"
)
HOVER = "t %{x:.2f}: %{y:.3g}<extra></extra>"


def color(name: str) -> str:
    """A token name or a literal colour."""
    return TOKENS.get(name, name)


def root_css() -> str:
    return ":root{" + ";".join(f"--{k}:{v}" for k, v in TOKENS.items()) + "}"


def font(role: str, size: float, color: str = "color-text", **kw) -> dict:
    return dict(family=TOKENS[role], size=size, color=TOKENS[color], **kw)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


CSS = (Path(__file__).parent / "hallsim.css").read_text()
