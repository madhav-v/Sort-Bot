# app_dash.py - Professional SortBot Dashboard
"""
Advanced SortBot Dashboard with comprehensive features:
- Real-time metrics and analytics
- Environmental impact calculator
- Performance trends
- System health monitoring
- Export functionality
- Advanced filtering and analysis
"""

import os
import io
import csv
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(__file__) or "."
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "events.csv")
INTERVAL_SECONDS = 2
TAIL_LINES = 5000
CATEGORIES = ["Plastic", "Burnable", "Cans", "Bottles", "Others"]
APP_PORT = 8050

# Category colors (matching main app)
CATEGORY_COLORS = {
    "Plastic": "rgb(230, 120, 90)",
    "Burnable": "rgb(60, 160, 255)",
    "Cans": "rgb(200, 200, 220)",
    "Bottles": "rgb(100, 220, 110)",
    "Others": "rgb(140, 140, 160)",
}

# Environmental impact data (kg CO2 saved per kg of recycled material)
CO2_SAVINGS = {
    "Plastic": 1.5,
    "Burnable": 0.5,
    "Cans": 9.0,
    "Bottles": 0.3,
    "Others": 0.8,
}


# ---------- Helpers ----------
def tail_lines(path: str, n: int) -> str:
    """Efficiently read last n lines from file"""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        avg_line_size = 200
        to_read = n * avg_line_size
        try:
            f.seek(-to_read, os.SEEK_END)
        except OSError:
            f.seek(0, os.SEEK_SET)
        data = f.read().decode(errors="replace")
    lines = data.splitlines()
    if len(lines) == 0:
        return ""
    first = lines[0].lower()
    if "ts" in first and "category" in first:
        return "\n".join(lines)
    else:
        try:
            with open(path, "r", encoding="utf8", errors="replace") as fh:
                header = fh.readline().strip()
                if "ts" in header.lower() and "category" in header.lower():
                    return header + "\n" + "\n".join(lines[-n:])
        except Exception:
            pass
    header = "ts,category,confidence,source,note"
    return header + "\n" + "\n".join(lines[-n:])


def load_recent_df(path: str, tail_lines_n: int = TAIL_LINES) -> pd.DataFrame:
    """Load recent events from CSV"""
    text = tail_lines(path, tail_lines_n)
    if not text:
        return pd.DataFrame(columns=["ts", "category", "confidence", "source", "note"])
    try:
        df = pd.read_csv(io.StringIO(text), parse_dates=["ts"])
    except Exception:
        df = pd.read_csv(io.StringIO(text), header=0, dtype=str)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        if "confidence" in df.columns:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(
                0.0
            )

    for c in ["category", "source", "note"]:
        if c not in df.columns:
            df[c] = ""
    if "confidence" not in df.columns:
        df["confidence"] = 0.0

    df = df.dropna(subset=["ts", "category"])
    df["category"] = df["category"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    if pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    return df


def calculate_environmental_impact(df: pd.DataFrame) -> dict:
    """Calculate environmental impact metrics"""
    if df.empty:
        return {"co2_saved": 0, "trees_saved": 0, "energy_saved": 0}

    # Assume each item weighs ~50g = 0.05kg
    total_co2 = 0
    for cat in CATEGORIES:
        count = len(df[df["category"] == cat])
        weight_kg = count * 0.05
        total_co2 += weight_kg * CO2_SAVINGS.get(cat, 0.5)

    # 1 tree absorbs ~21kg CO2 per year
    trees_saved = total_co2 / 21

    # Energy saved (kWh) - rough estimate
    energy_saved = total_co2 * 0.5

    return {
        "co2_saved": round(total_co2, 2),
        "trees_saved": round(trees_saved, 3),
        "energy_saved": round(energy_saved, 2),
    }


# ---------- Build Dash app ----------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Modern dark theme colors
THEME = {
    "bg_main": "#0a0a0f",
    "bg_card": "#14141f",
    "bg_card_light": "#1a1a2e",
    "accent": "#00ffc8",
    "accent_dim": "#00a382",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0b0",
    "border": "#2a2a3e",
}


# Helper functions for styling
def card_style():
    return {
        "backgroundColor": THEME["bg_card"],
        "borderRadius": "12px",
        "padding": "16px",
        "border": f"1px solid {THEME['border']}",
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
    }


def button_style():
    return {
        "backgroundColor": THEME["accent_dim"],
        "color": THEME["text_primary"],
        "border": "none",
        "padding": "8px 16px",
        "borderRadius": "6px",
        "cursor": "pointer",
        "fontSize": "13px",
        "fontWeight": "500",
        "transition": "all 0.2s",
    }


def kpi_card(title, value, subtitle="", icon=""):
    return html.Div(
        [
            html.Div(
                icon + " " + title,
                style={
                    "fontSize": "13px",
                    "color": THEME["text_secondary"],
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "color": THEME["accent"],
                    "marginBottom": "4px",
                },
            ),
            html.Div(
                subtitle, style={"fontSize": "11px", "color": THEME["text_secondary"]}
            ),
        ]
    )


app.layout = html.Div(
    style={
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        "backgroundColor": THEME["bg_main"],
        "color": THEME["text_primary"],
        "minHeight": "100vh",
        "padding": "0",
        "margin": "0",
    },
    children=[
        # Header
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {THEME['bg_card']} 0%, {THEME['bg_card_light']} 100%)",
                "padding": "24px 40px",
                "borderBottom": f"2px solid {THEME['accent']}",
                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                    },
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "🤖 SORTBOT ANALYTICS DASHBOARD",
                                    style={
                                        "margin": "0 0 8px 0",
                                        "fontSize": "32px",
                                        "fontWeight": "700",
                                        "background": f"linear-gradient(90deg, {THEME['accent']} 0%, {THEME['text_primary']} 100%)",
                                        "WebkitBackgroundClip": "text",
                                        "WebkitTextFillColor": "transparent",
                                    },
                                ),
                                html.Div(
                                    "Real-time waste sorting intelligence • AI-powered analytics",
                                    style={
                                        "color": THEME["text_secondary"],
                                        "fontSize": "14px",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(
                                    id="live-status",
                                    children="● LIVE",
                                    style={
                                        "backgroundColor": THEME["bg_card"],
                                        "padding": "10px 20px",
                                        "borderRadius": "25px",
                                        "border": f"2px solid {THEME['accent']}",
                                        "fontSize": "14px",
                                        "fontWeight": "600",
                                        "color": THEME["accent"],
                                    },
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
        # Main content
        html.Div(
            style={"padding": "24px 40px"},
            children=[
                # KPI Cards Row
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                        "gap": "16px",
                        "marginBottom": "24px",
                    },
                    children=[
                        html.Div(id="kpi-total", style=card_style()),
                        html.Div(id="kpi-rate", style=card_style()),
                        html.Div(id="kpi-accuracy", style=card_style()),
                        html.Div(id="kpi-uptime", style=card_style()),
                        html.Div(id="kpi-co2", style=card_style()),
                    ],
                ),
                # Charts Grid
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "2fr 1fr",
                        "gap": "16px",
                        "marginBottom": "24px",
                    },
                    children=[
                        # Left column - main charts
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id="main-bar-chart",
                                        config={"displayModeBar": False},
                                    ),
                                    style=card_style(),
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="trend-chart",
                                        config={"displayModeBar": False},
                                    ),
                                    style={**card_style(), "marginTop": "16px"},
                                ),
                            ]
                        ),
                        # Right column - pie and stats
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id="pie-chart", config={"displayModeBar": False}
                                    ),
                                    style=card_style(),
                                ),
                                html.Div(
                                    id="hourly-stats",
                                    style={
                                        **card_style(),
                                        "marginTop": "16px",
                                        "padding": "20px",
                                    },
                                ),
                            ]
                        ),
                    ],
                ),
                # Environmental Impact & Performance
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px",
                        "marginBottom": "24px",
                    },
                    children=[
                        html.Div(
                            id="environmental-impact",
                            style={**card_style(), "padding": "20px"},
                        ),
                        html.Div(
                            id="performance-metrics",
                            style={**card_style(), "padding": "20px"},
                        ),
                    ],
                ),
                # Recent Events Table
                html.Div(
                    style=card_style(),
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "16px",
                            },
                            children=[
                                html.H3(
                                    "📋 Recent Events",
                                    style={
                                        "margin": "0",
                                        "fontSize": "18px",
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "↻ Refresh",
                                            id="btn-refresh",
                                            n_clicks=0,
                                            style=button_style(),
                                        ),
                                        html.Button(
                                            "📥 Export CSV",
                                            id="btn-export",
                                            n_clicks=0,
                                            style={
                                                **button_style(),
                                                "marginLeft": "8px",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dash_table.DataTable(
                            id="recent-table",
                            columns=[
                                {"name": "Timestamp", "id": "ts"},
                                {"name": "Category", "id": "category"},
                                {"name": "Confidence", "id": "confidence"},
                                {"name": "Source", "id": "source"},
                                {"name": "Note", "id": "note"},
                            ],
                            data=[],
                            style_table={"overflowY": "auto", "maxHeight": "400px"},
                            style_cell={
                                "textAlign": "left",
                                "backgroundColor": THEME["bg_main"],
                                "color": THEME["text_primary"],
                                "border": f"1px solid {THEME['border']}",
                                "padding": "12px",
                                "fontSize": "13px",
                            },
                            style_header={
                                "backgroundColor": THEME["bg_card_light"],
                                "fontWeight": "600",
                                "border": f"1px solid {THEME['border']}",
                                "color": THEME["accent"],
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": THEME["bg_card"],
                                }
                            ],
                            page_size=15,
                        ),
                    ],
                ),
                # Settings Panel
                html.Div(
                    style={**card_style(), "marginTop": "24px"},
                    children=[
                        html.H3(
                            "⚙️ Dashboard Settings",
                            style={"margin": "0 0 16px 0", "fontSize": "18px"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "24px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Label(
                                            "Auto-refresh interval:",
                                            style={
                                                "color": THEME["text_secondary"],
                                                "marginBottom": "8px",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.Slider(
                                            id="interval-slider",
                                            min=1,
                                            max=10,
                                            step=1,
                                            value=INTERVAL_SECONDS,
                                            marks={i: f"{i}s" for i in [1, 2, 5, 10]},
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                        ),
                                        html.Div(
                                            id="interval-display",
                                            style={
                                                "marginTop": "8px",
                                                "color": THEME["accent"],
                                                "fontSize": "13px",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Data window (minutes):",
                                            style={
                                                "color": THEME["text_secondary"],
                                                "marginBottom": "8px",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.Slider(
                                            id="window-slider",
                                            min=10,
                                            max=240,
                                            step=10,
                                            value=60,
                                            marks={
                                                10: "10m",
                                                60: "1h",
                                                120: "2h",
                                                240: "4h",
                                            },
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                        ),
                                        html.Div(
                                            id="window-display",
                                            style={
                                                "marginTop": "8px",
                                                "color": THEME["accent"],
                                                "fontSize": "13px",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Footer
        html.Div(
            style={
                "backgroundColor": THEME["bg_card"],
                "padding": "16px 40px",
                "marginTop": "40px",
                "borderTop": f"1px solid {THEME['border']}",
                "textAlign": "center",
                "color": THEME["text_secondary"],
                "fontSize": "12px",
            },
            children=[
                html.Div(
                    "SortBot Analytics Dashboard v2.0 • Powered by AI • Real-time waste intelligence"
                ),
                html.Div(id="footer-time", style={"marginTop": "4px"}),
            ],
        ),
        # Hidden components
        dcc.Interval(
            id="poll-interval", interval=INTERVAL_SECONDS * 1000, n_intervals=0
        ),
        dcc.Store(id="app-start-time", data=datetime.now().isoformat()),
        dcc.Download(id="download-csv"),
    ],
)


def card_style():
    return {
        "backgroundColor": THEME["bg_card"],
        "borderRadius": "12px",
        "padding": "16px",
        "border": f"1px solid {THEME['border']}",
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
    }


def button_style():
    return {
        "backgroundColor": THEME["accent_dim"],
        "color": THEME["text_primary"],
        "border": "none",
        "padding": "8px 16px",
        "borderRadius": "6px",
        "cursor": "pointer",
        "fontSize": "13px",
        "fontWeight": "500",
        "transition": "all 0.2s",
    }


def kpi_card(title, value, subtitle="", icon=""):
    return html.Div(
        [
            html.Div(
                icon + " " + title,
                style={
                    "fontSize": "13px",
                    "color": THEME["text_secondary"],
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "color": THEME["accent"],
                    "marginBottom": "4px",
                },
            ),
            html.Div(
                subtitle, style={"fontSize": "11px", "color": THEME["text_secondary"]}
            ),
        ]
    )


# ---------- Callbacks ----------
@app.callback(
    Output("interval-display", "children"),
    Output("poll-interval", "interval"),
    Input("interval-slider", "value"),
)
def update_interval_settings(sec):
    return f"Refreshing every {sec} seconds", int(sec) * 1000


@app.callback(
    Output("window-display", "children"),
    Input("window-slider", "value"),
)
def update_window_display(minutes):
    return f"Showing last {minutes} minutes of data"


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(n_clicks):
    df = load_recent_df(DATA_FILE)
    return dcc.send_data_frame(
        df.to_csv,
        f"sortbot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )


@app.callback(
    [
        Output("main-bar-chart", "figure"),
        Output("pie-chart", "figure"),
        Output("trend-chart", "figure"),
        Output("recent-table", "data"),
        Output("kpi-total", "children"),
        Output("kpi-rate", "children"),
        Output("kpi-accuracy", "children"),
        Output("kpi-uptime", "children"),
        Output("kpi-co2", "children"),
        Output("hourly-stats", "children"),
        Output("environmental-impact", "children"),
        Output("performance-metrics", "children"),
        Output("footer-time", "children"),
    ],
    [
        Input("poll-interval", "n_intervals"),
        Input("window-slider", "value"),
        Input("btn-refresh", "n_clicks"),
    ],
    State("app-start-time", "data"),
)
def update_dashboard(n_intervals, window_minutes, refresh_clicks, app_start_time):
    df = load_recent_df(DATA_FILE)

    if df.empty:
        return create_empty_dashboard()

    # Filter to window
    now = pd.Timestamp.now(tz="UTC")
    window_start = now - pd.Timedelta(minutes=window_minutes)
    df_window = df[df["ts"] >= window_start].copy()

    if df_window.empty:
        df_window = df.tail(100).copy()

    # Calculate metrics
    total_events = len(df_window)
    events_per_min = total_events / window_minutes if window_minutes > 0 else 0
    avg_confidence = df_window["confidence"].mean() * 100

    # Uptime calculation
    if app_start_time:
        start = pd.to_datetime(app_start_time)
        uptime = (datetime.now() - start).total_seconds() / 3600
        uptime_str = f"{uptime:.1f}h"
    else:
        uptime_str = "N/A"

    # Environmental impact
    impact = calculate_environmental_impact(df_window)

    # Category counts
    counts = df_window["category"].value_counts().reindex(CATEGORIES, fill_value=0)

    # Main bar chart
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker=dict(
                    color=[CATEGORY_COLORS.get(cat, "#888") for cat in counts.index],
                    line=dict(color=THEME["border"], width=1),
                ),
                text=counts.values,
                textposition="outside",
            )
        ]
    )
    bar_fig.update_layout(
        title="Items Sorted by Category",
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_main"],
        font=dict(color=THEME["text_primary"], size=12),
        margin=dict(t=40, b=40, l=40, r=40),
        height=300,
        showlegend=False,
    )

    # Pie chart
    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.4,
                marker=dict(
                    colors=[CATEGORY_COLORS.get(cat, "#888") for cat in counts.index]
                ),
                textinfo="label+percent",
            )
        ]
    )
    pie_fig.update_layout(
        title="Distribution",
        paper_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text_primary"], size=12),
        margin=dict(t=40, b=20, l=20, r=20),
        height=350,
    )

    # Trend chart
    try:
        df_ts = df.set_index("ts").resample("5min").size().reset_index(name="count")
        df_ts = df_ts[df_ts["ts"] >= (now - pd.Timedelta(hours=4))]

        trend_fig = go.Figure(
            data=[
                go.Scatter(
                    x=df_ts["ts"],
                    y=df_ts["count"],
                    mode="lines+markers",
                    line=dict(color=THEME["accent"], width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba(0, 255, 200, 0.1)",
                )
            ]
        )
        trend_fig.update_layout(
            title="Activity Timeline (5-min intervals)",
            paper_bgcolor=THEME["bg_card"],
            plot_bgcolor=THEME["bg_main"],
            font=dict(color=THEME["text_primary"], size=12),
            margin=dict(t=40, b=40, l=40, r=40),
            height=300,
            xaxis=dict(gridcolor=THEME["border"]),
            yaxis=dict(gridcolor=THEME["border"]),
        )
    except Exception:
        trend_fig = go.Figure()

    # Recent table
    df_recent = df.sort_values("ts", ascending=False).head(50).copy()
    df_recent["ts"] = df_recent["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_recent["confidence"] = df_recent["confidence"].round(2)
    table_data = df_recent.to_dict("records")

    # KPI cards
    kpi_total = kpi_card(
        "Total Items", f"{total_events:,}", f"Last {window_minutes}min", "📊"
    )
    kpi_rate = kpi_card(
        "Processing Rate", f"{events_per_min:.1f}/min", "Items per minute", "⚡"
    )
    kpi_accuracy = kpi_card(
        "Avg Confidence", f"{avg_confidence:.1f}%", "AI accuracy", "🎯"
    )
    kpi_uptime = kpi_card("System Uptime", uptime_str, "Current session", "🟢")
    kpi_co2 = kpi_card(
        "CO₂ Saved", f"{impact['co2_saved']:.1f}kg", "Environmental impact", "🌱"
    )

    # Hourly stats
    hourly_counts = df_window.set_index("ts").resample("1h").size()
    hourly_html = html.Div(
        [
            html.H4(
                "📈 Hourly Breakdown",
                style={"margin": "0 0 12px 0", "fontSize": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                f"{hour.strftime('%H:00')}",
                                style={
                                    "color": THEME["text_secondary"],
                                    "fontSize": "12px",
                                },
                            ),
                            html.Span(
                                f"{count} items",
                                style={
                                    "float": "right",
                                    "color": THEME["accent"],
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                },
                            ),
                        ],
                        style={
                            "padding": "8px 0",
                            "borderBottom": f"1px solid {THEME['border']}",
                        },
                    )
                    for hour, count in hourly_counts.tail(6).items()
                ]
            ),
        ]
    )

    # Environmental impact card
    env_html = html.Div(
        [
            html.H4(
                "🌍 Environmental Impact",
                style={"margin": "0 0 16px 0", "fontSize": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "CO₂ Emissions Saved",
                                style={
                                    "fontSize": "12px",
                                    "color": THEME["text_secondary"],
                                },
                            ),
                            html.Div(
                                f"{impact['co2_saved']:.2f} kg",
                                style={
                                    "fontSize": "20px",
                                    "color": THEME["accent"],
                                    "fontWeight": "600",
                                },
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Equivalent Trees Saved",
                                style={
                                    "fontSize": "12px",
                                    "color": THEME["text_secondary"],
                                },
                            ),
                            html.Div(
                                f"{impact['trees_saved']:.3f} trees",
                                style={
                                    "fontSize": "20px",
                                    "color": THEME["accent"],
                                    "fontWeight": "600",
                                },
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Energy Saved",
                                style={
                                    "fontSize": "12px",
                                    "color": THEME["text_secondary"],
                                },
                            ),
                            html.Div(
                                f"{impact['energy_saved']:.2f} kWh",
                                style={
                                    "fontSize": "20px",
                                    "color": THEME["accent"],
                                    "fontWeight": "600",
                                },
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )

    # Performance metrics
    perf_html = html.Div(
        [
            html.H4(
                "📊 Performance Metrics",
                style={"margin": "0 0 16px 0", "fontSize": "16px"},
            ),
            html.Div(
                [
                    metric_row("Total Processed", f"{len(df):,} items", "All time"),
                    metric_row(
                        "Success Rate",
                        f"{(counts.sum() / max(1, total_events) * 100):.1f}%",
                        "Classification accuracy",
                    ),
                    metric_row(
                        "Active Sources",
                        f"{df_window['source'].nunique()}",
                        "Unique devices",
                    ),
                    metric_row(
                        "Peak Hour",
                        f"{df_window.set_index('ts').resample('1h').size().idxmax().strftime('%H:00') if not df_window.empty else 'N/A'}",
                        "Most active time",
                    ),
                ]
            ),
        ]
    )

    # Footer time
    footer_time = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return (
        bar_fig,
        pie_fig,
        trend_fig,
        table_data,
        kpi_total,
        kpi_rate,
        kpi_accuracy,
        kpi_uptime,
        kpi_co2,
        hourly_html,
        env_html,
        perf_html,
        footer_time,
    )


def metric_row(label, value, subtitle):
    return html.Div(
        [
            html.Div(
                label, style={"fontSize": "12px", "color": THEME["text_secondary"]}
            ),
            html.Div(
                [
                    html.Span(
                        value,
                        style={
                            "fontSize": "18px",
                            "color": THEME["accent"],
                            "fontWeight": "600",
                        },
                    ),
                    html.Span(
                        f" {subtitle}",
                        style={
                            "fontSize": "11px",
                            "color": THEME["text_secondary"],
                            "marginLeft": "8px",
                        },
                    ),
                ]
            ),
        ],
        style={"padding": "8px 0", "borderBottom": f"1px solid {THEME['border']}"},
    )


def create_empty_dashboard():
    """Return empty dashboard components"""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor=THEME["bg_card"], font=dict(color=THEME["text_primary"])
    )

    return (
        empty_fig,
        empty_fig,
        empty_fig,
        [],
        kpi_card("Total Items", "0", "", "📊"),
        kpi_card("Processing Rate", "0.0/min", "", "⚡"),
        kpi_card("Avg Confidence", "0.0%", "", "🎯"),
        kpi_card("System Uptime", "0h", "", "🟢"),
        kpi_card("CO₂ Saved", "0kg", "", "🌱"),
        html.Div("No data available"),
        html.Div("No environmental data"),
        html.Div("No performance data"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ---------- Run ----------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"🚀 Starting Professional SortBot Dashboard on http://127.0.0.1:{APP_PORT}")
    print(
        f"📊 Dashboard features: Real-time analytics, Environmental impact, Performance tracking"
    )
    print(f"🔄 Auto-refresh enabled • Press CTRL+C to stop")
app.run(debug=False, port=APP_PORT)
