import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dash import Dash, Input, Output, State, dcc, html, no_update
import plotly.graph_objects as go
from PIL import Image

from services.influx_writer import InfluxWriter

ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_IMAGE_PATH = ASSETS_DIR / "crystal_sample.png"
MEASUREMENT_NAME = os.getenv("COORD_MEASUREMENT", "crystal_annotations")
influx_writer = InfluxWriter()


def _generate_fallback_image() -> np.ndarray:
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)
    tile = np.tile(gradient, (256, 1))
    image = np.stack([tile, tile, tile], axis=-1)
    return image


def _resolve_image_path(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    path_override = os.getenv("CRYSTAL_IMAGE_PATH")
    if path_override:
        return Path(path_override).expanduser()
    return DEFAULT_IMAGE_PATH


def _load_image_array(explicit_path: Optional[str] = None) -> Tuple[np.ndarray, str]:
    candidate = _resolve_image_path(explicit_path)
    if candidate.exists():
        with Image.open(candidate) as image:
            array = np.array(image.convert("RGB"))
        return array, f"Image source: {candidate}"
    fallback = _generate_fallback_image()
    return fallback, f"Placeholder gradient (file not found: {candidate})"


def create_placeholder_image_figure(
    explicit_path: Optional[str] = None,
) -> Tuple[go.Figure, str, Tuple[int, int]]:
    """
    Placeholder figure that mimics a crystal snapshot. Hovering/clicking anywhere
    on this graph will provide x/y coordinates that we can later map to real
    image pixels once OPC UA snapshots are wired in.
    """
    image_array, label = _load_image_array(explicit_path)
    height, width = image_array.shape[0], image_array.shape[1]
    display_array = np.flipud(image_array)
    fig = go.Figure(data=[go.Image(z=display_array)])
    fig.update_layout(
        title="Crystal image preview",
        xaxis=dict(range=[0, width], constrain="domain", title="X coordinate"),
        yaxis=dict(
            range=[0, height],
            autorange=False,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            title="Y coordinate",
            tick0=0,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode="zoom",
        height=400,
        clickmode="event+select",
    )
    return fig, label, (width, height)


def discover_asset_images() -> List[Dict[str, str]]:
    if not ASSETS_DIR.exists():
        return []
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    images: List[Dict[str, str]] = []
    for pattern in extensions:
        for path in ASSETS_DIR.glob(pattern):
            images.append({"label": path.name, "value": str(path.resolve())})
    images.sort(key=lambda entry: entry["label"])
    return images


def build_layout() -> html.Div:
    """
    Compose the base layout for the UI. Every section currently contains
    placeholder components that will be wired up once the requirements are
    finalized.
    """
    asset_images = discover_asset_images()
    default_path = asset_images[0]["value"] if asset_images else None
    image_figure, image_label, _ = create_placeholder_image_figure(default_path)
    image_context = {"path": default_path, "label": image_label}

    def section(title: str, children) -> html.Div:
        return html.Div(
            [
                html.H3(title, className="section-title"),
                html.Div(children, className="section-body"),
            ],
            className="section",
        )

    return html.Div(
        [
            html.Header(
                [
                    html.H1("Process Monitoring & Control UI"),
                    html.P(
                        "Dash layout scaffold. Hook data sources, callbacks, and styling later."
                    ),
                ],
                className="page-header",
            ),
            html.Main(
                [
                    section(
                        "Process Controls",
                        [
                            html.Div("TODO: Start/Stop buttons and parameter inputs."),
                            html.Div("TODO: OPC UA command widgets."),
                        ],
                    ),
                    section(
                        "Crystal Imaging",
                        [
                            html.P(
                                "Hover or click on the placeholder image to capture coordinates. "
                                "This will be replaced with actual OPC UA snapshots later."
                            ),
                            html.Div(
                                [
                                    html.Label("选择 assets 目录中的图片"),
                                    dcc.Dropdown(
                                        id="crystal-image-select",
                                        options=asset_images,
                                        value=default_path,
                                        placeholder="选择一张测试图�?,
                                        clearable=True,
                                    ),
                                    html.Label("或输入绝对路�?自定义文�?),
                                    dcc.Input(
                                        id="crystal-image-path-input",
                                        type="text",
                                        placeholder="e.g. /path/to/image.png",
                                        debounce=True,
                                    ),
                                ],
                                className="image-selector",
                            ),
                            dcc.Graph(
                                id="crystal-image",
                                figure=image_figure,
                                config={"displayModeBar": False},
                            ),
                            html.Small(
                                image_label,
                                id="crystal-image-source",
                                className="text-muted",
                            ),
                            html.Div(
                                id="crystal-coords-display",
                                className="coords-display",
                                children="Move the cursor over the image to see coordinates here.",
                            ),
                            dcc.Store(
                                id="crystal-coords",
                                data={"event": None, "x": None, "y": None},
                            ),
                            dcc.Store(
                                id="crystal-image-context",
                                data=image_context,
                            ),
                        ],
                    ),
                    section(
                        "Live Data",
                        [
                            dcc.Graph(
                                id="live-trend",
                                figure={
                                    "layout": {
                                        "title": "Live trend placeholder",
                                        "xaxis": {"title": "Time"},
                                        "yaxis": {"title": "Value"},
                                        "annotations": [
                                            {
                                                "text": "Connect to RabbitMQ/Influx streams",
                                                "xref": "paper",
                                                "yref": "paper",
                                                "showarrow": False,
                                                "font": {"size": 14},
                                            }
                                        ],
                                    }
                                },
                            )
                        ],
                    ),
                    section(
                        "Historical Analysis",
                        [
                            html.Div("TODO: Date range selector and historical plots."),
                            html.Div("TODO: Metrics and aggregations from InfluxDB."),
                        ],
                    ),
                    section(
                        "Configuration",
                        [
                            html.Div("TODO: Forms bound to .ini/.txt configuration files."),
                            html.Div("TODO: Status indicators for Telegraf/RabbitMQ/InfluxDB."),
                        ],
                    ),
                ],
                className="page-body",
            ),
        ],
        className="page",
    )


def create_app() -> Dash:
    app = Dash(__name__)
    app.title = "MLDeploy UI"
    app.layout = build_layout
    register_callbacks(app)
    return app


def _extract_xy(event_payload: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not event_payload:
        return None
    points = event_payload.get("points")
    if not points:
        return None
    point = points[0]
    return point.get("x"), point.get("y")


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("crystal-image", "figure"),
        Output("crystal-image-source", "children"),
        Output("crystal-image-context", "data"),
        Input("crystal-image-select", "value"),
        Input("crystal-image-path-input", "value"),
        State("crystal-image-context", "data"),
        prevent_initial_call=True,
    )
    def reload_image(
        selected_asset: Optional[str],
        manual_path: Optional[str],
        context: Optional[Dict[str, Any]],
    ):
        candidate = manual_path or selected_asset or None
        figure, label, _ = create_placeholder_image_figure(candidate)
        context = context or {}
        context.update({"path": candidate, "label": label})
        return figure, label, context

    @app.callback(
        Output("crystal-coords", "data"),
        Output("crystal-coords-display", "children"),
        Input("crystal-image", "clickData"),
        State("crystal-image-context", "data"),
        State("crystal-coords", "data"),
        prevent_initial_call=True,
    )
    def update_coords(
        click_data: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        stored: Optional[Dict[str, Any]],
    ):
        stored = stored or {"event": None, "x": None, "y": None}
        context = context or {}
        image_label = context.get("label") or "unknown"
        image_path = context.get("path") or "unknown"

        coords = None
        event = None
        if click_data:
            coords = _extract_xy(click_data)
            event = "click"

        if not coords:
            return stored, no_update

        x, y = coords
        new_data = {"event": event, "x": x, "y": y}
        message = f"Last {event}: ({x:.2f}, {y:.2f})"
        tags = {"event": event or "unknown", "image_label": image_label}
        fields = {"x": float(x), "y": float(y)}
        try:
            influx_writer.write_point(
                measurement=MEASUREMENT_NAME,
                fields=fields,
                tags=tags | {"image_path": image_path},
            )
        except Exception as exc:
            message += f" (write failed: {exc})"
        return new_data, message


if __name__ == "__main__":
    dash_app = create_app()
    port = int(os.getenv("UI_PORT", "8050"))
    dash_app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "0") == "1")
