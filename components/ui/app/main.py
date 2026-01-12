# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import plotly.graph_objects as go
from PIL import Image

from services.influx_writer import InfluxWriter
from services.recover_3d import calc_foot_point

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


def _empty_3d_figure(title: str = "3D preview") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
    )
    return fig


def _default_guide_state() -> Dict[str, Any]:
    return {
        "enabled": False,
        "step_idx": 0,
        "steps": [
            "m",
            "u",
            "v",
            "w",
            "k_c1",
            "k_c2",
            "k_c3",
            "k_c4",
            "k_o1",
            "k_o2",
            "k_o3",
            "k_o4",
        ],
    }


def _calc_intersect(
    p1: Dict[str, float],
    p2: Dict[str, float],
    q1: Dict[str, float],
    q2: Dict[str, float],
) -> Optional[Dict[str, float]]:
    try:
        A = np.array([p1["x"], p1["y"]], dtype=float)
        B = np.array([p2["x"], p2["y"]], dtype=float)
        C = np.array([q1["x"], q1["y"]], dtype=float)
        D = np.array([q2["x"], q2["y"]], dtype=float)
        mat = np.column_stack((B - A, C - D))
        rhs = C - A
        sol = np.linalg.solve(mat, rhs)
        E = A + sol[0] * (B - A)
        return {"x": float(E[0]), "y": float(E[1])}
    except Exception:
        return None


def _point_to_array(point: Dict[str, float]) -> np.ndarray:
    return np.array([point["x"], point["y"], 0.0], dtype=float)


def _apply_get_points_overlay(fig: go.Figure, data: Dict[str, Any]) -> go.Figure:
    points = data.get("points", {})
    lines = data.get("lines", [])
    if points:
        xs = [pt["x"] for pt in points.values()]
        ys = [pt["y"] for pt in points.values()]
        labels = list(points.keys())
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=6, color="#e63946"),
                name="points",
            )
        )
    if lines:
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        for line in lines:
            p1 = line["p1"]
            p2 = line["p2"]
            shapes.append(
                dict(
                    type="line",
                    x0=p1["x"],
                    y0=p1["y"],
                    x1=p2["x"],
                    y1=p2["y"],
                    line=dict(color="#1d3557", width=2),
                    layer="above",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[p1["x"], p2["x"]],
                    y=[p1["y"], p2["y"]],
                    mode="lines",
                    line=dict(color="#1d3557", width=2),
                    name="lines",
                    showlegend=False,
                )
            )
        fig.update_layout(shapes=shapes)
    return fig


def _rebuild_get_points_data(points: Dict[str, Dict[str, float]], is_full: bool) -> Dict[str, Any]:
    lines: List[Dict[str, Dict[str, float]]] = []
    derived: Dict[str, Dict[str, float]] = {}

    def add_line(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]):
        if a and b:
            lines.append({"p1": a, "p2": b})

    u_ad_t = points.get("u_ad_t")
    u_ad_e = points.get("u_ad_e")
    v_ad_t = points.get("v_ad_t")
    v_ad_e = points.get("v_ad_e")

    if u_ad_t and u_ad_e and v_ad_t and v_ad_e:
        m = _calc_intersect(u_ad_t, u_ad_e, v_ad_t, v_ad_e)
        if m:
            derived["m"] = m

    if not is_full:
        if derived.get("m") and u_ad_t and u_ad_e:
            m_arr = _point_to_array(derived["m"])
            u = calc_foot_point(m_arr, {"t": _point_to_array(u_ad_t), "e": _point_to_array(u_ad_e)})
            derived["u"] = {"x": float(u[0]), "y": float(u[1])}
        if derived.get("m") and v_ad_t and v_ad_e:
            m_arr = _point_to_array(derived["m"])
            v = calc_foot_point(m_arr, {"t": _point_to_array(v_ad_t), "e": _point_to_array(v_ad_e)})
            derived["v"] = {"x": float(v[0]), "y": float(v[1])}
        if points.get("w"):
            derived["w"] = points["w"]
        add_line(derived.get("m"), derived.get("u"))
        add_line(derived.get("m"), derived.get("v"))
        add_line(derived.get("m"), derived.get("w"))
    else:
        u_op_t = points.get("u_op_t")
        u_op_e = points.get("u_op_e")
        v_op_t = points.get("v_op_t")
        v_op_e = points.get("v_op_e")
        if u_op_t and u_op_e and u_ad_t and u_ad_e:
            u = _calc_intersect(u_op_t, u_op_e, u_ad_t, u_ad_e)
            if u:
                derived["u"] = u
        if v_op_t and v_op_e and v_ad_t and v_ad_e:
            v = _calc_intersect(v_op_t, v_op_e, v_ad_t, v_ad_e)
            if v:
                derived["v"] = v
        if u_op_t and u_op_e and v_op_t and v_op_e:
            w = _calc_intersect(u_op_t, u_op_e, v_op_t, v_op_e)
            if w:
                derived["w"] = w
        add_line(derived.get("m"), derived.get("u"))
        add_line(derived.get("m"), derived.get("v"))
        add_line(derived.get("m"), derived.get("w"))
        add_line(derived.get("u"), derived.get("w"))
        add_line(derived.get("v"), derived.get("w"))

    k_c = [points.get(f"k_c{i}") for i in range(1, 5)]
    if all(k_c):
        for idx in range(4):
            add_line(k_c[idx], k_c[(idx + 1) % 4])

    merged_points = dict(points)
    merged_points.update(derived)
    return {"points": merged_points, "lines": lines}
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
                    html.P("Dash UI 骨架，后续接入数据源与回调。"),
                ],
                className="page-header",
            ),
            html.Main(
                [
                    section(
                        "工艺控制",
                        [
                            html.Div("TODO: 启动/停止按钮与参数输入。"),
                            html.Div("TODO: OPC UA 控制组件。"),
                        ],
                    ),
                    section(
                        "晶体图像标注",
                        [
                            html.P("点击图像记录坐标，后续将接入 OPC UA 实时图像。"),
                            html.Div(
                                [
                                    html.Label("选择 assets 里的图片"),
                                    dcc.Dropdown(
                                        id="crystal-image-select",
                                        options=asset_images,
                                        value=default_path,
                                        placeholder="选择一张测试图片",
                                        clearable=True,
                                    ),
                                    html.Label("或输入自定义图片绝对路径"),
                                    dcc.Input(
                                        id="crystal-image-path-input",
                                        type="text",
                                        placeholder="例如：/path/to/image.png",
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
                                children="点击图像获取坐标。",
                            ),
                            html.Div(
                                [
                                    html.Label("点位角色 (tag)"),
                                    dcc.Input(
                                        id="point-role",
                                        type="text",
                                        placeholder="例如：m/u/v/w/k_c1",
                                        debounce=True,
                                    ),
                                ],
                                className="point-role-controls",
                            ),
                            html.Div(
                                [
                                    html.Label("Full 模式"),
                                    dcc.Checklist(
                                        id="is-full-toggle",
                                        options=[{"label": "is_full", "value": "full"}],
                                        value=[],
                                    ),
                                ],
                                className="full-mode-controls",
                            ),
                            html.Div(
                                [
                                    html.Button("启用引导", id="guide-toggle", n_clicks=0),
                                    html.Button("撤回一步", id="guide-undo", n_clicks=0),
                                    html.Button("重置引导", id="guide-reset", n_clicks=0),
                                    html.Div(
                                        "引导未启用。",
                                        id="guide-status",
                                        className="text-muted",
                                    ),
                                ],
                                className="guide-controls",
                            ),
                            html.Button("撤回上一个点", id="coords-undo", n_clicks=0, className="coords-undo"),
                            html.Div(
                                [
                                    html.Button("生成 3D", id="recover-3d-btn", n_clicks=0),
                                    html.Div(
                                        "3D 占位：点齐后点击生成。",
                                        id="recover-3d-placeholder",
                                        className="text-muted",
                                    ),
                                ],
                                className="recovery-3d-controls",
                            ),
                            html.Div(
                                [
                                    html.Button("开始 get_points", id="get-points-start", n_clicks=0),
                                    html.Button("撤回 get_points", id="get-points-undo", n_clicks=0),
                                    html.Div(
                                        "get_points 空闲。",
                                        id="get-points-status",
                                        className="text-muted",
                                    ),
                                ],
                                className="get-points-controls",
                            ),
                            dcc.Store(
                                id="crystal-coords",
                                data={"event": None, "x": None, "y": None},
                            ),
                            dcc.Store(
                                id="crystal-coords-history",
                                data=[],
                            ),
                            dcc.Store(
                                id="crystal-image-context",
                                data=image_context,
                            ),
                            dcc.Store(
                                id="point-guide-state",
                                data=_default_guide_state(),
                            ),
                            dcc.Store(
                                id="get-points-state",
                                data={"enabled": False, "step_idx": 0, "steps": []},
                            ),
                            dcc.Store(
                                id="get-points-data",
                                data={"points": {}, "lines": []},
                            ),
                        ],
                    ),
                    section(
                        "实时数据",
                        [
                            dcc.Graph(
                                id="live-trend",
                                figure={
                                    "layout": {
                                            "title": "实时趋势占位",
                                        "xaxis": {"title": "Time"},
                                        "yaxis": {"title": "Value"},
                                        "annotations": [
                                            {
                                                "text": "连接 RabbitMQ/Influx 数据流",
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
                        "历史分析",
                        [
                            html.Div("TODO: 日期范围选择与历史曲线。"),
                            html.A(
                                "打开 Grafana",
                                id="grafana-link",
                                href=os.getenv("GRAFANA_URL", "http://localhost:3000"),
                                target="_blank",
                                className="grafana-link",
                            ),
                            html.Div("TODO: InfluxDB 指标与聚合。"),
                        ],
                    ),
                    section(
                        "配置",
                        [
                            html.Div("TODO: 绑定 .ini/.txt 配置表单。"),
                            html.Div("TODO: Telegraf/RabbitMQ/InfluxDB 状态指示。"),
                        ],
                    ),
                    section(
                        "调试",
                        [
                            html.Details(
                                [
                                    html.Summary("查看 get_points / coords 的前端内存"),
                                    html.Div("get-points-data:"),
                                    html.Pre(id="debug-get-points", className="debug-json"),
                                    html.Div("crystal-coords-history:"),
                                    html.Pre(id="debug-coords-history", className="debug-json"),
                                ],
                                open=False,
                            )
                        ],
                    ),
                    html.Div(
                        id="recover-3d-modal",
                        className="recover-3d-modal",
                        style={"display": "none"},
                        children=[
                            html.Div(
                                className="recover-3d-modal-content",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("3D 预览"),
                                            html.Button("关闭", id="recover-3d-close", n_clicks=0),
                                        ],
                                        className="recover-3d-modal-header",
                                    ),
                                    dcc.Graph(
                                        id="recover-3d-graph",
                                        figure=_empty_3d_figure(),
                                        config={"displayModeBar": False},
                                    ),
                                    html.Div(
                                        id="recover-3d-status",
                                        className="text-muted",
                                        children="等待生成 3D 结果。",
                                    ),
                                ],
                            )
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
        Input("get-points-data", "data"),
        State("crystal-image-context", "data"),
        prevent_initial_call=True,
    )
    def reload_image(
        selected_asset: Optional[str],
        manual_path: Optional[str],
        get_points_data: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ):
        candidate = manual_path or selected_asset or None
        figure, label, _ = create_placeholder_image_figure(candidate)
        context = context or {}
        context.update({"path": candidate, "label": label})
        if get_points_data:
            figure = _apply_get_points_overlay(figure, get_points_data)
        return figure, label, context

    @app.callback(
        Output("crystal-coords", "data"),
        Output("crystal-coords-display", "children"),
        Output("crystal-coords-history", "data"),
        Output("point-guide-state", "data"),
        Output("guide-status", "children"),
        Input("crystal-image", "clickData"),
        Input("coords-undo", "n_clicks"),
        Input("guide-toggle", "n_clicks"),
        Input("guide-undo", "n_clicks"),
        Input("guide-reset", "n_clicks"),
        State("crystal-image-context", "data"),
        State("crystal-coords", "data"),
        State("crystal-coords-history", "data"),
        State("point-role", "value"),
        State("point-guide-state", "data"),
        prevent_initial_call=True,
    )
    def update_coords(
        click_data: Optional[Dict[str, Any]],
        undo_clicks: int,
        guide_toggle: int,
        guide_undo: int,
        guide_reset: int,
        context: Optional[Dict[str, Any]],
        stored: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]],
        point_role: Optional[str],
        guide_state: Optional[Dict[str, Any]],
    ):
        stored = stored or {"event": None, "x": None, "y": None}
        context = context or {}
        image_label = context.get("label") or "unknown"
        image_path = context.get("path") or "unknown"
        history = history or []
        guide_state = guide_state or _default_guide_state()
        guide_state = dict(guide_state)
        steps = guide_state.get("steps") or []
        step_idx = int(guide_state.get("step_idx", 0))

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

        if trigger == "guide-toggle":
            guide_state["enabled"] = not guide_state.get("enabled", False)
            status = "引导已启用。" if guide_state["enabled"] else "引导已关闭。"
            return stored, no_update, history, guide_state, status

        if trigger == "guide-reset":
            guide_state["step_idx"] = 0
            return stored, no_update, history, guide_state, "引导已重置。"

        if trigger in ("coords-undo", "guide-undo"):
            if not history:
                status = "暂无可撤回的点。"
                return stored, status, history, guide_state, status
            last = history[-1]
            history = history[:-1]
            tags = last.get("tags") or {"event": "unknown", "image_label": image_label, "image_path": image_path}
            timestamp = last.get("time")
            delete_note = ""
            if timestamp:
                try:
                    influx_writer.delete_points(
                        measurement=MEASUREMENT_NAME,
                        timestamp=timestamp,
                        tags=tags,
                    )
                    delete_note = "（数据库回滚成功）"
                except Exception as exc:
                    delete_note = f"（数据库回滚失败: {exc}）"
            if guide_state.get("enabled") and step_idx > 0:
                guide_state["step_idx"] = step_idx - 1
            if history:
                prev = history[-1]
                message = f"上一次 {prev.get('event')}: ({prev.get('x'):.2f}, {prev.get('y'):.2f}){delete_note}"
                status = (
                    f"引导回退到第 {guide_state['step_idx'] + 1} 步。"
                    if guide_state.get("enabled")
                    else message
                )
                return prev, message, history, guide_state, status
            status = f"已撤回。当前无点位{delete_note}"
            return {"event": None, "x": None, "y": None, "time": None, "tags": tags}, status, history, guide_state, status

        coords = None
        event = None
        if trigger == "crystal-image" and click_data:
            coords = _extract_xy(click_data)
            event = "click"

        if not coords:
            status = "引导已启用。" if guide_state.get("enabled") else "引导已关闭。"
            return stored, no_update, history, guide_state, status

        x, y = coords
        now_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        role = point_role or ""
        if guide_state.get("enabled"):
            if step_idx >= len(steps):
                status = "引导已完成，无需继续记录。"
                return stored, status, history, guide_state, status
            role = steps[step_idx]
        tags = {
            "event": event or "unknown",
            "image_label": image_label,
            "image_path": image_path,
            "point_role": role,
        }
        new_data = {"event": event, "x": x, "y": y, "time": now_ts, "tags": tags}
        message = f"最新 {event}: ({x:.2f}, {y:.2f})"
        fields = {"x": float(x), "y": float(y)}
        try:
            influx_writer.write_point(
                measurement=MEASUREMENT_NAME,
                fields=fields,
                tags=tags,
                timestamp=now_ts,
            )
        except Exception as exc:
            message += f" (write failed: {exc})"
        history = history + [new_data]
        if guide_state.get("enabled"):
            guide_state["step_idx"] = step_idx + 1
            status = f"引导中：已记录 {role}，下一步 {guide_state['step_idx'] + 1}/{len(steps)}"
        else:
            status = "引导未启用。"
        return new_data, message, history, guide_state, status

    def _get_role_points(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        role_points: Dict[str, Dict[str, float]] = {}
        for entry in history:
            tags = entry.get("tags") or {}
            role = tags.get("point_role")
            if not role:
                continue
            role_points[role] = {"x": entry.get("x"), "y": entry.get("y")}
        return role_points

    @app.callback(
        Output("recover-3d-modal", "style"),
        Output("recover-3d-graph", "figure"),
        Output("recover-3d-status", "children"),
        Input("recover-3d-btn", "n_clicks"),
        Input("recover-3d-close", "n_clicks"),
        State("crystal-coords-history", "data"),
        prevent_initial_call=True,
    )
    def toggle_3d_modal(
        open_clicks: int,
        close_clicks: int,
        history: Optional[List[Dict[str, Any]]],
    ):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if trigger == "recover-3d-close":
            return {"display": "none"}, _empty_3d_figure(), "已关闭。"

        history = history or []
        role_points = _get_role_points(history)
        required_roles = ("m", "w", "u", "v")
        missing = [role for role in required_roles if role not in role_points]
        if missing:
            missing_str = ", ".join(missing)
            return (
                {"display": "block"},
                _empty_3d_figure(),
                f"缺少点位：{missing_str}",
            )

        fig = _empty_3d_figure()
        fig.add_trace(
            go.Scatter3d(
                x=[role_points["m"]["x"], role_points["w"]["x"], role_points["u"]["x"], role_points["v"]["x"]],
                y=[role_points["m"]["y"], role_points["w"]["y"], role_points["u"]["y"], role_points["v"]["y"]],
                z=[0, 0, 0, 0],
                mode="markers+text",
                text=["m", "w", "u", "v"],
                marker=dict(size=4, color="#e63946"),
            )
        )
        return {"display": "block"}, fig, "已生成 3D 占位图。"

    @app.callback(
        Output("get-points-state", "data"),
        Output("get-points-data", "data"),
        Output("get-points-status", "children"),
        Input("get-points-start", "n_clicks"),
        Input("get-points-undo", "n_clicks"),
        Input("crystal-image", "clickData"),
        State("is-full-toggle", "value"),
        State("get-points-state", "data"),
        State("get-points-data", "data"),
        prevent_initial_call=True,
    )
    def run_get_points(
        start_clicks: int,
        undo_clicks: int,
        click_data: Optional[Dict[str, Any]],
        is_full_value: Optional[List[str]],
        state: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
    ):
        state = state or {"enabled": False, "step_idx": 0, "steps": []}
        data = data or {"points": {}, "lines": []}
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        is_full = bool(is_full_value and "full" in is_full_value)

        if trigger == "get-points-start":
            steps = [
                "u_ad_t",
                "u_ad_e",
                "u_ad_o",
                "v_ad_t",
                "v_ad_e",
                "v_ad_o",
            ]
            if is_full:
                steps += [
                    "u_op_t",
                    "u_op_e",
                    "u_op_o",
                    "v_op_t",
                    "v_op_e",
                    "v_op_o",
                ]
            else:
                steps += ["w"]
            steps += ["k_c1", "k_c2", "k_c3", "k_c4", "k_o1", "k_o2", "k_o3", "k_o4"]
            state = {"enabled": True, "step_idx": 0, "steps": steps, "is_full": is_full}
            data = {"points": {}, "lines": []}
            return state, data, "get_points 已开始。"

        if trigger == "get-points-undo":
            points = dict(data.get("points", {}))
            if not points:
                return state, data, "暂无可撤回的步骤。"
            step_idx = max(int(state.get("step_idx", 0)) - 1, 0)
            steps = state.get("steps") or []
            if step_idx < len(steps):
                points.pop(steps[step_idx], None)
            state["step_idx"] = step_idx
            data["points"] = points
            return state, _rebuild_get_points_data(points, is_full), "已撤回一步。"

        if trigger == "crystal-image" and state.get("enabled"):
            coords = _extract_xy(click_data)
            if not coords:
                return state, data, "未捕获点位。"
            step_idx = int(state.get("step_idx", 0))
            steps = state.get("steps") or []
            if step_idx >= len(steps):
                return state, data, "get_points 已完成。"
            role = steps[step_idx]
            points = dict(data.get("points", {}))
            points[role] = {"x": coords[0], "y": coords[1]}
            state["step_idx"] = step_idx + 1
            data = _rebuild_get_points_data(points, is_full)
            status = f"已记录 {role}。"
            return state, data, status

        return state, data, "get_points 空闲。"



    @app.callback(
        Output("debug-get-points", "children"),
        Output("debug-coords-history", "children"),
        Input("get-points-data", "data"),
        Input("crystal-coords-history", "data"),
    )
    def render_debug_panel(get_points_data, coords_history):
        get_points_text = json.dumps(get_points_data or {}, indent=2)
        coords_text = json.dumps(coords_history or [], indent=2)
        return get_points_text, coords_text


if __name__ == "__main__":
    dash_app = create_app()
    port = int(os.getenv("UI_PORT", "8050"))
    dash_app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "0") == "1")










