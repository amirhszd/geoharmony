import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
import webbrowser, threading, time, requests, os

# ---------- Helper ----------
def wait_then_open(port=8050, path=""):
    url = f"http://127.0.0.1:{port}/{path}"
    for _ in range(60):
        try:
            if requests.get(url).status_code == 200:
                webbrowser.open(url)
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.1)


# ---------- Safe cast ----------
def to_uint8(arr):
    """
    Cast an arbitrary-dtype image array to uint8.
      • float in [0,1]  → scale ×255
      • float in [0, >1]→ clip to [0,255]
      • int16/int32     → clip & cast
      • uint8           → returned unchanged
    """
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)

    # If data look like reflectance (0-1), scale up
    if arr_f.max() <= 1.0 + 1e-3:
        arr_f *= 255.0

    return np.clip(arr_f, 0, 255).astype(np.uint8)

# ---------- Core app ----------
def make_blend_app(ref_arr, tgt_arr, results_queue):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

    def make_composite(alpha):
        ref = to_uint8(ref_arr)  # ← ensure uint8
        tgt = to_uint8(tgt_arr)
        # blend: α·ref + (1-α)·tgt
        comp = alpha * ref.astype(np.float32) + (1 - alpha) * tgt.astype(np.float32)
        return np.clip(comp, 0, 255).astype(np.uint8)

    # initial figure (α = 0.5)
    fig = px.imshow(make_composite(0.5))
    fig = go.FigureWidget(fig)
    fig.update_layout(
        autosize=True, dragmode="zoom",
        margin=dict(l=60, r=60, t=20, b=60),
        uirevision="fixed"
    )

    # ---------- Layout ----------
    app.layout = dbc.Container(fluid=True, children=[
        dbc.Row(dbc.Col(html.H2("IMAGE MATCHMAKER", className="text-center fw-bold"), className="mt-3")),
        dbc.Row(dbc.Col(html.H5("Adjust the slider to blend reference and warped target images.",
                                className="text-center text-muted mb-3"))),

        # Image
        dbc.Row(dbc.Col(
            dbc.Card(dcc.Graph(id="img", figure=fig, config={"responsive": True, "scrollZoom": True},
                               style={"height": "70vh"}), className="shadow")
        )),

        # Alpha slider
        dbc.Row(dbc.Col(
            dcc.Slider(id='alpha-slider', min=0, max=1, step=0.1, value=0.5,
                       tooltip={"placement": "bottom", "always_visible": True},
                       style={"width": "50%"}),
            className="my-3 px-5"
        )),

        # Buttons
        dbc.Row(dbc.Col([
            dbc.Button("Redo", id="redo-btn", color="danger", className="me-2 px-5"),
            dbc.Button("Done", id="done-btn", color="success", className="px-5")
        ], className="text-center my-3")),

        # Feedback
        dbc.Row(dbc.Col(dbc.Alert(id="msg", is_open=False, color="info",
                                  className="text-center fw-bold")))
    ])

    # ---------- Callbacks ----------
    @app.callback(
        Output("img", "figure"),
        Input("alpha-slider", "value"),
        prevent_initial_call=True
    )
    def update_blend(alpha):
        comp = make_composite(alpha)
        new_fig = px.imshow(comp)
        new_fig.update_layout(
            autosize=True, dragmode="zoom",
            margin=dict(l=60, r=60, t=20, b=60),
            uirevision="fixed"
        )
        return new_fig

    @app.callback(
        Output("msg", "children"),
        Output("msg", "is_open"),
        Input("done-btn", "n_clicks"),
        Input("redo-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def handle_buttons(done, redo):
        trig = dash.callback_context.triggered_id
        def delayed_exit(flag):
            time.sleep(1.5)
            os._exit(0)

        if trig == "done-btn":
            results_queue.put(True)
            threading.Thread(target=delayed_exit, args=(True,), daemon=True).start()
            return "Done – redirecting …", True
        elif trig == "redo-btn":
            results_queue.put(False)
            threading.Thread(target=delayed_exit, args=(False,), daemon=True).start()
            return "Redo selected – restarting …", True
        raise dash.exceptions.PreventUpdate

    return app

# ---------- Runner ----------
def run_show_warped_app(ref_arr, tgt_arr, results_queue):
    app = make_blend_app(ref_arr, tgt_arr, results_queue)
    threading.Thread(target=lambda: wait_then_open(8050), daemon=True).start()
    app.run(debug=False)

# ---------- Stand-alone test ----------
if __name__ == "__main__":
    import cv2
    ref = cv2.cvtColor(cv2.imread("ref.jpg"), cv2.COLOR_BGR2RGB)
    tgt = cv2.cvtColor(cv2.imread("tgt.jpg"), cv2.COLOR_BGR2RGB)
    q = None  # replace with mp.Queue() when using from parent script
    run_blend_app(ref, tgt, q)