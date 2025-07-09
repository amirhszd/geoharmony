import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import webbrowser
from dash import ctx
import time
import threading
import requests
def wait_then_open(port=8050, path=""):
    url = f"http://127.0.0.1:{port}/{path}"
    for _ in range(50):  # retry for ~5 seconds
        try:
            r = requests.get(url)
            if r.status_code == 200:
                webbrowser.open(url)
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.1)


def make_single_image_app(image, results_queue):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

    def make_base_fig(image):
        fig = px.imshow(image)
        fig.update_layout(
            autosize=True,
            dragmode='zoom',
            margin=dict(l=60, r=60, t=20, b=60),
            uirevision='fixed'
        )
        return fig

    fig = make_base_fig(image)

    app.layout = dbc.Container(fluid=True, children=[

        # ---------- Title ----------
        dbc.Row([
            dbc.Col(html.H2("IMAGE MATCHMAKER", className="text-center fw-bold"))
        ], className="mt-3"),

        dbc.Row([
            dbc.Col(html.H5("Warped target image overlayed on the reference image.",
            className="text-center text-muted mb-1"))
        ]),

        dbc.Row([
            dbc.Col(html.H5("If you are satisfied with the result, click 'Done'. If not, click 'Redo' to restart.",
            className="text-center text-muted mb-3"))
        ]),

        # ---------- Image ----------
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dcc.Graph(
                        id="image-display",
                        figure=fig,
                        config={"responsive": True, "scrollZoom": True},
                        style={"height": "70vh"}
                    )
                ], className="shadow")
            ])
        ]),

        # ---------- Buttons ----------
        dbc.Row([
            dbc.Col([
                dbc.Button("Redo", id="redo-button", color="danger", className="me-2 px-5"),
                dbc.Button("Done", id="done-button", color="success", className="px-5")
            ], className="text-center my-4")
        ]),

        # ---------- Feedback Message ----------
        dbc.Row([
            dbc.Col(
                dbc.Alert(id="success-div", is_open=False, color="info", className="text-center fw-bold")
            )
        ])
    ])

    # ✅ Callback is defined BEFORE return
    @app.callback(
        Output("success-div", "children"),
        Output("success-div", "is_open"),
        Input("done-button", "n_clicks"),
        Input("redo-button", "n_clicks"),
        prevent_initial_call=True
    )
    def handle_done_or_redo(done_clicks, redo_clicks):
        triggered_id = ctx.triggered_id
        import threading, os, time
        def delayed_exit():
            time.sleep(1.5)
            os._exit(0)

        if triggered_id == "done-button":
            results_queue.put(True)
            threading.Thread(target=delayed_exit, daemon=True).start()
            return "Done – You are being redirected to the original page.", True

        elif triggered_id == "redo-button":
            results_queue.put(False)
            threading.Thread(target=delayed_exit, daemon=True).start()
            return "Restarting...", True

        raise dash.exceptions.PreventUpdate

    return app  # dash.callback_context is deprecated; use ctx in Dash v2.12+

# =========================
#      RUN DASH APP
# =========================
def run_show_warped_app(image, results_queue):
    app = make_single_image_app(image, results_queue)
    threading.Thread(target=lambda: wait_then_open(8050), daemon=True).start()
    app.run(debug=False)

if __name__ == "__main__":
    import numpy as np
    image = np.random.rand(800,800)
    app = make_single_image_app(image)
    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=False)
