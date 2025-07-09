import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import webbrowser
import dash_bootstrap_components as dbc

# =========================
#      MAIN DASH APP
# =========================
def make_dash_app(image1, image2, results_queue):
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY]  # pick any of the 30+ themes
    )
    # =========================
    #      BASE FIGURE MAKER
    # =========================
    def make_base_fig(image):
        fig = px.imshow(image)
        fig.update_layout(
            clickmode='event+select',
            dragmode='zoom',
            autosize=True,  # let Plotly fill its parent
            margin=dict(l=60, r=60, t=20, b=60),
            uirevision='fixed'
        )
        return fig

    # =========================
    #      INITIAL FIGURES
    # =========================
    fig1 = make_base_fig(image1)
    fig2 = make_base_fig(image2)

    def serve_layout(fig1, fig2):
        return dbc.Container(fluid=True, children=[

            # ---------- Header ----------
            dbc.Row([
                dbc.Col(html.H2("IMAGE MATCHMAKER", className="text-center fw-bold"))
            ], className="mt-3"),

            dbc.Row([
                dbc.Col(html.H5("Use the left mouse button to drop X marks.",
                                className="text-center text-muted mb-3"))
            ]),

            # ---------- Two responsive columns ----------
            dbc.Row([

                # ----- Reference side -----
                dbc.Col(md=6, children=[
                    html.H4("REFERENCE", className="text-center text-primary fw-bold"),
                    dbc.Card([
                        dcc.Graph(
                            id="image-left",
                            figure=fig1,
                            config={"responsive": True, "scrollZoom": True},
                            style={"height": "60vh"}  # fill 60 % of viewport height
                        ),
                        dbc.CardBody(
                            dbc.ButtonGroup([
                                dbc.Button("Undo", id="undo-left", color="info", n_clicks=0),
                                dbc.Button("Clear", id="clear-left", color="info", n_clicks=0)
                            ], className="d-flex justify-content-center")
                        )
                    ], className="shadow")
                ]),

                # ----- Target side -----
                dbc.Col(md=6, children=[
                    html.H4("TARGET", className="text-center text-primary fw-bold"),
                    dbc.Card([
                        dcc.Graph(
                            id="image-right",
                            figure=fig2,
                            config={"responsive": True, "scrollZoom": True},
                            style={"height": "60vh"}
                        ),
                        dbc.CardBody(
                            dbc.ButtonGroup([
                                dbc.Button("Undo", id="undo-right", color="info", n_clicks=0),
                                dbc.Button("Clear", id="clear-right", color="info", n_clicks=0)
                            ], className="d-flex justify-content-center")
                        )
                    ], className="shadow")
                ])
            ], className="gx-4"),  # horizontal gutter

            # ---------- Submit button ----------
            dbc.Row([
                dbc.Col(
                    dbc.Button("Submit", id="submit-button",
                               color="success", disabled=True, className="px-5"),
                    className="text-center my-4"
                )
            ]),

            # ---------- Warning message ----------
            dbc.Row([
                dbc.Col(
                    dbc.Alert(id="warning-div", color="warning",
                              is_open=False,  # start hidden
                              className="text-center fw-bold"),
                )
            ]),

            dbc.Row([
                dbc.Col(
                    dbc.Alert(id="success-div",
                              color="success",
                              is_open=False,
                              className="text-center fw-bold"),
                )
            ]),


            # Stores
            dcc.Store(id="store-left", data=[]),
            dcc.Store(id="store-right", data=[])
        ])

    app.layout = serve_layout(fig1, fig2)

    # =========================
    #      LEFT IMAGE CALLBACK
    # =========================
    @app.callback(
        Output('image-left', 'figure'),
        Output('store-left', 'data'),
        Input('image-left', 'clickData'),
        Input('undo-left', 'n_clicks'),
        Input('clear-left', 'n_clicks'),
        State('image-left', 'relayoutData'),
        State('store-left', 'data'),
        prevent_initial_call=True
    )
    def update_left(clickData, undo, clear, relayout, points):
        return update_image_core(clickData, undo, clear, relayout, points, image1)

    # =========================
    #      RIGHT IMAGE CALLBACK
    # =========================
    @app.callback(
        Output('image-right', 'figure'),
        Output('store-right', 'data'),
        Input('image-right', 'clickData'),
        Input('undo-right', 'n_clicks'),
        Input('clear-right', 'n_clicks'),
        State('image-right', 'relayoutData'),
        State('store-right', 'data'),
        prevent_initial_call=True
    )
    def update_right(clickData, undo, clear, relayout, points):
        return update_image_core(clickData, undo, clear, relayout, points, image2)

    # =========================
    #      IMAGE CORE LOGIC
    # =========================
    def update_image_core(clickData, undo_clicks, clear_clicks, relayoutData, stored_points, image):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if 'image' in triggered_id and clickData:
            point = clickData['points'][0]
            stored_points.append({'x': point['x'], 'y': point['y']})
        elif 'undo' in triggered_id:
            if stored_points:
                stored_points.pop()
        elif 'clear' in triggered_id:
            stored_points = []

        fig_new = make_base_fig(image)

        if stored_points:
            fig_new.add_trace(
                go.Scatter(
                    x=[p['x'] for p in stored_points],
                    y=[p['y'] for p in stored_points],
                    mode='markers+text',
                    marker=dict(size=10, color='red', symbol='x', line=dict(width=2), opacity=1),
                    text=[str(i + 1) for i in range(len(stored_points))],
                    textposition='top center',
                    textfont=dict(size=14, color='black', family='Arial Black'),
                    name='Clicks',
                    opacity=1
                )
            )

        if relayoutData:
            x_range = relayoutData.get('xaxis.range[0]'), relayoutData.get('xaxis.range[1]')
            y_range = relayoutData.get('yaxis.range[0]'), relayoutData.get('yaxis.range[1]')
            if None not in x_range:
                fig_new.update_xaxes(range=x_range)
            if None not in y_range:
                fig_new.update_yaxes(range=y_range)

        return fig_new, stored_points



    # =========================
    #      SUBMIT CALLBACK
    # =========================
    @app.callback(
        Output("success-div", "children"),
        Output("success-div", "is_open"),
        Input("submit-button", "n_clicks"),
        State("store-left", "data"),
        State("store-right", "data"),
        prevent_initial_call=True
    )
    def do_submit(n, left_pts, right_pts):
        if not n:
            raise dash.exceptions.PreventUpdate

        # 1) Put the results on the queue
        results_queue.put((left_pts, right_pts))

        # 2) Show “Success” message for the user
        success_msg = "Points submitted – you may close this window."

        # 3) OPTIONAL: exit the Dash process after a brief delay
        #    (gives the user time to see the message)
        import threading, os, time
        def delayed_exit():
            time.sleep(2)  # 2-second grace
            os._exit(0)

        threading.Thread(target=delayed_exit, daemon=True).start()

        return success_msg, True

    @app.callback(
        Output("submit-button", "disabled"),
        Output("warning-div", "children"),
        Output("warning-div", "is_open"),
        Input("store-left", "data"),
        Input("store-right", "data")
    )
    def validate(left_pts, right_pts):
        if len(left_pts) < 4 or len(right_pts) < 4:
            return True, "Pick at least 4 points in each image.", True
        if len(left_pts) != len(right_pts):
            return True, "Both images must have the same number of points.", True
        return False, "", False



    return app

# =========================
#      RUN DASH APP
# =========================
def run_dash(image1, image2, results_queue):
    app = make_dash_app(image1, image2, results_queue)
    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=False)