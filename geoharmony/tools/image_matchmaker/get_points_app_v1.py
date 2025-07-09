import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import webbrowser

# =========================
#      MAIN DASH APP
# =========================
def make_dash_app(image1, image2, results_queue):
    app = dash.Dash(__name__)
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

    # =========================
    #      BUTTON STYLES
    # =========================
    button_style = {
        'fontSize': '2em',
        'padding': '10px 20px',
        'borderRadius': '15px',
        'border': '2px solid #333',
        'backgroundColor': '#009CBD',
        'cursor': 'pointer',
        'margin': '10px'
    }
    green_button_style = button_style.copy()
    green_button_style.pop('cursor')
    green_button_style.update({
        'backgroundColor': '#28a745',
        'color': 'white'
    })

    # --- 1. Row wrapper must be allowed to grow/shrink ---
    row_style = {
        "display": "flex",
        "flexDirection": "row",
        "flex": "1 1 0%",  # ① let the row stretch to fill the page
        "minHeight": "0",  # ② and allow it to shrink on phone width
        "width": "100%",
        "padding": "10px",
        "border": "4px solid #F76902",
        "borderRadius": "20px",
        "marginTop": "20px",
    }

    # --- 2. Each LEFT / RIGHT column must also grow ---
    col_style = {
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "flex": "1 1 0%",  # ③ column grows with row
        "minWidth": "0",
        "minHeight": "0"
    }

    # --- 3. The graphs fill the column completely ---
    graph_style = {
        "flex": "1 1 0%",  # ④ take all remaining column space
        "minWidth": "0",
        "minHeight": "0",
        "width": "100%",
        "height": "100%"
    }





    # =========================
    #      DASH LAYOUT
    # =========================
    app.layout = html.Div(
        [
            # =========================
            #      HEADER
            # =========================
            html.H2(
                "IMAGE MATCHMAKER",
                style={
                    'textAlign': 'center',
                    'fontSize': '2.2em',
                    'color': '#333',
                    'fontFamily': 'Georgia, serif',
                    'fontWeight': '600',
                    'letterSpacing': '0.75px',
                    'marginBottom': '0px',
                    'marginTop': '5px'
                }
            ),

            html.H3(
                "Use the left mouse button to drop X marks.",
                style={
                    'textAlign': 'center',
                    'fontSize': '1.5em',
                    'color': '#3a3a3a',  # ← darker gray
                    'fontFamily': 'Georgia, serif',
                    'fontWeight': '600',
                    'marginBottom': '10px'
                }
            ),


            html.Div(
                [
                    # =========================
                    #      LEFT IMAGE
                    # =========================
                    html.Div(
                        [
                            html.H3(
                                "REFERENCE",
                                style={
                                    'textAlign': 'center',
                                    'fontWeight': '600',
                                    'fontSize': '1.5em',
                                    'color': '#F76902',
                                    'fontFamily': 'Georgia, serif',
                                    'letterSpacing': '0.3px'
                                }
                            ),
                            dcc.Graph(
                                id='image-left',
                                figure=fig1,
                                config={'responsive': True},
                                style=graph_style
                            ),
                            # =========================
                            #      LEFT BUTTONS
                            # =========================
                            html.Div(
                                [
                                    html.Button("undo", id='undo-left', n_clicks=0, className='click-button', style= button_style),
                                    html.Button("clear", id='clear-left', n_clicks=0, className='click-button', style= button_style),
                                ],
                                style={'display': 'flex', 'justifyContent': 'center'}
                            )
                        ],
                        style=col_style
                    ),

                    # =========================
                    #      RIGHT IMAGE
                    # =========================
                    html.Div(
                        [
                            html.H3(
                                "TARGET",
                                style={
                                    'textAlign': 'center',
                                    'fontWeight': '600',
                                    'fontSize': '1.5em',
                                    'color': '#F76902',
                                    'fontFamily': 'Georgia, serif',
                                    'letterSpacing': '0.3px'
                                }
                            ),
                            dcc.Graph(
                                id='image-right',
                                figure=fig2,
                                config={'responsive': True},
                                style=graph_style
                            ),
                            # =========================
                            #      RIGHT BUTTONS
                            # =========================
                            html.Div(
                                [
                                    html.Button("undo", id='undo-right', n_clicks=0, className='click-button', style= button_style),
                                    html.Button("clear", id='clear-right', n_clicks=0, className='click-button', style= button_style),
                                ],
                                style={'display': 'flex', 'justifyContent': 'center'}
                            )
                        ],
                        style=col_style
                    ),
                ],
                style=row_style,
            ),

            # =========================
            #      SUBMIT BUTTON
            # =========================
            html.Div(
                html.Button(
                    "submit",
                    id='submit-button',
                    n_clicks=0,
                    className='submit-button',
                    disabled=True,
                    style=green_button_style
                ),
                style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'marginTop': '10px'
                }
            ),
            html.Div(
                id='warning-div',
                style={
                    'color': 'crimson',
                    'textAlign': 'center',
                    'fontWeight': '600'
                }
            ),

            # =========================
            #      DATA STORES
            # =========================
            dcc.Store(id='store-left', data=[]),
            dcc.Store(id='store-right', data=[])
        ],
        style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center',
            'height':'100vh',  # fill the browser window
            'margin':20,
            'padding':20
        }
    )

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
        Output('submit-button', 'children'),  # dummy output
        Input('submit-button', 'n_clicks'),
        State('store-left', 'data'),
        State('store-right', 'data'),
        prevent_initial_call=True
    )
    def do_submit(n_clicks, left_pts, right_pts):
        # by the time we get here, validation has passed and n_clicks>0
        results_queue.put((left_pts, right_pts))
        os._exit(0)
        return dash.no_update


    @app.callback(
        Output('submit-button', 'disabled'),
        Output('warning-div',    'children'),
        Input('store-left',       'data'),
        Input('store-right',      'data'),
        prevent_initial_call=False
    )
    def validate_points(left_pts, right_pts):
        if len(left_pts) < 4 or len(right_pts) < 4:
            return True,  "Pick at least 4 points in each image."
        if len(left_pts) != len(right_pts):
            return True,  "Both images must have the same number of points."
        return False, ""  # enables button and clears warning



    return app

# =========================
#      RUN DASH APP
# =========================
def run_dash(image1, image2, results_queue):
    app = make_dash_app(image1, image2, results_queue)
    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=False)