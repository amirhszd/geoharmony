import os
import geoharmony.tools.gdalimage as gi
import dash
from dash import callback_context
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, no_update, Patch
import plotly.express as px
import plotly.graph_objects as go
from flask import request
import threading
import dash_leaflet as dl
from io import BytesIO
import base64
from PIL import Image
import numpy as np

text_style = {
    "backgroundColor": "#f8f9fa",  # light gray
    "border": "1px solid #ddd",
    "borderRadius": "6px",
    "padding": "12px",
    "marginBottom": "10px",
    "fontSize": "16px",
    "lineHeight": "1.5",
    "color": "#333"
}

step1_text  = """
**📍 Step 1 – Getting Started**  

Provide the required file paths and choose your processing options:  

---

**1️⃣ Input your files**  
- **VNIR**, **SWIR**, and **Micasense** images are required.  
- *(Optional)* Include the **Micasense DSM** if available — often generated in Pix4D.  

**2️⃣ Units don’t matter**  
- The algorithms use **space-independent metrics** (e.g., mutual information) to align datasets.  

**3️⃣ Speed up your workflow**  
- Increase the **number of threads** for faster processing.  
- Toggle *Save to Memory* to store intermediate files in RAM instead of disk  
  _(faster, but uses more RAM)_.  

---
"""

step2_text = """
**📍 Step 2 – Band Selection for Rigid Coregistration**  

Choose your RGB representatives and preview registration results:  

---

**1️⃣ Select representative bands**  
- Use the drop-down menus to pick **blue**, **green**, and **red** band representatives for:  
  - Your **VNIR** dataset  
  - Your **SWIR** dataset  
  - Your **MicaSense** dataset  

**2️⃣ Coregistration process**  
- These selected bands are used for **rigid coregistration** between:  
  - **VNIR ↔ MicaSense**  
  - **SWIR ↔ MicaSense**  

**3️⃣ Use Available Homography**  
- Toggle this on if **homography files** already exist in your output folder.  
- If none exist, **new files will be generated**.  

**4️⃣ Preview layers on the map**  
- Use the **layer control** (top-right corner of the map) to toggle datasets on/off.  
- You can visually check **how well the datasets align** prior to coregistration.  

---
"""

step3_text =     """
**📍 Step 3 – ShapeShifter Alignment**  

Fine-tune alignment using pixel shifting and band selection:  

---

**1️⃣ Pixel Shift**  
- Moves each **waterfall row** in the orthorectified image left/right to correct roll error.  
- Suggested range: **3–7 pixels** depending on misregistration severity.  

**2️⃣ Kernel Size**  
- Sets the **median filter** window to smooth misplaced pixels after shifting.  

**3️⃣ Band Selection for ShapeShifting**  
- **VNIR → MicaSense**: Select a **VNIR band range** (averaged) and a **single MicaSense band**.  
- **SWIR → MicaSense**: Select a **SWIR band range** and a **single MicaSense band**.  

**💡 Suggested settings:**  
- **VNIR**: Green band range (~500–600 nm) and MicaSense Band 2.  
- **SWIR**: Use VNIR–SWIR overlap region; choose the **last MicaSense band** (closest to SWIR sensor).  

---
"""

step4_text =     """
**📍 Step 4 – Elastic Registration**  

Align the **Sphere** image to the **VNIR (Veneer)** image using representative overlap bands:  

---

**1️⃣ Band Selection**  
- **VNIR Overlap Bands** and **SWIR Overlap Bands** are averaged to reduce noise in low SNR regions.  
- These mean-band images are used for the registration process.  

**2️⃣ Bilateral Filtering**  
- **Bilateral Filter Kernel Size**: Controls the smoothing window.  
- **Bilateral Filter Sigma Color** & **Bilateral Filter Sigma Space**: Adjust color-based and spatial smoothing.  
- Purpose: **Remove noise** while **preserving fine texture**.  
- More details: [OpenCV Bilateral Filter Docs](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)  

**3️⃣ Registration Parameters**  
- **Internal Size, Steps, Flow Sigma**: Advanced ANTsPy registration settings.  
- Fixed image: **VNIR **  
- Moving image: **SWIR**  
- Transform type: **SyN**  
- More info: [ANTsPy Registration Docs](https://antspy.readthedocs.io/en/latest/registration.html)  

**4️⃣ Performance**  
- **CPU Cores**: Number of cores for parallel processing (CPU-bound, no extra memory usage).  

---
"""

# Helper for RGB image display using Plotly
# Helper for RGB image display using Plotly
def make_base_fig(image):
    fig = px.imshow(image)
    fig = go.Figure(fig)
    fig.update_layout(
        clickmode='event+select',
        dragmode='zoom',
        autosize=True,
        margin=dict(l=60, r=60, t=20, b=60),
        uirevision='fixed'
    )
    return fig

results_queue = None

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.COSMO,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)



### IMAGE PATHS CONTAINER
# creating input path filename for each image with
labels = ["VNIR", "SWIR", "Micasense", "Micasense DSM"]

# creating input path filename for RIGID COregisteratION
def make_dropdown_row(label):
    id_map = {
        "VNIR": "coreg_vnir_band_indices",
        "SWIR": "coreg_swir_band_indices",
        "Micasense": "coreg_mica_band_indices"
    }
    dropdown_id = id_map.get(label, f"band-dropdown-{label}")
    return html.Div([
        html.Label(f"{label} Bands", className="form-label"),
        dcc.Dropdown(
            id=dropdown_id,
            multi=True,
            placeholder="Pick 3: Green, Blue, Red",
            className="mb-3"
        )
    ])



# --- Card Components for each section ---
card_image_paths = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            # Left: compact form-grid version
            dbc.Col(
                dbc.Form(
                    [
                        dbc.Row([
                            dbc.Label("VNIR Path", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Input(
                                    placeholder="Path to the hyperspectral VNIR data cube...",
                                    id="vnir_path",
                                    type="text",
                                    persistence=True,
                                    persistence_type='memory',
                                    invalid=True,
                                    size="sm",
                                    style={"maxWidth": 520}
                                ),
                                width=8
                            )
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("SWIR Path", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Input(
                                    placeholder="Path to the hyperspectral SWIR data cube...",
                                    id="swir_path",
                                    type="text",
                                    persistence=True,
                                    persistence_type='memory',
                                    invalid=True,
                                    size="sm",
                                    style={"maxWidth": 520}
                                ),
                                width=8
                            )
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Micasense Path", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Input(
                                    placeholder="Path to the multispectral data...",
                                    id="mica_path",
                                    type="text",
                                    persistence=True,
                                    persistence_type='memory',
                                    invalid=True,
                                    size="sm",
                                    style={"maxWidth": 520}
                                ),
                                width=8
                            )
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Micasense DSM Path (optional)", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Input(
                                    placeholder="Path to the digital surface model (DSM) generated from multispectral data...",
                                    id="mica_dsm_path",
                                    type="text",
                                    persistence=True,
                                    persistence_type='memory',
                                    size="sm",
                                    style={"maxWidth": 520}
                                ),
                                width=8
                            )
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Output Folder", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Input(
                                    placeholder="Directory in which algorithm outputs are written...",
                                    id="out_folder",
                                    type="text",
                                    debounce=True,
                                    persistence=True,
                                    persistence_type='memory',
                                    invalid=True,
                                    size="sm",
                                    style={"maxWidth": 520}
                                ),
                                width=8
                            )
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("CPU Cores", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Input(
                                    id="num_threads",
                                    type="number",
                                    min=1,
                                    max=os.cpu_count(),
                                    step=1,
                                    persistence=True,
                                    persistence_type='memory',
                                    value=os.cpu_count(),
                                    size="sm",
                                    style={"maxWidth": 200}
                                ),
                                width=8
                            )
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Save to Memory", width=4, className="col-form-label"),
                            dbc.Col(
                                dbc.Checklist(
                                    options=[{"label": "", "value":"save"}],
                                    value=["save"],
                                    id="save_to_memory",
                                    switch=True,
                                ),
                                width=8
                            )
                        ], className="g-2"),
                    ],
                    className="form-narrow"
                ),
                width=6
            ),
            # Right: instruction block
            dbc.Col(dcc.Markdown(step1_text  , style = text_style), width=6),
        ])
    ])
], style={"borderRadius": "0.5rem", "padding": "1rem", "marginBottom": "2rem"})

card_coreg = dbc.Card([
    dbc.CardBody([
        # --- Row 1: (1) Controls stacked, (2) Instructions text box ---
        dbc.Row([
            # (1) VNIR/SWIR/Micasense band dropdowns stacked + Homography switch
            dbc.Col(
                dbc.Form([
                    dbc.Row([
                        dbc.Label("VNIR Bands", width=5, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="coreg_vnir_band_indices",
                                multi=True,
                                placeholder="Pick 3: Blue, Green, Red",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=7
                        )
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Label("SWIR Bands", width=5, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="coreg_swir_band_indices",
                                multi=True,
                                placeholder="Pick 3: Blue, Green, Red",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=7
                        )
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Label("Micasense Bands", width=5, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="coreg_mica_band_indices",
                                multi=True,
                                placeholder="Pick 3: Blue, Green, Red",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=7
                        )
                    ], className="g-2 mb-3"),

                    dbc.Row([
                        dbc.Label("Use available homography", width=5, className="col-form-label"),
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "", "value": "use_homography"}],
                                value=[],
                                id="use_available_homography",
                                switch=True
                            ),
                            width=7
                        )
                    ], className="g-2"),
                ], className="form-narrow"),
                md=6, lg=5
            ),
            dbc.Col(dcc.Markdown(step2_text, style=text_style), md=6, lg=7),
        ]),

        # --- Row 2: (3) Map, (4) Map — implemented as one full-width map to avoid duplicate IDs ---
        dbc.Row([
            dbc.Col(
                dl.Map([
                    dl.LayersControl([], id="lc")
                ],
                    id="coreg-map",
                    center=[30, -95],
                    zoom=4,
                    style={"width": "100%", "height": "50vh"}
                ),
                width=12
            )
        ], className="mb-3"),
    ])
], style={"borderRadius": "0.5rem", "padding": "1rem", "marginBottom": "2rem"})

card_shape = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col(
                dbc.Form([
                    dbc.Row([
                        dbc.Label("Pixel Shift", width=6, className="col-form-label"),
                        dbc.Col(
                            dcc.Input(
                                id="pixel-shift",
                                type="number",
                                min=1,
                                value=3,
                                persistence=True,
                                persistence_type='memory',
                                className="form-control",
                                **{"size": "sm"}
                            ),
                            width=6
                        )
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Label("Median Filter Kernel Size", width=6, className="col-form-label"),
                        dbc.Col(
                            dcc.Input(
                                id="kernel-size",
                                type="number",
                                min=3,
                                step=2,
                                value=3,
                                persistence=True,
                                persistence_type='memory',
                                className="form-control",
                                **{"size": "sm"}
                            ),
                            width=6
                        )
                    ], className="g-2 mb-3"),

                    dbc.Row([
                        dbc.Label("VNIR band range to use for shapeshifting with Micasense", width=6, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ss_vnir_band_range",
                                multi=True,
                                placeholder="Pick start and end of range",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=6
                        )
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Label("Micasense Band for Shape-shifting VNIR", width=6, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ss_vnir_mica_band_index",
                                placeholder="Select Band",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=6
                        )
                    ], className="g-2 mb-3"),

                    dbc.Row([
                        dbc.Label("SWIR band range to use for shapeshifting Micasense", width=6, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ss_swir_band_range",
                                multi=True,
                                placeholder="Pick start and end of range",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=6
                        )
                    ], className="g-2 mb-2"),

                    dbc.Row([
                        dbc.Label("Micasense Band for Shape-shifting SWIR", width=6, className="col-form-label"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ss_swir_mica_band_index",
                                placeholder="Select Band",
                                persistence=True,
                                persistence_type='memory',
                                style={"width": "100%", "maxWidth": 420}
                            ),
                            width=6
                        )
                    ], className="g-2"),
                ], className="form-narrow"),
                md=6, lg=5
            ),
            dbc.Col(dcc.Markdown(step3_text, style=text_style), md=6, lg=7),
        ], className="align-items-start")
    ])
], style={"borderRadius": "0.5rem", "padding": "1rem", "marginBottom": "2rem"})

card_elastic = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dbc.Form(
                    [
                        dbc.Row([
                            dbc.Label("Bilateral Filter Kernel Size", width=5, className="col-form-label"),
                            dbc.Col(dbc.Input(id="elastic-kernel", type="number", min=3, value=20, size="sm"),
                                    width=7)
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Bilateral Filter Sigma Color", width=5, className="col-form-label"),
                            dbc.Col(dbc.Input(id="elastic-sigma-color", type="number", min=1, value=15, size="sm"),
                                    width=7)
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Bilateral Filter Sigma Space", width=5, className="col-form-label"),
                            dbc.Col(dbc.Input(id="elastic-sigma-space", type="number", min=1, value=20, size="sm"),
                                    width=7)
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("CPU Cores", width=5, className="col-form-label"),
                            dbc.Col(dbc.Input(id="elastic-steps", type="number", min=1, value = 8, step=1, size="sm"),
                                    width=7)
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("Flow Sigma", width=5, className="col-form-label"),
                            dbc.Col(dbc.Input(id="elastic-flow-sigma", type="number", min=1, value=3, size="sm"),
                                    width=7)
                        ], className="g-2 mb-3"),

                        dbc.Row([
                            dbc.Label("VNIR Overlap Bands", width=5, className="col-form-label"),
                            dbc.Col(dcc.Dropdown(id="ereg_vnir_overlap_band_indices",
                                                 multi=True,
                                                 placeholder="Pick start & end",
                                                 style={"width": "100%", "maxWidth": 420}),
                                    width=7)
                        ], className="g-2 mb-2"),

                        dbc.Row([
                            dbc.Label("SWIR Overlap Bands", width=5, className="col-form-label"),
                            dbc.Col(dcc.Dropdown(id="ereg_swir_overlap_band_indices",
                                                 multi=True,
                                                 placeholder="Pick start & end",
                                                 style={"width": "100%", "maxWidth": 420}),
                                    width=7)
                        ], className="g-2"),
                    ],
                    className="form-narrow"
                ), md=6, lg=5),
            dbc.Col(dcc.Markdown(step4_text, style = text_style), md=6, lg=5),
        ])
    ])
], style={"borderRadius": "0.5rem", "padding": "1rem", "marginBottom": "2rem"})

# RGB Card
rgb_card = dbc.Card([
    dbc.CardBody([
        # Label row
        dbc.Row([
            dbc.Col(html.H5("VNIR", className="text-center"), width=4),
            dbc.Col(html.H5("SWIR", className="text-center"), width=4),
            dbc.Col(html.H5("Micasense", className="text-center"), width=4),
        ], className="mb-2"),
        # Graph row
        dbc.Row([
            dbc.Col(dcc.Graph(id="vnir-image", config={"responsive": True}), width=4),
            dbc.Col(dcc.Graph(id="swir-image", config={"responsive": True}), width=4),
            dbc.Col(dcc.Graph(id="mica-image", config={"responsive": True}), width=4),
        ], className="mb-3")
    ])
], style={"borderRadius": "0.5rem", "padding": "1rem", "marginBottom": "2rem"})

app.layout = html.Div(style={"paddingLeft": "150px", "paddingRight": "150px", "paddingTop": "2rem", "paddingBottom": "2rem"}, children=[
    dbc.Container(fluid=True, children=[
        # --- Custom Header with Logo and Title ---
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Img(src="/assets/splash_logo.png", height="60px"),  # Adjust path if needed
                    width="auto"
                ),
                dbc.Col(
                    html.Div([
                        html.H2("Multi Modal Hyperspectral Pixel Aligner", className="display-6", style={
                            "fontWeight": "500", "paddingBottom": "100px", "paddingTop": "100px"
                        })
                    ]),
                    style={"display": "flex", "alignItems": "center"}
                ),
                dbc.Col(
                    html.A(
                        html.I(className="fab fa-github fa-2x", style={"color": "black"}),
                        href="https://github.com/amirhszd/geoharmony",  # Replace with actual repo URL
                        target="_blank",
                        style={"marginLeft": "auto", "marginRight": "10px"}
                    ),
                    width="auto"
                )
            ], align="center", className="mb-4")
        ], style={"backgroundColor": "#f0f0f0", "padding": "1rem", "borderRadius": "0.5rem"}),
        # Wizard: step storage
        dcc.Store(id="step", data=1),
        # Wizard header
        html.Div([
            html.H5(id="wizard-title"),
            dbc.Progress(id="wizard-progress", striped=True, animated=True, className="mb-4")
        ]),
        # Wizard content
        html.Div(id="wizard-content", children=[
            html.Div(card_image_paths, id="container-step-1"),
            html.Div(card_coreg,        id="container-step-2"),
            html.Div(card_shape,        id="container-step-3"),
            html.Div(card_elastic,      id="container-step-4"),
        ]),
        # Hidden output Div for run callback
        html.Div(id="run-output", style={"display": "none"}),
        # Navigation buttons
        dbc.Row([
            dbc.Col(dbc.Button("Back", id="btn-back", color="secondary", className="me-2"), width="auto"),
            dbc.Col(dbc.Button("Next", id="btn-next", color="primary", className="me-2"), width="auto"),
            dbc.Col(
                dbc.Button(
                    "Run",
                    id="btn-finish",
                    color="success",
                    style={"display": "none"}
                ),
                width="auto"
            )
        ], className="mt-3"),

        dbc.Row(id="spinner-ui", style={"display": "none"}, children=[
            dbc.Col([
                html.H4("Processing...", className="text-center mt-4"),
                dbc.Spinner(size="lg", color="primary", fullscreen=True)
            ])
        ]),

        html.Hr(),
        # (Wizard removes static RGB alert and card from layout)
    ])
])

# Dynamically register callbacks for each dataset
lbl = labels[0]
@app.callback(
    Output("coreg_vnir_band_indices", "options"),
    Output("coreg_vnir_band_indices", "value"),
    Input("vnir_path", "value"),
    Input("coreg_vnir_band_indices", "value"),
    prevent_initial_call=True
)
def update_dropdown(path, selected, _lbl=lbl):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    # Build options from GDAL image if path is valid
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [
                {"label": desc or f"Band {i+1}", "value": i}
                for i, desc in enumerate(img.band_description)
            ]
        except Exception:
            opts = []
    else:
        opts = []
    # If path changed, reset to default first 3 bands
    if trigger == "vnir_path":
        default = [0, 1, 2] if len(opts) >= 3 else list(range(len(opts)))
        return opts, default
    # If dropdown changed, clamp to max of 3 selections
    elif trigger == "coreg_vnir_band_indices":
        sel = selected or []
        if len(sel) > 3:
            sel = sel[:3]
        return opts, sel
    # Fallback
    return opts, selected or []


lbl = labels[1]
@app.callback(
    Output("coreg_swir_band_indices", "options"),
    Output("coreg_swir_band_indices", "value"),
    Input("swir_path", "value"),
    Input("coreg_swir_band_indices", "value"),
    prevent_initial_call=True
)
def update_dropdown(path, selected, _lbl=lbl):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    # Build options from GDAL image if path is valid
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [
                {"label": desc or f"Band {i+1}", "value": i}
                for i, desc in enumerate(img.band_description)
            ]
        except Exception:
            opts = []
    else:
        opts = []
    # If path changed, reset to default first 3 bands
    if trigger == "swir_path":
        default = [0, 1, 2] if len(opts) >= 3 else list(range(len(opts)))
        return opts, default
    # If dropdown changed, clamp to max of 3 selections
    elif trigger == "coreg_swir_band_indices":
        sel = selected or []
        if len(sel) > 3:
            sel = sel[:3]
        return opts, sel
    # Fallback
    return opts, selected or []

lbl = labels[2]
@app.callback(
    Output("coreg_mica_band_indices", "options"),
    Output("coreg_mica_band_indices", "value"),
    Input("mica_path", "value"),
    Input("coreg_mica_band_indices", "value"),
    prevent_initial_call=True
)
def update_dropdown(path, selected, _lbl=lbl):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    # Build options from GDAL image if path is valid
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [
                {"label": desc or f"Band {i+1}", "value": i}
                for i, desc in enumerate(img.band_description)
            ]
        except Exception:
            opts = []
    else:
        opts = []
    # If path changed, reset to default first 3 bands
    if trigger == "mica_path":
        default = [0, 1, 2] if len(opts) >= 3 else list(range(len(opts)))
        return opts, default
    # If dropdown changed, clamp to max of 3 selections
    elif trigger == "coreg_mica_band_indices":
        sel = selected or []
        if len(sel) > 3:
            sel = sel[:3]
        return opts, sel
    # Fallback
    return opts, selected or []




# --- Wizard callbacks ---

# Wizard header: title and progress
@app.callback(
    Output("wizard-title", "children"),
    Output("wizard-progress", "value"),
    Input("step", "data")
)
def update_wizard_header(step):
    titles = {
        1: "Step 1 of 4: Image Paths",
        2: "Step 2 of 4: Rigid Coregisteration",
        3: "Step 3 of 4: Shape Shifter",
        4: "Step 4 of 4: Elastic registeration"
    }
    progress = step * 25
    return titles.get(step, ""), progress



# Wizard navigation
@app.callback(
    Output("step", "data"),
    Output("btn-back", "disabled"),
    Output("btn-next", "disabled"),
    Output("btn-finish", "style"),
    Input("btn-back", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("step", "data"),
    Input("vnir_path", "value"),
    Input("swir_path", "value"),
    Input("mica_path", "value"),
    Input("out_folder", "value")
)
def navigate(prev_clicks, next_clicks, step, vnir_path, swir_path, mica_path, out_folder):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    # Only change step on button clicks
    if trigger == "btn-next" and step < 4:
        step += 1
    elif trigger == "btn-back" and step > 1:
        step -= 1

    # Disable "Back" on first step
    back_disabled = (step == 1)
    # Disable "Next" if on last step or required fields missing on step 1
    if step == 1:
        next_disabled = not all([vnir_path, swir_path, mica_path, out_folder])
    elif step == 4:
        next_disabled = True
    else:
        next_disabled = False
    # Show "Run" button only on last step
    finish_style = {"display": "block"} if step == 4 else {"display": "none"}

    return step, back_disabled, next_disabled, finish_style


# --- Combined input validation for file/folder paths ---
@app.callback(
    Output("vnir_path", "invalid"),
    Output("vnir_path", "valid"),
    Output("swir_path", "invalid"),
    Output("swir_path", "valid"),
    Output("mica_path", "invalid"),
    Output("mica_path", "valid"),
    Output("out_folder", "invalid"),
    Output("out_folder", "valid"),
    Input("vnir_path", "value"),
    Input("swir_path", "value"),
    Input("mica_path", "value"),
    Input("out_folder", "value")
)
def validate_inputs(vnir, swir, mica, out):
    # VNIR path
    vnir_ok = bool(vnir and os.path.isfile(vnir))
    # SWIR path
    swir_ok = bool(swir and os.path.isfile(swir))
    # Micasense path
    mica_ok = bool(mica and os.path.isfile(mica))
    # Output folder must be a valid directory
    out_ok = bool(out and os.path.isdir(out))

    return (
        not vnir_ok, vnir_ok,
        not swir_ok, swir_ok,
        not mica_ok, mica_ok,
        not out_ok, out_ok
    )

# Show/hide the step containers
@app.callback(
    Output("container-step-1", "style"),
    Output("container-step-2", "style"),
    Output("container-step-3", "style"),
    Output("container-step-4", "style"),
    Input("step", "data")
)
def update_container_visibility(step):
    return (
        {"display": "block"} if step == 1 else {"display": "none"},
        {"display": "block"} if step == 2 else {"display": "none"},
        {"display": "block"} if step == 3 else {"display": "none"},
        {"display": "block"} if step == 4 else {"display": "none"},
    )


def make_overlay(path, bands, label):
    if path and bands and len(bands)==3 and os.path.isfile(path):
        img = gi.GdalImage(path)
        arr = img.read_display_image(bands, band_last=True)
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        bounds = img.get_wgs84_bounds()
        overlay = dl.Overlay(
            dl.LayerGroup(
                dl.ImageOverlay(url=uri, bounds=bounds)
            ),
            name=label,
            checked=True
        )
        return overlay, bounds
    return None, None


@app.callback(
    Output("lc", "children"),
    Output("coreg-map", "viewport"),
    Input("vnir_path", "value"),
    Input("coreg_vnir_band_indices", "value"),
    Input("swir_path", "value"),
    Input("coreg_swir_band_indices", "value"),
    Input("mica_path", "value"),
    Input("coreg_mica_band_indices", "value"),
    prevent_initial_call=True
)
def update_coreg_layers(
    vnir_path, vnir_bands,
    swir_path, swir_bands,
    mica_path, mica_bands
):
    # Base layer
    base = dl.BaseLayer(
        dl.TileLayer(
            url="http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}",
            maxZoom=64
        ),
        name="Google Satellite",
        checked=True
    )
    children = [base]
    all_bounds = []

    # VNIR
    ov, b = make_overlay(vnir_path, vnir_bands, "VNIR Image")
    if ov:
        children.append(ov); all_bounds.append(b)

    # SWIR
    ov, b = make_overlay(swir_path, swir_bands, "SWIR Image")
    if ov:
        children.append(ov); all_bounds.append(b)

    # Micasense
    ov, b = make_overlay(mica_path, mica_bands, "Micasense Image")
    if ov:
        children.append(ov); all_bounds.append(b)

    return children, dict(bounds = b)



@app.callback(
    Output("ss_vnir_mica_band_index", "options"),
    Output("ss_swir_mica_band_index", "options"),
    Input("mica_path", "value"),
    prevent_initial_call=True
)
def update_shift_mica(path):
    if not path or not os.path.isfile(path):
        return [], []
    img = gi.GdalImage(path)
    opts = [{"label": desc or f"Band {i+1}", "value": i}
            for i, desc in enumerate(img.band_description)]
    return opts, opts

@app.callback(
    Output("kernel-size", "value"),
    Input("kernel-size", "value"),
    prevent_initial_call=True
)
def enforce_odd_kernel(val):
    if val is None:
        return val
    if val % 2 == 0:
        return val + 1
    return val

# --- Elastic registeration overlap-band dropdown callbacks ---
@app.callback(
    Output("ereg_vnir_overlap_band_indices", "options"),
    Output("ereg_vnir_overlap_band_indices", "value"),
    Input("vnir_path", "value"),
    Input("ereg_vnir_overlap_band_indices", "value"),
    prevent_initial_call=True
)
def update_ereg_vnir(path, selected):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    opts = []
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [{"label": desc or f"Band {i+1}", "value": i}
                    for i, desc in enumerate(img.band_description)]
        except Exception:
            opts = []
    if trigger == "vnir_path":
        default = [0, 1] if len(opts) >= 2 else list(range(len(opts)))
        return opts, default
    elif trigger == "ereg_vnir_overlap_band_indices":
        sel = selected or []
        if len(sel) > 2:
            sel = sel[:2]
        return opts, sel
    return opts, selected or []

# --- Shape-shifter band range dropdowns ---
@app.callback(
    Output("ss_vnir_band_range", "options"),
    Output("ss_vnir_band_range", "value"),
    Input("vnir_path", "value"),
    Input("ss_vnir_band_range", "value"),
    prevent_initial_call=True
)
def update_ss_vnir_range(path, selected):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    opts = []
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [{"label": desc or f"Band {i+1}", "value": i}
                    for i, desc in enumerate(img.band_description)]
        except Exception:
            opts = []
    if trigger == "vnir_path":
        default = [0, 1] if len(opts) >= 2 else list(range(len(opts)))
        return opts, default
    elif trigger == "ss_vnir_band_range":
        sel = selected or []
        if len(sel) > 2:
            sel = sel[:2]
        return opts, sel
    return opts, selected or []

@app.callback(
    Output("ss_swir_band_range", "options"),
    Output("ss_swir_band_range", "value"),
    Input("swir_path", "value"),
    Input("ss_swir_band_range", "value"),
    prevent_initial_call=True
)
def update_ss_swir_range(path, selected):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    opts = []
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [{"label": desc or f"Band {i+1}", "value": i}
                    for i, desc in enumerate(img.band_description)]
        except Exception:
            opts = []
    if trigger == "swir_path":
        default = [0, 1] if len(opts) >= 2 else list(range(len(opts)))
        return opts, default
    elif trigger == "ss_swir_band_range":
        sel = selected or []
        if len(sel) > 2:
            sel = sel[:2]
        return opts, sel
    return opts, selected or []

@app.callback(
    Output("ereg_swir_overlap_band_indices", "options"),
    Output("ereg_swir_overlap_band_indices", "value"),
    Input("swir_path", "value"),
    Input("ereg_swir_overlap_band_indices", "value"),
    prevent_initial_call=True
)
def update_ereg_swir(path, selected):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    opts = []
    if path and os.path.isfile(path):
        try:
            img = gi.GdalImage(path)
            opts = [{"label": desc or f"Band {i+1}", "value": i}
                    for i, desc in enumerate(img.band_description)]
        except Exception:
            opts = []
    if trigger == "swir_path":
        default = [0, 1] if len(opts) >= 2 else list(range(len(opts)))
        return opts, default
    elif trigger == "ereg_swir_overlap_band_indices":
        sel = selected or []
        if len(sel) > 2:
            sel = sel[:2]
        return opts, sel
    return opts, selected or []


# --- Collect-and-shutdown callback ---
@app.callback(
    Output("run-output", "children"),
    Output("spinner-ui", "style"),  # ← NEW
    Input("btn-finish", "n_clicks"),
    State("vnir_path", "value"),
    State("swir_path", "value"),
    State("mica_path", "value"),
    State("mica_dsm_path", "value"),
    State("out_folder", "value"),
    State("num_threads", "value"),
    State("save_to_memory", "value"),
    State("use_available_homography", "value"),
    State("coreg_vnir_band_indices", "value"),
    State("coreg_swir_band_indices", "value"),
    State("coreg_mica_band_indices", "value"),
    State("pixel-shift", "value"),
    State("kernel-size", "value"),
    State("ss_vnir_band_range", "value"),
    State("ss_swir_band_range", "value"),
    State("ss_vnir_mica_band_index", "value"),
    State("ss_swir_mica_band_index", "value"),
    State("elastic-kernel", "value"),
    State("elastic-sigma-color", "value"),
    State("elastic-sigma-space", "value"),
    State("elastic-steps", "value"),
    State("elastic-flow-sigma", "value"),
    State("ereg_vnir_overlap_band_indices", "value"),
    State("ereg_swir_overlap_band_indices", "value"),
    prevent_initial_call=True
)
def on_run(n_clicks,
           vnir_path, swir_path, mica_path, mica_dsm_path,
           out_folder, num_threads, save_to_memory,
           use_available_homography, coreg_vnir, coreg_swir, coreg_mica,
           pixel_shift, kernel_size, ss_vnir_range, ss_swir_range,
           ss_vnir_mica, ss_swir_mica,
           elastic_kernel, elastic_sigma_color, elastic_sigma_space,
           elastic_steps, elastic_flow_sigma, ereg_vnir, ereg_swir):
    spinner_style = {"display": "none"}
    if n_clicks:
        results = {
            "vnir_path": vnir_path,
            "swir_path": swir_path,
            "mica_path": mica_path,
            "mica_dsm_path": mica_dsm_path,
            "out_folder": out_folder,
            "num_threads": num_threads,
            "save_to_memory": bool(save_to_memory),
            "use_available_homography": bool(use_available_homography),
            "coreg_vnir_band_indices": coreg_vnir,
            "coreg_swir_band_indices": coreg_swir,
            "coreg_mica_band_indices": coreg_mica,
            "pixel_shift": pixel_shift,
            "kernel_size": kernel_size,
            "ss_vnir_band_indices": ss_vnir_range,
            "ss_swir_band_indices": ss_swir_range,
            "ss_vnir_mica_band_index": ss_vnir_mica,
            "ss_swir_mica_band_index": ss_swir_mica,
            "elastic_kernel": elastic_kernel,
            "elastic_sigma_color": elastic_sigma_color,
            "elastic_sigma_space": elastic_sigma_space,
            "elastic_steps": elastic_steps,
            "elastic_flow_sigma": elastic_flow_sigma,
            "ereg_vnir_overlap_band_indices": ereg_vnir,
            "ereg_swir_overlap_band_indices": ereg_swir
        }

        spinner_style = {"display": "block"}
        results_queue.put(results)

        import threading, os, time
        def delayed_exit():
            time.sleep(2)  # 2-second grace
            os._exit(0)
        threading.Thread(target=delayed_exit, daemon=True).start()

    return "", spinner_style


# --- Function to run the splash app with a queue and port ---
def wait_then_open(port):
    import time, webbrowser
    time.sleep(1)
    webbrowser.open(f"http://127.0.0.1:{port}")

def run_splash_app(queue, port=8050):
    global results_queue
    results_queue = queue
    # open browser after server starts
    threading.Thread(target=lambda: wait_then_open(port), daemon=True).start()
    app.run(debug=False, port=port)

if __name__ == "__main__":
    # Run the app directly
    # This is useful for testing the app without multiprocessing
    # In production, you would use the run_splash_app function
    app.run(debug=False, port=8050)