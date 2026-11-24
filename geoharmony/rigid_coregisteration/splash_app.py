"""Local settings interface for running SPLASH on a new scene."""
from pathlib import Path
import threading

from dash import Dash, Input, Output, State, dcc, html

from geoharmony.splash.config import SplashConfig


LABELS = {
    "vnir_path": "VNIR raster (including acquisition-row band)",
    "swir_path": "SWIR raster (including acquisition-row band)",
    "mica_path": "MSI reference raster",
    "mica_dsm_path": "Optional MSI DSM (heights in metres)",
    "out_folder": "Output directory (separate from inputs)",
    "coreg_vnir_band_indices": "VNIR display channels (choose 3)",
    "coreg_swir_band_indices": "SWIR display channels (choose 3)",
    "coreg_mica_band_indices": "MSI display channels (choose 3)",
    "ss_vnir_band_range": "VNIR scoring range (start, end)",
    "ss_swir_band_range": "SWIR scoring range (start, end)",
    "ss_vnir_mica_band_index": "MSI band for VNIR scoring",
    "ss_swir_mica_band_index": "MSI band for SWIR scoring",
    "pixel_shift_vnir": "Maximum VNIR displacement (pixels)",
    "pixel_shift_swir": "Maximum SWIR displacement (pixels)",
    "kernel_size": "Spatial median filter width (odd pixels; 1 disables)",
    "ereg_vnir_band_range": "VNIR spectral-overlap range (start, end)",
    "ereg_swir_band_range": "SWIR spectral-overlap range (start, end)",
    "elastic_kernel": "Bilateral filter diameter (pixels)",
    "elastic_sigma_color": "Bilateral intensity sigma (0–255 image scale)",
    "elastic_sigma_space": "Bilateral spatial sigma (pixels)",
    "elastic_flow_sigma": "SyN update-field smoothing sigma (pixel-grid units)",
    "num_threads": "ShapeShifter worker processes",
    "elastic_workers": "Elastic band-application workers",
    "itk_threads": "ANTs registration threads",
    "random_seed": "Random seed",
    "save_to_memory": "Keep intermediate rasters in RAM (requires more memory)",
    "use_available_homography": "Reuse matching saved manual homographies",
}

SECTIONS = [
    ("1. Inputs and outputs", "Select local raster files on the computer running SPLASH. ENVI inputs need their binary and matching .hdr. CRS and acquisition-row IDs must already be present.",
     ["vnir_path", "swir_path", "mica_path", "mica_dsm_path", "out_folder"]),
    ("2. Manual coregistration", "Display channels only control point-picking previews. Select corresponding stable features across the scene. New scenes require manual VNIR-to-MSI and SWIR-to-MSI registration.",
     ["coreg_vnir_band_indices", "coreg_swir_band_indices", "coreg_mica_band_indices", "use_available_homography"]),
    ("3. ShapeShifter", "Select a spectrally comparable MSI band and hyperspectral range for each sensor. Range endpoints are inclusive. Choose these from the sensor wavelengths; defaults are placeholders.",
     ["ss_vnir_band_range", "ss_vnir_mica_band_index", "ss_swir_band_range", "ss_swir_mica_band_index", "pixel_shift_vnir", "pixel_shift_swir", "kernel_size"]),
    ("4. Elastic registration and execution", "Select VNIR/SWIR bands covering the same wavelengths. SWIR is registered to the corrected VNIR grid. More workers increase memory use. Follow progress and errors in the terminal.",
     ["ereg_vnir_band_range", "ereg_swir_band_range", "elastic_kernel", "elastic_sigma_color", "elastic_sigma_space", "elastic_flow_sigma", "num_threads", "elastic_workers", "itk_threads", "random_seed", "save_to_memory"]),
]


def make_splash_app(results_queue=None, initial=None):
    cfg = SplashConfig.from_dict(initial or {}).to_dict()
    app = Dash(__name__, assets_folder=str(Path(__file__).with_name("assets")))
    names = list(LABELS)

    def field(name):
        value = cfg[name]
        if name in ("save_to_memory", "use_available_homography"):
            control = dcc.Checklist(id=name, options=[{"label": "Enabled", "value": "on"}], value=["on"] if value else [])
        elif name.endswith("band_indices") or name.endswith("band_range") or name.endswith("mica_band_index"):
            multiple = not name.endswith("mica_band_index")
            control = dcc.Dropdown(id=name, options=[], value=list(value) if multiple else value,
                                   multi=multiple, clearable=False, placeholder="Load a raster, then choose bands")
        elif name.endswith("path") or name == "out_folder":
            control = dcc.Input(id=name, value=value or "", type="text", debounce=True, placeholder="Local path", style={"width": "100%"})
        else:
            control = dcc.Input(id=name, value=value, type="number", min=0 if name.startswith("pixel_shift") or name == "random_seed" else 1,
                                step=1 if isinstance(value, int) else "any")
        return html.Div([html.Label(LABELS[name], htmlFor=name), control], className="splash-field")

    app.layout = html.Main([
        html.Header([html.Img(src="/assets/splashlogo.png", style={"height": "64px"}),
                     html.Div([html.H1("SPLASH"), html.P("Spatial registration of VNIR and SWIR imagery with an MSI reference")])], className="splash-header"),
        html.P("Band labels show both the one-based band number and the zero-based configuration index. The final hyperspectral acquisition-row band cannot be selected as a spectral band.", className="splash-note"),
        *[html.Section([html.H2(title), html.P(description), html.Div([field(name) for name in fields], className="splash-grid")], className="splash-section") for title, description, fields in SECTIONS],
        html.Section([html.H2("Display previews"), html.Div([dcc.Graph(id=f"preview-{sensor}", config={"displayModeBar": False}) for sensor in ("vnir", "swir", "mica")], className="splash-previews")], className="splash-section"),
        html.Div([html.Button("Validate and run SPLASH", id="run", n_clicks=0), html.Button("Cancel", id="cancel", n_clicks=0, className="secondary")], className="splash-actions"),
        html.Pre(id="run-status", role="status"),
    ], className="splash-main")

    for sensor in ("vnir", "swir", "mica"):
        fields = [f"coreg_{sensor}_band_indices"]
        if sensor == "mica":
            fields += ["ss_vnir_mica_band_index", "ss_swir_mica_band_index"]
        else:
            fields += [f"ss_{sensor}_band_range", f"ereg_{sensor}_band_range"]

        def register_band_options(sensor, fields):
            @app.callback(*[Output(name, "options") for name in fields], Input(f"{sensor}_path", "value"))
            def options(path):
                from osgeo import gdal
                choices = []
                if path and Path(path).is_file():
                    try:
                        ds = gdal.Open(str(path))
                        if ds:
                            count = ds.RasterCount - (sensor != "mica")
                            choices = [{"label": f"Band {i+1} (index {i}): {ds.GetRasterBand(i+1).GetDescription() or 'unnamed'}", "value": i} for i in range(count)]
                    except RuntimeError:
                        pass
                return [choices for _ in fields]
        register_band_options(sensor, fields)

        def register_preview(sensor):
            @app.callback(Output(f"preview-{sensor}", "figure"), Input(f"{sensor}_path", "value"), Input(f"coreg_{sensor}_band_indices", "value"))
            def preview(path, bands):
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from osgeo import gdal
                if not path or not Path(path).is_file() or not bands or len(bands) != 3:
                    return go.Figure().update_layout(title=f"{sensor.upper()}: select input and 3 display bands")
                try:
                    ds = gdal.Open(str(path))
                    scale = min(1.0, 900 / max(ds.RasterXSize, ds.RasterYSize))
                    channels = []
                    for index in bands:
                        band = ds.GetRasterBand(int(index)+1)
                        values = band.ReadAsArray(buf_xsize=max(1, int(ds.RasterXSize*scale)), buf_ysize=max(1, int(ds.RasterYSize*scale))).astype(float)
                        valid = np.isfinite(values)
                        if band.GetNoDataValue() is not None:
                            valid &= values != band.GetNoDataValue()
                        low, high = np.percentile(values[valid], [2, 98]) if np.any(valid) else (0, 1)
                        image = np.zeros_like(values)
                        if high > low:
                            image[valid] = np.clip((values[valid]-low)/(high-low)*255, 0, 255)
                        channels.append(image.astype(np.uint8))
                    return px.imshow(np.stack(channels, axis=-1)).update_layout(title=sensor.upper(), margin=dict(l=0, r=0, t=30, b=0))
                except (RuntimeError, ValueError, AttributeError):
                    return go.Figure().update_layout(title=f"{sensor.upper()}: input or band selection is invalid")
        register_preview(sensor)

    @app.callback(Output("run-status", "children"), Input("run", "n_clicks"), Input("cancel", "n_clicks"),
                  *[State(name, "value") for name in names], prevent_initial_call=True)
    def run(_run, _cancel, *values):
        from dash import ctx
        if ctx.triggered_id == "cancel":
            if results_queue is not None:
                results_queue.put({"cancelled": True})
            return "Cancelled."
        try:
            data = dict(zip(names, values))
            for key in ("save_to_memory", "use_available_homography"):
                data[key] = bool(data[key])
            if not data["mica_dsm_path"]:
                data["mica_dsm_path"] = None
            result = SplashConfig.from_dict(data)
            result.validate_inputs()
            if results_queue is None:
                return "Start the complete workflow using geoharmony-splash."
            results_queue.put(result.to_dict())
            return "Inputs validated. Follow processing in the terminal; the manual registration page will open next when required."
        except (ValueError, OSError, RuntimeError) as exc:
            return str(exc)

    return app


def run_splash_app(initial, queue, port=8050, open_browser=True):
    from geoharmony.rigid_coregisteration.get_points_app import wait_then_open
    app = make_splash_app(queue, initial)
    if open_browser:
        threading.Thread(target=lambda: wait_then_open(port), daemon=True).start()
    app.run(debug=False, host="127.0.0.1", port=port, use_reloader=False)
