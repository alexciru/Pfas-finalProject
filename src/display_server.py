import numpy as np
import plotly.graph_objects as go
import cv2

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import json
from utils.utils import ROOT_DIR

DATA_PATH = ROOT_DIR / "src/analysis_results.json"

app = dash.Dash(__name__)

app.layout = html.Div(
    html.Div(
        [
            html.H4("Live Feed"),
            html.Div(id="live-update-text"),
            dcc.Graph(
                id="live-update-graph",
                style={"width": "80vh", "height": "80vh", "textAlign": "center"},
            ),
            dcc.Interval(
                id="interval-component",
                interval=1 * 1000,  # in milliseconds
                n_intervals=0,
            ),
        ],
        style={"textAlign": "center"},
    )
)


@app.callback(
    Output("live-update-graph", "figure"), Input("interval-component", "n_intervals")
)
def update_graph_live(n):
    # read newly written data as dict obj and update in web
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    fig = get_3d_plot(data)
    return fig


def get_3d_plot(data):
    MAX_FRAMES = 200
    if data == {}:
        expected_x = np.array([])
        predicted_x = np.array([])
        expected_y = np.array([])
        predicted_y = np.array([])
    else:
        expected_x = data["expected_x"]
        predicted_x = data["predicted_x"]
        expected_y = data["expected_y"]
        predicted_y = data["predicted_y"]

    time = [i for i in range(0, len(expected_x))]
    marker_data_pred = go.Scatter3d(
        x=time,
        y=predicted_x,
        z=predicted_y,
        marker=go.scatter3d.Marker(size=2),
        mode="markers",
        marker_color="red",
    )
    marker_data_expected = go.Scatter3d(
        x=time,
        y=expected_x,
        z=expected_y,
        marker=go.scatter3d.Marker(size=2),
        mode="markers",
        marker_color="blue",
    )
    marker_data_pred.name = "predicted"
    marker_data_expected.name = "expected"
    fig = go.Figure(data=[marker_data_pred, marker_data_expected])
    val_range = 4
    fig.update_layout(
        scene=dict(
            xaxis_title="time",
            yaxis_title="x",
            zaxis_title="y",
            xaxis=dict(range=[0, MAX_FRAMES]),
            yaxis=dict(range=[-val_range, val_range]),
            zaxis=dict(range=[-val_range, val_range]),
        )
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
