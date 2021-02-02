import dash_html_components as html
import dash
import pandas as pd
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import numpy as np
import math

# Data preprocessing
nominal_flights = pd.read_csv("../Data/nominal_flights.csv")
nominal_flights.Anomaly = "Nominal"

default_HS = pd.read_csv("../Data/adverse_flight_1.csv")
default_HS.Anomaly = "Adverse"

nominal_flights.index = nominal_flights["Unnamed: 0"]
default_HS.index = default_HS["Unnamed: 0"]

distance_to_event = nominal_flights[nominal_flights.flight_id==0].index.values
marks_slider = {}
# labels for every 4 unit of distance (assuming nautical miles)
labels_distance = distance_to_event[::8]
for label in labels_distance[::-1]:
    marks_slider.update({int(label): f"{label} nm"})

parameters = []
for param in nominal_flights.columns:
    tmp = {"label":param, "value":param}
    parameters.append(tmp)

# Create app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.title = "Flight Data Analysis"

## Buttons, dropdowns, etc..
fig_dropdown = dcc.Dropdown(id="dropdown_params",
                options=parameters,
                multi=True,
                placeholder="Select flight parameters",
                value= ["ALTITUDE", "COMPUTED AIRSPEED LSP"]
            )
html_dropdown_wrapper = html.Div(
    [fig_dropdown, html.Div(className="class-selected-params",
                            id="selected-params")]
)
distance_slider = dbc.FormGroup(
        [
            dbc.Label("Distance to event"),
            dcc.Slider(
                id="distance-slider",
                min= distance_to_event[-1],
                max= distance_to_event[0],
                step= 0.25,
                value= 2,
                marks=marks_slider,
                included=False
            ),
        ]
    )
distance_slider_html = html.Div(
    [distance_slider, html.Div(id="slider-output-container")]
)
update_graph_button = html.Button(html.Span("Update Graph",
                                            style={"marginRight":"10px"}),
                                            className="btn update-button",
                                            n_clicks_timestamp=0,
                                            id="update_button")

scatter_plots = dcc.Graph(id="scatter-plots", style={"height":"500px"})

# Layout
app.layout = html.Div(className="master_container",
children= [
        html.Div(className="top_banner",
        children=[html.H2("Flight Data Analysis")]),
        html.Hr(style={"height":"15px"}),
        html.Div(
            id="cmd-select",
            children=[html_dropdown_wrapper, distance_slider_html, update_graph_button]
                ),
        html.Div(
            id="div scatter-plots",
            children= [scatter_plots]
                )
            ])

# Callbacks and functions
# @app.callback(
#     dash.dependencies.Output('slider-output-container', 'children'),
#     [dash.dependencies.Input('distance-slider', 'value')])
# def update_slider(value):
#     return "Selected distance: {} nm".format(value)

# @app.callback(
#     dash.dependencies.Output('selected-param', 'children'),
#     [dash.dependencies.Input('parameters_dropdown', 'value')])
# def update_selected_param(value):
#     return sorted(value)

@app.callback(
    [Output("scatter-plots", "figure"),
    Output('selected-params', 'children'),
    Output('slider-output-container', 'children')
    ],

    [Input("update_button", "n_clicks_timestamp"),
     Input('distance-slider', 'value'),
     # Input('dropdown_params', 'value')
     ],

    [
        #State('distance-slider', 'value'),
    State('dropdown_params', 'value')]
    )

def plot_new_scatter_plots(submit_clicks, dist_val, selected_params):
    out_slider = "Selected distance: {} nm".format(dist_val)
    if submit_clicks >0:
        df = pd.concat([nominal_flights.loc[dist_val,:],
                        default_HS.loc[[dist_val]]])

        fig = px.scatter_matrix(df, dimensions=selected_params,
                                color="Anomaly",
                                opacity=0.6,
                                # height=min(len(selected_params)*5, 10)
                                )
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(font=dict(size=7))
        # fig.update_layout({"font":{"size":5}}) --> Causing bugs?
        return fig, selected_params, out_slider
    else:
        return go.Figure(), selected_params, out_slider


if __name__ == '__main__':
    app.run_server(debug=True)