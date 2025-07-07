import numpy as np
from utils.load_data import load_scaler, load_signals, load_autoencoder
import utils.constants as cons
from utils.data_visualization import graph_signals
import matplotlib.pyplot as plt

import torch

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ---------------------------
# Data parameters
# ---------------------------
type_signal = 'bipolar'
labels = None
subsampling = 2
window_size = 500
window_overlap = 0
seed = 42
batch_size = 64

# ---------------------------
# Model parameters
# ---------------------------
layer_channels = [64, 64]
kernel_size = 3
s = 1
padding = 1
pooling = True
norm = True
drop = 0.25
latent_dim = 16

# Load scaler, signals and model
scaler = load_scaler(type_signal, layer_channels, kernel_size, s, padding, pooling, norm, drop, latent_dim)
_, _, x_test = load_signals(type_signal, seed, window_size, window_overlap, subsampling, subsets=['test'])
num_signals = len(x_test)

model = load_autoencoder(type_signal, window_size, subsampling, layer_channels, kernel_size, s, padding, pooling,
                         norm, drop, latent_dim)
device = next(model.parameters()).device

# Init app layout
app = dash.Dash(__name__)
app.title = "Interactive Graph"


# Create one vertical slider per latent dimension
def build_sliders(lat_dim):
    sliders = []
    for dim in range(lat_dim):
        sliders.append(
            html.Div(style=cons.styles['slider-container'], children=[
                html.Label(f"Dim {dim}", style=cons.styles['slider-label']),
                dcc.Slider(
                    id=f'dim_{dim}',
                    min=-1, max=1, step=0.001, value=0,
                    marks={-1: '-1', 0: '0', 1: '1'},
                    vertical=True,
                    tooltip={"placement": "right", "always_visible": True},
                    updatemode='drag',
                    included=False
                )
            ])
        )
    return sliders


# App layout: title, controls, graph, and sliders
app.layout = html.Div(style=cons.styles['container'], children=[
    html.H1("📊 EGM embedding interpretation", style=cons.styles['title']),
    html.Div(style=cons.styles['body'], children=[

        # Input for signal index
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '10px'}, children=[
            html.Label("Signal index:", style={'fontSize': '1rem', 'fontWeight': 'bold', 'color': '#333'}),
            dcc.Input(
                id="signal_index", type="number", value=625,
                min=0, max=num_signals, step=1,
                style={'width': '80px', 'textAlign': 'center'}
            )
        ]),

        # Checklist for signal types to display
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '10px'}, children=[
            html.Label("Type signals:", style={'fontSize': '1rem', 'fontWeight': 'bold', 'color': '#333'}),
            dcc.Checklist(
                ['Original', 'Reconstruction', 'Modified'],
                ['Original', 'Reconstruction', 'Modified'],
                inline=True,
                id='type_signals',
            )
        ]),

        # Plot container
        html.Div(style=cons.styles['graficas'], children=[
            dcc.Graph(id='egm_graph', style={'height': '100%'})
        ]),

        # Latent space sliders
        html.Div(style=cons.styles['controls'], children=build_sliders(latent_dim))
    ])
])


# Reset sliders when signal changes
@app.callback(
    [Output(f'dim_{dim}', 'value') for dim in range(latent_dim)],
    [Output(f'dim_{dim}', 'marks') for dim in range(latent_dim)],
    Input('signal_index', 'value'),
)
def rest_state(index):
    # Get signal and generate embedding
    signal = torch.Tensor(x_test[index]).squeeze().unsqueeze(0).unsqueeze(0).to(device)
    embedding = model(signal)[0].detach().cpu().numpy()

    # Normalize embedding and build slider marks
    norm_embedding = scaler.transform(embedding).tolist()[0]
    marks = [{-1: '-1', 0: '0', 1: '1', i: f'o'} for i in norm_embedding]
    all_outputs = norm_embedding + marks
    return all_outputs


# Update graph based on signal and slider state
@app.callback(
    Output('egm_graph', 'figure'),
    Input('signal_index', 'value'),
    Input('type_signals', 'value'),
    [Input(f'dim_{dim}', 'value') for dim in range(latent_dim)],
)
def update_graph(index, type_signals, *inputs):
    # Get original signal and forward through model
    original = x_test[index].squeeze()
    signal = torch.Tensor(original).squeeze().unsqueeze(0).unsqueeze(0).to(device)
    og_encoded, og_decoded, pool_indices = model(signal)
    og_decoded = og_decoded.squeeze().detach().cpu().numpy()

    # Replace embedding with slider values and decode
    embedding_sliders = torch.Tensor(scaler.inverse_transform(np.array([inputs]))).to(device)
    x = model.fc2(embedding_sliders)
    x = x.view(x.size(0), *model.spatial_dims)
    for i, (layer, unpool) in enumerate(zip(model.decoder, model.unpool_layers)):
        if unpool:
            x = unpool(x, pool_indices[::-1][i], output_size=model.intermediate_shapes[-2 - 2 * i])
        x = layer(x)
    decoded = x.squeeze().detach().cpu().numpy()

    # Create the figure
    fig = go.Figure()
    fig.update_layout(
        title="EGM signal: orignal vs reconstruction",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333', size=14),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Add original, reconstruction, and modified signals to plot
    if 'Original' in type_signals:
        fig = graph_signals(fig, original, cons.offset, cons.colors, 'Originals')
    if 'Reconstruction' in type_signals:
        fig = graph_signals(fig, og_decoded, cons.offset, cons.colors, 'Reconstructed')
    if 'Modified' in type_signals:
        fig = graph_signals(fig, decoded, cons.offset, cons.colors, 'Modified')

    return fig


# Update graph based on signal and slider state
@app.callback(
    [Input(f'dim_{dim}', 'value') for dim in range(latent_dim)],
)
def update_bar_chart(*inputs):
    # X-axis labels
    x_labels = [f'{i}' for i in range(len(inputs))]


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
