import numpy as np
import plotly.graph_objs as go


def graph_signals(figure, signals, offset, colors, label):
    """
    Adds a group of stacked signals to a Plotly figure, each shifted vertically by a given offset.

    Parameters:
    - figure: a Plotly figure object (go.Figure) to which the signals will be added.
    - signals (np.ndarray): 2D array of signals [channels, time].
    - offset (float): vertical offset between each channel.
    - colors (dict): dictionary mapping labels to color codes.
    - label (str): label to use for the legend and color.

    Returns:
    - figure: the updated Plotly figure with signal traces added.
    """

    # Time axis from 0 to signal length
    t = np.arange(0, signals.shape[1])

    # Add a dummy trace to show the legend entry with the correct color
    figure.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name=label,
        legendgroup=label,
        showlegend=True,
        line=dict(color=colors[label])
    ))

    if label == 'Reconstructed':
        # Add each signal trace, stacked vertically using the offset
        for num, signal in enumerate(signals):
            figure.add_trace(go.Scatter(
                x=t,
                y=signal - num * offset,
                mode='lines',
                legendgroup=label,
                showlegend=False,
                hoverinfo='x+y',
                line=dict(color=colors[label], dash='dash')
            ))
    else:
        # Add each signal trace, stacked vertically using the offset
        for num, signal in enumerate(signals):
            figure.add_trace(go.Scatter(
                x=t,
                y=signal - num * offset,
                mode='lines',
                legendgroup=label,
                showlegend=False,
                hoverinfo='x+y',
                line=dict(color=colors[label])
            ))

    return figure
