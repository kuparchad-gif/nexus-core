import plotly.graph_objects as go
import pandas as pd

def equity_curve_figure(equity: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Value", template="plotly_dark")
    return fig

def drawdown_figure(drawdown: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines", name="Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", template="plotly_dark")
    return fig

def weights_heatmap(weights: pd.DataFrame):
    fig = go.Figure(data=go.Heatmap(z=weights.T.values, x=weights.index, y=weights.columns, colorbar=dict(title="Weight")))
    fig.update_layout(title="Weights Heatmap", xaxis_title="Date", yaxis_title="Symbol", template="plotly_dark")
    return fig
