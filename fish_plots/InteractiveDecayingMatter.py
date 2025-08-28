import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output


# Compute energy densities
def compute_energy_densities(Gamma, omega_dm0):
    one_plus_z = np.logspace(0, 8, 500)
    omega_r = 9.02e-5
    omega_lambda = 0.685
    omega_stable_m = 0.315 - omega_dm0

    # Approximate cosmic time using a proxy variable (better than 1/(1+z))
    tau = 1 / one_plus_z**1.5  # heuristic proxy for cosmic time, since t ∝ (1+z)^{-3/2} in matter-dom.

    decay_factor = np.exp(-Gamma * tau)

    rho_dm = omega_dm0 * one_plus_z**3 * decay_factor
    rho_m = omega_stable_m * one_plus_z**3
    rho_r = omega_r * one_plus_z**4 + omega_dm0 * one_plus_z**4 * (1 - decay_factor)
    rho_lambda = np.full_like(one_plus_z, omega_lambda)

    rho_total = rho_m + rho_dm + rho_r + rho_lambda
    omega_m = rho_m / rho_total
    omega_dm = rho_dm / rho_total
    omega_r = rho_r / rho_total
    omega_l = rho_lambda / rho_total

    omega_m_total = (rho_m + rho_dm) / rho_total  # Total matter (stable + decay)

    return one_plus_z, omega_m, omega_dm, omega_r, omega_l, omega_m_total


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Decaying Matter Energy Density"

app.layout = html.Div([
    html.H1("Energy Density Evolution with Decaying Matter", style={'textAlign': 'center'}),

    dcc.Graph(id='energy-density-plot'),

    html.Div([
        html.Label("Gamma (Γ):", style={'marginRight': '10px'}),
        dcc.Slider(
            id='gamma-slider',
            min=0,
            max=5,
            step=0.1,
            value=0.1,
            marks={i: str(i) for i in range(6)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '20px'}),

    html.Div([
        html.Label("Initial Decaying Matter Density (Ω_decay₀):", style={'marginRight': '10px'}),
        dcc.Slider(
            id='omega-dm-slider',
            min=0.0,
            max=0.2,
            step=0.005,
            value=0.05,
            marks={i / 20: f"{i / 20:.2f}" for i in range(0, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '20px'}),
])


@app.callback(
    Output('energy-density-plot', 'figure'),
    Input('gamma-slider', 'value'),
    Input('omega-dm-slider', 'value')
)
def update_figure(Gamma, omega_dm0):
    z, omega_m, omega_dm, omega_r, omega_l, omega_m_total = compute_energy_densities(Gamma, omega_dm0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=omega_m, mode='lines', name='Ωₘ (stable)', line=dict(color='steelblue', dash='dash')))
    fig.add_trace(go.Scatter(x=z, y=omega_dm, mode='lines', name='Ω_decay (decaying)', line=dict(color='cyan', dash='dash')))
    fig.add_trace(go.Scatter(x=z, y=omega_m_total, mode='lines', name='Ωₘ, total (stable + decay)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=z, y=omega_r, mode='lines', name='Ωᵣ (with decay)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=z, y=omega_l, mode='lines', name='ΩΛ', line=dict(color='gold')))

    fig.update_layout(
        xaxis=dict(
            title='1 + z',
            type='log',
            range=[0, 8],
            tickvals=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
            ticktext=["10⁰", "10¹", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷", "10⁸"],
        ),
        yaxis=dict(title='Fraction of Total Energy Density', range=[0, 1]),
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
        margin=dict(l=60, r=60, t=50, b=50),
        height=600,
        template='plotly_white'
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)
