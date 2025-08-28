import numpy as np
from scipy.integrate import solve_ivp
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Constants
H0 = 70                         # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)   # in 1/s

# Default parameters
omega_m0 = 0.315
omega_r0 = 9.02e-5
omega_lambda0 = 1 - omega_m0 - omega_r0


# Core calculation
def compute_Hubble(Gamma, omega_ddm0):
    z_i = 1e4
    z_eval = np.logspace(np.log10(z_i), 0, 500)
    a_eval = 1 / (1 + z_eval)

    rho_m0 = omega_m0 - omega_ddm0
    rho_dm0 = omega_ddm0

    y0 = [rho_dm0 * (1 + z_i) ** 3, omega_r0 * (1 + z_i) ** 4]

    def d_rho_dz(z, y):
        rho_x, rho_r = y
        rho_m = rho_m0 * (1 + z) ** 3
        rho_lambda = omega_lambda0
        rho_total = rho_m + rho_x + rho_r + rho_lambda
        H = H0_s * np.sqrt(max(rho_total, 1e-30))
        drho_x_dz = (3 * rho_x + (Gamma / H) * rho_x) / (1 + z)
        drho_r_dz = (4 * rho_r - (Gamma / H) * rho_x) / (1 + z)
        return [drho_x_dz, drho_r_dz]

    sol = solve_ivp(
        d_rho_dz, (z_i, 1), y0, t_eval=z_eval,
        method='BDF', rtol=1e-8, atol=1e-10
    )

    z = sol.t
    a_vals = 1 / (1 + z)
    rho_x = np.maximum(sol.y[0], 1e-30)
    rho_r = np.maximum(sol.y[1], 1e-30)
    rho_m = rho_m0 * a_vals ** -3

    # Clamp H to avoid log(0)
    H_decay = np.sqrt(np.maximum(rho_m + rho_x + rho_r + omega_lambda0, 1e-30)) * H0
    H_lcdm = np.sqrt(omega_m0 * a_vals ** -3 + omega_r0 * a_vals ** -4 + omega_lambda0) * H0

    return 1 + z, H_decay, H_lcdm


# Initialize app
app = dash.Dash(__name__)
app.title = "Hubble Parameter Interactive"

# Layout
app.layout = html.Div([
    html.H2("H(z) with Decaying Dark Matter", style={'textAlign': 'center'}),

    dcc.Graph(id='hubble-plot'),

    html.Div([
        html.Label("log₁₀(Γ) [log₁₀(1/s)]"),
        dcc.Slider(
            id='loggamma-slider',
            min=-18,
            max=-14,
            step=0.1,
            value=-16,
            marks={i: f"{i}" for i in range(-18, -13)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '30px'}),

    html.Div([
        html.Label("Ω_decay₀:"),
        dcc.Slider(
            id='omega-ddm-slider',
            min=0.0,
            max=0.2,
            step=0.01,
            value=0.05,
            marks={i / 20: f"{i / 20:.2f}" for i in range(5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '30px'}),
])


# Callback
@app.callback(
    Output('hubble-plot', 'figure'),
    Input('loggamma-slider', 'value'),
    Input('omega-ddm-slider', 'value')
)
def update_plot(log10_gamma, omega_ddm0):
    Gamma = 10 ** log10_gamma
    z_vals, H_decay, H_lcdm = compute_Hubble(Gamma, omega_ddm0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_vals, y=H_lcdm,
        mode='lines', name='ΛCDM',
        line=dict(color='black', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=z_vals, y=H_decay,
        mode='lines', name='Decaying DM',
        line=dict(color='blue')
    ))

    fig.update_layout(
        xaxis=dict(
            title='1 + z',
            type='log',
            tickvals=[1e0, 1e1, 1e2, 1e3, 1e4],
            ticktext=["10⁰", "10¹", "10²", "10³", "10⁴"]
        ),
        yaxis=dict(
            title='H(z) [km/s/Mpc]',
            type='log',
            tickvals=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
            ticktext=["10⁰", "10¹", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷", "10⁸"]
        ),
        height=600,
        template='plotly_white',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=60, t=50, b=50)
    )

    return fig


# Run server
if __name__ == '__main__':
    app.run(debug=True)
