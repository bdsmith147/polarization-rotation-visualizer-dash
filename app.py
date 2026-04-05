# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Dash app wiring for PolariViz.
#
# THIS FILE IS CONVERSION SCAFFOLDING.
# It exists to let you test physics.py and traces.py interactively in Python.
# The final deployed product is a static Plotly.js site on GitHub Pages.
# When converting:
#   - app.layout   → HTML structure in index.html
#   - @callback    → addEventListener + updateAll() function in app.js
#   - dcc.Slider   → <input type="range"> in index.html
#   - dcc.Tabs     → <div class="tab"> structure in index.html
#   - dcc.Checklist→ <input type="checkbox"> in index.html
#
# STRUCTURE:
#   1. Imports
#   2. Slider/control definitions  (reusable specs)
#   3. Layout
#   4. Callback
#   5. Run
# ─────────────────────────────────────────────────────────────────────────────

from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from math_helpers import degrees_to_radians
from physics import compute_all
from traces import (
    make_3d_figure,
    make_level_figure,
    make_density_figure,
    make_ellipse_figure,
    make_poincare_figure,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. APP INIT
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(__name__, title='PolariViz')


# ══════════════════════════════════════════════════════════════════════════════
# 2. SLIDER / CONTROL SPECS
# ══════════════════════════════════════════════════════════════════════════════
# Each slider is defined as a dict so the spec lives in one place.
# JS conversion: each entry maps to one <input type="range"> element.

SLIDERS = {
    # Beam direction (ZYZ Euler angles)
    'theta':  dict(min=0,   max=180, step=1, value=45,  label='θ (polar)'),
    'phi':    dict(min=0,   max=360, step=1, value=0,   label='φ (azimuthal)'),
    'chi':    dict(min=0,   max=360, step=1, value=0,   label='χ (roll)'),
    # Quantization axis
    'theta_b': dict(min=0,  max=180, step=1, value=0,   label='θ_B (polar)'),
    'phi_b':   dict(min=0,  max=360, step=1, value=0,   label='φ_B (azimuthal)'),
    # Waveplate chain
    'alpha1': dict(min=0,   max=180, step=1, value=0,   label='α₁  QWP₁ fast axis'),
    'alpha2': dict(min=0,   max=180, step=1, value=0,   label='α₂  HWP  fast axis'),
    'alpha3': dict(min=0,   max=180, step=1, value=0,   label='α₃  QWP₂ fast axis'),
}


def _slider_block(slider_id, spec):
    """Return a labelled slider as a Div — label + dcc.Slider + value display.

    JS conversion:
        <div class="slider-block">
            <label>{spec['label']}: <span id="{slider_id}-val">0</span>°</label>
            <input type="range" id="{slider_id}" min=... max=... step=... value=...>
        </div>
    """
    return html.Div([
        html.Div([
            html.Span(spec['label'] + ': ',
                      style={'color': '#B0B0C0', 'fontSize': '12px'}),
            html.Span(id=f'{slider_id}-val',
                      style={'color': '#E0E0E0', 'fontSize': '12px',
                             'fontWeight': 'bold'}),
            html.Span('°', style={'color': '#B0B0C0', 'fontSize': '12px'}),
        ], style={'marginBottom': '2px'}),
        dcc.Slider(
            id=f'slider-{slider_id}',
            min=spec['min'],
            max=spec['max'],
            step=spec['step'],
            value=spec['value'],
            marks=None,
            tooltip={'placement': 'bottom', 'always_visible': False},
            updatemode='drag',
        ),
    ], style={'marginBottom': '12px'})


# ══════════════════════════════════════════════════════════════════════════════
# 3. LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared styles ─────────────────────────────────────────────────────────────
STYLE_PAGE = {
    'backgroundColor': '#1A1A2E',
    'color': '#E0E0E0',
    'fontFamily': 'Inter, Segoe UI, sans-serif',
    'padding': '12px',
    'minHeight': '100vh',
}

STYLE_ROW = {
    'display': 'flex',
    'flexDirection': 'row',
    'gap': '10px',
    'marginBottom': '10px',
}

STYLE_PANEL = {
    'backgroundColor': '#16213E',
    'borderRadius': '8px',
    'padding': '12px',
    'border': '1px solid #2A2A4A',
}

STYLE_SECTION_LABEL = {
    'color': '#7EC8E3',
    'fontSize': '11px',
    'fontWeight': 'bold',
    'letterSpacing': '0.08em',
    'textTransform': 'uppercase',
    'marginBottom': '6px',
    'marginTop': '10px',
}

TAB_STYLE = {
    'backgroundColor': '#1A1A2E',
    'color': '#B0B0C0',
    'border': '1px solid #2A2A4A',
    'borderRadius': '4px 4px 0 0',
    'padding': '6px 12px',
    'fontSize': '12px',
}

TAB_SELECTED_STYLE = {
    'backgroundColor': '#16213E',
    'color': '#E0E0E0',
    'border': '1px solid #457B9D',
    'borderBottom': '1px solid #16213E',
    'borderRadius': '4px 4px 0 0',
    'padding': '6px 12px',
    'fontSize': '12px',
}

# ── Controls panel ────────────────────────────────────────────────────────────
controls_panel = html.Div([

    html.H3('PolariViz', style={
        'color': '#7EC8E3', 'fontSize': '16px',
        'margin': '0 0 12px 0', 'letterSpacing': '0.1em',
    }),

    # ── Beam direction ────────────────────────────────────────────────────────
    html.Div('Beam Direction', style=STYLE_SECTION_LABEL),
    _slider_block('theta',  SLIDERS['theta']),
    _slider_block('phi',    SLIDERS['phi']),
    _slider_block('chi',    SLIDERS['chi']),

    # ── Quantization axis ─────────────────────────────────────────────────────
    html.Div('Quantization Axis', style=STYLE_SECTION_LABEL),
    _slider_block('theta_b', SLIDERS['theta_b']),
    _slider_block('phi_b',   SLIDERS['phi_b']),

    # ── Polarization input (tabbed) ───────────────────────────────────────────
    html.Div('Polarization Input', style=STYLE_SECTION_LABEL),
    dcc.Tabs(
        id='pol-tabs',
        value='basis',
        children=[
            dcc.Tab(
                label='Basis States',
                value='basis',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[
                    html.Div([
                        dcc.RadioItems(
                            id='basis-radio',
                            options=[
                                {'label': ' σ+', 'value': 'sigma_plus'},
                                {'label': ' π',  'value': 'pi'},
                                {'label': ' σ−', 'value': 'sigma_minus'},
                            ],
                            value='sigma_plus',
                            labelStyle={
                                'display': 'inline-block',
                                'marginRight': '14px',
                                'color': '#E0E0E0',
                                'fontSize': '13px',
                                'cursor': 'pointer',
                            },
                            style={'marginTop': '10px'},
                        ),
                    ]),
                ],
            ),
            dcc.Tab(
                label='Waveplate Chain',
                value='waveplate',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[
                    html.Div([
                        html.Div(
                            '↓ vertical  →  QWP₁  →  HWP  →  QWP₂',
                            style={'color': '#7EC8E3', 'fontSize': '11px',
                                   'marginTop': '8px', 'marginBottom': '4px'},
                        ),
                        _slider_block('alpha1', SLIDERS['alpha1']),
                        _slider_block('alpha2', SLIDERS['alpha2']),
                        _slider_block('alpha3', SLIDERS['alpha3']),
                    ]),
                ],
            ),
        ],
        style={'marginBottom': '0px'},
    ),

    # ── Visibility checkboxes ─────────────────────────────────────────────────
    html.Div('Visibility', style=STYLE_SECTION_LABEL),
    dcc.Checklist(
        id='visibility-checks',
        options=[
            {'label': ' Show polarization ellipse', 'value': 'ellipse'},
            {'label': ' Show ê₁, ê₂ axes',          'value': 'eaxes'},
        ],
        value=['ellipse', 'eaxes'],
        labelStyle={
            'display': 'block',
            'color': '#E0E0E0',
            'fontSize': '12px',
            'marginBottom': '6px',
            'cursor': 'pointer',
        },
    ),

], style={**STYLE_PANEL, 'width': '38%', 'boxSizing': 'border-box',
          'overflowY': 'auto', 'maxHeight': '62vh'})

# ── 3D scene ──────────────────────────────────────────────────────────────────
scene_panel = html.Div([
    dcc.Graph(
        id='plot-3d',
        style={'height': '60vh'},
        config={'displayModeBar': True, 'scrollZoom': True},
    ),
], style={**STYLE_PANEL, 'width': '60%', 'boxSizing': 'border-box',
          'padding': '6px'})

# ── Bottom row: level diagram ─────────────────────────────────────────────────
level_panel = html.Div([
    dcc.Graph(
        id='plot-level',
        style={'height': '35vh'},
        config={'displayModeBar': False},
    ),
], style={**STYLE_PANEL, 'width': '30%', 'boxSizing': 'border-box',
          'padding': '6px'})

# ── Bottom row: density matrix ────────────────────────────────────────────────
density_panel = html.Div([
    dcc.Graph(
        id='plot-density',
        style={'height': '35vh'},
        config={'displayModeBar': False},
    ),
], style={**STYLE_PANEL, 'width': '30%', 'boxSizing': 'border-box',
          'padding': '6px'})

# ── Bottom row: tabbed panel (ellipse / Poincaré) ─────────────────────────────
tabbed_panel = html.Div([
    dcc.Tabs(
        id='plot-tabs',
        value='ellipse',
        children=[
            dcc.Tab(
                label='Polarization Ellipse',
                value='ellipse',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[dcc.Graph(
                    id='plot-ellipse',
                    style={'height': '32vh'},
                    config={'displayModeBar': False},
                )],
            ),
            dcc.Tab(
                label='Poincaré Sphere',
                value='poincare',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[dcc.Graph(
                    id='plot-poincare',
                    style={'height': '32vh'},
                    config={'displayModeBar': False},
                )],
            ),
        ],
    ),
], style={**STYLE_PANEL, 'width': '38%', 'boxSizing': 'border-box',
          'padding': '6px'})

# ── Full layout ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    # Row 1: 3D scene + controls
    html.Div([
        scene_panel,
        controls_panel,
    ], style=STYLE_ROW),

    # Row 2: level diagram + density matrix + tabbed plots
    html.Div([
        level_panel,
        density_panel,
        tabbed_panel,
    ], style=STYLE_ROW),

], style=STYLE_PAGE)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CALLBACK
# ══════════════════════════════════════════════════════════════════════════════
# Single callback: all inputs → all outputs.
# JS conversion: this entire function becomes one updateAll() in app.js,
# called by addEventListener on every slider/radio/checkbox change.

@callback(
    # ── Outputs: all 5 figures ────────────────────────────────────────────────
    Output('plot-3d',      'figure'),
    Output('plot-level',   'figure'),
    Output('plot-density', 'figure'),
    Output('plot-ellipse', 'figure'),
    Output('plot-poincare','figure'),
    # ── Slider value displays ─────────────────────────────────────────────────
    Output('theta-val',   'children'),
    Output('phi-val',     'children'),
    Output('chi-val',     'children'),
    Output('theta_b-val', 'children'),
    Output('phi_b-val',   'children'),
    Output('alpha1-val',  'children'),
    Output('alpha2-val',  'children'),
    Output('alpha3-val',  'children'),
    # ── Inputs: beam sliders ──────────────────────────────────────────────────
    Input('slider-theta',   'value'),
    Input('slider-phi',     'value'),
    Input('slider-chi',     'value'),
    # Quantization axis
    Input('slider-theta_b', 'value'),
    Input('slider-phi_b',   'value'),
    # Polarization mode
    Input('pol-tabs',       'value'),
    Input('basis-radio',    'value'),
    # Waveplate sliders
    Input('slider-alpha1',  'value'),
    Input('slider-alpha2',  'value'),
    Input('slider-alpha3',  'value'),
    # Visibility
    Input('visibility-checks', 'value'),
)
def update_all(
    theta, phi, chi,
    theta_b, phi_b,
    pol_mode, basis_state,
    alpha1, alpha2, alpha3,
    visibility,
):
    # ── Visibility flags ──────────────────────────────────────────────────────
    show_ellipse = 'ellipse' in (visibility or [])
    show_eaxes   = 'eaxes'   in (visibility or [])

    # ── Run physics pipeline ──────────────────────────────────────────────────
    # JS conversion: this block becomes the body of updateAll() in app.js
    result = compute_all(
        theta_rad   = degrees_to_radians(theta),
        phi_rad     = degrees_to_radians(phi),
        chi_rad     = degrees_to_radians(chi),
        theta_B_rad = degrees_to_radians(theta_b),
        phi_B_rad   = degrees_to_radians(phi_b),
        input_mode  = pol_mode,
        basis_state = basis_state if pol_mode == 'basis' else None,
        alpha1_rad  = degrees_to_radians(alpha1),
        alpha2_rad  = degrees_to_radians(alpha2),
        alpha3_rad  = degrees_to_radians(alpha3),
    )

    # ── Build figures ─────────────────────────────────────────────────────────
    # JS conversion: each make_*_figure() call becomes its JS equivalent
    fig_3d       = make_3d_figure(result, show_ellipse, show_eaxes)
    fig_level    = make_level_figure(result)
    fig_density  = make_density_figure(result)
    fig_ellipse  = make_ellipse_figure(result)
    fig_poincare = make_poincare_figure(result)

    return (
        fig_3d, fig_level, fig_density, fig_ellipse, fig_poincare,
        # Slider value displays
        str(theta), str(phi), str(chi),
        str(theta_b), str(phi_b),
        str(alpha1), str(alpha2), str(alpha3),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, port=8050)