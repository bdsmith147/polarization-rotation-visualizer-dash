# traces.py
# ─────────────────────────────────────────────────────────────────────────────
# Plotly figure builders for PolariViz.
#
# Every function here is PURE:
#   - Takes numbers / plain lists (from physics.compute_all() result dict)
#   - Returns a go.Figure object
#   - No sliders, no callbacks, no global state, no NumPy
#
# Called exclusively by app.py inside the single master callback.
#
# FILE STRUCTURE:
#   1. Constants & color scheme
#   2. 3D scene helpers  (arrow, sphere, axes — return traces, not figures)
#   3. make_3d_figure()
#   4. make_level_figure()
#   5. make_density_figure()
#   6. make_ellipse_figure()
#   7. make_poincare_figure()
# ─────────────────────────────────────────────────────────────────────────────

import math
import plotly.graph_objects as go
from math_helpers import scale, add, subtract, normalize, norm, cross, linspace


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & COLOR SCHEME
# ══════════════════════════════════════════════════════════════════════════════

# ── Arrow / scene geometry ────────────────────────────────────────────────────
L_ARROW      = 1.0    # length of k̂ and quantization axis arrows
L_EAXES      = 0.5    # length of ê₁, ê₂ axes
R_SPHERE     = 0.15   # atom cloud radius
SCENE_RANGE  = 1.6    # ± axis range for 3D scene
ELLIPSE_SCALE = 0.35  # max radius of polarization ellipse (in scene units)

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_K_ARROW     = '#E63946'   # red      — k̂ vector
COLOR_QUANT       = '#457B9D'   # blue     — quantization axis
COLOR_EAXES       = '#2A9D8F'   # teal     — ê₁ and ê₂ (same color for both)
COLOR_ELLIPSE     = '#E9C46A'   # gold     — polarization ellipse in 3D
COLOR_SPHERE      = '#A8DADC'   # pale blue — atom cloud
COLOR_LAB_AXES    = '#999999'   # grey     — x, y, z reference lines
COLOR_SIGMA_PLUS  = '#E63946'   # red      — σ+ transitions
COLOR_PI          = '#2A9D8F'   # teal     — π  transitions
COLOR_SIGMA_MINUS = '#457B9D'   # blue     — σ- transitions
COLOR_BG          = '#1A1A2E'   # dark navy — figure background
COLOR_PAPER       = '#16213E'   # slightly lighter — paper background
COLOR_TEXT        = '#E0E0E0'   # light grey — axis labels and annotations

# ── Level diagram layout (in normalized figure units 0–1) ─────────────────────
LEVEL_J0_Y     = 0.15   # y position of J=0 ground level line
LEVEL_J1_Y     = 0.80   # y position of J=1 excited level line
LEVEL_MJ_XS    = {      # x positions of mJ sublevels
    'plus1':  0.25,
    'zero':   0.50,
    'minus1': 0.75,
}
LEVEL_LINE_HALF_W = 0.10  # half-width of each level line segment

# ── Density matrix bar chart ──────────────────────────────────────────────────
DM_BAR_WIDTH  = 0.6    # width of each bar in x and y (bar occupies 0.6 of cell)
DM_BAR_GAP    = 1.0    # cell size (bars are spaced 1.0 apart)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 3D SCENE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _arrow_traces(tail, tip, color, name, line_width=5, cone_size=0.07):
    """Build a 3D arrow as a Scatter3d line (shaft) + Cone (head).

    Returns a list of two traces: [shaft, head].
    The cone is placed at the tip pointing in the direction tail→tip.

    tail, tip : real 3-vectors [x, y, z]
    """
    direction = subtract(tip, tail)
    mag = norm(direction)
    if mag < 1e-10:
        return []

    # Shaft: line from tail to tip
    shaft = go.Scatter3d(
        x=[tail[0], tip[0]],
        y=[tail[1], tip[1]],
        z=[tail[2], tip[2]],
        mode='lines',
        line=dict(color=color, width=line_width),
        name=name,
        showlegend=True,
        hoverinfo='name',
    )

    # Head: cone at tip
    u, v, w = direction[0], direction[1], direction[2]
    head = go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[u], v=[v], w=[w],
        sizemode='absolute',
        sizeref=cone_size,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        showlegend=False,
        hoverinfo='name',
    )
    return [shaft, head]


def _dashed_line_trace(tail, tip, color, name, opacity=0.6, width=3):
    """A dashed 3D line from tail to tip — used for ê₁, ê₂ axes.

    Plotly Scatter3d doesn't natively support dashed lines in 3D,
    so we approximate by placing dots along the line.
    Returns a single go.Scatter3d trace.
    """
    n = 20  # number of dots
    ts = linspace(0, 1, n)
    xs = [tail[0] + t * (tip[0] - tail[0]) for t in ts]
    ys = [tail[1] + t * (tip[1] - tail[1]) for t in ts]
    zs = [tail[2] + t * (tip[2] - tail[2]) for t in ts]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=2, color=color, opacity=opacity),
        name=name,
        showlegend=True,
        hoverinfo='name',
    )


def _sphere_surface_trace(radius, color, opacity, name):
    """A smooth sphere surface centered at origin using go.Surface.

    Parameterized as:
        x = r sinθ cosφ
        y = r sinθ sinφ
        z = r cosθ

    Returns a single go.Surface trace.
    """
    n_theta = 24
    n_phi   = 24
    thetas  = linspace(0, math.pi, n_theta)
    phis    = linspace(0, 2 * math.pi, n_phi)

    xs = [[radius * math.sin(th) * math.cos(ph)
           for ph in phis] for th in thetas]
    ys = [[radius * math.sin(th) * math.sin(ph)
           for ph in phis] for th in thetas]
    zs = [[radius * math.cos(th)
           for ph in phis] for th in thetas]

    return go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name=name,
        hoverinfo='name',
        lighting=dict(ambient=0.8, diffuse=0.5),
    )


def _lab_axes_traces():
    """Three thin grey reference lines along lab x, y, z axes.

    Returns a list of 3 go.Scatter3d traces with endpoint labels.
    """
    axes = [
        ([1, 0, 0], 'x'),
        ([0, 1, 0], 'y'),
        ([0, 0, 1], 'z'),
    ]
    traces = []
    for direction, label in axes:
        tip = scale(direction, SCENE_RANGE * 0.9)
        traces.append(go.Scatter3d(
            x=[0, tip[0]],
            y=[0, tip[1]],
            z=[0, tip[2]],
            mode='lines+text',
            line=dict(color=COLOR_LAB_AXES, width=1),
            text=['', label],
            textposition='top center',
            textfont=dict(color=COLOR_TEXT, size=12),
            name=f'{label}-axis',
            showlegend=False,
            hoverinfo='none',
        ))
    return traces


def _ellipse_traces(xs, ys, zs, k_tail, color, opacity, name):
    """Polarization ellipse as a closed Scatter3d line.

    The ellipse coordinates (xs, ys, zs) are centered at the origin
    from physics.py. Here we shift them to be centered at k_tail,
    and scale them to ELLIPSE_SCALE.

    Returns a single go.Scatter3d trace.
    """
    # Find max extent for normalization
    max_r = max(
        math.sqrt(xs[i]**2 + ys[i]**2 + zs[i]**2)
        for i in range(len(xs))
    )
    if max_r < 1e-10:
        max_r = 1.0

    scale_factor = ELLIPSE_SCALE / max_r

    # Shift to k_tail
    ex = [k_tail[0] + xs[i] * scale_factor for i in range(len(xs))]
    ey = [k_tail[1] + ys[i] * scale_factor for i in range(len(ys))]
    ez = [k_tail[2] + zs[i] * scale_factor for i in range(len(zs))]

    # Close the loop
    ex.append(ex[0])
    ey.append(ey[0])
    ez.append(ez[0])

    return go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='lines',
        line=dict(color=color, width=2),
        opacity=opacity,
        name=name,
        showlegend=True,
        hoverinfo='name',
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. make_3d_figure()
# ══════════════════════════════════════════════════════════════════════════════

def make_3d_figure(result, show_ellipse=True, show_eaxes=True):
    """Build the main 3D scene figure.

    result:        dict from physics.compute_all()
    show_ellipse:  bool — show polarization ellipse
    show_eaxes:    bool — show ê₁, ê₂ frame axes

    Returns a go.Figure.
    """
    k_hat     = result['k_hat']
    e1        = result['e1']
    e2        = result['e2']
    q_axis    = result['quant_axis']
    ellipse_xs = result['ellipse_xs']
    ellipse_ys = result['ellipse_ys']
    ellipse_zs = result['ellipse_zs']

    # ── Derived geometry ──────────────────────────────────────────────────────
    k_tail = scale(k_hat, -L_ARROW)          # tail: -L × k̂
    k_tip  = scale(k_hat,  R_SPHERE)         # tip: sphere surface

    q_tip  = scale(q_axis, L_ARROW)          # quantization axis tip

    e1_tail = k_tail
    e1_tip  = add(k_tail, scale(e1, L_EAXES))
    e2_tail = k_tail
    e2_tip  = add(k_tail, scale(e2, L_EAXES))

    # ── Assemble traces ───────────────────────────────────────────────────────
    traces = []

    # 1. Lab reference axes (background, unobtrusive)
    traces += _lab_axes_traces()

    # 2. Atom cloud sphere
    traces.append(_sphere_surface_trace(
        R_SPHERE, COLOR_SPHERE, opacity=0.5, name='Atom cloud'))

    # 3. Quantization axis
    traces += _arrow_traces(
        [0, 0, 0], q_tip,
        color=COLOR_QUANT, name='Quantization axis',
        line_width=5, cone_size=0.07)

    # 4. k̂ vector (most prominent arrow)
    traces += _arrow_traces(
        k_tail, k_tip,
        color=COLOR_K_ARROW, name='k̂ (beam)',
        line_width=6, cone_size=0.08)

    # 5. ê₁, ê₂ axes (subtle, checkbox-gated)
    if show_eaxes:
        traces.append(_dashed_line_trace(
            e1_tail, e1_tip,
            color=COLOR_EAXES, name='ê₁', opacity=0.6))
        traces.append(_dashed_line_trace(
            e2_tail, e2_tip,
            color=COLOR_EAXES, name='ê₂', opacity=0.6))

    # 6. Polarization ellipse (subtle, checkbox-gated)
    if show_ellipse:
        traces.append(_ellipse_traces(
            ellipse_xs, ellipse_ys, ellipse_zs,
            k_tail=k_tail,
            color=COLOR_ELLIPSE, opacity=0.7,
            name='Polarization ellipse'))

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = go.Layout(
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text='3D Scene',
            font=dict(color=COLOR_TEXT, size=13),
            x=0.5,
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT, size=11),
            bgcolor='rgba(0,0,0,0)',
            x=0.01, y=0.99,
        ),
        scene=dict(
            bgcolor=COLOR_BG,
            xaxis=dict(
                range=[-SCENE_RANGE, SCENE_RANGE],
                showticklabels=False,
                showgrid=True, gridcolor='#333355',
                zeroline=False, title='',
            ),
            yaxis=dict(
                range=[-SCENE_RANGE, SCENE_RANGE],
                showticklabels=False,
                showgrid=True, gridcolor='#333355',
                zeroline=False, title='',
            ),
            zaxis=dict(
                range=[-SCENE_RANGE, SCENE_RANGE],
                showticklabels=False,
                showgrid=True, gridcolor='#333355',
                zeroline=False, title='',
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.4, y=1.0, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
    )

    return go.Figure(data=traces, layout=layout)


# ══════════════════════════════════════════════════════════════════════════════
# 4. make_level_figure()
# ══════════════════════════════════════════════════════════════════════════════

def make_level_figure(result):
    """Build the J=0 → J=1 energy level diagram.

    Draws horizontal level lines and vertical transition arrows.
    Arrow width ∝ absorption strength.
    Arrow color encodes transition type (σ+/π/σ-).
    Percentage label next to each arrowhead.

    Returns a go.Figure.
    """
    absorption = result['absorption']

    sp  = absorption['mJ_plus1']
    pi  = absorption['mJ_0']
    sm  = absorption['mJ_minus1']
    total = sp + pi + sm if (sp + pi + sm) > 0 else 1.0

    # Normalize to percentages
    pct = {
        'plus1':  100 * sp  / total,
        'zero':   100 * pi  / total,
        'minus1': 100 * sm  / total,
    }

    # Arrow widths (scaled to max)
    max_abs = max(sp, pi, sm) if max(sp, pi, sm) > 0 else 1.0
    max_width = 12
    min_width = 1

    def arrow_width(strength):
        return max(min_width, (strength / max_abs) * max_width)

    widths = {
        'plus1':  arrow_width(sp),
        'zero':   arrow_width(pi),
        'minus1': arrow_width(sm),
    }

    colors = {
        'plus1':  COLOR_SIGMA_PLUS,
        'zero':   COLOR_PI,
        'minus1': COLOR_SIGMA_MINUS,
    }

    labels = {
        'plus1':  'mJ=+1  (σ+)',
        'zero':   'mJ=0   (π)',
        'minus1': 'mJ=-1  (σ-)',
    }
    
    angles = {
        'plus1':  -35,
        'zero':   0,
        'minus1': 35,
        }

    traces = []

    # ── J=0 ground level line ─────────────────────────────────────────────────
    traces.append(go.Scatter(
        x=[0.5 - LEVEL_LINE_HALF_W, 0.5 + LEVEL_LINE_HALF_W], y=[LEVEL_J0_Y, LEVEL_J0_Y],
        mode='lines',
        line=dict(color=COLOR_TEXT, width=2),
        name='J=0',
        showlegend=False,
        hoverinfo='none',
    ))
    traces.append(go.Scatter(
        x=[0.05], y=[LEVEL_J0_Y],
        mode='text',
        text=['J=0'],
        textfont=dict(color=COLOR_TEXT, size=12),
        showlegend=False,
        hoverinfo='none',
    ))

    # ── J=1 excited level lines (one per mJ sublevel) ─────────────────────────
    for key, x_center in LEVEL_MJ_XS.items():
        x0 = x_center - LEVEL_LINE_HALF_W
        x1 = x_center + LEVEL_LINE_HALF_W
        traces.append(go.Scatter(
            x=[x0, x1], y=[LEVEL_J1_Y, LEVEL_J1_Y],
            mode='lines',
            line=dict(color=COLOR_TEXT, width=2),
            showlegend=False,
            hoverinfo='none',
        ))

    # ── mJ sublevel labels ────────────────────────────────────────────────────
    for key, x_center in LEVEL_MJ_XS.items():
        mj_label = {'plus1': 'mJ=+1', 'zero': 'mJ=0', 'minus1': 'mJ=-1'}[key]
        traces.append(go.Scatter(
            x=[x_center], y=[LEVEL_J1_Y + 0.08],
            mode='text',
            text=[mj_label],
            textfont=dict(color=COLOR_TEXT, size=11),
            showlegend=False,
            hoverinfo='none',
        ))

    # J=1 label
    traces.append(go.Scatter(
        x=[0.05], y=[LEVEL_J1_Y],
        mode='text',
        text=['J=1'],
        textfont=dict(color=COLOR_TEXT, size=12),
        showlegend=False,
        hoverinfo='none',
    ))

    # ── Transition arrows ─────────────────────────────────────────────────────
    for key, x_center in LEVEL_MJ_XS.items():
        color = colors[key]
        width = widths[key]
        pct_val = pct[key]
        ang = angles[key]

        # Arrow shaft (vertical line)
        traces.append(go.Scatter(
            x=[0.5, x_center],
            y=[LEVEL_J0_Y, LEVEL_J1_Y - 0.08],
            mode='lines',
            line=dict(color=color, width=width),
            name=labels[key],
            showlegend=True,
            hoverinfo='name',
        ))

        # Arrowhead (triangle marker at tip)
        traces.append(go.Scatter(
            x=[x_center],
            y=[LEVEL_J1_Y - 0.08],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=max(6, width * 1.5),
                color=color,
                angle=ang,
            ),
            showlegend=False,
            hoverinfo='none',
        ))

        # Percentage label next to arrowhead
        traces.append(go.Scatter(
            x=[x_center + 0.07],
            y=[LEVEL_J1_Y - 0.05],
            mode='text',
            text=[f'{pct_val:.1f}%'],
            textfont=dict(color=color, size=12),
            showlegend=False,
            hoverinfo='none',
        ))

    layout = go.Layout(
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(
            text='J=0 → J=1 Transitions',
            font=dict(color=COLOR_TEXT, size=13),
            x=0.5,
        ),
        xaxis=dict(
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT, size=11),
            bgcolor='rgba(0,0,0,0)',
            x=0.01, y=0.5,
        ),
        showlegend=True,
    )

    return go.Figure(data=traces, layout=layout)


# ══════════════════════════════════════════════════════════════════════════════
# 5. make_density_figure()
# ══════════════════════════════════════════════════════════════════════════════

def _bar_mesh(x_center, y_center, height, color):
    """Build a single rectangular prism (3D bar) using go.Mesh3d.

    The bar is centered at (x_center, y_center), has a fixed
    footprint of DM_BAR_WIDTH × DM_BAR_WIDTH, and rises from z=0
    to z=height.

    Returns a single go.Mesh3d trace.
    """
    hw = DM_BAR_WIDTH / 2
    x0, x1 = x_center - hw, x_center + hw
    y0, y1 = y_center - hw, y_center + hw
    z0, z1 = 0.0, max(height, 0.01)  # always at least a sliver

    # 8 vertices of the box
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]

    # 12 triangles (2 per face × 6 faces)
    i = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 4]
    j = [1, 3, 2, 5, 3, 6, 0, 7, 4, 5, 5, 7]
    k = [2, 2, 5, 6, 6, 7, 7, 4, 5, 1, 6, 6]  # renamed to avoid clash

    return go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=i, j=j, k=k,
        color=color,
        opacity=0.85,
        flatshading=True,
        showlegend=False,
        hoverinfo='skip',
    )


def _phase_to_color(phase_rad):
    """Map a phase angle in [-π, π] to an RGB hex color using HSV.

    Hue cycles through the full color wheel as phase goes 0 → 2π.
    Returns a hex color string '#RRGGBB'.
    """
    # Normalize phase to [0, 1] for hue
    hue = (phase_rad % (2 * math.pi)) / (2 * math.pi)
    # HSV → RGB (saturation=0.9, value=0.9)
    s, v = 0.9, 0.9
    h6 = hue * 6.0
    hi = int(h6) % 6
    f  = h6 - int(h6)
    p  = v * (1 - s)
    q  = v * (1 - s * f)
    t  = v * (1 - s * (1 - f))
    rgb_map = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]
    r, g, b = rgb_map[hi]
    ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
    return f'#{ri:02x}{gi:02x}{bi:02x}'


def make_density_figure(result):
    """Build the 3×3 density matrix visualization.

    Each element ρᵢⱼ is drawn as a 3D rectangular bar:
      - Height = |ρᵢⱼ|  (amplitude)
      - Color  = arg(ρᵢⱼ) mapped to HSV hue  (phase)

    Axes labeled [σ+, π, σ-] on both x and y.

    Returns a go.Figure.
    """
    rho_amps   = result['rho_amps']
    rho_phases = result['rho_phases']

    labels = ['σ+', 'π', 'σ-']
    traces = []

    for i in range(3):
        for j in range(3):
            amp   = rho_amps[i][j]
            phase = rho_phases[i][j]
            color = _phase_to_color(phase)
            # x → column (j), y → row (i)
            traces.append(_bar_mesh(
                x_center = j * DM_BAR_GAP,
                y_center = i * DM_BAR_GAP,
                height   = amp,
                color    = color,
            ))

    # ── Axis tick annotations ─────────────────────────────────────────────────
    # Add invisible scatter traces at tick positions to get axis labels
    for idx, lbl in enumerate(labels):
        # x-axis label (bottom)
        traces.append(go.Scatter3d(
            x=[idx * DM_BAR_GAP], y=[-0.7], z=[0],
            mode='text',
            text=[lbl],
            textfont=dict(color=COLOR_TEXT, size=12),
            showlegend=False,
            hoverinfo='none',
        ))
        # y-axis label (left)
        traces.append(go.Scatter3d(
            x=[-0.7], y=[idx * DM_BAR_GAP], z=[0],
            mode='text',
            text=[lbl],
            textfont=dict(color=COLOR_TEXT, size=12),
            showlegend=False,
            hoverinfo='none',
        ))

    layout = go.Layout(
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text='Density Matrix  (height=|ρ|, color=phase)',
            font=dict(color=COLOR_TEXT, size=13),
            x=0.5,
        ),
        scene=dict(
            bgcolor=COLOR_BG,
            xaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=labels,
                tickfont=dict(color=COLOR_TEXT),
                title=dict(text='', font=dict(color=COLOR_TEXT)),
                showgrid=True, gridcolor='#333355',
                range=[-0.8, 2.8],
            ),
            yaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=labels,
                tickfont=dict(color=COLOR_TEXT),
                title=dict(text='', font=dict(color=COLOR_TEXT)),
                showgrid=True, gridcolor='#333355',
                range=[-0.8, 2.8],
            ),
            zaxis=dict(
                range=[0, 1.1],
                tickfont=dict(color=COLOR_TEXT),
                title=dict(text='|ρ|', font=dict(color=COLOR_TEXT, size=11)),
                showgrid=True, gridcolor='#333355',
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
    )

    return go.Figure(data=traces, layout=layout)


# ══════════════════════════════════════════════════════════════════════════════
# 6. make_ellipse_figure()
# ══════════════════════════════════════════════════════════════════════════════

def make_ellipse_figure(result):
    """Build the 2D polarization ellipse in the {ê₁, ê₂} transverse plane.

    Axes:
      x → component along ê₁
      y → component along ê₂

    The ellipse is normalized so its maximum radius = 1.

    Returns a go.Figure.
    """
    e1_vals = result['ellipse_e1']
    e2_vals = result['ellipse_e2']
    stokes  = result['stokes']

    # Normalize ellipse to max radius = 1
    max_r = max(
        math.sqrt(e1_vals[i]**2 + e2_vals[i]**2)
        for i in range(len(e1_vals))
    )
    if max_r < 1e-10:
        max_r = 1.0

    e1_norm = [v / max_r for v in e1_vals]
    e2_norm = [v / max_r for v in e2_vals]

    # Close the loop
    e1_closed = e1_norm + [e1_norm[0]]
    e2_closed = e2_norm + [e2_norm[0]]

    # Handedness label from S3
    S3 = stokes['S3']
    if abs(S3) < 0.05:
        handedness = 'Linear'
    elif S3 > 0:
        handedness = 'RHC'
    else:
        handedness = 'LHC'

    traces = [
        # Reference circle (unit circle guide)
        go.Scatter(
            x=[math.cos(t) for t in linspace(0, 2*math.pi, 60)],
            y=[math.sin(t) for t in linspace(0, 2*math.pi, 60)],
            mode='lines',
            line=dict(color='#333355', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='none',
        ),
        # ê₁ axis reference line
        go.Scatter(
            x=[-1.1, 1.1], y=[0, 0],
            mode='lines',
            line=dict(color=COLOR_LAB_AXES, width=1),
            showlegend=False,
            hoverinfo='none',
        ),
        # ê₂ axis reference line
        go.Scatter(
            x=[0, 0], y=[-1.1, 1.1],
            mode='lines',
            line=dict(color=COLOR_LAB_AXES, width=1),
            showlegend=False,
            hoverinfo='none',
        ),
        # Polarization ellipse
        go.Scatter(
            x=e1_closed,
            y=e2_closed,
            mode='lines',
            line=dict(color=COLOR_ELLIPSE, width=2),
            fill='toself',
            fillcolor='rgba(233,196,106,0.1)',
            name='Polarization ellipse',
            hoverinfo='none',
        ),
        # Origin dot
        go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=5, color=COLOR_TEXT),
            showlegend=False,
            hoverinfo='none',
        ),
    ]

    layout = go.Layout(
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(
            text=f'Polarization Ellipse  ({handedness})',
            font=dict(color=COLOR_TEXT, size=13),
            x=0.5,
        ),
        xaxis=dict(
            range=[-1.3, 1.3],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=dict(text='ê₁', font=dict(color=COLOR_EAXES, size=12)),
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-1.3, 1.3],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=dict(text='ê₂', font=dict(color=COLOR_EAXES, size=12)),
        ),
        showlegend=False,
        annotations=[
            dict(
                x=1.15, y=0.05,
                xref='x', yref='y',
                text='ê₁',
                showarrow=False,
                font=dict(color=COLOR_EAXES, size=12),
            ),
            dict(
                x=0.05, y=1.15,
                xref='x', yref='y',
                text='ê₂',
                showarrow=False,
                font=dict(color=COLOR_EAXES, size=12),
            ),
        ],
    )

    return go.Figure(data=traces, layout=layout)


# ══════════════════════════════════════════════════════════════════════════════
# 7. make_poincare_figure()
# ══════════════════════════════════════════════════════════════════════════════

def make_poincare_figure(result):
    """Build the Poincaré sphere showing current polarization state.

    Traces:
      1. Unit sphere surface (low opacity)
      2. S1, S2, S3 axis lines with pole labels
      3. Current state: bold point on sphere surface
      4. Line from origin to point

    Returns a go.Figure.
    """
    stokes = result['stokes']
    p = stokes['poincare']   # [S1/S0, S2/S0, S3/S0]

    traces = []

    # ── Sphere surface ────────────────────────────────────────────────────────
    traces.append(_sphere_surface_trace(
        radius=1.0, color='#334466', opacity=0.15,
        name='Poincaré sphere'))

    # ── Axis lines with pole labels ───────────────────────────────────────────
    pole_axes = [
        ([1,0,0], [-1,0,0], 'H', 'V',   COLOR_SIGMA_PLUS),
        ([0,1,0], [0,-1,0], '+45°', '-45°', COLOR_PI),
        ([0,0,1], [0,0,-1], 'RHC', 'LHC', COLOR_SIGMA_MINUS),
    ]
    for pos_tip, neg_tip, pos_lbl, neg_lbl, color in pole_axes:
        # Axis line
        traces.append(go.Scatter3d(
            x=[neg_tip[0], pos_tip[0]],
            y=[neg_tip[1], pos_tip[1]],
            z=[neg_tip[2], pos_tip[2]],
            mode='lines+text',
            line=dict(color=color, width=2),
            text=[neg_lbl, pos_lbl],
            textposition=['bottom center', 'top center'],
            textfont=dict(color=color, size=11),
            showlegend=False,
            hoverinfo='none',
        ))

    # ── Line from origin to current state point ───────────────────────────────
    traces.append(go.Scatter3d(
        x=[0, p[0]], y=[0, p[1]], z=[0, p[2]],
        mode='lines',
        line=dict(color=COLOR_TEXT, width=2, dash='dot'),
        showlegend=False,
        hoverinfo='none',
    ))

    # ── Current state point ───────────────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode='markers',
        marker=dict(size=10, color=COLOR_ELLIPSE, symbol='circle'),
        name='Polarization state',
        hovertemplate=(
            f'S1/S0: {p[0]:.3f}<br>'
            f'S2/S0: {p[1]:.3f}<br>'
            f'S3/S0: {p[2]:.3f}<extra></extra>'
        ),
    ))

    layout = go.Layout(
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text='Poincaré Sphere',
            font=dict(color=COLOR_TEXT, size=13),
            x=0.5,
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT, size=11),
            bgcolor='rgba(0,0,0,0)',
        ),
        scene=dict(
            bgcolor=COLOR_BG,
            xaxis=dict(
                range=[-1.3, 1.3],
                showticklabels=False,
                showgrid=True, gridcolor='#333355',
                zeroline=False, title='',
            ),
            yaxis=dict(
                range=[-1.3, 1.3],
                showticklabels=False,
                showgrid=True, gridcolor='#333355',
                zeroline=False, title='',
            ),
            zaxis=dict(
                range=[-1.3, 1.3],
                showticklabels=False,
                showgrid=True, gridcolor='#333355',
                zeroline=False, title='',
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.4, y=1.0, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
    )

    return go.Figure(data=traces, layout=layout)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
# Run directly:  python traces.py

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from math_helpers import degrees_to_radians
    from physics import compute_all

    print('Running traces.py self-test...')

    result = compute_all(
        theta_rad   = degrees_to_radians(45),
        phi_rad     = degrees_to_radians(30),
        chi_rad     = degrees_to_radians(0),
        theta_B_rad = degrees_to_radians(20),
        phi_B_rad   = degrees_to_radians(60),
        input_mode  = 'waveplate',
        alpha1_rad  = degrees_to_radians(0),
        alpha2_rad  = degrees_to_radians(45),
        alpha3_rad  = degrees_to_radians(0),
    )

    # Build all figures — just check they return without error
    fig_3d      = make_3d_figure(result, show_ellipse=True, show_eaxes=True)
    fig_level   = make_level_figure(result)
    fig_density = make_density_figure(result)
    fig_ellipse = make_ellipse_figure(result)
    fig_poincare = make_poincare_figure(result)

    assert isinstance(fig_3d,       go.Figure), '3D figure failed'
    assert isinstance(fig_level,    go.Figure), 'Level figure failed'
    assert isinstance(fig_density,  go.Figure), 'Density figure failed'
    assert isinstance(fig_ellipse,  go.Figure), 'Ellipse figure failed'
    assert isinstance(fig_poincare, go.Figure), 'Poincaré figure failed'

    # Check 3D figure has at least the fixed traces
    assert len(fig_3d.data) >= 5, \
        f'3D figure should have ≥5 traces, got {len(fig_3d.data)}'

    # Check level figure has percentage labels
    texts = [str(t.text) for t in fig_level.data if hasattr(t, 'text')]
    assert any('%' in str(t) for t in texts), \
        'Level figure missing percentage labels'

    # Check density figure has 9 bars
    mesh_traces = [t for t in fig_density.data
                   if isinstance(t, go.Mesh3d)]
    assert len(mesh_traces) == 9, \
        f'Density figure should have 9 Mesh3d bars, got {len(mesh_traces)}'

    # Basis state mode
    result_basis = compute_all(
        theta_rad=0, phi_rad=0, chi_rad=0,
        theta_B_rad=0, phi_B_rad=0,
        input_mode='basis',
        basis_state='sigma_plus',
    )
    fig_3d_basis = make_3d_figure(result_basis,
                                   show_ellipse=False, show_eaxes=False)
    assert isinstance(fig_3d_basis, go.Figure), 'Basis state 3D figure failed'

    print('All tests passed.')