"""
PySide6 waveform-drawing tool for jbubble.

Top panel:   draw a driving-pressure waveform by hand (rightward-only).
Bottom panel: jbubble simulation result (radius response).
Right panel:  all configurable bubble / simulation parameters.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QLabel,
    QScrollArea,
    QSplitter,
    QStatusBar,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

jax.config.update("jax_enable_x64", True)

from jbubble import Bubble, Waveform, Units, SaveSpec, run_simulation


# ---------------------------------------------------------------------------
# Drawing canvas (top)
# ---------------------------------------------------------------------------

class DrawingCanvas(FigureCanvas):
    """Matplotlib canvas that captures mouse strokes (rightward-only)."""

    def __init__(self, duration_us: float = 20.0, pressure_range_kpa: float = 300.0,
                 num_samples: int = 2048, parent=None):
        self.fig = Figure(figsize=(9, 3), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.duration_us = duration_us
        self.pressure_range_kpa = pressure_range_kpa
        self.num_samples = num_samples

        # raw stroke data
        self._raw_x: list[float] = []
        self._raw_y: list[float] = []
        self._is_drawing = False
        self._last_x = -np.inf  # rightward gatekeeper

        # processed signal (uniform grid, in kPa)
        self._signal_kpa = np.zeros(num_samples)

        # blitting background (captured after first full draw)
        self._bg = None

        self._setup_axes()
        self._connect_events()

    # -- axis helpers ---------------------------------------------------------

    def _setup_axes(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Draw driving pressure (click & drag rightward)")
        self.ax.set_xlabel("Time (μs)")
        self.ax.set_ylabel("Pressure (kPa)")
        self.ax.set_xlim(0, self.duration_us)
        self.ax.set_ylim(-self.pressure_range_kpa, self.pressure_range_kpa)
        self.ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        self.ax.grid(True, ls="--", alpha=0.4)

        self._stroke_line, = self.ax.plot([], [], "r-", lw=1, label="Hand-drawn",
                                           animated=True)
        self._processed_line, = self.ax.plot([], [], "b-", lw=2, alpha=0.7,
                                              label="Interpolated", animated=True)
        self.ax.legend(loc="upper right", fontsize=8)

    def _capture_background(self):
        """Capture the static background for blitting."""
        self.draw()
        self._bg = self.copy_from_bbox(self.ax.bbox)

    def _blit_lines(self):
        """Restore background and redraw only the two animated lines."""
        if self._bg is None:
            self._capture_background()
        self.restore_region(self._bg)
        self.ax.draw_artist(self._stroke_line)
        self.ax.draw_artist(self._processed_line)
        self.blit(self.ax.bbox)
        self.flush_events()

    def _connect_events(self):
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event):
        """Re-capture background whenever the canvas is fully redrawn (e.g. resize)."""
        self._bg = self.copy_from_bbox(self.ax.bbox)

    # -- mouse callbacks ------------------------------------------------------

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._is_drawing = True
        self._raw_x = [event.xdata]
        self._raw_y = [event.ydata]
        self._last_x = event.xdata
        self._stroke_line.set_data(self._raw_x, self._raw_y)
        # Re-enable animated mode for fast blitting during the stroke
        self._stroke_line.set_animated(True)
        self._processed_line.set_animated(True)
        self._bg = None  # force background recapture without the lines
        self._blit_lines()

    def _on_move(self, event):
        if not self._is_drawing or event.inaxes != self.ax:
            return
        # Rightward-only constraint
        if event.xdata <= self._last_x:
            return
        self._raw_x.append(event.xdata)
        self._raw_y.append(event.ydata)
        self._last_x = event.xdata
        self._stroke_line.set_data(self._raw_x, self._raw_y)
        self._process()  # update interpolated trace live while drawing

    def _on_release(self, event):
        if not self._is_drawing:
            return
        self._is_drawing = False
        self._process()
        # Disable animated so the lines are included in the normal full draw
        self._stroke_line.set_animated(False)
        self._processed_line.set_animated(False)
        self.draw()

    # -- interpolation --------------------------------------------------------

    def _process(self):
        if len(self._raw_x) < 2:
            return

        x = np.asarray(self._raw_x)
        y = np.asarray(self._raw_y)

        # Ensure strictly increasing (should already be due to rightward-only)
        order = np.argsort(x)
        x, y = x[order], y[order]
        xu, idx = np.unique(x, return_index=True)
        yu = y[idx]

        target_t = np.linspace(0, self.duration_us, self.num_samples)
        f = interp1d(xu, yu, kind="linear", bounds_error=False, fill_value=0.0)
        self._signal_kpa = f(target_t)

        self._processed_line.set_data(target_t, self._signal_kpa)
        self._blit_lines()

    # -- public interface -----------------------------------------------------

    def get_signal_kpa(self) -> np.ndarray:
        """Return the interpolated pressure waveform in kPa."""
        return self._signal_kpa.copy()

    def get_waveform(self) -> Waveform:
        """Build a jbubble Waveform (SI units: Pa, s)."""
        pressure_pa = self._signal_kpa * 1e3  # kPa -> Pa
        dt_s = (self.duration_us * 1e-6) / (self.num_samples - 1)  # μs -> s
        samples = jnp.array(pressure_pa)
        return Waveform(samples=samples, dt=dt_s, t0=0.0)

    def clear(self):
        self._raw_x.clear()
        self._raw_y.clear()
        self._last_x = -np.inf
        self._signal_kpa[:] = 0.0
        self._stroke_line.set_data([], [])
        self._processed_line.set_data([], [])
        self._bg = None  # force background recapture
        self.draw()

    def update_axes(self, duration_us: float, pressure_range_kpa: float,
                    num_samples: int):
        self.duration_us = duration_us
        self.pressure_range_kpa = pressure_range_kpa
        self.num_samples = num_samples
        self._signal_kpa = np.zeros(num_samples)
        self.ax.set_xlim(0, duration_us)
        self.ax.set_ylim(-pressure_range_kpa, pressure_range_kpa)
        self._bg = None  # force background recapture
        self.clear()


# ---------------------------------------------------------------------------
# Response canvas (bottom)
# ---------------------------------------------------------------------------

class ResponseCanvas(FigureCanvas):
    """Two-subplot canvas: driving pressure + bubble radius."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 4), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_p = self.fig.add_subplot(211)
        self.ax_r = self.fig.add_subplot(212, sharex=self.ax_p)

        self.ax_p.set_ylabel("Pressure (kPa)")
        self.ax_p.set_title("Driving pressure")
        self.ax_p.grid(True, ls="--", alpha=0.4)

        self.ax_r.set_xlabel("Time (μs)")
        self.ax_r.set_ylabel("Radius (μm)")
        self.ax_r.set_title("Bubble radius response")
        self.ax_r.grid(True, ls="--", alpha=0.4)

        self._line_p, = self.ax_p.plot([], [], "b-", lw=1.2)
        self._line_r, = self.ax_r.plot([], [], "r-", lw=1.2)

    def plot_result(self, ts_us: np.ndarray, pressure_kpa: np.ndarray,
                    radius_um: np.ndarray, R0_um: float):
        self._line_p.set_data(ts_us, pressure_kpa)
        self.ax_p.relim()
        self.ax_p.autoscale_view()

        self._line_r.set_data(ts_us, radius_um)
        self.ax_r.axhline(R0_um, color="gray", ls=":", lw=0.8)
        self.ax_r.relim()
        self.ax_r.autoscale_view()

        self.draw_idle()

    def clear(self):
        self._line_p.set_data([], [])
        self._line_r.set_data([], [])
        # remove any existing R0 reference lines
        for line in list(self.ax_r.lines):
            if line is not self._line_r:
                line.remove()
        self.ax_p.relim()
        self.ax_p.autoscale_view()
        self.ax_r.relim()
        self.ax_r.autoscale_view()
        self.draw_idle()


# ---------------------------------------------------------------------------
# Parameter panel (right sidebar)
# ---------------------------------------------------------------------------

def _double_spin(value: float, minimum: float, maximum: float,
                 decimals: int = 4, step: float = 0.01,
                 suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(minimum, maximum)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(f"  {suffix}")
    return sb


def _int_spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(minimum, maximum)
    sb.setValue(value)
    return sb


class ParameterPanel(QScrollArea):
    """Right-hand sidebar with all tuneable knobs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(290)
        self.setMaximumWidth(360)

        container = QWidget()
        layout = QVBoxLayout(container)

        # -- Bubble parameters ------------------------------------------------
        grp_bubble = QGroupBox("Bubble (Marmottant)")
        form_b = QFormLayout()

        self.spin_R0         = _double_spin(3.0,   0.1, 50.0,  2, 0.1,  "μm")
        self.spin_R_buckle   = _double_spin(0.99,  0.5, 1.0,   3, 0.005, "× R₀")
        self.spin_gamma      = _double_spin(1.07,  1.0, 2.0,   2, 0.01)
        self.spin_chi        = _double_spin(0.38,  0.0, 5.0,   3, 0.01, "N/m")
        self.spin_mu_L       = _double_spin(0.89,  0.01, 10.0, 3, 0.01, "mPa·s")
        self.spin_kappa_s    = _double_spin(2.4,   0.0, 100.0, 2, 0.1,  "nN·s/m")
        self.spin_rho_L      = _double_spin(1000,  500, 2000,  1, 10,   "kg/m³")
        self.spin_c_L        = _double_spin(1498,  1000, 2000, 1, 10,   "m/s")
        self.spin_P_amb      = _double_spin(101.3, 50, 200,    1, 1,    "kPa")
        self.spin_sigma_L    = _double_spin(72,    0,  100,    1, 1,    "mN/m")
        self.spin_vdw_div    = _double_spin(5.61,  2,  20,     2, 0.1)

        form_b.addRow("R₀", self.spin_R0)
        form_b.addRow("R_buckle", self.spin_R_buckle)
        form_b.addRow("γ (polytropic)", self.spin_gamma)
        form_b.addRow("χ (shell elast.)", self.spin_chi)
        form_b.addRow("μ_L (liq. visc.)", self.spin_mu_L)
        form_b.addRow("κ_s (shell visc.)", self.spin_kappa_s)
        form_b.addRow("ρ_L (liq. dens.)", self.spin_rho_L)
        form_b.addRow("c_L (sound spd.)", self.spin_c_L)
        form_b.addRow("P_amb", self.spin_P_amb)
        form_b.addRow("σ_L (surf. tens.)", self.spin_sigma_L)
        form_b.addRow("vdW divisor", self.spin_vdw_div)
        grp_bubble.setLayout(form_b)
        layout.addWidget(grp_bubble)

        # -- Waveform / canvas parameters -------------------------------------
        grp_wave = QGroupBox("Waveform canvas")
        form_w = QFormLayout()

        self.spin_duration   = _double_spin(20.0, 1.0, 200.0, 1, 1.0, "μs")
        self.spin_prange     = _double_spin(300,  10,  5000,  0, 50,  "kPa")
        self.spin_nsamples   = _int_spin(2048, 256, 16384)

        form_w.addRow("Duration", self.spin_duration)
        form_w.addRow("Pressure range", self.spin_prange)
        form_w.addRow("Samples", self.spin_nsamples)
        grp_wave.setLayout(form_w)
        layout.addWidget(grp_wave)

        # -- Simulation parameters --------------------------------------------
        grp_sim = QGroupBox("Simulation")
        form_s = QFormLayout()

        self.spin_window     = _double_spin(30.0, 1.0, 500.0, 1, 5,   "μs")
        self.spin_max_steps  = _int_spin(20000, 1000, 200000)
        self.spin_save_pts   = _int_spin(2048, 256, 16384)

        form_s.addRow("Window", self.spin_window)
        form_s.addRow("Max steps", self.spin_max_steps)
        form_s.addRow("Save points", self.spin_save_pts)
        grp_sim.setLayout(form_s)
        layout.addWidget(grp_sim)

        # -- Action buttons ---------------------------------------------------
        self.btn_simulate = QPushButton("▶  Simulate")
        self.btn_simulate.setStyleSheet(
            "QPushButton { background-color: #2d8cf0; color: white; "
            "font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a6fd1; }"
        )
        self.btn_clear = QPushButton("Clear")
        self.btn_apply_axes = QPushButton("Apply canvas settings")

        layout.addWidget(self.btn_simulate)
        layout.addWidget(self.btn_apply_axes)
        layout.addWidget(self.btn_clear)
        layout.addStretch()

        self.setWidget(container)

    # -- convenience readers --------------------------------------------------

    def build_bubble(self) -> Bubble:
        """Construct a Bubble from the current spin-box values."""
        R0 = self.spin_R0.value() * 1e-6                     # μm -> m
        return Bubble(
            R0=R0,
            R_buckle=self.spin_R_buckle.value() * R0,         # fraction of R0
            gamma=self.spin_gamma.value(),
            chi=self.spin_chi.value(),                        # N/m
            mu_L=self.spin_mu_L.value() * 1e-3,              # mPa·s -> Pa·s
            kappa_s=self.spin_kappa_s.value() * 1e-9,         # nN·s/m -> kg/s
            rho_L=self.spin_rho_L.value(),                    # kg/m³
            c_L=self.spin_c_L.value(),                        # m/s
            P_amb=self.spin_P_amb.value() * 1e3,              # kPa -> Pa
            sigma_L=self.spin_sigma_L.value() * 1e-3,         # mN/m -> N/m
            vdw_divisor=self.spin_vdw_div.value(),
        )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("jbubble – Waveform Drawer")
        self.resize(1200, 750)

        # -- widgets ----------------------------------------------------------
        self.draw_canvas = DrawingCanvas(
            duration_us=20.0, pressure_range_kpa=300.0, num_samples=2048
        )
        self.response_canvas = ResponseCanvas()
        self.params = ParameterPanel()

        # -- layout -----------------------------------------------------------
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.addWidget(self.draw_canvas)
        plot_splitter.addWidget(self.response_canvas)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 1)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(plot_splitter)
        main_splitter.addWidget(self.params)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)

        self.setCentralWidget(main_splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Draw a waveform, configure the bubble, then hit Simulate.")

        # -- connections ------------------------------------------------------
        self.params.btn_simulate.clicked.connect(self._run_simulation)
        self.params.btn_clear.clicked.connect(self._clear)
        self.params.btn_apply_axes.clicked.connect(self._apply_canvas_settings)

    # -- slots ----------------------------------------------------------------

    def _apply_canvas_settings(self):
        self.draw_canvas.update_axes(
            duration_us=self.params.spin_duration.value(),
            pressure_range_kpa=self.params.spin_prange.value(),
            num_samples=self.params.spin_nsamples.value(),
        )
        self.status.showMessage("Canvas settings applied – previous drawing cleared.")

    def _clear(self):
        self.draw_canvas.clear()
        self.response_canvas.clear()
        self.status.showMessage("Cleared.")

    def _run_simulation(self):
        self.status.showMessage("Running simulation …")
        QApplication.processEvents()

        try:
            bubble = self.params.build_bubble()
            waveform = self.draw_canvas.get_waveform()
            units = Units()
            save_spec = SaveSpec(num_samples=self.params.spin_save_pts.value())
            window_s = self.params.spin_window.value() * 1e-6  # μs -> s

            result = run_simulation(
                bubble=bubble,
                pulse=waveform,
                units=units,
                save_spec=save_spec,
                window_s=window_s,
                max_steps=self.params.spin_max_steps.value(),
            )

            ts_us = np.asarray(result.ts) * 1e6               # s -> μs
            pressure_kpa = np.asarray(result.driving_pressure) * 1e-3  # Pa -> kPa
            radius_um = np.asarray(result.radius) * 1e6        # m -> μm
            R0_um = bubble.R0 * 1e6

            converged = bool(result.converged)
            self.response_canvas.plot_result(ts_us, pressure_kpa, radius_um, R0_um)
            tag = "✓ converged" if converged else "✗ did NOT converge"
            self.status.showMessage(f"Simulation complete ({tag}).")

        except Exception as exc:
            self.status.showMessage(f"Simulation failed: {exc}")
            raise

    # -- OS dark-mode passthrough (optional) -----------------------------------

    def _apply_style(self):
        """Minimal stylesheet so it looks decent on light and dark desktops."""
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win._apply_style()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
