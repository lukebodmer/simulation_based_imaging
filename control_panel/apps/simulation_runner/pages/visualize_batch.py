"""Visualize Simulation Batch page."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import toml
from pyvista.trame.ui import plotter_ui
from trame.widgets import matplotlib as mpl_widgets
from trame.widgets import vuetify3 as v3

from sbimaging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path("/data/simulations")

# PyVista offscreen rendering for trame
pv.OFF_SCREEN = True

# Available colormaps for visualization
COLORMAP_OPTIONS = [
    {"title": "Seismic (diverging)", "value": "seismic"},
    {"title": "Coolwarm (diverging)", "value": "coolwarm"},
    {"title": "RdBu (diverging)", "value": "RdBu_r"},
    {"title": "Viridis", "value": "viridis"},
    {"title": "Plasma", "value": "plasma"},
    {"title": "Inferno", "value": "inferno"},
    {"title": "Magma", "value": "magma"},
    {"title": "Cividis", "value": "cividis"},
    {"title": "Turbo", "value": "turbo"},
    {"title": "Jet", "value": "jet"},
    {"title": "Hot", "value": "hot"},
    {"title": "Cool", "value": "cool"},
    {"title": "Gray", "value": "gray"},
]

# Opacity presets for different visualization needs
OPACITY_PRESETS = [
    {
        "title": "Symmetric (hide zero)",
        "value": "symmetric",
        "opacity": [0.9, 0.7, 0.5, 0.3, 0, 0.3, 0.5, 0.7, 0.9],
    },
    {
        "title": "Linear",
        "value": "linear",
        "opacity": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    },
    {
        "title": "Uniform",
        "value": "uniform",
        "opacity": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    },
    {
        "title": "High values only",
        "value": "high_only",
        "opacity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0],
    },
    {
        "title": "Low values only",
        "value": "low_only",
        "opacity": [1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "title": "Extremes only",
        "value": "extremes",
        "opacity": [1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.1, 0.5, 1.0],
    },
]


class VisualizeBatchPage:
    """Page for visualizing completed simulation batches."""

    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        # PyVista plotters
        self.plotter_sim = pv.Plotter()  # Simulation pressure data
        self.plotter_wave = pv.Plotter()  # Wave speed visualization

        # Matplotlib figure widgets (will be set during UI build)
        self.sensor_matrix_widget = None
        self.energy_widget = None

        # Data storage
        self.mesh_data = None
        self.timestep_data = None
        self.sensor_data = None
        self.energy_data = None

        self._setup_state()

    def _setup_state(self):
        """Initialize state variables for this page."""
        self.state.available_batches = []
        self.state.selected_batch = ""
        self.state.simulation_list = []
        self.state.selected_simulation = ""
        self.state.timestep_list = []
        self.state.selected_timestep = ""
        self.state.simulation_parameters_dict = {}
        self.state.viz_active_tab = "wave"

        # Colormap and visualization controls
        self.state.colormap_options = COLORMAP_OPTIONS
        self.state.opacity_presets = OPACITY_PRESETS
        self.state.selected_colormap = "seismic"
        self.state.selected_opacity_preset = "symmetric"
        self.state.clim_min = -1.0
        self.state.clim_max = 1.0
        self.state.auto_clim = False
        self.state.point_size = 8

        # Refresh batch list on init
        self._refresh_batch_list()

        # Register state change handlers
        self.state.change("selected_batch")(self._on_batch_selected)
        self.state.change("selected_simulation")(self._on_simulation_selected)
        self.state.change("selected_timestep")(self._on_timestep_selected)
        self.state.change("selected_colormap")(self._on_visualization_setting_changed)
        self.state.change("selected_opacity_preset")(
            self._on_visualization_setting_changed
        )
        self.state.change("clim_min")(self._on_visualization_setting_changed)
        self.state.change("clim_max")(self._on_visualization_setting_changed)
        self.state.change("auto_clim")(self._on_visualization_setting_changed)
        self.state.change("point_size")(self._on_visualization_setting_changed)

    def _refresh_batch_list(self):
        """Scan /data/simulations for available batches."""
        batches = []
        if DEFAULT_DATA_DIR.exists():
            for item in sorted(DEFAULT_DATA_DIR.iterdir()):
                if item.is_dir():
                    has_params = (item / "parameter_files").exists()
                    has_sims = (item / "simulations").exists()
                    if has_params or has_sims:
                        batches.append({"title": item.name, "value": item.name})

        self.state.available_batches = batches
        logger.info(f"Found {len(batches)} simulation batches")

    def _on_batch_selected(self, selected_batch, **kwargs):
        """Handle batch selection - refresh simulation list."""
        self.state.simulation_list = []
        self.state.selected_simulation = ""
        self.state.timestep_list = []
        self.state.selected_timestep = ""
        self.state.simulation_parameters_dict = {}

        if not selected_batch:
            return

        sim_dir = DEFAULT_DATA_DIR / selected_batch / "simulations"
        if sim_dir.exists():
            sims = [
                {"title": d.name, "value": d.name}
                for d in sorted(sim_dir.iterdir())
                if d.is_dir()
            ]
            self.state.simulation_list = sims
            logger.info(f"Found {len(sims)} simulations in batch {selected_batch}")

            # Auto-select the first simulation
            if sims:
                self.state.selected_simulation = sims[0]["value"]

    def _on_simulation_selected(self, selected_simulation, **kwargs):
        """Handle simulation selection - load parameters and timesteps."""
        self.state.timestep_list = []
        self.state.selected_timestep = ""
        self.state.simulation_parameters_dict = {}

        # Clear cached data from previous simulation
        self.mesh_data = None
        self.timestep_data = None
        self.sensor_data = None
        self.energy_data = None

        if not selected_simulation or not self.state.selected_batch:
            return

        batch_name = self.state.selected_batch

        # Load simulation parameters
        param_file = (
            DEFAULT_DATA_DIR
            / batch_name
            / "parameter_files"
            / f"{selected_simulation}.toml"
        )
        if param_file.exists():
            try:
                raw_params = toml.load(param_file)
                sim_params = {
                    section: {k: self._prettify_value(v) for k, v in content.items()}
                    for section, content in raw_params.items()
                }
                self.state.simulation_parameters_dict = sim_params
            except Exception as e:
                logger.error(f"Failed to load parameters: {e}")
                self.state.simulation_parameters_dict = {"Error": {"message": str(e)}}

        # Load timestep list
        sim_dir = DEFAULT_DATA_DIR / batch_name / "simulations" / selected_simulation
        data_dir = sim_dir / "data"

        # Load sensor and energy data first (these are saved once at end of simulation)
        self._load_sensor_data(sim_dir)
        self._load_energy_data(sim_dir)

        if data_dir.exists():
            timesteps = []
            for f in sorted(data_dir.glob("*.pkl")):
                try:
                    display_num = str(int(f.stem.split("_t")[-1]))
                except (ValueError, IndexError):
                    display_num = f.name
                timesteps.append({"title": display_num, "value": f.name})
            # Sort numerically
            timesteps.sort(key=lambda x: int(x["title"]) if x["title"].isdigit() else 0)
            self.state.timestep_list = timesteps
            logger.info(f"Found {len(timesteps)} timesteps")

            # Auto-select the first timestep and load its data
            if timesteps:
                first_timestep = timesteps[0]["value"]
                self.state.selected_timestep = first_timestep
                # Explicitly load and visualize since state change may not fire
                self._load_timestep_and_visualize(sim_dir, first_timestep)

    def _load_sensor_data(self, sim_dir: Path):
        """Load sensor data from simulation directory."""
        sensor_file = sim_dir / "sensor_data.pkl"
        if sensor_file.exists():
            try:
                with open(sensor_file, "rb") as f:
                    self.sensor_data = pickle.load(f)
                logger.info(f"Loaded sensor data from {sensor_file}")
            except Exception as e:
                logger.error(f"Failed to load sensor data: {e}")
                self.sensor_data = None
        else:
            self.sensor_data = None

    def _load_energy_data(self, sim_dir: Path):
        """Load energy data from simulation directory."""
        energy_file = sim_dir / "energy_data.pkl"
        if energy_file.exists():
            try:
                with open(energy_file, "rb") as f:
                    self.energy_data = pickle.load(f)
                logger.info(f"Loaded energy data from {energy_file}")
            except Exception as e:
                logger.error(f"Failed to load energy data: {e}")
                self.energy_data = None
        else:
            self.energy_data = None

    def _on_timestep_selected(self, selected_timestep, **kwargs):
        """Handle timestep selection - load and visualize data."""
        if not selected_timestep:
            return

        batch_name = self.state.selected_batch
        sim_hash = self.state.selected_simulation
        if not batch_name or not sim_hash:
            return

        sim_dir = DEFAULT_DATA_DIR / batch_name / "simulations" / sim_hash
        self._load_timestep_and_visualize(sim_dir, selected_timestep)

    def _on_visualization_setting_changed(self, **kwargs):
        """Handle changes to visualization settings (colormap, clim, etc.)."""
        # Only update if we have data loaded
        if self.timestep_data is not None and self.mesh_data is not None:
            self._update_simulation_plot()

    def _load_timestep_and_visualize(self, sim_dir: Path, timestep_filename: str):
        """Load timestep data and update all visualizations."""
        data_dir = sim_dir / "data"
        batch_name = self.state.selected_batch
        sim_hash = self.state.selected_simulation

        # Load timestep data
        timestep_path = data_dir / timestep_filename
        try:
            with open(timestep_path, "rb") as f:
                self.timestep_data = pickle.load(f)
            logger.info(f"Loaded timestep data from {timestep_path}")
        except Exception as e:
            logger.error(f"Failed to load timestep data: {e}")
            return

        # Load mesh data
        self._load_mesh_data(batch_name, sim_hash)

        # Update visualizations
        self._update_visualizations()

    def _load_mesh_data(self, batch_name: str, sim_hash: str):
        """Load mesh data for the simulation."""
        batch_dir = DEFAULT_DATA_DIR / batch_name
        sim_dir = batch_dir / "simulations" / sim_hash

        # First try to load from simulation output directory
        sim_mesh_file = sim_dir / "mesh.pkl"
        if sim_mesh_file.exists():
            try:
                with open(sim_mesh_file, "rb") as f:
                    self.mesh_data = pickle.load(f)
                logger.info(f"Loaded mesh data from {sim_mesh_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load mesh from sim dir: {e}")

        # Fall back to looking up mesh hash from parameter file
        try:
            from sbimaging.batch.planner import BatchPlanner

            planner = BatchPlanner(batch_dir)
            mesh_hash = planner.get_mesh_hash_for_simulation(sim_hash)

            if mesh_hash is None:
                logger.error(f"No mesh hash found for simulation {sim_hash}")
                return

            mesh_file = batch_dir / "meshes" / mesh_hash / "mesh.pkl"
            if mesh_file.exists():
                with open(mesh_file, "rb") as f:
                    self.mesh_data = pickle.load(f)
                logger.info(f"Loaded mesh data from {mesh_file}")
            else:
                logger.error(f"Mesh pickle not found: {mesh_file}")

        except Exception as e:
            logger.error(f"Failed to load mesh data: {e}")

    def _update_visualizations(self):
        """Update all visualizations with current data."""
        if self.timestep_data is None:
            return

        # Update PyVista plotters
        self._update_wave_speed_plot()
        self._update_simulation_plot()

        # Update matplotlib plots
        self._update_sensor_plot()
        self._update_energy_plot()

    def _update_wave_speed_plot(self):
        """Update the wave speed visualization."""
        if self.mesh_data is None:
            return

        self.plotter_wave.clear()

        try:
            # Build tetrahedral grid from mesh
            if hasattr(self.mesh_data, "num_cells"):
                num_cells = self.mesh_data.num_cells
                cell_to_vertices = self._to_numpy(self.mesh_data.cell_to_vertices)
                vertex_coordinates = self._to_numpy(self.mesh_data.vertex_coordinates)
                speed = self._to_numpy(self.mesh_data.speed[0, :])
            else:
                num_cells = self.mesh_data["num_cells"]
                cell_to_vertices = self._to_numpy(self.mesh_data["cell_to_vertices"])
                vertex_coordinates = self._to_numpy(
                    self.mesh_data["vertex_coordinates"]
                )
                speed = self._to_numpy(self.mesh_data["speed_per_cell"])

            cell_conn = np.hstack(
                [np.full((num_cells, 1), 4), cell_to_vertices]
            ).ravel()
            cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

            grid = pv.UnstructuredGrid(cell_conn, cell_types, vertex_coordinates)
            grid.cell_data["speed"] = speed

            # Create uniform grid for volume rendering
            bounds = grid.bounds
            resolution = (50, 50, 50)
            nx, ny, nz = resolution
            image = pv.ImageData()
            image.dimensions = resolution
            image.origin = (bounds[0], bounds[2], bounds[4])
            image.spacing = (
                (bounds[1] - bounds[0]) / (nx - 1),
                (bounds[3] - bounds[2]) / (ny - 1),
                (bounds[5] - bounds[4]) / (nz - 1),
            )

            sampled = image.sample(grid)
            self.plotter_wave.add_volume(
                sampled,
                scalars="speed",
                cmap="viridis",
                opacity="sigmoid",
            )
            self.plotter_wave.show_grid()
            self.plotter_wave.reset_camera()
            self.plotter_wave.render()

            # Push update to trame UI
            self.ctrl.view_update_wave()

        except Exception as e:
            logger.error(f"Failed to update wave speed plot: {e}")

    def _update_simulation_plot(self):
        """Update the simulation pressure visualization."""
        if self.mesh_data is None or self.timestep_data is None:
            return

        self.plotter_sim.clear()

        try:
            # Get node coordinates
            if hasattr(self.mesh_data, "x"):
                x = self._to_numpy(self.mesh_data.x.ravel(order="F"))
                y = self._to_numpy(self.mesh_data.y.ravel(order="F"))
                z = self._to_numpy(self.mesh_data.z.ravel(order="F"))
            else:
                x = self._to_numpy(self.mesh_data["x"].ravel(order="F"))
                y = self._to_numpy(self.mesh_data["y"].ravel(order="F"))
                z = self._to_numpy(self.mesh_data["z"].ravel(order="F"))

            node_coordinates = np.column_stack((x, y, z))

            # Get pressure field
            fields = self.timestep_data.get("fields", {})
            if "p" in fields:
                pressure = self._to_numpy(fields["p"]).ravel(order="F")

                # Get visualization settings from state
                colormap = self.state.selected_colormap
                point_size = int(self.state.point_size)

                # Get opacity from preset
                opacity_preset = self.state.selected_opacity_preset
                opacity = [0.9, 0.7, 0.5, 0.3, 0, 0.3, 0.5, 0.7, 0.9]  # default
                for preset in OPACITY_PRESETS:
                    if preset["value"] == opacity_preset:
                        opacity = preset["opacity"]
                        break

                # Determine color limits
                if self.state.auto_clim:
                    clim_max = float(np.abs(pressure).max()) or 1.0
                    clim = [-clim_max, clim_max]
                else:
                    clim = [float(self.state.clim_min), float(self.state.clim_max)]

                self.plotter_sim.add_points(
                    node_coordinates,
                    scalars=pressure,
                    cmap=colormap,
                    opacity=opacity,
                    clim=clim,
                    point_size=point_size,
                    render_points_as_spheres=True,
                )

            # Add sensors if available
            sensor_coords = self.timestep_data.get("sensor_coordinates", [])
            if sensor_coords:
                sensor_points = np.array(sensor_coords)
                self.plotter_sim.add_points(
                    sensor_points,
                    color="black",
                    point_size=10,
                    render_points_as_spheres=True,
                )

            self.plotter_sim.show_grid()
            self.plotter_sim.reset_camera()
            self.plotter_sim.render()

            # Push update to trame UI
            self.ctrl.view_update_sim()

        except Exception as e:
            logger.error(f"Failed to update simulation plot: {e}")

    def _update_sensor_plot(self):
        """Update the sensor data matrix plot."""
        if self.sensor_matrix_widget is None:
            return

        if self.sensor_data is None or "pressure" not in self.sensor_data:
            logger.debug("No sensor pressure data available")
            return

        try:
            data_matrix = self._to_numpy(self.sensor_data["pressure"])

            fig, ax = plt.subplots(figsize=(8, 4))
            vmax = np.abs(data_matrix).max() or 1.0
            cax = ax.imshow(
                data_matrix,
                aspect="auto",
                cmap="seismic",
                origin="lower",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_ylabel("Sensor Index")
            ax.set_xlabel("Time Step Index")
            fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.01)
            plt.tight_layout()

            self.sensor_matrix_widget.update(fig)
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to update sensor plot: {e}")

    def _update_energy_plot(self):
        """Update the energy plot."""
        if self.energy_widget is None:
            return

        if self.energy_data is None:
            logger.debug("No energy data available")
            return

        try:
            total_energy = self.energy_data.get("total", [])
            kinetic_energy = self.energy_data.get("kinetic", [])
            potential_energy = self.energy_data.get("potential", [])

            if len(total_energy) == 0:
                return

            # Get dt from timestep data if available
            dt = self.timestep_data.get("dt", 1.0) if self.timestep_data else 1.0
            time_array = np.arange(len(total_energy)) * dt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(
                time_array,
                self._to_numpy(total_energy),
                marker="o",
                markersize=2,
                label="Total Energy",
            )
            if len(kinetic_energy) > 0:
                ax.plot(
                    time_array,
                    self._to_numpy(kinetic_energy),
                    marker="x",
                    markersize=2,
                    label="KE",
                )
            if len(potential_energy) > 0:
                ax.plot(
                    time_array,
                    self._to_numpy(potential_energy),
                    marker="*",
                    markersize=2,
                    label="PE",
                )
            ax.legend()
            ax.set_title("Global Energy")
            ax.set_ylabel("Energy")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            self.energy_widget.update(fig)
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to update energy plot: {e}")

    def _to_numpy(self, array):
        """Convert CuPy array to NumPy if needed."""
        if hasattr(array, "get"):
            return array.get()
        return np.asarray(array)

    def _prettify_value(self, value):
        """Format values for display."""
        if isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):
                return ", ".join(str(x) for x in value)
            elif all(isinstance(x, list) for x in value):
                return "\n".join(str(row) for row in value)
        return str(value)

    def _go_to_previous_simulation(self):
        """Navigate to the previous simulation."""
        simulations = self.state.simulation_list
        current = self.state.selected_simulation
        if not simulations or not current:
            return

        current_index = next(
            (i for i, s in enumerate(simulations) if s["value"] == current), -1
        )
        if current_index > 0:
            self.state.selected_simulation = simulations[current_index - 1]["value"]

    def _go_to_next_simulation(self):
        """Navigate to the next simulation."""
        simulations = self.state.simulation_list
        current = self.state.selected_simulation
        if not simulations or not current:
            return

        current_index = next(
            (i for i, s in enumerate(simulations) if s["value"] == current), -1
        )
        if current_index < len(simulations) - 1:
            self.state.selected_simulation = simulations[current_index + 1]["value"]

    def _go_to_previous_timestep(self):
        """Navigate to the previous timestep."""
        timesteps = self.state.timestep_list
        current = self.state.selected_timestep
        if not timesteps or not current:
            return

        current_index = next(
            (i for i, t in enumerate(timesteps) if t["value"] == current), -1
        )
        if current_index > 0:
            self.state.selected_timestep = timesteps[current_index - 1]["value"]

    def _go_to_next_timestep(self):
        """Navigate to the next timestep."""
        timesteps = self.state.timestep_list
        current = self.state.selected_timestep
        if not timesteps or not current:
            return

        current_index = next(
            (i for i, t in enumerate(timesteps) if t["value"] == current), -1
        )
        if current_index < len(timesteps) - 1:
            self.state.selected_timestep = timesteps[current_index + 1]["value"]

    def build_ui(self):
        """Build the visualization page UI."""
        with v3.VContainer(fluid=True, classes="fill-height"):
            with v3.VRow(classes="fill-height"):
                # Left sidebar - Selection controls
                with v3.VCol(cols=3, classes="fill-height"):
                    self._build_selection_panel()

                # Right content - Visualizations
                with v3.VCol(cols=9, classes="fill-height"):
                    self._build_visualization_panel()

    def _build_selection_panel(self):
        """Build the left selection panel."""
        with v3.VCard(classes="fill-height"):
            v3.VCardTitle("Selection")
            with v3.VCardText():
                # Batch selection
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=10):
                        v3.VSelect(
                            v_model=("selected_batch",),
                            items=("available_batches",),
                            label="Batch",
                            density="compact",
                            clearable=True,
                        )
                    with v3.VCol(cols=2):
                        v3.VBtn(
                            icon="mdi-refresh",
                            variant="text",
                            density="compact",
                            click=self._refresh_batch_list,
                        )

                # Simulation selection with navigation buttons
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=2, classes="pa-0"):
                        v3.VBtn(
                            icon="mdi-chevron-left",
                            variant="text",
                            density="compact",
                            click=self._go_to_previous_simulation,
                            disabled=(
                                "!selected_simulation || simulation_list.length === 0",
                            ),
                        )
                    with v3.VCol(cols=8, classes="pa-0"):
                        v3.VSelect(
                            v_model=("selected_simulation",),
                            items=("simulation_list",),
                            label="Simulation",
                            density="compact",
                            clearable=True,
                            disabled=("!selected_batch",),
                            hide_details=True,
                        )
                    with v3.VCol(cols=2, classes="pa-0"):
                        v3.VBtn(
                            icon="mdi-chevron-right",
                            variant="text",
                            density="compact",
                            click=self._go_to_next_simulation,
                            disabled=(
                                "!selected_simulation || simulation_list.length === 0",
                            ),
                        )

                # Timestep selection with navigation buttons
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=2, classes="pa-0"):
                        v3.VBtn(
                            icon="mdi-chevron-left",
                            variant="text",
                            density="compact",
                            click=self._go_to_previous_timestep,
                            disabled=(
                                "!selected_timestep || timestep_list.length === 0",
                            ),
                        )
                    with v3.VCol(cols=8, classes="pa-0"):
                        v3.VSelect(
                            v_model=("selected_timestep",),
                            items=("timestep_list",),
                            label="Timestep",
                            item_title="title",
                            item_value="value",
                            density="compact",
                            clearable=True,
                            disabled=("!selected_simulation",),
                            hide_details=True,
                        )
                    with v3.VCol(cols=2, classes="pa-0"):
                        v3.VBtn(
                            icon="mdi-chevron-right",
                            variant="text",
                            density="compact",
                            click=self._go_to_next_timestep,
                            disabled=(
                                "!selected_timestep || timestep_list.length === 0",
                            ),
                        )

                v3.VDivider(classes="my-3")

                # Parameters display
                v3.VCardSubtitle("Simulation Parameters")
                with v3.VExpansionPanels(multiple=True, density="compact"):
                    with v3.VExpansionPanel(
                        v_for="([section, params]) in Object.entries(simulation_parameters_dict)",
                        key=("section",),
                    ):
                        v3.VExpansionPanelTitle("{{ section }}")
                        with v3.VExpansionPanelText():
                            with v3.VList(density="compact"):
                                with v3.VListItem(
                                    v_for="([key, value]) in Object.entries(params)",
                                    key=("key",),
                                ):
                                    v3.VListItemTitle(
                                        "{{ key }}",
                                        classes="text-caption font-weight-bold",
                                    )
                                    v3.VListItemSubtitle(
                                        "{{ value }}",
                                        style="white-space: pre-wrap; font-family: monospace; font-size: 11px;",
                                    )

    def _build_visualization_panel(self):
        """Build the right visualization panel."""
        with v3.VCard(classes="fill-height"):
            # Tabs for different views
            with v3.VTabs(v_model=("viz_active_tab",), density="compact"):
                v3.VTab(value="wave", text="Wave Speed")
                v3.VTab(value="simulation", text="Simulation")
                v3.VTab(value="data", text="Data")

            with v3.VCardText(classes="fill-height pa-0"):
                with v3.VWindow(
                    v_model=("viz_active_tab",),
                    style="height: calc(100vh - 180px);",
                ):
                    # Wave Speed tab
                    with v3.VWindowItem(value="wave", style="height: 100%;"):
                        view_wave = plotter_ui(
                            self.plotter_wave,
                            server=self.server,
                            add_menu=False,
                        )
                        self.ctrl.view_update_wave = view_wave.update

                    # Simulation tab
                    with v3.VWindowItem(value="simulation", style="height: 100%;"):
                        # Visualization controls toolbar
                        with v3.VToolbar(density="compact", color="surface"):
                            with v3.VRow(dense=True, align="center", classes="px-2"):
                                with v3.VCol(cols="auto"):
                                    v3.VSelect(
                                        v_model=("selected_colormap",),
                                        items=("colormap_options",),
                                        label="Colormap",
                                        density="compact",
                                        hide_details=True,
                                        style="min-width: 180px;",
                                    )
                                with v3.VCol(cols="auto"):
                                    v3.VSelect(
                                        v_model=("selected_opacity_preset",),
                                        items=("opacity_presets",),
                                        label="Opacity",
                                        density="compact",
                                        hide_details=True,
                                        style="min-width: 160px;",
                                    )
                                with v3.VCol(cols="auto"):
                                    v3.VTextField(
                                        v_model=("clim_min",),
                                        label="Min",
                                        type="number",
                                        step="0.1",
                                        density="compact",
                                        hide_details=True,
                                        disabled=("auto_clim",),
                                        style="max-width: 80px;",
                                    )
                                with v3.VCol(cols="auto"):
                                    v3.VTextField(
                                        v_model=("clim_max",),
                                        label="Max",
                                        type="number",
                                        step="0.1",
                                        density="compact",
                                        hide_details=True,
                                        disabled=("auto_clim",),
                                        style="max-width: 80px;",
                                    )
                                with v3.VCol(cols="auto"):
                                    v3.VCheckbox(
                                        v_model=("auto_clim",),
                                        label="Auto",
                                        density="compact",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols="auto"):
                                    v3.VSlider(
                                        v_model=("point_size",),
                                        label="Pt Size",
                                        min=1,
                                        max=20,
                                        step=1,
                                        density="compact",
                                        hide_details=True,
                                        thumb_label=True,
                                        style="min-width: 120px;",
                                    )
                        # PyVista view - use calc to subtract toolbar height
                        with v3.VSheet(style="height: calc(100% - 64px); width: 100%;"):
                            view_sim = plotter_ui(
                                self.plotter_sim,
                                server=self.server,
                                add_menu=False,
                            )
                            self.ctrl.view_update_sim = view_sim.update

                    # Data tab
                    with v3.VWindowItem(value="data", style="height: 100%;"):
                        with v3.VContainer(fluid=True, classes="fill-height"):
                            with v3.VRow(classes="flex-grow-1"):
                                with v3.VCol(
                                    cols=12,
                                    classes="d-flex align-center justify-center",
                                ):
                                    self.sensor_matrix_widget = mpl_widgets.Figure(
                                        figure=None
                                    )
                                    self.sensor_matrix_widget.update(
                                        plt.figure(figsize=(10, 4))
                                    )
                            with v3.VRow(classes="flex-grow-1"):
                                with v3.VCol(
                                    cols=12,
                                    classes="d-flex align-center justify-center",
                                ):
                                    self.energy_widget = mpl_widgets.Figure(figure=None)
                                    self.energy_widget.update(
                                        plt.figure(figsize=(10, 4))
                                    )
