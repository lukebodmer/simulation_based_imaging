"""Run Simulation Batch page."""

import asyncio
from pathlib import Path

import toml
from trame.widgets import html
from trame.widgets import vuetify3 as v3

from sbimaging.batch.executor import resume_batch, run_batch
from sbimaging.batch.generator import ParameterRange, ParameterSpace
from sbimaging.batch.planner import BatchPlanner
from sbimaging.config import get_base_config_path, list_presets, load_preset
from sbimaging.config.simulation import (
    MeshConfig,
    OuterMaterialConfig,
    OutputConfig,
    ReceiverConfig,
    SimulationConfig,
    SolverConfig,
    SourceConfig,
)
from sbimaging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path("/data/simulations")

INCLUSION_TYPES = [
    {"title": "Ellipsoid", "value": "ellipsoid"},
    {"title": "Sphere", "value": "sphere"},
    {"title": "Multiple Cubes", "value": "multi_cubes"},
    {"title": "Cube in Ellipsoid", "value": "cube_in_ellipsoid"},
]


class RunBatchPage:
    """Page for configuring and running simulation batches."""

    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        self._presets = self._load_presets()
        self._setup_state()

    def _load_presets(self) -> dict:
        """Load all available presets."""
        presets = {}
        for name in list_presets():
            try:
                presets[name] = load_preset(name)
            except Exception as e:
                logger.warning(f"Failed to load preset {name}: {e}")
        return presets

    def _setup_state(self):
        """Initialize reactive state variables."""
        preset_items = [{"title": "-- Start from scratch --", "value": ""}]
        for name, preset in self._presets.items():
            preset_items.append(
                {
                    "title": f"{preset.name}",
                    "value": name,
                }
            )

        self.state.preset_items = preset_items
        self.state.selected_preset = ""
        self.state.inclusion_type_items = INCLUSION_TYPES

        # Batch settings
        self.state.batch_name = "my_batch"
        self.state.batch_description = ""
        self.state.batch_dir = str(DEFAULT_DATA_DIR)
        self.state.num_samples = 100

        # Material property ranges
        self.state.wave_speed_min = 1.5
        self.state.wave_speed_max = 4.0
        self.state.density_min = 1.5
        self.state.density_max = 4.0

        # Inclusion geometry - scaling ranges for each axis
        self.state.scaling_x_min = 0.1
        self.state.scaling_x_max = 0.3
        self.state.scaling_y_min = 0.1
        self.state.scaling_y_max = 0.3
        self.state.scaling_z_min = 0.1
        self.state.scaling_z_max = 0.3

        # Inclusion behavior
        self.state.allow_rotation = False
        self.state.allow_movement = False

        # Inclusion type
        self.state.inclusion_type = "ellipsoid"

        # Cube parameters (for multi_cubes and cube_in_ellipsoid)
        self.state.cube_quantity_min = 1
        self.state.cube_quantity_max = 3
        self.state.cube_width_min = 0.05
        self.state.cube_width_max = 0.2

        # Geometry
        self.state.boundary_buffer = 0.05

        # Sources (per-source control)
        self.state.source_count = 6
        default_sources = [
            {
                "center": [0.5, 0.5, 0.0],
                "frequency": 3.0,
                "amplitude": 1.0,
                "radius": 0.05,
            },
            {
                "center": [0.5, 0.5, 1.0],
                "frequency": 3.0,
                "amplitude": 1.0,
                "radius": 0.05,
            },
            {
                "center": [0.5, 0.0, 0.5],
                "frequency": 3.0,
                "amplitude": 1.0,
                "radius": 0.05,
            },
            {
                "center": [0.5, 1.0, 0.5],
                "frequency": 3.0,
                "amplitude": 1.0,
                "radius": 0.05,
            },
            {
                "center": [0.0, 0.5, 0.5],
                "frequency": 3.0,
                "amplitude": 1.0,
                "radius": 0.05,
            },
            {
                "center": [1.0, 0.5, 0.5],
                "frequency": 3.0,
                "amplitude": 1.0,
                "radius": 0.05,
            },
        ]
        self.state.sources = default_sources
        self.state.all_source_frequency = 3.0
        self.state.all_source_amplitude = 1.0
        self.state.all_source_radius = 0.05

        # Outer material (fixed, not swept)
        self.state.outer_wave_speed = 2.0
        self.state.outer_density = 2.0

        # Mesh settings
        self.state.grid_size = 0.04
        self.state.box_size = 1.0
        self.state.inclusion_center_x = 0.5
        self.state.inclusion_center_y = 0.5
        self.state.inclusion_center_z = 0.5

        # Solver settings
        self.state.polynomial_order = 1
        self.state.solver_time_mode = "timesteps"  # "timesteps" or "total_time"
        self.state.number_of_timesteps = 10000
        self.state.total_simulation_time = 1.0  # seconds

        # Receivers
        self.state.sensors_per_face = 25

        # Output intervals
        self.state.output_image_interval = 1000
        self.state.output_data_interval = 1000
        self.state.output_points_interval = 10
        self.state.output_energy_interval = 500
        self.state.save_last_timestep_only = False  # Only save image/data on final step

        # UI state for expansion panels
        self.state.expanded_panels = [0, 1, 2, 3]
        self.state.expanded_fixed_panels = [0, 1, 2, 3, 4, 5]

        # Advanced: custom base config
        self.state.use_custom_base_config = False
        self.state.custom_base_config = ""
        self.state.mesh_file = ""

        # Execution state
        self.state.is_running = False
        self.state.progress_message = ""
        self.state.completed_count = 0
        self.state.failed_count = 0
        self.state.pending_count = 0
        self.state.total_simulations = 0
        self.state.progress_percent = 0
        self.state.existing_param_count = 0
        self.state.existing_completed_count = 0
        self.state.existing_pending_count = 0
        self.state.can_resume_batch = False
        self.state.log_messages = []

        # Existing batches list
        self.state.existing_batches = []
        self._scan_existing_batches()

        # Save preset dialog state
        self.state.show_save_preset_dialog = False
        self.state.save_preset_name = ""
        self.state.save_preset_description = ""
        self.state.save_preset_message = ""

        self.state.change("selected_preset")(self._on_preset_change)
        self.state.change("batch_name")(self._on_batch_name_change)
        self.state.change("batch_dir")(self._on_batch_name_change)

        self._update_existing_count()

    def _on_batch_name_change(self, **kwargs):
        """Update existing count when batch name changes."""
        self._update_existing_count()

    def _scan_existing_batches(self):
        """Scan for all existing batches in the data directory."""
        base_dir = (
            Path(self.state.batch_dir) if self.state.batch_dir else DEFAULT_DATA_DIR
        )

        batches = []
        if base_dir.exists():
            for d in sorted(base_dir.iterdir()):
                if not d.is_dir():
                    continue

                param_dir = d / "parameter_files"
                sim_dir = d / "simulations"
                metadata_file = d / "batch_metadata.toml"

                # Only include directories that look like batches (have parameter files)
                if not param_dir.exists():
                    continue

                param_count = len(list(param_dir.glob("*.toml")))
                if param_count == 0:
                    continue

                # Count completed simulations
                completed_count = 0
                if sim_dir.exists():
                    for sim in sim_dir.iterdir():
                        if sim.is_dir() and (sim / "sensor_data.pkl").exists():
                            completed_count += 1

                pending_count = param_count - completed_count
                is_complete = pending_count == 0
                can_resume = metadata_file.exists() and pending_count > 0

                batches.append(
                    {
                        "name": d.name,
                        "total": param_count,
                        "completed": completed_count,
                        "pending": pending_count,
                        "is_complete": is_complete,
                        "can_resume": can_resume,
                    }
                )

        self.state.existing_batches = batches

    def _select_existing_batch(self, batch_name: str):
        """Select an existing batch by name."""
        self.state.batch_name = batch_name
        self._update_existing_count()

    def _on_batch_click(self, batch_name: str):
        """Handle click on an existing batch in the list."""
        self.state.batch_name = batch_name
        self._update_existing_count()

    def _update_existing_count(self):
        """Count existing parameter files and completed simulations."""
        batch_dir = self._get_batch_output_dir()
        param_dir = batch_dir / "parameter_files"
        sim_dir = batch_dir / "simulations"
        metadata_file = batch_dir / "batch_metadata.toml"

        # Count parameter files
        if param_dir.exists():
            param_count = len(list(param_dir.glob("*.toml")))
            self.state.existing_param_count = param_count
        else:
            param_count = 0
            self.state.existing_param_count = 0

        # Count completed simulations (those with sensor_data.pkl)
        completed_count = 0
        if sim_dir.exists():
            for d in sim_dir.iterdir():
                if d.is_dir() and (d / "sensor_data.pkl").exists():
                    completed_count += 1

        self.state.existing_completed_count = completed_count
        self.state.existing_pending_count = param_count - completed_count

        # Can resume if we have metadata and pending simulations
        self.state.can_resume_batch = (
            metadata_file.exists() and param_count > 0 and completed_count < param_count
        )

    def _on_preset_change(self, selected_preset, **kwargs):
        """Populate all fields from the selected preset."""
        if not selected_preset or selected_preset not in self._presets:
            return

        preset = self._presets[selected_preset]
        inc = preset.inclusion
        cubes = preset.cubes

        # Batch settings
        self.state.batch_name = preset.name
        self.state.batch_description = preset.description
        self.state.num_samples = preset.default_num_samples

        # Material properties
        self.state.wave_speed_min = inc.wave_speed_range[0]
        self.state.wave_speed_max = inc.wave_speed_range[1]
        self.state.density_min = inc.density_range[0]
        self.state.density_max = inc.density_range[1]

        # Scaling ranges
        self.state.scaling_x_min = inc.scaling_range[0][0]
        self.state.scaling_x_max = inc.scaling_range[0][1]
        self.state.scaling_y_min = inc.scaling_range[1][0]
        self.state.scaling_y_max = inc.scaling_range[1][1]
        self.state.scaling_z_min = inc.scaling_range[2][0]
        self.state.scaling_z_max = inc.scaling_range[2][1]

        # Behavior
        self.state.allow_rotation = inc.allow_rotation
        self.state.allow_movement = inc.allow_movement

        # Inclusion type
        if inc.is_sphere:
            self.state.inclusion_type = "sphere"
        elif inc.is_multi_cubes:
            self.state.inclusion_type = "multi_cubes"
        elif inc.is_cube_in_ellipsoid:
            self.state.inclusion_type = "cube_in_ellipsoid"
        else:
            self.state.inclusion_type = "ellipsoid"

        # Cube parameters
        self.state.cube_quantity_min = cubes.quantity_range[0]
        self.state.cube_quantity_max = cubes.quantity_range[1]
        self.state.cube_width_min = cubes.width_range[0]
        self.state.cube_width_max = cubes.width_range[1]

        # Geometry
        self.state.boundary_buffer = preset.boundary_buffer

        # Load optional fixed parameters from preset
        if preset.sources:
            self.state.source_count = preset.sources.number
            self.state.all_source_frequency = preset.sources.frequency
            self.state.all_source_amplitude = preset.sources.amplitude
            self.state.all_source_radius = preset.sources.radius
            self._apply_to_all_sources()

        if preset.outer_material:
            self.state.outer_wave_speed = preset.outer_material.wave_speed
            self.state.outer_density = preset.outer_material.density

        if preset.mesh:
            self.state.grid_size = preset.mesh.grid_size
            self.state.box_size = preset.mesh.box_size

        if preset.solver:
            self.state.polynomial_order = preset.solver.polynomial_order
            if preset.solver.total_time is not None:
                self.state.solver_time_mode = "total_time"
                self.state.total_simulation_time = preset.solver.total_time
            else:
                self.state.solver_time_mode = "timesteps"
                self.state.number_of_timesteps = (
                    preset.solver.number_of_timesteps or 10000
                )

        if preset.receivers:
            self.state.sensors_per_face = preset.receivers.sensors_per_face

        if preset.output:
            self.state.output_image_interval = preset.output.image
            self.state.output_data_interval = preset.output.data
            self.state.output_points_interval = preset.output.points
            self.state.output_energy_interval = preset.output.energy
            self.state.save_last_timestep_only = getattr(
                preset.output, "save_last_timestep_only", False
            )

        self._update_existing_count()

    def build_ui(self):
        """Build the run batch page UI."""
        with v3.VContainer(fluid=True):
            # Top row: Status, Run button, Save preset
            with v3.VRow():
                with v3.VCol(cols=12):
                    self._build_status_and_actions()

            # Expand/Collapse buttons
            with v3.VRow(classes="mb-2"):
                with v3.VCol(cols=12, classes="d-flex justify-end"):
                    v3.VBtn(
                        "Expand All",
                        variant="text",
                        density="compact",
                        prepend_icon="mdi-unfold-more-horizontal",
                        click=self._expand_all_panels,
                        classes="mr-2",
                    )
                    v3.VBtn(
                        "Collapse All",
                        variant="text",
                        density="compact",
                        prepend_icon="mdi-unfold-less-horizontal",
                        click=self._collapse_all_panels,
                    )

            # Parameter columns
            with v3.VRow():
                # Left column: Sweep parameters
                with v3.VCol(cols=12, md=6):
                    self._build_sweep_parameters_panel()
                # Right column: Fixed parameters
                with v3.VCol(cols=12, md=6):
                    self._build_fixed_parameters_panel()

            # Log at the bottom
            with v3.VRow():
                with v3.VCol(cols=12):
                    self._build_log_card()

    def build_toolbar_actions(self):
        """Build toolbar action buttons for this page."""
        pass

    def _expand_all_panels(self):
        """Expand all parameter panels."""
        self.state.expanded_panels = [0, 1, 2, 3]
        self.state.expanded_fixed_panels = [0, 1, 2, 3, 4, 5, 6]

    def _collapse_all_panels(self):
        """Collapse all parameter panels."""
        self.state.expanded_panels = []
        self.state.expanded_fixed_panels = []

    def _build_sweep_parameters_panel(self):
        """Build the sweep parameters panel with collapsible sections."""
        with v3.VExpansionPanels(multiple=True, v_model=("expanded_panels",)):
            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Batch Settings")
                with v3.VExpansionPanelText():
                    self._build_batch_settings_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Material Properties")
                with v3.VExpansionPanelText():
                    self._build_material_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Inclusion Geometry")
                with v3.VExpansionPanelText():
                    self._build_geometry_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Cube Parameters")
                with v3.VExpansionPanelText():
                    self._build_cube_content()

    def _build_fixed_parameters_panel(self):
        """Build the fixed parameters panel with collapsible sections."""
        with v3.VExpansionPanels(
            multiple=True, v_model=("expanded_fixed_panels",), classes="mb-4"
        ):
            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Sources")
                with v3.VExpansionPanelText():
                    self._build_sources_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Outer Material")
                with v3.VExpansionPanelText():
                    self._build_outer_material_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Mesh Settings")
                with v3.VExpansionPanelText():
                    self._build_mesh_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Solver")
                with v3.VExpansionPanelText():
                    self._build_solver_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Receivers")
                with v3.VExpansionPanelText():
                    self._build_receivers_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Output Intervals")
                with v3.VExpansionPanelText():
                    self._build_output_intervals_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Advanced")
                with v3.VExpansionPanelText():
                    self._build_advanced_content()

    def _build_batch_settings_content(self):
        """Build the batch settings content."""
        v3.VSelect(
            v_model=("selected_preset",),
            items=("preset_items",),
            label="Load from Preset",
            hint="Load settings from a preset, then customize below",
            persistent_hint=True,
            clearable=True,
            density="compact",
        )

        v3.VDivider(classes="my-3")

        v3.VTextField(
            v_model=("batch_name",),
            label="Batch Name",
            hint="Identifier for this batch run",
            persistent_hint=True,
            density="compact",
        )

        v3.VAlert(
            v_if=("existing_param_count > 0",),
            type="warning",
            variant="tonal",
            density="compact",
            classes="mt-2",
            text=(
                "'This batch has ' + existing_param_count + ' parameter files: ' + "
                "existing_completed_count + ' completed, ' + "
                "existing_pending_count + ' pending.'",
            ),
        )

        v3.VAlert(
            v_if=("can_resume_batch",),
            type="info",
            variant="tonal",
            density="compact",
            classes="mt-2",
            text="Use 'Resume Batch' to continue from where you left off.",
        )

        v3.VTextarea(
            v_model=("batch_description",),
            label="Description",
            rows=2,
            density="compact",
            classes="mt-3",
        )

        v3.VTextField(
            v_model=("num_samples",),
            label="Number of Samples",
            type="number",
            hint="Parameter files to generate",
            persistent_hint=True,
            density="compact",
            classes="mt-3",
        )

        v3.VTextField(
            v_model=("batch_dir",),
            label="Output Directory",
            hint="Root directory for batch data",
            persistent_hint=True,
            density="compact",
            classes="mt-3",
        )

    def _build_material_content(self):
        """Build the material properties content."""
        html.Div(
            "Inclusion material properties are swept (vary per sample).",
            classes="text-caption mb-3",
        )

        html.Div("Wave Speed Range", classes="text-subtitle-2 mb-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("wave_speed_min",),
                    label="Min",
                    type="number",
                    step="0.1",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("wave_speed_max",),
                    label="Max",
                    type="number",
                    step="0.1",
                    density="compact",
                )

        html.Div("Density Range", classes="text-subtitle-2 mb-2 mt-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("density_min",),
                    label="Min",
                    type="number",
                    step="0.1",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("density_max",),
                    label="Max",
                    type="number",
                    step="0.1",
                    density="compact",
                )

    def _build_geometry_content(self):
        """Build the inclusion geometry content."""
        v3.VSelect(
            v_model=("inclusion_type",),
            items=("inclusion_type_items",),
            label="Inclusion Type",
            density="compact",
        )

        v3.VDivider(classes="my-3")

        html.Div("Scaling Range (X-axis)", classes="text-subtitle-2 mb-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("scaling_x_min",),
                    label="Min",
                    type="number",
                    step="0.01",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("scaling_x_max",),
                    label="Max",
                    type="number",
                    step="0.01",
                    density="compact",
                )

        html.Div("Scaling Range (Y-axis)", classes="text-subtitle-2 mb-2 mt-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("scaling_y_min",),
                    label="Min",
                    type="number",
                    step="0.01",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("scaling_y_max",),
                    label="Max",
                    type="number",
                    step="0.01",
                    density="compact",
                )

        html.Div("Scaling Range (Z-axis)", classes="text-subtitle-2 mb-2 mt-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("scaling_z_min",),
                    label="Min",
                    type="number",
                    step="0.01",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("scaling_z_max",),
                    label="Max",
                    type="number",
                    step="0.01",
                    density="compact",
                )

        v3.VDivider(classes="my-3")

        v3.VCheckbox(
            v_model=("allow_rotation",),
            label="Allow Rotation",
            density="compact",
            hide_details=True,
        )

        v3.VCheckbox(
            v_model=("allow_movement",),
            label="Allow Movement from Center",
            density="compact",
            hide_details=True,
        )

        v3.VTextField(
            v_model=("boundary_buffer",),
            label="Boundary Buffer",
            type="number",
            step="0.01",
            hint="Minimum distance from domain boundary",
            persistent_hint=True,
            density="compact",
            classes="mt-3",
        )

    def _build_cube_content(self):
        """Build the cube parameters content."""
        html.Div(
            "For multi-cube and cube-in-ellipsoid types.",
            classes="text-caption mb-3",
        )

        html.Div("Cube Quantity Range", classes="text-subtitle-2 mb-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("cube_quantity_min",),
                    label="Min",
                    type="number",
                    step="1",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("cube_quantity_max",),
                    label="Max",
                    type="number",
                    step="1",
                    density="compact",
                )

        html.Div("Cube Width Range", classes="text-subtitle-2 mb-2 mt-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("cube_width_min",),
                    label="Min",
                    type="number",
                    step="0.01",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("cube_width_max",),
                    label="Max",
                    type="number",
                    step="0.01",
                    density="compact",
                )

    def _build_advanced_content(self):
        """Build the advanced settings content."""
        v3.VCheckbox(
            v_model=("use_custom_base_config",),
            label="Use Custom Base Config",
            density="compact",
            hide_details=True,
        )

        v3.VTextField(
            v_if=("use_custom_base_config",),
            v_model=("custom_base_config",),
            label="Base Config Path",
            hint="Path to custom TOML config",
            persistent_hint=True,
            density="compact",
            classes="mt-2",
        )

        v3.VTextField(
            v_model=("mesh_file",),
            label="Mesh File (optional)",
            hint="Use pre-generated .msh file",
            persistent_hint=True,
            density="compact",
            classes="mt-3",
        )

    def _build_sources_content(self):
        """Build the sources configuration content."""
        html.Div(
            "Configure acoustic sources on domain boundary faces.",
            classes="text-caption mb-3",
        )

        html.Div("Apply to All Sources", classes="text-subtitle-2 mb-2")
        with v3.VRow(dense=True):
            with v3.VCol(cols=4):
                v3.VTextField(
                    v_model=("all_source_frequency",),
                    label="Frequency",
                    type="number",
                    step="0.5",
                    density="compact",
                    suffix="Hz",
                )
            with v3.VCol(cols=4):
                v3.VTextField(
                    v_model=("all_source_amplitude",),
                    label="Amplitude",
                    type="number",
                    step="0.1",
                    density="compact",
                )
            with v3.VCol(cols=4):
                v3.VTextField(
                    v_model=("all_source_radius",),
                    label="Radius",
                    type="number",
                    step="0.01",
                    density="compact",
                )

        v3.VBtn(
            "Apply to All",
            color="primary",
            variant="outlined",
            density="compact",
            click=self._apply_to_all_sources,
            classes="mb-3",
        )

        v3.VDivider(classes="my-3")

        v3.VTextField(
            v_model=("source_count",),
            label="Number of Sources",
            type="number",
            step="1",
            density="compact",
            hint="Typically 6 (one per face)",
            persistent_hint=True,
        )

    def _apply_to_all_sources(self):
        """Apply the quick-set values to all sources."""
        freq = float(self.state.all_source_frequency)
        amp = float(self.state.all_source_amplitude)
        radius = float(self.state.all_source_radius)

        updated_sources = []
        for src in self.state.sources:
            updated_sources.append(
                {
                    "center": src["center"],
                    "frequency": freq,
                    "amplitude": amp,
                    "radius": radius,
                }
            )
        self.state.sources = updated_sources

    def _build_outer_material_content(self):
        """Build the outer material configuration content."""
        html.Div(
            "Background material properties (same for all samples).",
            classes="text-caption mb-3",
        )

        v3.VTextField(
            v_model=("outer_wave_speed",),
            label="Wave Speed",
            type="number",
            step="0.1",
            density="compact",
        )

        v3.VTextField(
            v_model=("outer_density",),
            label="Density",
            type="number",
            step="0.1",
            density="compact",
            classes="mt-2",
        )

    def _build_mesh_content(self):
        """Build the mesh settings content."""
        html.Div(
            "Mesh generation parameters.",
            classes="text-caption mb-3",
        )

        v3.VTextField(
            v_model=("grid_size",),
            label="Grid Size",
            type="number",
            step="0.01",
            density="compact",
            hint="Target element size",
            persistent_hint=True,
        )

        v3.VTextField(
            v_model=("box_size",),
            label="Box Size",
            type="number",
            step="0.1",
            density="compact",
            hint="Domain size (cubic)",
            persistent_hint=True,
            classes="mt-2",
        )

        html.Div("Inclusion Center", classes="text-subtitle-2 mb-2 mt-3")
        with v3.VRow(dense=True):
            with v3.VCol(cols=4):
                v3.VTextField(
                    v_model=("inclusion_center_x",),
                    label="X",
                    type="number",
                    step="0.1",
                    density="compact",
                )
            with v3.VCol(cols=4):
                v3.VTextField(
                    v_model=("inclusion_center_y",),
                    label="Y",
                    type="number",
                    step="0.1",
                    density="compact",
                )
            with v3.VCol(cols=4):
                v3.VTextField(
                    v_model=("inclusion_center_z",),
                    label="Z",
                    type="number",
                    step="0.1",
                    density="compact",
                )

    def _build_solver_content(self):
        """Build the solver settings content."""
        html.Div(
            "DG solver configuration.",
            classes="text-caption mb-3",
        )

        v3.VTextField(
            v_model=("polynomial_order",),
            label="Polynomial Order",
            type="number",
            step="1",
            density="compact",
            hint="Higher = more accurate but slower",
            persistent_hint=True,
        )

        # Time mode selection
        with v3.VRadioGroup(
            v_model=("solver_time_mode",),
            inline=True,
            density="compact",
            classes="mt-2",
        ):
            v3.VRadio(label="Timesteps", value="timesteps")
            v3.VRadio(label="Final Time", value="total_time")

        # Timesteps input (shown when mode is "timesteps")
        v3.VTextField(
            v_model=("number_of_timesteps",),
            v_show="solver_time_mode === 'timesteps'",
            label="Number of Timesteps",
            type="number",
            step="1000",
            density="compact",
            hint="Total simulation timesteps",
            persistent_hint=True,
        )

        # Total time input (shown when mode is "total_time")
        v3.VTextField(
            v_model=("total_simulation_time",),
            v_show="solver_time_mode === 'total_time'",
            label="Final Simulation Time (seconds)",
            type="number",
            step="0.1",
            density="compact",
            hint="Timesteps computed after mesh generation",
            persistent_hint=True,
        )

    def _build_receivers_content(self):
        """Build the receivers configuration content."""
        html.Div(
            "Sensor/receiver placement.",
            classes="text-caption mb-3",
        )

        v3.VTextField(
            v_model=("sensors_per_face",),
            label="Sensors per Face",
            type="number",
            step="1",
            density="compact",
            hint="Number of sensors on each domain face",
            persistent_hint=True,
        )

    def _build_output_intervals_content(self):
        """Build the output intervals content."""
        html.Div(
            "How often to write outputs during simulation.",
            classes="text-caption mb-3",
        )

        # Checkbox to save only last timestep for image/data
        v3.VCheckbox(
            v_model=("save_last_timestep_only",),
            label="Only save last timestep (image/data)",
            density="compact",
            hint="Saves image and data only on the final timestep",
            persistent_hint=True,
        )

        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("output_image_interval",),
                    label="Image Interval",
                    type="number",
                    step="100",
                    density="compact",
                    disabled=("save_last_timestep_only",),
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("output_data_interval",),
                    label="Data Interval",
                    type="number",
                    step="100",
                    density="compact",
                    disabled=("save_last_timestep_only",),
                )

        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("output_points_interval",),
                    label="Points Interval",
                    type="number",
                    step="10",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("output_energy_interval",),
                    label="Energy Interval",
                    type="number",
                    step="100",
                    density="compact",
                )

    def _build_status_and_actions(self):
        """Build the status display and action buttons at the top."""
        with v3.VCard(classes="mb-4"):
            with v3.VCardText():
                with v3.VRow(dense=True):
                    # Left side: Existing batches list
                    with v3.VCol(cols=12, md=5):
                        html.Div("Existing Batches", classes="text-subtitle-1 mb-2")
                        with v3.VSheet(
                            color="grey-darken-4",
                            rounded=True,
                            classes="pa-2",
                            style="max-height: 150px; overflow-y: auto;",
                        ):
                            # Refresh button
                            v3.VBtn(
                                "Refresh",
                                variant="text",
                                density="compact",
                                size="small",
                                prepend_icon="mdi-refresh",
                                click=self._scan_existing_batches,
                                classes="mb-2",
                            )

                            # Show message if no batches
                            html.Div(
                                "No batches found",
                                v_if=("existing_batches.length === 0",),
                                classes="text-caption text-grey",
                            )

                            # Batch list
                            with v3.VList(
                                density="compact",
                                v_if=("existing_batches.length > 0",),
                                bg_color="transparent",
                            ):
                                with v3.VListItem(
                                    v_for="batch in existing_batches",
                                    key=("batch.name",),
                                    click=(self._on_batch_click, "[batch.name]"),
                                    disabled=("batch.is_complete",),
                                    classes="pa-1",
                                ):
                                    with v3.VListItemTitle(
                                        classes="d-flex align-center"
                                    ):
                                        # Batch name
                                        html.Span(
                                            "{{ batch.name }}",
                                            classes="text-body-2",
                                        )
                                        v3.VSpacer()
                                        # Status indicator
                                        v3.VIcon(
                                            "mdi-check-circle",
                                            v_if=("batch.is_complete",),
                                            color="success",
                                            size="small",
                                            classes="ml-2",
                                        )
                                        v3.VChip(
                                            v_if=("!batch.is_complete",),
                                            size="x-small",
                                            color="warning",
                                            classes="ml-2",
                                            __properties=[
                                                ("text", "batch.pending + ' pending'")
                                            ],
                                        )

                    # Middle: Status counters
                    with v3.VCol(cols=12, md=4):
                        html.Div("Current Run Status", classes="text-subtitle-1 mb-2")
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=4):
                                with v3.VCard(color="info", variant="tonal"):
                                    with v3.VCardText(classes="text-center pa-2"):
                                        html.Div(
                                            "{{ pending_count }}", classes="text-h5"
                                        )
                                        html.Div("Pending", classes="text-caption")
                            with v3.VCol(cols=4):
                                with v3.VCard(color="success", variant="tonal"):
                                    with v3.VCardText(classes="text-center pa-2"):
                                        html.Div(
                                            "{{ completed_count }}", classes="text-h5"
                                        )
                                        html.Div("Done", classes="text-caption")
                            with v3.VCol(cols=4):
                                with v3.VCard(color="error", variant="tonal"):
                                    with v3.VCardText(classes="text-center pa-2"):
                                        html.Div(
                                            "{{ failed_count }}", classes="text-h5"
                                        )
                                        html.Div("Failed", classes="text-caption")

                    # Right side: Action buttons
                    with v3.VCol(cols=12, md=3):
                        html.Div("Actions", classes="text-subtitle-1 mb-2")
                        v3.VBtn(
                            "Run New Batch",
                            color="primary",
                            block=True,
                            disabled=("is_running || !batch_name",),
                            click=self._run_batch,
                            classes="mb-2",
                        )
                        v3.VBtn(
                            "Resume Batch",
                            color="secondary",
                            block=True,
                            disabled=("is_running || !can_resume_batch",),
                            click=self._resume_batch,
                            classes="mb-2",
                        )
                        v3.VBtn(
                            "Save Preset",
                            variant="outlined",
                            block=True,
                            prepend_icon="mdi-content-save",
                            click=self._open_save_preset_dialog,
                        )

                # Progress bar and message
                v3.VProgressLinear(
                    v_if=("is_running",),
                    model_value=("progress_percent",),
                    color="primary",
                    height=8,
                    classes="mt-3",
                )
                v3.VAlert(
                    v_if=("progress_message",),
                    text=("progress_message",),
                    type="info",
                    density="compact",
                    classes="mt-3",
                )

        # Save preset dialog
        with v3.VDialog(v_model=("show_save_preset_dialog",), max_width="500"):
            with v3.VCard():
                v3.VCardTitle("Save as Preset")
                with v3.VCardText():
                    v3.VTextField(
                        v_model=("save_preset_name",),
                        label="Preset Name",
                        hint="Lowercase with underscores (e.g., my_preset)",
                        persistent_hint=True,
                        density="compact",
                    )
                    v3.VTextField(
                        v_model=("save_preset_description",),
                        label="Description",
                        hint="Brief description of this preset",
                        persistent_hint=True,
                        density="compact",
                        classes="mt-3",
                    )
                    v3.VAlert(
                        v_if=("save_preset_message",),
                        text=("save_preset_message",),
                        type="success",
                        density="compact",
                        classes="mt-3",
                    )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        variant="text",
                        click=self._close_save_preset_dialog,
                    )
                    v3.VBtn(
                        "Save",
                        color="primary",
                        variant="flat",
                        click=self._save_preset,
                        disabled=("!save_preset_name",),
                    )

    def _build_log_card(self):
        """Build the log output card."""
        with v3.VCard():
            v3.VCardTitle("Log")
            with v3.VCardText():
                with v3.VSheet(
                    color="grey-darken-4",
                    rounded=True,
                    classes="pa-3",
                    style="max-height: 250px; overflow-y: auto; font-family: monospace;",
                ):
                    html.Pre(
                        "{{ log_messages.join('\\n') }}",
                        style="margin: 0; white-space: pre-wrap; font-size: 12px;",
                    )

    def _log(self, message: str):
        """Add a message to the log display."""
        self.state.log_messages = [*self.state.log_messages, message]
        logger.info(message)

    def _build_parameter_space(self) -> ParameterSpace:
        """Build ParameterSpace from current UI state."""
        return ParameterSpace(
            inclusion_density=ParameterRange(
                float(self.state.density_min),
                float(self.state.density_max),
            ),
            inclusion_speed=ParameterRange(
                float(self.state.wave_speed_min),
                float(self.state.wave_speed_max),
            ),
            inclusion_scaling=ParameterRange(
                float(self.state.scaling_x_min),
                float(self.state.scaling_x_max),
            ),
            cube_width=ParameterRange(
                float(self.state.cube_width_min),
                float(self.state.cube_width_max),
            ),
            cube_count=(
                int(self.state.cube_quantity_min),
                int(self.state.cube_quantity_max),
            ),
            boundary_buffer=float(self.state.boundary_buffer),
        )

    def _build_simulation_config(self) -> SimulationConfig:
        """Build SimulationConfig from current UI state."""
        source_count = int(self.state.source_count)
        sources = self.state.sources[:source_count]
        source_config = SourceConfig(
            number=source_count,
            centers=[s["center"] for s in sources],
            radii=[s["radius"] for s in sources],
            amplitudes=[s["amplitude"] for s in sources],
            frequencies=[s["frequency"] for s in sources],
        )

        return SimulationConfig(
            sources=source_config,
            outer_material=OuterMaterialConfig(
                wave_speed=float(self.state.outer_wave_speed),
                density=float(self.state.outer_density),
            ),
            mesh=MeshConfig(
                grid_size=float(self.state.grid_size),
                box_size=float(self.state.box_size),
                inclusion_center=[
                    float(self.state.inclusion_center_x),
                    float(self.state.inclusion_center_y),
                    float(self.state.inclusion_center_z),
                ],
            ),
            solver=SolverConfig(
                polynomial_order=int(self.state.polynomial_order),
                number_of_timesteps=(
                    int(self.state.number_of_timesteps)
                    if self.state.solver_time_mode == "timesteps"
                    else None
                ),
                total_time=(
                    float(self.state.total_simulation_time)
                    if self.state.solver_time_mode == "total_time"
                    else None
                ),
            ),
            receivers=ReceiverConfig(
                sensors_per_face=int(self.state.sensors_per_face),
            ),
            output=OutputConfig(
                image=int(self.state.output_image_interval),
                data=int(self.state.output_data_interval),
                points=int(self.state.output_points_interval),
                energy=int(self.state.output_energy_interval),
                save_last_timestep_only=bool(self.state.save_last_timestep_only),
            ),
            batch_name=self.state.batch_name,
            batch_description=self.state.batch_description,
            num_samples=int(self.state.num_samples),
        )

    def _open_save_preset_dialog(self):
        """Open the save preset dialog."""
        # Pre-fill with batch name if available
        self.state.save_preset_name = self.state.batch_name.lower().replace(" ", "_")
        self.state.save_preset_description = self.state.batch_description
        self.state.save_preset_message = ""
        self.state.show_save_preset_dialog = True

    def _close_save_preset_dialog(self):
        """Close the save preset dialog."""
        self.state.show_save_preset_dialog = False

    def _get_solver_time_toml(self) -> str:
        """Get the solver time configuration as TOML string."""
        if self.state.solver_time_mode == "total_time":
            return f"total_time = {float(self.state.total_simulation_time)}"
        else:
            return f"number_of_timesteps = {int(self.state.number_of_timesteps)}"

    def _save_preset(self):
        """Save current parameters as a preset."""
        from importlib import resources

        preset_name = self.state.save_preset_name.lower().replace(" ", "_")
        description = self.state.save_preset_description or f"Preset: {preset_name}"

        # Determine inclusion type flags
        inc_type = self.state.inclusion_type
        is_sphere = inc_type == "sphere"
        is_multi_cubes = inc_type == "multi_cubes"
        is_cube_in_ellipsoid = inc_type == "cube_in_ellipsoid"

        preset_content = f'''# {preset_name} Preset
# {description}

[preset]
name = "{preset_name}"
description = "{description}"
base_config = "base_parameters.toml"
default_num_samples = {int(self.state.num_samples)}

[sweep.inclusion]
# Material property ranges
wave_speed_range = [{float(self.state.wave_speed_min)}, {float(self.state.wave_speed_max)}]
density_range = [{float(self.state.density_min)}, {float(self.state.density_max)}]

# Geometry ranges [min, max] for each axis
scaling_range = [
    [{float(self.state.scaling_x_min)}, {float(self.state.scaling_x_max)}],
    [{float(self.state.scaling_y_min)}, {float(self.state.scaling_y_max)}],
    [{float(self.state.scaling_z_min)}, {float(self.state.scaling_z_max)}],
]

# Inclusion behavior
allow_rotation = {str(self.state.allow_rotation).lower()}
allow_movement = {str(self.state.allow_movement).lower()}

# Inclusion type
is_sphere = {str(is_sphere).lower()}
is_ellipsoid_of_revolution = false
is_multi_cubes = {str(is_multi_cubes).lower()}
is_cube_in_ellipsoid = {str(is_cube_in_ellipsoid).lower()}

[sweep.cubes]
quantity_range = [{int(self.state.cube_quantity_min)}, {int(self.state.cube_quantity_max)}]
width_range = [{float(self.state.cube_width_min)}, {float(self.state.cube_width_max)}]

[sweep.geometry]
boundary_buffer = {float(self.state.boundary_buffer)}

[fixed.sources]
number = {int(self.state.source_count)}
frequency = {float(self.state.all_source_frequency)}
amplitude = {float(self.state.all_source_amplitude)}
radius = {float(self.state.all_source_radius)}

[fixed.outer_material]
wave_speed = {float(self.state.outer_wave_speed)}
density = {float(self.state.outer_density)}

[fixed.mesh]
grid_size = {float(self.state.grid_size)}
box_size = {float(self.state.box_size)}

[fixed.solver]
polynomial_order = {int(self.state.polynomial_order)}
{self._get_solver_time_toml()}

[fixed.receivers]
sensors_per_face = {int(self.state.sensors_per_face)}

[fixed.output]
image = {int(self.state.output_image_interval)}
data = {int(self.state.output_data_interval)}
points = {int(self.state.output_points_interval)}
energy = {int(self.state.output_energy_interval)}
save_last_timestep_only = {str(bool(self.state.save_last_timestep_only)).lower()}
'''

        # Get the presets directory path
        presets_pkg = resources.files("sbimaging.config.presets")
        with resources.as_file(presets_pkg) as presets_dir:
            preset_path = presets_dir / f"{preset_name}.toml"
            with open(preset_path, "w") as f:
                f.write(preset_content)
            logger.info(f"Saved preset to {preset_path}")

        self.state.save_preset_message = f"Saved preset: {preset_name}"

        # Reload presets to include the new one
        self._presets = self._load_presets()
        preset_items = [{"title": "-- Start from scratch --", "value": ""}]
        for name, preset in self._presets.items():
            preset_items.append({"title": preset.name, "value": name})
        self.state.preset_items = preset_items

    def _get_batch_output_dir(self) -> Path:
        """Compute the batch output directory."""
        base_dir = (
            Path(self.state.batch_dir) if self.state.batch_dir else DEFAULT_DATA_DIR
        )
        batch_name = self.state.batch_name or "unnamed_batch"
        return base_dir / batch_name

    def _save_batch_config(self, batch_dir: Path):
        """Save the complete batch configuration to a TOML file."""
        config = {
            "batch": {
                "name": self.state.batch_name,
                "description": self.state.batch_description,
                "num_samples": int(self.state.num_samples),
                "inclusion_type": self.state.inclusion_type,
            },
            "sweep_parameters": {
                "material": {
                    "wave_speed_range": [
                        float(self.state.wave_speed_min),
                        float(self.state.wave_speed_max),
                    ],
                    "density_range": [
                        float(self.state.density_min),
                        float(self.state.density_max),
                    ],
                },
                "geometry": {
                    "scaling_x_range": [
                        float(self.state.scaling_x_min),
                        float(self.state.scaling_x_max),
                    ],
                    "scaling_y_range": [
                        float(self.state.scaling_y_min),
                        float(self.state.scaling_y_max),
                    ],
                    "scaling_z_range": [
                        float(self.state.scaling_z_min),
                        float(self.state.scaling_z_max),
                    ],
                    "allow_rotation": self.state.allow_rotation,
                    "allow_movement": self.state.allow_movement,
                    "boundary_buffer": float(self.state.boundary_buffer),
                },
                "cubes": {
                    "quantity_range": [
                        int(self.state.cube_quantity_min),
                        int(self.state.cube_quantity_max),
                    ],
                    "width_range": [
                        float(self.state.cube_width_min),
                        float(self.state.cube_width_max),
                    ],
                },
            },
            "fixed_parameters": {
                "sources": {
                    "number": int(self.state.source_count),
                    "frequency": float(self.state.all_source_frequency),
                    "amplitude": float(self.state.all_source_amplitude),
                    "radius": float(self.state.all_source_radius),
                    "centers": [s["center"] for s in self.state.sources],
                },
                "outer_material": {
                    "wave_speed": float(self.state.outer_wave_speed),
                    "density": float(self.state.outer_density),
                },
                "mesh": {
                    "grid_size": float(self.state.grid_size),
                    "box_size": float(self.state.box_size),
                    "inclusion_center": [
                        float(self.state.inclusion_center_x),
                        float(self.state.inclusion_center_y),
                        float(self.state.inclusion_center_z),
                    ],
                },
                "solver": {
                    "polynomial_order": int(self.state.polynomial_order),
                    "number_of_timesteps": int(self.state.number_of_timesteps),
                },
                "receivers": {
                    "sensors_per_face": int(self.state.sensors_per_face),
                },
                "output_intervals": {
                    "image": int(self.state.output_image_interval),
                    "data": int(self.state.output_data_interval),
                    "points": int(self.state.output_points_interval),
                    "energy": int(self.state.output_energy_interval),
                    "save_last_timestep_only": bool(self.state.save_last_timestep_only),
                },
            },
        }

        config_path = batch_dir / "batch_config.toml"
        with open(config_path, "w") as f:
            toml.dump(config, f)
        logger.info(f"Saved batch configuration to {config_path}")

    def _run_batch(self):
        """Start the simulation batch execution."""
        asyncio.create_task(self._run_batch_async())

    async def _run_batch_async(self):
        """Execute the simulation batch asynchronously."""
        batch_dir = self._get_batch_output_dir()
        if not batch_dir.exists():
            batch_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"Created batch directory: {batch_dir}")

        # Save batch configuration
        self._save_batch_config(batch_dir)

        self.state.is_running = True
        self.state.progress_message = "Starting batch..."
        self.state.log_messages = []
        self.state.pending_count = 0
        self.state.completed_count = 0
        self.state.failed_count = 0
        self.state.total_simulations = 0
        self.state.progress_percent = 0

        self._log(f"Batch: {self.state.batch_name}")
        if self.state.batch_description:
            self._log(f"Description: {self.state.batch_description}")
        self._log(f"Output directory: {batch_dir}")
        self._log(f"Samples: {self.state.num_samples}")
        self._log(f"Inclusion type: {self.state.inclusion_type}")
        self._log(
            f"Wave speed: [{self.state.wave_speed_min}, {self.state.wave_speed_max}]"
        )
        self._log(f"Density: [{self.state.density_min}, {self.state.density_max}]")

        mesh_file = Path(self.state.mesh_file) if self.state.mesh_file else None
        num_samples = int(self.state.num_samples)

        if self.state.use_custom_base_config and self.state.custom_base_config:
            base_config = Path(self.state.custom_base_config)
            self._log(f"Using custom base config: {base_config}")
        else:
            try:
                base_config = get_base_config_path()
                self._log("Using default base config")
            except FileNotFoundError as e:
                self._log(f"Error: {e}")
                self.state.is_running = False
                return

        parameter_space = self._build_parameter_space()
        simulation_config = self._build_simulation_config()

        if mesh_file:
            self._log(f"Mesh file: {mesh_file}")

        self._log(f"Sources: {self.state.source_count}")
        self._log(
            f"Outer material: speed={self.state.outer_wave_speed}, density={self.state.outer_density}"
        )
        self._log(
            f"Mesh: grid_size={self.state.grid_size}, box_size={self.state.box_size}"
        )
        self._log(
            f"Solver: poly_order={self.state.polynomial_order}, timesteps={self.state.number_of_timesteps}"
        )

        # Allow UI to update before starting heavy computation
        await asyncio.sleep(0)

        try:
            self.state.progress_message = "Scanning for pending simulations..."
            await asyncio.sleep(0)

            planner = BatchPlanner(batch_dir)
            planner.compute_mesh_hashes()
            pending = planner.find_pending_simulations()
            self.state.pending_count = len(pending)
            self._log(f"Found {len(pending)} existing pending simulations")
            await asyncio.sleep(0)

            self.state.progress_message = "Running simulations..."

            # Get the event loop and server for state updates
            loop = asyncio.get_event_loop()
            server = self.server

            def progress_callback(pending: int, completed: int, failed: int):
                """Update UI state from batch executor (called from thread pool)."""
                total = self.state.total_simulations
                if total > 0:
                    # 10% for mesh generation, 90% for simulations
                    sim_progress = (completed + failed) / total * 90
                    percent = 10 + sim_progress
                else:
                    percent = 10

                def update_state():
                    self.state.pending_count = pending
                    self.state.completed_count = completed
                    self.state.failed_count = failed
                    self.state.progress_percent = int(percent)
                    self.state.progress_message = (
                        f"Running simulations... ({completed + failed}/{total})"
                    )
                    # Force state push to client
                    server.state.flush()

                # Schedule state update on the main event loop
                loop.call_soon_threadsafe(update_state)

            def mesh_progress_callback(generated: int, total: int):
                """Update UI during mesh generation (called from thread pool)."""
                # Mesh generation is 10% of total progress
                percent = (generated / total * 10) if total > 0 else 0

                def update_state():
                    self.state.progress_message = (
                        f"Generating meshes... ({generated}/{total})"
                    )
                    self.state.progress_percent = int(percent)
                    server.state.flush()

                loop.call_soon_threadsafe(update_state)

            def total_simulations_callback(total: int):
                """Set total simulations count (called from thread pool)."""

                def update_state():
                    self.state.total_simulations = total
                    self.state.pending_count = total
                    self.state.progress_message = f"Running simulations... (0/{total})"
                    self.state.progress_percent = 10
                    server.state.flush()

                loop.call_soon_threadsafe(update_state)

            # Run the batch in a thread pool to avoid blocking the event loop
            completed, failed = await loop.run_in_executor(
                None,
                lambda: run_batch(
                    batch_dir=batch_dir,
                    base_config_path=base_config,
                    num_samples=num_samples,
                    mesh_file=mesh_file,
                    parameter_space=parameter_space,
                    geometry_type=self.state.inclusion_type,
                    simulation_config=simulation_config,
                    progress_callback=progress_callback,
                    mesh_progress_callback=mesh_progress_callback,
                    total_simulations_callback=total_simulations_callback,
                ),
            )

            self.state.completed_count = completed
            self.state.failed_count = failed
            self.state.pending_count = 0
            self.state.progress_percent = 100
            self.state.progress_message = (
                f"Complete: {completed} succeeded, {failed} failed"
            )
            self._log(f"Batch finished: {completed} completed, {failed} failed")

        except Exception as e:
            self.state.progress_message = f"Error: {e}"
            self._log(f"Error: {e}")
            logger.exception("Batch execution failed")

        finally:
            self.state.is_running = False
            self._update_existing_count()
            self._scan_existing_batches()

    def _resume_batch(self):
        """Resume a batch from where it left off."""
        asyncio.create_task(self._resume_batch_async())

    async def _resume_batch_async(self):
        """Resume batch execution asynchronously."""
        batch_dir = self._get_batch_output_dir()

        self.state.is_running = True
        self.state.progress_message = "Resuming batch..."
        self.state.log_messages = []
        self.state.pending_count = 0
        self.state.completed_count = 0
        self.state.failed_count = 0
        self.state.total_simulations = 0
        self.state.progress_percent = 0

        self._log(f"Resuming batch: {self.state.batch_name}")
        self._log(f"Output directory: {batch_dir}")

        # Allow UI to update
        await asyncio.sleep(0)

        try:
            # Get the event loop and server for state updates
            loop = asyncio.get_event_loop()
            server = self.server

            def progress_callback(pending: int, completed: int, failed: int):
                """Update UI state from batch executor (called from thread pool)."""
                total = self.state.total_simulations
                if total > 0:
                    percent = (completed + failed) / total * 100
                else:
                    percent = 0

                def update_state():
                    self.state.pending_count = pending
                    self.state.completed_count = completed
                    self.state.failed_count = failed
                    self.state.progress_percent = int(percent)
                    self.state.progress_message = (
                        f"Running simulations... ({completed + failed}/{total})"
                    )
                    server.state.flush()

                loop.call_soon_threadsafe(update_state)

            def total_simulations_callback(total: int):
                """Set total simulations count (called from thread pool)."""

                def update_state():
                    self.state.total_simulations = total
                    self.state.pending_count = total
                    self.state.progress_message = f"Resuming: {total} pending"
                    self.state.progress_percent = 0
                    server.state.flush()

                loop.call_soon_threadsafe(update_state)

            self._log("Scanning for pending simulations...")

            # Run the resume in a thread pool
            completed, failed = await loop.run_in_executor(
                None,
                lambda: resume_batch(
                    batch_dir=batch_dir,
                    progress_callback=progress_callback,
                    total_simulations_callback=total_simulations_callback,
                ),
            )

            self.state.completed_count = completed
            self.state.failed_count = failed
            self.state.pending_count = 0
            self.state.progress_percent = 100
            self.state.progress_message = (
                f"Complete: {completed} succeeded, {failed} failed"
            )
            self._log(f"Batch resumed: {completed} completed, {failed} failed")

        except FileNotFoundError as e:
            self.state.progress_message = f"Cannot resume: {e}"
            self._log(f"Error: {e}")
            logger.error(f"Cannot resume batch: {e}")

        except Exception as e:
            self.state.progress_message = f"Error: {e}"
            self._log(f"Error: {e}")
            logger.exception("Batch resume failed")

        finally:
            self.state.is_running = False
            self._update_existing_count()
            self._scan_existing_batches()
