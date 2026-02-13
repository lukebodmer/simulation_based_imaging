"""Run Simulation Batch page."""

from pathlib import Path

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from sbimaging.batch.executor import run_batch
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
        self.state.number_of_timesteps = 10000

        # Receivers
        self.state.sensors_per_face = 25

        # Output intervals
        self.state.output_image_interval = 1000
        self.state.output_data_interval = 1000
        self.state.output_points_interval = 10
        self.state.output_energy_interval = 500

        # UI state for expansion panels
        self.state.expanded_panels = [0, 1, 2]

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
        self.state.existing_param_count = 0
        self.state.log_messages = []

        self.state.change("selected_preset")(self._on_preset_change)
        self.state.change("batch_name")(self._on_batch_name_change)
        self.state.change("batch_dir")(self._on_batch_name_change)

        self._update_existing_count()

    def _on_batch_name_change(self, **kwargs):
        """Update existing count when batch name changes."""
        self._update_existing_count()

    def _update_existing_count(self):
        """Count existing parameter files in the batch directory."""
        batch_dir = self._get_batch_output_dir()
        param_dir = batch_dir / "parameter_files"
        if param_dir.exists():
            count = len(list(param_dir.glob("*.toml")))
            self.state.existing_param_count = count
        else:
            self.state.existing_param_count = 0

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
            self.state.number_of_timesteps = preset.solver.number_of_timesteps

        if preset.receivers:
            self.state.sensors_per_face = preset.receivers.sensors_per_face

        if preset.output:
            self.state.output_image_interval = preset.output.image
            self.state.output_data_interval = preset.output.data
            self.state.output_points_interval = preset.output.points
            self.state.output_energy_interval = preset.output.energy

        self._update_existing_count()

    def build_ui(self):
        """Build the run batch page UI."""
        with v3.VContainer(fluid=True):
            with v3.VRow():
                # Left column: Sweep parameters
                with v3.VCol(cols=12, md=6):
                    self._build_sweep_parameters_panel()
                # Right column: Fixed parameters + Status
                with v3.VCol(cols=12, md=6):
                    self._build_fixed_parameters_panel()
                    self._build_status_card()

            with v3.VRow():
                with v3.VCol(cols=12):
                    self._build_log_card()

    def build_toolbar_actions(self):
        """Build toolbar action buttons for this page."""
        v3.VBtn(
            "Run Batch",
            color="primary",
            disabled=("is_running || !batch_name",),
            click=self._run_batch,
        )

    def _build_sweep_parameters_panel(self):
        """Build the sweep parameters panel with collapsible sections."""
        with v3.VExpansionPanels(multiple=True, v_model=("expanded_panels",)):
            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Batch Settings")
                with v3.VExpansionPanelText():
                    self._build_batch_settings_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Material Properties (Swept)")
                with v3.VExpansionPanelText():
                    self._build_material_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Inclusion Geometry (Swept)")
                with v3.VExpansionPanelText():
                    self._build_geometry_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Cube Parameters")
                with v3.VExpansionPanelText():
                    self._build_cube_content()

    def _build_fixed_parameters_panel(self):
        """Build the fixed parameters panel with collapsible sections."""
        with v3.VExpansionPanels(multiple=True, classes="mb-4"):
            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Sources (Fixed)")
                with v3.VExpansionPanelText():
                    self._build_sources_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Outer Material (Fixed)")
                with v3.VExpansionPanelText():
                    self._build_outer_material_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Mesh Settings (Fixed)")
                with v3.VExpansionPanelText():
                    self._build_mesh_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Solver (Fixed)")
                with v3.VExpansionPanelText():
                    self._build_solver_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Receivers (Fixed)")
                with v3.VExpansionPanelText():
                    self._build_receivers_content()

            with v3.VExpansionPanel():
                v3.VExpansionPanelTitle("Output Intervals (Fixed)")
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
                "'This batch already has ' + existing_param_count + ' parameter files.'",
            ),
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

        v3.VTextField(
            v_model=("number_of_timesteps",),
            label="Number of Timesteps",
            type="number",
            step="1000",
            density="compact",
            hint="Total simulation timesteps",
            persistent_hint=True,
            classes="mt-2",
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

        with v3.VRow(dense=True):
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("output_image_interval",),
                    label="Image Interval",
                    type="number",
                    step="100",
                    density="compact",
                )
            with v3.VCol(cols=6):
                v3.VTextField(
                    v_model=("output_data_interval",),
                    label="Data Interval",
                    type="number",
                    step="100",
                    density="compact",
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

    def _build_status_card(self):
        """Build the status display card."""
        with v3.VCard():
            v3.VCardTitle("Status")
            with v3.VCardText():
                v3.VProgressLinear(
                    indeterminate=("is_running",),
                    color="primary",
                    height=6,
                )
                with v3.VRow(classes="mt-3", dense=True):
                    with v3.VCol(cols=4):
                        with v3.VCard(color="info", variant="tonal"):
                            with v3.VCardText(classes="text-center pa-2"):
                                html.Div("{{ pending_count }}", classes="text-h5")
                                html.Div("Pending", classes="text-caption")
                    with v3.VCol(cols=4):
                        with v3.VCard(color="success", variant="tonal"):
                            with v3.VCardText(classes="text-center pa-2"):
                                html.Div("{{ completed_count }}", classes="text-h5")
                                html.Div("Done", classes="text-caption")
                    with v3.VCol(cols=4):
                        with v3.VCard(color="error", variant="tonal"):
                            with v3.VCardText(classes="text-center pa-2"):
                                html.Div("{{ failed_count }}", classes="text-h5")
                                html.Div("Failed", classes="text-caption")

                v3.VAlert(
                    v_if=("progress_message",),
                    text=("progress_message",),
                    type="info",
                    density="compact",
                    classes="mt-3",
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
        sources = self.state.sources
        source_config = SourceConfig(
            number=int(self.state.source_count),
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
                number_of_timesteps=int(self.state.number_of_timesteps),
            ),
            receivers=ReceiverConfig(
                sensors_per_face=int(self.state.sensors_per_face),
            ),
            output=OutputConfig(
                image=int(self.state.output_image_interval),
                data=int(self.state.output_data_interval),
                points=int(self.state.output_points_interval),
                energy=int(self.state.output_energy_interval),
            ),
            batch_name=self.state.batch_name,
            batch_description=self.state.batch_description,
            num_samples=int(self.state.num_samples),
        )

    def _get_batch_output_dir(self) -> Path:
        """Compute the batch output directory."""
        base_dir = (
            Path(self.state.batch_dir) if self.state.batch_dir else DEFAULT_DATA_DIR
        )
        batch_name = self.state.batch_name or "unnamed_batch"
        return base_dir / batch_name

    def _run_batch(self):
        """Execute the simulation batch."""
        batch_dir = self._get_batch_output_dir()
        if not batch_dir.exists():
            batch_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"Created batch directory: {batch_dir}")

        self.state.is_running = True
        self.state.progress_message = "Starting batch..."
        self.state.log_messages = []

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

        try:
            self.state.progress_message = "Scanning for pending simulations..."

            planner = BatchPlanner(batch_dir)
            planner.compute_mesh_hashes()
            pending = planner.find_pending_simulations()
            self.state.pending_count = len(pending)
            self._log(f"Found {len(pending)} existing pending simulations")

            self.state.progress_message = "Running simulations..."
            completed, failed = run_batch(
                batch_dir=batch_dir,
                base_config_path=base_config,
                num_samples=num_samples,
                mesh_file=mesh_file,
                parameter_space=parameter_space,
                geometry_type=self.state.inclusion_type,
                simulation_config=simulation_config,
            )

            self.state.completed_count = completed
            self.state.failed_count = failed
            self.state.pending_count = 0
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
