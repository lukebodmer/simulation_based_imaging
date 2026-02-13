"""Batch Parameters page."""

from pathlib import Path

import toml
from trame.widgets import html
from trame.widgets import vuetify3 as v3

from sbimaging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path("/data/simulations")


class BatchParametersPage:
    """Page for viewing batch and simulation parameters."""

    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        self._setup_state()

    def _setup_state(self):
        """Initialize state variables for this page."""
        self.state.params_available_batches = []
        self.state.params_selected_batch = ""
        self.state.params_batch_config = ""
        self.state.params_batch_metadata = ""
        self.state.params_simulation_list = []
        self.state.params_selected_simulation = ""
        self.state.params_simulation_config = ""

        # Refresh batch list on init
        self._refresh_batch_list()

        # Register state change handlers
        self.state.change("params_selected_batch")(self._on_batch_selected)
        self.state.change("params_selected_simulation")(self._on_simulation_selected)

    def _refresh_batch_list(self):
        """Scan for available batches."""
        batches = []
        if DEFAULT_DATA_DIR.exists():
            for item in sorted(DEFAULT_DATA_DIR.iterdir()):
                if item.is_dir():
                    has_metadata = (item / "batch_metadata.toml").exists()
                    has_params = (item / "parameter_files").exists()
                    if has_metadata or has_params:
                        batches.append({"title": item.name, "value": item.name})

        self.state.params_available_batches = batches
        logger.info(f"Found {len(batches)} batches with parameters")

    def _on_batch_selected(self, params_selected_batch, **kwargs):
        """Handle batch selection - load config, metadata, and simulation list."""
        self.state.params_batch_config = ""
        self.state.params_batch_metadata = ""
        self.state.params_simulation_list = []
        self.state.params_selected_simulation = ""
        self.state.params_simulation_config = ""

        if not params_selected_batch:
            return

        batch_dir = DEFAULT_DATA_DIR / params_selected_batch

        # Load batch configuration (sweep and fixed parameters)
        config_file = batch_dir / "batch_config.toml"
        if config_file.exists():
            try:
                self.state.params_batch_config = config_file.read_text()
                logger.info(f"Loaded batch config from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load batch config: {e}")
                self.state.params_batch_config = f"Error loading config: {e}"
        else:
            self.state.params_batch_config = "No batch_config.toml found (batch created before config saving was added)"

        # Load batch metadata (mesh info, global_dt)
        metadata_file = batch_dir / "batch_metadata.toml"
        if metadata_file.exists():
            try:
                self.state.params_batch_metadata = metadata_file.read_text()
                logger.info(f"Loaded batch metadata from {metadata_file}")
            except Exception as e:
                logger.error(f"Failed to load batch metadata: {e}")
                self.state.params_batch_metadata = f"Error loading metadata: {e}"
        else:
            self.state.params_batch_metadata = "No batch_metadata.toml found"

        # Load simulation list from parameter files
        param_dir = batch_dir / "parameter_files"
        if param_dir.exists():
            sims = [
                {"title": f.stem, "value": f.stem}
                for f in sorted(param_dir.glob("*.toml"))
            ]
            self.state.params_simulation_list = sims
            logger.info(
                f"Found {len(sims)} parameter files in batch {params_selected_batch}"
            )

            # Auto-select the first simulation
            if sims:
                self.state.params_selected_simulation = sims[0]["value"]

    def _on_simulation_selected(self, params_selected_simulation, **kwargs):
        """Handle simulation selection - load parameter file."""
        self.state.params_simulation_config = ""

        if not params_selected_simulation or not self.state.params_selected_batch:
            return

        param_file = (
            DEFAULT_DATA_DIR
            / self.state.params_selected_batch
            / "parameter_files"
            / f"{params_selected_simulation}.toml"
        )

        if param_file.exists():
            try:
                self.state.params_simulation_config = param_file.read_text()
                logger.info(f"Loaded simulation parameters from {param_file}")
            except Exception as e:
                logger.error(f"Failed to load simulation parameters: {e}")
                self.state.params_simulation_config = f"Error loading parameters: {e}"
        else:
            self.state.params_simulation_config = "Parameter file not found"

    def _go_to_previous_simulation(self):
        """Navigate to the previous simulation."""
        simulations = self.state.params_simulation_list
        current = self.state.params_selected_simulation
        if not simulations or not current:
            return

        current_index = next(
            (i for i, s in enumerate(simulations) if s["value"] == current), -1
        )
        if current_index > 0:
            self.state.params_selected_simulation = simulations[current_index - 1][
                "value"
            ]

    def _go_to_next_simulation(self):
        """Navigate to the next simulation."""
        simulations = self.state.params_simulation_list
        current = self.state.params_selected_simulation
        if not simulations or not current:
            return

        current_index = next(
            (i for i, s in enumerate(simulations) if s["value"] == current), -1
        )
        if current_index < len(simulations) - 1:
            self.state.params_selected_simulation = simulations[current_index + 1][
                "value"
            ]

    def build_ui(self):
        """Build the batch parameters page UI."""
        with v3.VContainer(fluid=True, classes="fill-height"):
            with v3.VRow(classes="fill-height"):
                # Left column - Batch metadata
                with v3.VCol(cols=12, md=6, classes="fill-height"):
                    self._build_batch_metadata_panel()

                # Right column - Simulation parameters
                with v3.VCol(cols=12, md=6, classes="fill-height"):
                    self._build_simulation_params_panel()

    def _build_batch_metadata_panel(self):
        """Build the batch configuration and metadata panel."""
        with v3.VCard(classes="fill-height"):
            v3.VCardTitle("Batch Configuration")
            with v3.VCardText():
                # Batch selection
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=10):
                        v3.VSelect(
                            v_model=("params_selected_batch",),
                            items=("params_available_batches",),
                            label="Select Batch",
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

                v3.VDivider(classes="my-3")

                # Batch config display (sweep and fixed parameters)
                html.Div("Sweep & Fixed Parameters", classes="text-subtitle-2 mb-2")
                with v3.VSheet(
                    color="grey-darken-4",
                    rounded=True,
                    classes="pa-3 mb-3",
                    style="height: calc(50vh - 200px); overflow-y: auto;",
                ):
                    html.Pre(
                        "{{ params_batch_config }}",
                        style="margin: 0; white-space: pre-wrap; font-family: monospace; font-size: 13px;",
                    )

                # Batch metadata display (mesh info, global_dt)
                html.Div("Computed Metadata", classes="text-subtitle-2 mb-2")
                with v3.VSheet(
                    color="grey-darken-4",
                    rounded=True,
                    classes="pa-3",
                    style="height: calc(50vh - 200px); overflow-y: auto;",
                ):
                    html.Pre(
                        "{{ params_batch_metadata }}",
                        style="margin: 0; white-space: pre-wrap; font-family: monospace; font-size: 13px;",
                    )

    def _build_simulation_params_panel(self):
        """Build the simulation parameters panel."""
        with v3.VCard(classes="fill-height"):
            v3.VCardTitle("Simulation Parameters")
            with v3.VCardText():
                # Simulation selection with navigation buttons
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=2, classes="pa-0"):
                        v3.VBtn(
                            icon="mdi-chevron-left",
                            variant="text",
                            density="compact",
                            click=self._go_to_previous_simulation,
                            disabled=(
                                "!params_selected_simulation || params_simulation_list.length === 0",
                            ),
                        )
                    with v3.VCol(cols=8, classes="pa-0"):
                        v3.VSelect(
                            v_model=("params_selected_simulation",),
                            items=("params_simulation_list",),
                            label="Select Simulation",
                            density="compact",
                            clearable=True,
                            disabled=("!params_selected_batch",),
                            hide_details=True,
                        )
                    with v3.VCol(cols=2, classes="pa-0"):
                        v3.VBtn(
                            icon="mdi-chevron-right",
                            variant="text",
                            density="compact",
                            click=self._go_to_next_simulation,
                            disabled=(
                                "!params_selected_simulation || params_simulation_list.length === 0",
                            ),
                        )

                v3.VDivider(classes="my-3")

                # Simulation parameters display
                with v3.VSheet(
                    color="grey-darken-4",
                    rounded=True,
                    classes="pa-3",
                    style="height: calc(100vh - 280px); overflow-y: auto;",
                ):
                    html.Pre(
                        "{{ params_simulation_config }}",
                        style="margin: 0; white-space: pre-wrap; font-family: monospace; font-size: 13px;",
                    )
