"""Trame app for simulation batch management.

A multi-page dashboard for running and visualizing simulation batches.
"""

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3

from apps.simulation_runner.pages.run_batch import RunBatchPage
from apps.simulation_runner.pages.visualize_batch import VisualizeBatchPage
from sbimaging.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

PAGES = [
    {"title": "Run Simulation Batch", "value": "run"},
    {"title": "Visualize Simulation Batch", "value": "visualize"},
]


class SimulationApp:
    """Multi-page Trame application for simulation batch management."""

    def __init__(self, server=None):
        self.server = server or get_server()
        self.state = self.server.state
        self.ctrl = self.server.controller

        # Initialize pages
        self.run_batch_page = RunBatchPage(self.server)
        self.visualize_batch_page = VisualizeBatchPage(self.server)

        self._setup_state()
        self._setup_ui()

    def _setup_state(self):
        """Initialize app-level state."""
        self.state.current_page = "run"
        self.state.page_items = PAGES

    def _setup_ui(self):
        """Build the user interface."""
        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("Simulation Batch Manager")

            with layout.toolbar:
                # Page navigation tabs
                with v3.VTabs(v_model=("current_page",), density="compact"):
                    v3.VTab(value="run", text="Run Simulation Batch")
                    v3.VTab(value="visualize", text="Visualize Simulation Batch")

                v3.VSpacer()

                # Page-specific toolbar actions (only show on run page)
                with v3.VContainer(
                    v_show=("current_page === 'run'",),
                    classes="d-flex pa-0",
                ):
                    self.run_batch_page.build_toolbar_actions()

            with layout.content:
                # Run Batch page
                with v3.VWindow(v_model=("current_page",)):
                    with v3.VWindowItem(value="run"):
                        self.run_batch_page.build_ui()

                    with v3.VWindowItem(value="visualize"):
                        self.visualize_batch_page.build_ui()


def main():
    """Entry point for the simulation app."""
    app = SimulationApp()
    app.server.start()


if __name__ == "__main__":
    main()
