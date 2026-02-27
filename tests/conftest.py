"""
Shared test fixtures for EnergyPlus MCP Server tests.

Provides mock configuration and sample data to avoid filesystem
side-effects from Config.__post_init__.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch


@pytest.fixture
def mock_config(tmp_path):
    """Build a Config with tmp_path-based directories, avoiding filesystem side-effects."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    sample_files = tmp_path / "sample_files"
    sample_files.mkdir()
    logs_dir = workspace / "logs"
    logs_dir.mkdir()

    env_overrides = {
        "EPLUS_IDD_PATH": str(tmp_path / "Energy+.idd"),
        "ENERGYPLUS_MCP_WORKSPACE": str(workspace),
    }

    # Create a fake IDD file so path validation doesn't warn excessively
    (tmp_path / "Energy+.idd").touch()

    with patch.dict(os.environ, env_overrides):
        # Import here to avoid triggering module-level config at import time
        from energyplus_mcp_server.config import Config, PathConfig

        paths = PathConfig(
            workspace_root=str(workspace),
            sample_files_path=str(sample_files),
            output_dir=str(output_dir),
        )
        config = Config.__new__(Config)
        # Manually set fields to avoid __post_init__ side-effects
        config.energyplus = type("EnergyPlusConfig", (), {
            "idd_path": str(tmp_path / "Energy+.idd"),
            "installation_path": str(tmp_path),
            "executable_path": str(tmp_path / "energyplus"),
            "version": "25.1.0",
            "weather_data_path": str(tmp_path / "WeatherData"),
            "default_weather_file": str(tmp_path / "WeatherData" / "default.epw"),
            "example_files_path": str(tmp_path / "ExampleFiles"),
        })()
        config.paths = paths
        config.server = type("ServerConfig", (), {
            "name": "test-server",
            "version": "0.1.0-test",
            "log_level": "DEBUG",
            "simulation_timeout": 10,
            "tool_timeout": 5,
        })()
        config.debug_mode = False

        yield config


@pytest.fixture
def sample_arrays():
    """Paired measured/simulated numpy arrays for calibration tests.

    12-month data with known statistical properties.
    Measured: typical commercial building monthly kWh.
    Simulated: measured + small controlled offsets.
    """
    measured = np.array([
        15000.0, 14500.0, 16000.0, 18000.0, 22000.0, 28000.0,
        32000.0, 30000.0, 25000.0, 19000.0, 15500.0, 14800.0,
    ])
    simulated = np.array([
        14800.0, 14200.0, 15800.0, 17500.0, 21500.0, 27000.0,
        31000.0, 29500.0, 24500.0, 18500.0, 15200.0, 14500.0,
    ])
    return measured, simulated
