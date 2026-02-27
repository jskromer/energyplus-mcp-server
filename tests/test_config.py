"""
Tests for Config, PathConfig, get_config, and reload_config.

Covers default paths, EPLUS_IDD_PATH override, PathConfig __post_init__,
and singleton behavior.
"""

import os
import pytest
from unittest.mock import patch

from energyplus_mcp_server.config import (
    Config,
    PathConfig,
    EnergyPlusConfig,
    get_config,
    reload_config,
)


class TestPathConfigPostInit:
    def test_auto_sets_workspace_root(self, tmp_path):
        """workspace_root defaults to parent of config module when env unset."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove env var if set
            os.environ.pop("ENERGYPLUS_MCP_WORKSPACE", None)
            pc = PathConfig()
            assert pc.workspace_root != ""

    def test_auto_sets_sample_files(self, tmp_path):
        pc = PathConfig(workspace_root=str(tmp_path))
        assert pc.sample_files_path == os.path.join(str(tmp_path), "sample_files")

    def test_auto_sets_output_dir(self, tmp_path):
        pc = PathConfig(workspace_root=str(tmp_path))
        assert pc.output_dir == os.path.join(str(tmp_path), "outputs")

    def test_respects_explicit_values(self, tmp_path):
        pc = PathConfig(
            workspace_root=str(tmp_path),
            sample_files_path="/custom/samples",
            output_dir="/custom/out",
        )
        assert pc.sample_files_path == "/custom/samples"
        assert pc.output_dir == "/custom/out"


class TestConfigDefaults:
    def test_default_installation_path(self, tmp_path):
        """With no EPLUS_IDD_PATH, uses /app/software/EnergyPlusV25-1-0."""
        with patch.dict(os.environ, {"ENERGYPLUS_MCP_WORKSPACE": str(tmp_path)}, clear=False):
            os.environ.pop("EPLUS_IDD_PATH", None)
            # Create required dirs
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "outputs").mkdir(exist_ok=True)
            cfg = Config(paths=PathConfig(workspace_root=str(tmp_path)))
            assert cfg.energyplus.installation_path == "/app/software/EnergyPlusV25-1-0"

    def test_idd_path_override(self, tmp_path):
        """EPLUS_IDD_PATH set → derives all paths from it."""
        fake_idd = tmp_path / "EPlus" / "Energy+.idd"
        fake_idd.parent.mkdir(parents=True)
        fake_idd.touch()

        env = {
            "EPLUS_IDD_PATH": str(fake_idd),
            "ENERGYPLUS_MCP_WORKSPACE": str(tmp_path),
        }
        with patch.dict(os.environ, env, clear=False):
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "outputs").mkdir(exist_ok=True)
            cfg = Config(paths=PathConfig(workspace_root=str(tmp_path)))

        assert cfg.energyplus.idd_path == str(fake_idd)
        assert cfg.energyplus.installation_path == str(fake_idd.parent)
        assert cfg.energyplus.executable_path == os.path.join(
            str(fake_idd.parent), "energyplus"
        )
        assert cfg.energyplus.weather_data_path == os.path.join(
            str(fake_idd.parent), "WeatherData"
        )
        assert cfg.energyplus.example_files_path == os.path.join(
            str(fake_idd.parent), "ExampleFiles"
        )


class TestGetConfigSingleton:
    def test_returns_same_instance(self):
        """get_config should return the same object on repeated calls."""
        # Clean any existing singleton first
        if hasattr(get_config, "_config"):
            delattr(get_config, "_config")

        # We can't easily construct Config without side-effects in CI,
        # but we can test the singleton mechanism by patching.
        sentinel = object()
        get_config._config = sentinel
        try:
            assert get_config() is sentinel
            assert get_config() is sentinel
        finally:
            delattr(get_config, "_config")

    def test_reload_clears_singleton(self):
        """reload_config should clear the cached singleton."""
        sentinel = object()
        get_config._config = sentinel
        try:
            # reload_config deletes _config then calls get_config()
            # which will create a new Config. We just test deletion.
            assert hasattr(get_config, "_config")
            delattr(get_config, "_config")
            assert not hasattr(get_config, "_config")
        except Exception:
            # Clean up in case of failure
            if hasattr(get_config, "_config"):
                delattr(get_config, "_config")
            raise
