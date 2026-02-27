"""
Tests for resolve_path in energyplus_mcp_server.utils.path_utils.

Covers empty paths, absolute paths, relative paths, extension validation,
search directory ordering, and must_exist=False output path construction.
"""

import os
import pytest

from energyplus_mcp_server.utils.path_utils import resolve_path


class TestResolvePathEmpty:
    def test_empty_string_raises(self, mock_config):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_path(mock_config, "", description="test file")

    def test_none_raises(self, mock_config):
        """None is falsy — same branch as empty string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_path(mock_config, None, description="test file")


class TestResolvePathAbsolute:
    def test_exists_returned_as_is(self, mock_config, tmp_path):
        f = tmp_path / "model.idf"
        f.write_text("!IDF Version;")
        result = resolve_path(mock_config, str(f), file_types=[".idf"])
        assert result == str(f)

    def test_not_exists_must_exist_raises(self, mock_config, tmp_path):
        missing = str(tmp_path / "missing.idf")
        with pytest.raises(FileNotFoundError):
            resolve_path(mock_config, missing, file_types=[".idf"], must_exist=True)

    def test_wrong_extension_raises(self, mock_config, tmp_path):
        f = tmp_path / "model.txt"
        f.write_text("not idf")
        with pytest.raises(ValueError, match="expected extension"):
            resolve_path(mock_config, str(f), file_types=[".idf"])


class TestResolvePathRelative:
    def test_found_in_sample_files(self, mock_config):
        """Relative path should be found in sample_files_path."""
        sample_dir = mock_config.paths.sample_files_path
        test_file = os.path.join(sample_dir, "test_model.idf")
        with open(test_file, "w") as fh:
            fh.write("!IDF Version;")

        result = resolve_path(mock_config, "test_model.idf", file_types=[".idf"])
        assert os.path.isabs(result)
        assert result == os.path.abspath(test_file)

    def test_must_exist_false_joins_with_output_dir(self, mock_config):
        """When must_exist=False, a bare filename goes to output_dir."""
        result = resolve_path(
            mock_config, "new_file.idf", must_exist=False
        )
        assert result == os.path.join(mock_config.paths.output_dir, "new_file.idf")

    def test_must_exist_false_with_default_dir(self, mock_config, tmp_path):
        custom_dir = str(tmp_path / "custom")
        result = resolve_path(
            mock_config, "output.csv", must_exist=False, default_dir=custom_dir
        )
        assert result == os.path.join(custom_dir, "output.csv")

    def test_must_exist_false_relative_with_separator(self, mock_config):
        """Relative path with separator → joins with workspace_root."""
        result = resolve_path(
            mock_config, "subdir/file.idf", must_exist=False
        )
        assert result == os.path.join(
            mock_config.paths.workspace_root, "subdir/file.idf"
        )


class TestResolvePathSearchOrder:
    def test_idf_searches_example_files(self, mock_config, tmp_path):
        """IDF search should include example_files_path."""
        example_dir = str(tmp_path / "ExampleFiles")
        os.makedirs(example_dir, exist_ok=True)
        mock_config.energyplus.example_files_path = example_dir

        test_file = os.path.join(example_dir, "5Zone.idf")
        with open(test_file, "w") as fh:
            fh.write("!IDF;")

        result = resolve_path(mock_config, "5Zone.idf", file_types=[".idf"])
        assert result == os.path.abspath(test_file)

    def test_epw_searches_weather_data(self, mock_config, tmp_path):
        """EPW search should include weather_data_path."""
        weather_dir = str(tmp_path / "WeatherData")
        os.makedirs(weather_dir, exist_ok=True)
        mock_config.energyplus.weather_data_path = weather_dir

        test_file = os.path.join(weather_dir, "SF.epw")
        with open(test_file, "w") as fh:
            fh.write("LOCATION,San Francisco")

        result = resolve_path(mock_config, "SF.epw", file_types=[".epw"])
        assert result == os.path.abspath(test_file)


class TestResolvePathNotFound:
    def test_file_not_found_raises(self, mock_config):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_path(mock_config, "nonexistent.idf", file_types=[".idf"])
