# Copilot Instructions for EnergyPlus MCP Server

## Project Overview
- This project implements an EnergyPlus Model Context Protocol (MCP) server for advanced building energy modeling and simulation workflows.
- The core server is in `energyplus_mcp_server/server.py` and is built on FastMCP (`mcp.server.fastmcp.FastMCP`).
- EnergyPlus-specific logic and tool APIs are in `energyplus_mcp_server/energyplus_tools.py` and the `utils/` submodules.
- Configuration is managed via `energyplus_mcp_server/config.py` using dataclasses for server, path, and EnergyPlus settings.

## Key Components
- **Tool Functions**: Each tool (e.g., `copy_file`, `run_energyplus_simulation`, `add_output_meters`) is exposed as an async function decorated with `@mcp.tool()` in `server.py` and implemented in `energyplus_tools.py` or `utils/`.
- **Utilities**: The `utils/` directory contains modular managers for people, lights, electric equipment, schedules, output variables/meters, and diagrams.
- **Configuration**: All paths, EnergyPlus versioning, and server settings are centralized in `config.py` and loaded at server startup.

## Developer Workflows
- **Install dependencies**: `pip install -e .` (requires Python 3.10+)
- **Run the server**: `python -m energyplus_mcp_server.server` (ensure EnergyPlus is installed and paths are set in config)
- **Test**: Tests are in `tests/` (e.g., `pytest tests/`).
- **Tool Reference**: See `TOOLS_REFERENCE.md` for detailed tool API docs and usage examples.

## Project Conventions
- All tool APIs return JSON-serializable results, often with detailed status, validation, or error info.
- File and directory paths are resolved relative to the workspace root unless absolute.
- IDF/weather/sample files are managed in `sample_files/`, `outputs/`, and `illustrative examples/`.
- Logging is configured at startup; logs are written to `logs/`.
- Prefer using the provided tool functions for file/model operations instead of direct file I/O.

## Integration & Extensibility
- New tools should be added as async functions in `server.py` and registered with `@mcp.tool()`.
- Utilities for new object types should be placed in `utils/` and imported in `energyplus_tools.py`.
- Configuration changes should be made via the dataclasses in `config.py`.

## Examples
- To add a new tool for modifying HVAC schedules, create a manager in `utils/`, add logic in `energyplus_tools.py`, and expose it in `server.py` with `@mcp.tool()`.
- To run a simulation: use the `run_energyplus_simulation` tool, passing the IDF and weather file paths.

## References
- See `TOOLS_REFERENCE.md` for all available tools and usage patterns.
- See `pyproject.toml` for dependencies and project metadata.
- Example IDF/weather files are in `sample_files/` and `illustrative examples/`.

---
If any section is unclear or missing key project-specific details, please provide feedback for further refinement.
