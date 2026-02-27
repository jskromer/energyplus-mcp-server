"""
EnergyPlus MCP Server with FastMCP

EnergyPlus Model Context Protocol Server (EnergyPlus-MCP)
Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

See License.txt in the parent directory for license details.
"""

import os
import sys
import platform
import asyncio
import logging
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Import FastMCP instead of the low-level Server
from mcp.server.fastmcp import FastMCP

# Import our EnergyPlus utilities and configuration
from energyplus_mcp_server.energyplus_tools import EnergyPlusManager
from energyplus_mcp_server.config import get_config, Config

logger = logging.getLogger(__name__)

# Record actual startup time for status reporting
_server_startup_time = datetime.now()

# Initialize configuration and set up logging
config = get_config()

# Initialize the FastMCP server with configuration
mcp = FastMCP(config.server.name)

# Initialize EnergyPlus manager with configuration
ep_manager = EnergyPlusManager(config)

logger.info("EnergyPlus MCP Server '%s' v%s initialized", config.server.name, config.server.version)


@mcp.tool()
async def copy_file(source_path: str, target_path: str, overwrite: bool = False, file_types: Optional[List[str]] = None) -> str:
    """
    Copy a file from source to target location with intelligent path resolution
    
    Args:
        source_path: Source file path. Can be:
                    - Absolute path: "/full/path/to/file.idf"
                    - Relative path: "models/mymodel.idf"
                    - Filename only: "1ZoneUncontrolled.idf" (searches in sample_files)
                    - Fuzzy name: Will search in sample_files, example_files, weather_data, etc.
        target_path: Target path for the copy. Can be:
                    - Absolute path: "/full/path/to/copy.idf"
                    - Relative path: "outputs/modified_file.idf"  
                    - Filename only: "my_copy.idf" (saves to outputs directory)
        overwrite: Whether to overwrite existing target file (default: False)
        file_types: List of acceptable file extensions (e.g., [".idf", ".epw"]). If None, accepts any file type.
    
    Returns:
        JSON string with copy operation results including resolved paths, file sizes, and validation status
        
    Examples:
        # Copy IDF file with validation
        copy_file("1ZoneUncontrolled.idf", "my_model.idf", file_types=[".idf"])
        
        # Copy weather file
        copy_file("USA_CA_San.Francisco.epw", "sf_weather.epw", file_types=[".epw"])
        
        # Copy any file type
        copy_file("sample.idf", "outputs/test.idf", overwrite=True)
        
        # Copy with fuzzy matching (e.g., city name for weather files)
        copy_file("san francisco", "my_weather.epw", file_types=[".epw"])
    """
    try:
        logger.info("Copying file: '%s' -> '%s' (overwrite=%s, file_types=%s)", source_path, target_path, overwrite, file_types)
        result = ep_manager.copy_file(source_path, target_path, overwrite, file_types)
        return result
    except ValueError as e:
        logger.warning("Invalid arguments for copy_file: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Unexpected error copying file: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def load_idf_model(idf_path: str) -> str:
    """
    Load and validate an EnergyPlus IDF file
    
    Args:
        idf_path: Path to the IDF file (can be absolute, relative, or just filename for sample files)
    
    Returns:
        JSON string with model information and loading status
    """
    try:
        logger.info("Loading IDF model: %s", idf_path)
        result = ep_manager.load_idf(idf_path)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid input for load_idf_model: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Unexpected error loading IDF %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_model_summary(idf_path: str) -> str:
    """
    Get basic model information (Building, Site, SimulationControl, Version)
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with model summary information
    """
    try:
        logger.info("Getting model summary: %s", idf_path)
        summary = ep_manager.get_model_basics(idf_path)
        return summary
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error getting model summary for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def check_simulation_settings(idf_path: str) -> str:
    """
    Check SimulationControl and RunPeriod settings with information about modifiable fields
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with current settings and descriptions of modifiable fields
    """
    try:
        logger.info("Checking simulation settings: %s", idf_path)
        settings = ep_manager.check_simulation_settings(idf_path)
        return settings
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error checking simulation settings for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def inspect_schedules(idf_path: str, include_values: bool = False) -> str:
    """
    Inspect and inventory all schedule objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
        include_values: Whether to extract actual schedule values (default: False)
    
    Returns:
        JSON string with detailed schedule inventory and analysis
    """
    try:
        logger.info("Inspecting schedules: %s (include_values=%s)", idf_path, include_values)
        schedules_info = ep_manager.inspect_schedules(idf_path, include_values)
        return schedules_info
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error inspecting schedules for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def inspect_people(idf_path: str) -> str:
    """
    Inspect and list all People objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with detailed People objects information including:
        - Name, zone, and schedule associations
        - Calculation method (People, People/Area, Area/Person)
        - Occupancy values and thermal comfort settings
        - Summary statistics by zone and calculation method
    """
    try:
        logger.info("Inspecting People objects: %s", idf_path)
        result = ep_manager.inspect_people(idf_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error inspecting People objects for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def modify_people(
    idf_path: str,
    modifications: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """
    Modify People objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the input IDF file
        modifications: List of modification specifications. Each item should contain:
                      - "target": Specifies which People objects to modify
                        - "all": Apply to all People objects
                        - "zone:ZoneName": Apply to People objects in specific zone
                        - "name:PeopleName": Apply to specific People object by name
                      - "field_updates": Dictionary of field names and new values
                        Valid fields include:
                        - Number_of_People_Schedule_Name
                        - Number_of_People_Calculation_Method (People, People/Area, Area/Person)
                        - Number_of_People
                        - People_per_Floor_Area
                        - Floor_Area_per_Person
                        - Fraction_Radiant
                        - Sensible_Heat_Fraction
                        - Activity_Level_Schedule_Name
                        - Carbon_Dioxide_Generation_Rate
                        - Clothing_Insulation_Schedule_Name
                        - Air_Velocity_Schedule_Name
                        - Thermal_Comfort_Model_1_Type
                        - Thermal_Comfort_Model_2_Type
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
        
    Examples:
        # Modify all People objects to use 0.1 people/m2
        modify_people("model.idf", [
            {
                "target": "all",
                "field_updates": {
                    "Number_of_People_Calculation_Method": "People/Area",
                    "People_per_Floor_Area": 0.1
                }
            }
        ])
        
        # Modify People objects in specific zone
        modify_people("model.idf", [
            {
                "target": "zone:Office Zone",
                "field_updates": {
                    "Number_of_People": 10,
                    "Activity_Level_Schedule_Name": "Office Activity"
                }
            }
        ])
        
        # Modify specific People object by name
        modify_people("model.idf", [
            {
                "target": "name:Office People",
                "field_updates": {
                    "Fraction_Radiant": 0.3,
                    "Sensible_Heat_Fraction": 0.6
                }
            }
        ])
    """
    try:
        logger.info("Modifying People objects: %s", idf_path)
        result = ep_manager.modify_people(idf_path, modifications, output_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid input for modify_people: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error modifying People objects for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def inspect_lights(idf_path: str) -> str:
    """
    Inspect and list all Lights objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with detailed Lights objects information including:
        - Name, zone, and schedule associations
        - Calculation method (LightingLevel, Watts/Area, Watts/Person)
        - Lighting power values and heat fraction settings
        - Summary statistics by zone and calculation method
    """
    try:
        logger.info("Inspecting Lights objects: %s", idf_path)
        result = ep_manager.inspect_lights(idf_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error inspecting Lights objects for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def modify_lights(
    idf_path: str,
    modifications: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """
    Modify Lights objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the input IDF file
        modifications: List of modification specifications. Each item should contain:
                      - "target": Specifies which Lights objects to modify
                        - "all": Apply to all Lights objects
                        - "zone:ZoneName": Apply to Lights objects in specific zone
                        - "name:LightsName": Apply to specific Lights object by name
                      - "field_updates": Dictionary of field names and new values
                        Valid fields include:
                        - Schedule_Name
                        - Design_Level_Calculation_Method (LightingLevel, Watts/Area, Watts/Person)
                        - Lighting_Level
                        - Watts_per_Floor_Area
                        - Watts_per_Person
                        - Return_Air_Fraction
                        - Fraction_Radiant
                        - Fraction_Visible
                        - Fraction_Replaceable
                        - EndUse_Subcategory
                        - Return_Air_Fraction_Calculated_from_Plenum_Temperature
                        - Return_Air_Fraction_Function_of_Plenum_Temperature_Coefficient_1
                        - Return_Air_Fraction_Function_of_Plenum_Temperature_Coefficient_2
                        - Return_Air_Heat_Gain_Node_Name
                        - Exhaust_Air_Heat_Gain_Node_Name
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
        
    Examples:
        # Modify all Lights objects to use 10 W/m2
        modify_lights("model.idf", [
            {
                "target": "all",
                "field_updates": {
                    "Design_Level_Calculation_Method": "Watts/Area",
                    "Watts_per_Floor_Area": 10.0
                }
            }
        ])
        
        # Modify Lights objects in specific zone
        modify_lights("model.idf", [
            {
                "target": "zone:Office Zone",
                "field_updates": {
                    "Lighting_Level": 2000,
                    "Schedule_Name": "Office Lighting Schedule"
                }
            }
        ])
        
        # Modify specific Lights object by name
        modify_lights("model.idf", [
            {
                "target": "name:Office Lights",
                "field_updates": {
                    "Fraction_Radiant": 0.42,
                    "Fraction_Visible": 0.18
                }
            }
        ])
    """
    try:
        logger.info("Modifying Lights objects: %s", idf_path)
        result = ep_manager.modify_lights(idf_path, modifications, output_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid input for modify_lights: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error modifying Lights objects for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def inspect_electric_equipment(idf_path: str) -> str:
    """
    Inspect and list all ElectricEquipment objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with detailed ElectricEquipment objects information including:
        - Name, zone, and schedule associations
        - Calculation method (EquipmentLevel, Watts/Area, Watts/Person)
        - Equipment power values and heat fraction settings
        - Summary statistics by zone and calculation method
    """
    try:
        logger.info("Inspecting ElectricEquipment objects: %s", idf_path)
        result = ep_manager.inspect_electric_equipment(idf_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error inspecting ElectricEquipment objects for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def modify_electric_equipment(
    idf_path: str,
    modifications: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """
    Modify ElectricEquipment objects in the EnergyPlus model
    
    Args:
        idf_path: Path to the input IDF file
        modifications: List of modification specifications. Each item should contain:
                      - "target": Specifies which ElectricEquipment objects to modify
                        - "all": Apply to all ElectricEquipment objects
                        - "zone:ZoneName": Apply to ElectricEquipment objects in specific zone
                        - "name:ElectricEquipmentName": Apply to specific ElectricEquipment object by name
                      - "field_updates": Dictionary of field names and new values
                        Valid fields include:
                        - Schedule_Name
                        - Design_Level_Calculation_Method (EquipmentLevel, Watts/Area, Watts/Person)
                        - Design_Level
                        - Watts_per_Floor_Area
                        - Watts_per_Person
                        - Fraction_Latent
                        - Fraction_Radiant
                        - Fraction_Lost
                        - EndUse_Subcategory
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
        
    Examples:
        # Modify all ElectricEquipment objects to use 15 W/m2
        modify_electric_equipment("model.idf", [
            {
                "target": "all",
                "field_updates": {
                    "Design_Level_Calculation_Method": "Watts/Area",
                    "Watts_per_Floor_Area": 15.0
                }
            }
        ])
        
        # Modify ElectricEquipment objects in specific zone
        modify_electric_equipment("model.idf", [
            {
                "target": "zone:Office Zone",
                "field_updates": {
                    "Design_Level": 3000,
                    "Schedule_Name": "Office Equipment Schedule"
                }
            }
        ])
        
        # Modify specific ElectricEquipment object by name
        modify_electric_equipment("model.idf", [
            {
                "target": "name:Office Equipment",
                "field_updates": {
                    "Fraction_Radiant": 0.3,
                    "Fraction_Latent": 0.1
                }
            }
        ])
    """
    try:
        logger.info("Modifying ElectricEquipment objects: %s", idf_path)
        result = ep_manager.modify_electric_equipment(idf_path, modifications, output_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid input for modify_electric_equipment: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error modifying ElectricEquipment objects for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def modify_simulation_control(
    idf_path: str, 
    field_updates: Dict[str, Any],  # Changed from str to Dict[str, Any]
    output_path: Optional[str] = None
) -> str:
    """
    Modify SimulationControl settings and save to a new file
    
    Args:
        idf_path: Path to the input IDF file
        field_updates: Dictionary with field names and new values (e.g., {"Run_Simulation_for_Weather_File_Run_Periods": "Yes"})
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
    """
    try:
        logger.info("Modifying SimulationControl: %s", idf_path)
        
        # No need to parse JSON since we're receiving a dict directly
        result = ep_manager.modify_simulation_settings(
            idf_path=idf_path,
            object_type="SimulationControl",
            field_updates=field_updates,  # Pass the dict directly
            output_path=output_path
        )
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error modifying SimulationControl for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def modify_run_period(
    idf_path: str, 
    field_updates: Dict[str, Any],  # Changed from str to Dict[str, Any]
    run_period_index: int = 0,
    output_path: Optional[str] = None
) -> str:
    """
    Modify RunPeriod settings and save to a new file
    
    Args:
        idf_path: Path to the input IDF file
        field_updates: Dictionary with field names and new values (e.g., {"Begin_Month": 1, "End_Month": 3})
        run_period_index: Index of RunPeriod to modify (default 0 for first RunPeriod)
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
    """
    try:
        logger.info("Modifying RunPeriod: %s", idf_path)
        
        # No need to parse JSON since we're receiving a dict directly
        result = ep_manager.modify_simulation_settings(
            idf_path=idf_path,
            object_type="RunPeriod",
            field_updates=field_updates,  # Pass the dict directly
            run_period_index=run_period_index,
            output_path=output_path
        )
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error modifying RunPeriod for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def change_infiltration_by_mult(
    idf_path: str, 
    mult: float,
    output_path: Optional[str] = None
) -> str:
    """
    Modify infiltration in ZoneInfiltration:DesignFlowRate and save to a new file
    
    Args:
        idf_path: Path to the input IDF file
        mult: Multiplicative factor to apply to all ZoneInfiltration:DesignFlowRate objects
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
    """
    try:
        logger.info("Modifying Infiltration: %s", idf_path)
        
        # No need to parse JSON since we're receiving a dict directly
        result = ep_manager.change_infiltration_by_mult(
            idf_path=idf_path,
            mult=mult,  # Pass the float directly
            output_path=output_path
        )
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error Infiltration modification for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def add_window_film_outside(
    idf_path: str,
    u_value: float = 4.94,
    shgc: float = 0.45,
    visible_transmittance: float = 0.66,
    output_path: Optional[str] = None
) -> str:
    """
    Add exterior window film to all exterior windows using WindowMaterial:SimpleGlazingSystem
    
    Args:
        idf_path: Path to the input IDF file
        u_value: U-value of the window film (default: 4.94 W/m²·K from CBES)
        shgc: Solar Heat Gain Coefficient of the window film (default: 0.45)
        visible_transmittance: Visible transmittance of the window film (default: 0.66)
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
    """
    try:
        logger.info("Adding window film to exterior windows: %s", idf_path)
        result = ep_manager.add_window_film_outside(
            idf_path=idf_path,
            u_value=u_value,
            shgc=shgc,
            visible_transmittance=visible_transmittance,
            output_path=output_path
        )
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error adding window film for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def add_coating_outside(
    idf_path: str,
    location: str,
    solar_abs: float = 0.4,
    thermal_abs: float = 0.9,
    output_path: Optional[str] = None
) -> str:
    """
    Add exterior coating to all exterior surfaces of the specified location (wall or roof)
    
    Args:
        idf_path: Path to the input IDF file
        location: Surface type - either "wall" or "roof"
        solar_abs: Solar Absorptance of the exterior coating (default: 0.4)
        thermal_abs: Thermal Absorptance of the exterior coating (default: 0.9)
        output_path: Optional path for output file (if None, creates one with _modified suffix)
    
    Returns:
        JSON string with modification results
    """
    try:
        logger.info("Adding exterior coating to %s surfaces: %s", location, idf_path)
        result = ep_manager.add_coating_outside(
            idf_path=idf_path,
            location=location,
            solar_abs=solar_abs,
            thermal_abs=thermal_abs,
            output_path=output_path
        )
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid location parameter: %s", location)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error adding exterior coating for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def list_zones(idf_path: str) -> str:
    """
    List all zones in the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with detailed zone information
    """
    try:
        logger.info("Listing zones: %s", idf_path)
        zones = ep_manager.list_zones(idf_path)
        return zones
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error listing zones for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_surfaces(idf_path: str) -> str:
    """
    Get detailed surface information from the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with surface details
    """
    try:
        logger.info("Getting surfaces: %s", idf_path)
        surfaces = ep_manager.get_surfaces(idf_path)
        return surfaces
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error getting surfaces for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})

@mcp.tool()
async def get_materials(idf_path: str) -> str:
    """
    Get material information from the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with material details
    """
    try:
        logger.info("Getting materials: %s", idf_path)
        materials = ep_manager.get_materials(idf_path)
        return materials
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error getting materials for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def validate_idf(idf_path: str) -> str:
    """
    Validate an EnergyPlus IDF file and return validation results
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with validation results, warnings, and errors
    """
    try:
        logger.info("Validating IDF: %s", idf_path)
        validation_result = ep_manager.validate_idf(idf_path)
        return validation_result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error validating IDF %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_output_variables(idf_path: str, discover_available: bool = False, run_days: int = 1) -> str:
    """
    Get output variables from the model - either configured variables or discover all available ones
    
    Args:
        idf_path: Path to the IDF file (can be absolute, relative, or just filename for sample files)
        discover_available: If True, runs a short simulation to discover all available variables. 
                          If False, returns currently configured variables in the IDF (default: False)
        run_days: Number of days to run for discovery simulation (default: 1, only used if discover_available=True)
    
    Returns:
        JSON string with output variables information. When discover_available=True, includes
        all possible variables with units, frequencies, and ready-to-use Output:Variable lines.
        When discover_available=False, shows only currently configured Output:Variable and Output:Meter objects.
    """
    try:
        logger.info("Getting output variables: %s (discover_available=%s)", idf_path, discover_available)
        result = ep_manager.get_output_variables(idf_path, discover_available, run_days)
        
        mode = "available variables discovery" if discover_available else "configured variables"
        return result
        
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error getting output variables for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_output_meters(idf_path: str, discover_available: bool = False, run_days: int = 1) -> str:
    """
    Get output meters from the model - either configured meters or discover all available ones
    
    Args:
        idf_path: Path to the IDF file (can be absolute, relative, or just filename for sample files)
        discover_available: If True, runs a short simulation to discover all available meters.
                          If False, returns currently configured meters in the IDF (default: False)
        run_days: Number of days to run for discovery simulation (default: 1, only used if discover_available=True)
    
    Returns:
        JSON string with meter information. When discover_available=True, includes
        all possible meters with units, frequencies, and ready-to-use Output:Meter lines.
        When discover_available=False, shows only currently configured Output:Meter objects.
    """
    try:
        logger.info("Getting output meters: %s (discover_available=%s)", idf_path, discover_available)
        result = ep_manager.get_output_meters(idf_path, discover_available, run_days)
        
        mode = "available meters discovery" if discover_available else "configured meters"
        return result
        
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error getting output meters for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def add_output_variables(
    idf_path: str,
    variables: List,  # Can be List[Dict], List[str], or mixed
    validation_level: str = "moderate",
    allow_duplicates: bool = False,
    output_path: Optional[str] = None
) -> str:
    """
    Add output variables to an EnergyPlus IDF file with intelligent validation
    
    Args:
        idf_path: Path to the input IDF file (can be absolute, relative, or filename for sample files)
        variables: List of variable specifications. Can be:
                  - Simple strings: ["Zone Air Temperature", "Surface Inside Face Temperature"] 
                  - [name, frequency] pairs: [["Zone Air Temperature", "hourly"], ["Surface Temperature", "daily"]]
                  - Full specifications: [{"key_value": "*", "variable_name": "Zone Air Temperature", "frequency": "hourly"}]
                  - Mixed formats in the same list
        validation_level: Validation strictness level:
                         - "strict": Full validation with model checking (recommended for beginners)
                         - "moderate": Basic validation with helpful warnings (default)
                         - "lenient": Minimal validation (for advanced users)
        allow_duplicates: Whether to allow duplicate output variable specifications (default: False)
        output_path: Optional path for output file (if None, creates one with _with_outputs suffix)
    
    Returns:
        JSON string with detailed results including validation report, added variables, and performance metrics
        
    Examples:
        # Simple usage
        add_output_variables("model.idf", ["Zone Air Temperature", "Zone Air Relative Humidity"])
        
        # With custom frequencies  
        add_output_variables("model.idf", [["Zone Air Temperature", "daily"], ["Surface Temperature", "hourly"]])
        
        # Full control
        add_output_variables("model.idf", [
            {"key_value": "Zone1", "variable_name": "Zone Air Temperature", "frequency": "hourly"},
            {"key_value": "*", "variable_name": "Surface Inside Face Temperature", "frequency": "daily"}
        ], validation_level="strict")
    """
    try:
        logger.info("Adding output variables: %s (%s variables, %s validation)", idf_path, len(variables), validation_level)
        
        result = ep_manager.add_output_variables(
            idf_path=idf_path,
            variables=variables,
            validation_level=validation_level,
            allow_duplicates=allow_duplicates,
            output_path=output_path
        )
        
        return result
        
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid arguments for add_output_variables: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error adding output variables: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def add_output_meters(
    idf_path: str,
    meters: List,  # Can be List[Dict], List[str], or mixed
    validation_level: str = "moderate",
    allow_duplicates: bool = False,
    output_path: Optional[str] = None
) -> str:
    """
    Add output meters to an EnergyPlus IDF file with intelligent validation
    
    Args:
        idf_path: Path to the input IDF file (can be absolute, relative, or filename for sample files)
        meters: List of meter specifications. Can be:
               - Simple strings: ["Electricity:Facility", "NaturalGas:Facility"] 
               - [name, frequency] pairs: [["Electricity:Facility", "hourly"], ["NaturalGas:Facility", "daily"]]
               - [name, frequency, type] triplets: [["Electricity:Facility", "hourly", "Output:Meter"]]
               - Full specifications: [{"meter_name": "Electricity:Facility", "frequency": "hourly", "meter_type": "Output:Meter"}]
               - Mixed formats in the same list
        validation_level: Validation strictness level:
                         - "strict": Full validation with model checking (recommended for beginners)
                         - "moderate": Basic validation with helpful warnings (default)
                         - "lenient": Minimal validation (for advanced users)
        allow_duplicates: Whether to allow duplicate output meter specifications (default: False)
        output_path: Optional path for output file (if None, creates one with _with_meters suffix)
    
    Returns:
        JSON string with detailed results including validation report, added meters, and performance metrics
        
    Examples:
        # Simple usage
        add_output_meters("model.idf", ["Electricity:Facility", "NaturalGas:Facility"])
        
        # With custom frequencies  
        add_output_meters("model.idf", [["Electricity:Facility", "daily"], ["NaturalGas:Facility", "hourly"]])
        
        # Full control with meter types
        add_output_meters("model.idf", [
            {"meter_name": "Electricity:Facility", "frequency": "hourly", "meter_type": "Output:Meter"},
            {"meter_name": "NaturalGas:Facility", "frequency": "daily", "meter_type": "Output:Meter:Cumulative"}
        ], validation_level="strict")
    """
    try:
        logger.info("Adding output meters: %s (%s meters, %s validation)", idf_path, len(meters), validation_level)
        
        result = ep_manager.add_output_meters(
            idf_path=idf_path,
            meters=meters,
            validation_level=validation_level,
            allow_duplicates=allow_duplicates,
            output_path=output_path
        )
        
        return result
        
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid arguments for add_output_meters: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error adding output meters: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def list_available_files(
    include_example_files: bool = False,
    include_weather_data: bool = False
) -> str:
    """
    List available files in specified directories
    
    Args:
        include_example_files: Whether to include EnergyPlus example files directory (default: False)
        include_weather_data: Whether to include EnergyPlus weather data directory (default: False)
    
    Returns:
        JSON string with available files organized by source and type. Always includes sample_files directory.
    """
    try:
        logger.info("Listing available files (example_files=%s, weather_data=%s)", include_example_files, include_weather_data)
        files = ep_manager.list_available_files(include_example_files, include_weather_data)
        return files
    except Exception as e:
        logger.error("Error listing available files: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_server_configuration() -> str:
    """
    Get current server configuration information
    
    Returns:
        JSON string with configuration details
    """
    try:
        logger.info("Getting server configuration")
        config_info = ep_manager.get_configuration_info()
        return config_info
    except Exception as e:
        logger.error("Error getting configuration: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_server_status() -> str:
    """
    Get current server status and health information
    
    Returns:
        JSON string with server status
    """
    try:
        status_info = {
            "server": {
                "name": config.server.name,
                "version": config.server.version,
                "status": "running",
                "startup_time": _server_startup_time.isoformat(),
                "debug_mode": config.debug_mode
            },
            "system": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture()[0]
            },
            "energyplus": {
                "version": config.energyplus.version,
                "idd_available": os.path.exists(config.energyplus.idd_path) if config.energyplus.idd_path else False,
                "executable_available": os.path.exists(config.energyplus.executable_path) if config.energyplus.executable_path else False
            },
            "paths": {
                "sample_files_available": os.path.exists(config.paths.sample_files_path),
                "temp_dir_available": os.path.exists(config.paths.temp_dir),
                "output_dir_available": os.path.exists(config.paths.output_dir)
            }
        }
        
        return json.dumps(status_info, indent=2)
        
    except Exception as e:
        logger.error("Error getting server status: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def discover_hvac_loops(idf_path: str) -> str:
    """
    Discover all HVAC loops (Plant, Condenser, Air) in the EnergyPlus model
    
    Args:
        idf_path: Path to the IDF file
    
    Returns:
        JSON string with all HVAC loops found, organized by type
    """
    try:
        logger.info("Discovering HVAC loops: %s", idf_path)
        loops = ep_manager.discover_hvac_loops(idf_path)
        return loops
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error discovering HVAC loops for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_loop_topology(idf_path: str, loop_name: str) -> str:
    """
    Get detailed topology information for a specific HVAC loop
    
    Args:
        idf_path: Path to the IDF file
        loop_name: Name of the specific loop to analyze
    
    Returns:
        JSON string with detailed loop topology including supply/demand sides, branches, and components
    """
    try:
        logger.info("Getting loop topology for '%s': %s", loop_name, idf_path)
        topology = ep_manager.get_loop_topology(idf_path, loop_name)
        return topology
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Loop not found: %s", loop_name)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error getting loop topology for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def visualize_loop_diagram(
    idf_path: str, 
    loop_name: Optional[str] = None,
    output_path: Optional[str] = None,
    format: str = "png",
    show_legend: bool = True
) -> str:
    """
    Generate and save a visual diagram of HVAC loop(s)
    
    Args:
        idf_path: Path to the IDF file
        loop_name: Optional specific loop name (if None, shows all loops)
        output_path: Optional custom output path (if None, creates one automatically)
        format: Image format for the diagram (png, jpg, pdf, svg)
        show_legend: Whether to include a legend in the diagram (default: True)
    
    Returns:
        JSON string with diagram generation results and file path
    """
    try:
        logger.info("Creating loop diagram for '%s': %s (show_legend=%s)", loop_name or 'all loops', idf_path, show_legend)
        result = ep_manager.visualize_loop_diagram(idf_path, loop_name, output_path, format, show_legend)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error creating loop diagram for %s: %s", idf_path, e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def run_energyplus_simulation(
    idf_path: str, 
    weather_file: Optional[str] = None,
    output_directory: Optional[str] = None,
    annual: bool = True,
    design_day: bool = False,
    readvars: bool = True,
    expandobjects: bool = True
) -> str:
    """
    Run EnergyPlus simulation with specified IDF and weather file
    
    Args:
        idf_path: Path to the IDF file (can be absolute, relative, or just filename for sample files)
        weather_file: Path to weather file (.epw) or city name (e.g., 'San Francisco'). If None, simulation runs without weather file
        output_directory: Directory for simulation outputs (if None, creates timestamped directory in outputs/)
        annual: Run annual simulation (default: True)
        design_day: Run design day only simulation (default: False) 
        readvars: Run ReadVarsESO after simulation to process outputs (default: True)
        expandobjects: Run ExpandObjects prior to simulation for HVAC templates (default: True)
    
    Returns:
        JSON string with simulation results, duration, and output file paths
    """
    try:
        logger.info("Running EnergyPlus simulation: %s", idf_path)
        if weather_file:
            logger.info("With weather file: %s", weather_file)
        
        result = ep_manager.run_simulation(
            idf_path=idf_path,
            weather_file=weather_file,
            output_directory=output_directory,
            annual=annual,
            design_day=design_day,
            readvars=readvars,
            expandobjects=expandobjects
        )
        return result
    except FileNotFoundError as e:
        logger.warning("File not found for simulation: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error running EnergyPlus simulation: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def create_interactive_plot(
    output_directory: str,
    idf_name: Optional[str] = None,
    file_type: str = "auto",
    custom_title: Optional[str] = None
) -> str:
    """
    Create interactive HTML plot from EnergyPlus output files (meter or variable outputs)
    
    Args:
        output_directory: Directory containing the CSV output files from simulation
        idf_name: Name of the IDF file (without extension). If None, auto-detects from files
        file_type: Type of file to plot - "meter", "variable", or "auto" (default: auto)
        custom_title: Custom title for the plot (optional)
    
    Returns:
        JSON string with plot creation results and file path
    """
    try:
        logger.info("Creating interactive plot from: %s", output_directory)
        result = ep_manager.create_interactive_plot(output_directory, idf_name, file_type, custom_title)
        return result
    except FileNotFoundError as e:
        logger.warning("Output files not found: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error creating interactive plot: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def convert_output_units(
    output_directory: str,
    idf_name: Optional[str] = None,
    target_energy_unit: str = "kWh",
    target_power_unit: str = "kW",
    save_converted: bool = True,
    file_type: str = "auto"
) -> str:
    """
    Convert EnergyPlus output from SI units (Joules, Watts) to natural units (kWh, kBtu, etc.)

    Args:
        output_directory: Directory containing the CSV output files from simulation
        idf_name: Name of the IDF file (without extension). If None, auto-detects from files
        target_energy_unit: Target unit for energy. Options: kWh, MWh, kBtu, MBtu, therm, GJ, MJ (default: kWh)
        target_power_unit: Target unit for power. Options: kW, MW, Btu/hr, kBtu/hr, ton (default: kW)
        save_converted: If True, saves a new CSV with "_converted" suffix (default: True)
        file_type: Type of file to convert - "meter", "variable", or "auto" (default: auto)

    Returns:
        JSON string with conversion results, statistics, and converted file path
    """
    try:
        logger.info("Converting output units in: %s", output_directory)
        result = ep_manager.convert_output_units(
            output_directory=output_directory,
            idf_name=idf_name,
            target_energy_unit=target_energy_unit,
            target_power_unit=target_power_unit,
            save_converted=save_converted,
            file_type=file_type
        )
        return result
    except FileNotFoundError as e:
        logger.warning("Output files not found: %s", e)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid parameter: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error converting output units: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_server_logs(lines: int = 50) -> str:
    """
    Get recent server log entries
    
    Args:
        lines: Number of recent log lines to return (default 50)
    
    Returns:
        Recent log entries as text
    """
    try:
        log_file = Path(config.paths.workspace_root) / "logs" / "energyplus_mcp_server.log"
        
        if not log_file.exists():
            return json.dumps({"message": "Log file not found. Server may be using console logging only."})
        
        # Read last N lines efficiently
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        log_content = {
            "log_file": str(log_file),
            "total_lines": len(all_lines),
            "showing_lines": len(recent_lines),
            "recent_logs": "".join(recent_lines)
        }
        
        return json.dumps(log_content, indent=2)
        
    except Exception as e:
        logger.error("Error reading server logs: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_error_logs(lines: int = 20) -> str:
    """
    Get recent error log entries
    
    Args:
        lines: Number of recent error lines to return (default 20)
    
    Returns:
        Recent error log entries as text
    """
    try:
        error_log_file = Path(config.paths.workspace_root) / "logs" / "energyplus_mcp_errors.log"
        
        if not error_log_file.exists():
            return json.dumps({"message": "Error log file not found. No errors logged yet."})
        
        with open(error_log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        error_content = {
            "error_log_file": str(error_log_file),
            "total_error_lines": len(all_lines),
            "showing_lines": len(recent_lines),
            "recent_errors": "".join(recent_lines)
        }
        
        return json.dumps(error_content, indent=2)
        
    except Exception as e:
        logger.error("Error reading error logs: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def clear_logs() -> str:
    """
    Clear/rotate current log files (creates backup)
    
    Returns:
        Status of log clearing operation
    """
    try:
        log_dir = Path(config.paths.workspace_root) / "logs"
        
        if not log_dir.exists():
            return json.dumps({"message": "No log directory found."})
        
        cleared_files = []
        
        # Main log file
        main_log = log_dir / "energyplus_mcp_server.log"
        if main_log.exists():
            backup_name = f"energyplus_mcp_server_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            main_log.rename(log_dir / backup_name)
            cleared_files.append(str(main_log))
        
        # Error log file
        error_log = log_dir / "energyplus_mcp_errors.log"
        if error_log.exists():
            backup_name = f"energyplus_mcp_errors_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            error_log.rename(log_dir / backup_name)
            cleared_files.append(str(error_log))
        
        result = {
            "success": True,
            "cleared_files": cleared_files,
            "backup_location": str(log_dir),
            "message": "Log files cleared and backed up successfully"
        }
        
        logger.info("Log files cleared and backed up")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("Error clearing logs: %s", e)
        return json.dumps({"error": str(e)})


# =============================================================================
# CALIBRATION TOOLS (ASHRAE Guideline 14)
# =============================================================================

# Initialize calibration orchestrator
from energyplus_mcp_server.utils.calibration_engine import CalibrationOrchestrator
calibration_orchestrator = CalibrationOrchestrator(config)


@mcp.tool()
async def compute_calibration_metrics(
    measured: List[float],
    simulated: List[float],
    standard: str = "ashrae_14_monthly"
) -> str:
    """
    Compute ASHRAE Guideline 14 calibration metrics (NMBE, CV(RMSE), R²)

    Args:
        measured: List of measured/utility values (e.g., monthly kWh from bills)
        simulated: List of simulated values from EnergyPlus (same units as measured)
        standard: Calibration standard to evaluate against. Options:
                 - "ashrae_14_monthly": NMBE ±5%, CV(RMSE) ≤15%, R² ≥0.75
                 - "ashrae_14_hourly": NMBE ±10%, CV(RMSE) ≤30%, R² ≥0.75
                 - "ipmvp_monthly": NMBE ±5%, CV(RMSE) ≤15%
                 - "ipmvp_hourly": NMBE ±10%, CV(RMSE) ≤20%
                 - "femp_monthly": NMBE ±5%, CV(RMSE) ≤15%
                 - "femp_hourly": NMBE ±10%, CV(RMSE) ≤30%

    Returns:
        JSON string with calibration metrics, pass/fail status, and interpretation

    Examples:
        # Compare 12 months of utility data to simulation
        compute_calibration_metrics(
            measured=[15000, 14500, 16000, 18000, 22000, 28000, 32000, 30000, 25000, 19000, 15500, 14800],
            simulated=[14800, 14200, 15800, 17500, 21500, 27000, 31000, 29500, 24500, 18500, 15200, 14500]
        )
    """
    try:
        logger.info("Computing calibration metrics: %s data points, standard=%s", len(measured), standard)
        result = calibration_orchestrator.compute_calibration_metrics(measured, simulated, standard)
        return result
    except Exception as e:
        logger.error("Error computing calibration metrics: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def compare_to_utility_data(
    utility_data: Dict[str, Any],
    simulated_values: List[float],
    standard: str = "ashrae_14_monthly"
) -> str:
    """
    Compare EnergyPlus simulation results against utility billing data

    Args:
        utility_data: Dictionary containing utility bill data with structure:
                     {
                         "fuel_type": "electricity",
                         "units": "kWh",
                         "bills": [
                             {"start_date": "2023-01-01", "end_date": "2023-01-31", "consumption": 15000},
                             {"start_date": "2023-02-01", "end_date": "2023-02-28", "consumption": 14500},
                             ...
                         ]
                     }
        simulated_values: List of simulated monthly values (same order as bills)
        standard: Calibration standard to use

    Returns:
        JSON string with detailed comparison, metrics, and calibration status

    Examples:
        compare_to_utility_data(
            utility_data={
                "fuel_type": "electricity",
                "units": "kWh",
                "bills": [
                    {"start_date": "2023-01-01", "end_date": "2023-01-31", "consumption": 15000},
                    {"start_date": "2023-02-01", "end_date": "2023-02-28", "consumption": 14500}
                ]
            },
            simulated_values=[14800, 14200]
        )
    """
    try:
        logger.info("Comparing simulation to utility data: %s periods", len(simulated_values))
        result = calibration_orchestrator.compare_utility_data(utility_data, simulated_values, standard)
        return result
    except Exception as e:
        logger.error("Error comparing to utility data: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def inspect_calibration_parameters(idf_path: str) -> str:
    """
    Inspect current values of standard calibration parameters in an IDF model

    Extracts values for key parameters used in ASHRAE Level 3 calibration:
    - Infiltration (ACH, flow per area)
    - Lighting power density (LPD)
    - Equipment power density (EPD)
    - Occupancy density
    - HVAC efficiencies (COP, burner efficiency)
    - Window properties (U-factor, SHGC)
    - Setpoint temperatures

    Args:
        idf_path: Path to the IDF file

    Returns:
        JSON string with current parameter values, bounds, and modification guidance
    """
    try:
        logger.info("Inspecting calibration parameters: %s", idf_path)
        result = calibration_orchestrator.inspect_parameters(idf_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error inspecting calibration parameters: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def modify_calibration_parameters(
    idf_path: str,
    modifications: Dict[str, float],
    mode: str = "set",
    output_path: Optional[str] = None
) -> str:
    """
    Modify standard calibration parameters in an IDF file

    Args:
        idf_path: Path to the input IDF file
        modifications: Dictionary of parameter names and values to apply.
                      Available parameters:
                      - infiltration_ach: Zone air changes per hour (0.1-2.0)
                      - infiltration_flow_per_area: Flow per exterior area (m3/s-m2)
                      - lighting_power_density: Watts per floor area (3-20 W/m2)
                      - equipment_power_density: Watts per floor area (3-30 W/m2)
                      - people_per_area: People per floor area (0.01-0.5 people/m2)
                      - cooling_cop: DX cooling COP (2.5-5.0)
                      - heating_efficiency: Gas burner efficiency (0.7-0.98)
                      - fan_efficiency: Fan total efficiency (0.4-0.75)
                      - window_u_factor: Window U-factor (0.5-6.0 W/m2-K)
                      - window_shgc: Solar heat gain coefficient (0.2-0.85)
        mode: "set" for absolute values, "multiply" for multipliers
        output_path: Optional output file path (auto-generated if None)

    Returns:
        JSON string with modification results and output file path

    Examples:
        # Set infiltration to 0.5 ACH and LPD to 10 W/m2
        modify_calibration_parameters("model.idf",
            {"infiltration_ach": 0.5, "lighting_power_density": 10.0})

        # Multiply infiltration by 0.8 (reduce by 20%)
        modify_calibration_parameters("model.idf",
            {"infiltration_ach": 0.8}, mode="multiply")
    """
    try:
        logger.info("Modifying calibration parameters: %s, mode=%s", idf_path, mode)
        result = calibration_orchestrator.modify_parameters(idf_path, modifications, mode, output_path)
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except ValueError as e:
        logger.warning("Invalid parameter modification: %s", e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error modifying calibration parameters: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_calibration_standards() -> str:
    """
    Get available calibration standards and their acceptance thresholds

    Returns information about ASHRAE Guideline 14, IPMVP, and FEMP standards
    for both monthly and hourly calibration data.

    Returns:
        JSON string with standards and their NMBE, CV(RMSE), and R² thresholds
    """
    try:
        logger.info("Getting calibration standards")
        result = calibration_orchestrator.get_calibration_standards()
        return result
    except Exception as e:
        logger.error("Error getting calibration standards: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def setup_sensitivity_analysis(
    idf_path: str,
    parameters: Optional[List[str]] = None,
    method: str = "local",
    perturbation_percent: float = 10.0
) -> str:
    """
    Set up sensitivity analysis for calibration parameter screening

    Sensitivity analysis identifies which parameters have the greatest
    influence on model outputs, allowing focused calibration efforts.

    Args:
        idf_path: Path to the IDF file
        parameters: List of parameter names to analyze (None = all standard parameters)
        method: Analysis method:
               - "local": One-at-a-time perturbation (no SALib required)
               - "morris": Morris screening method (requires SALib)
        perturbation_percent: Percent to perturb for local method (default: 10)

    Returns:
        JSON string with analysis setup, required simulations, and parameter cases

    Examples:
        # Set up local sensitivity for all parameters
        setup_sensitivity_analysis("model.idf")

        # Focus on specific parameters
        setup_sensitivity_analysis("model.idf",
            parameters=["infiltration_ach", "lighting_power_density", "equipment_power_density"])
    """
    try:
        logger.info("Setting up sensitivity analysis: %s, method=%s", idf_path, method)
        if method == "morris":
            result = calibration_orchestrator.sensitivity_manager.setup_morris_analysis(parameters)
        else:
            result = calibration_orchestrator.sensitivity_manager.setup_local_sensitivity(
                idf_path, parameters, perturbation_percent
            )
        return result
    except FileNotFoundError as e:
        logger.warning("IDF file not found: %s", idf_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error setting up sensitivity analysis: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_calibration_parameter_recommendations(
    category: Optional[str] = None
) -> str:
    """
    Get recommended parameters for calibration, organized by sensitivity category

    Returns parameters in recommended calibration order (highest sensitivity first):
    1. Schedules (occupancy, HVAC, lighting)
    2. Infiltration
    3. Internal loads (lighting, equipment)
    4. HVAC efficiency
    5. Envelope properties
    6. Setpoints

    Args:
        category: Optional category filter. Options:
                 - "schedules"
                 - "infiltration"
                 - "internal_loads"
                 - "hvac"
                 - "envelope"
                 - "setpoints"
                 - None (all categories)

    Returns:
        JSON string with parameter recommendations, bounds, and typical values
    """
    try:
        logger.info("Getting calibration parameter recommendations: category=%s", category)
        result = calibration_orchestrator.get_parameter_recommendations(category)
        return result
    except Exception as e:
        logger.error("Error getting parameter recommendations: %s", e)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def load_utility_data(
    csv_path: str,
    format: str = "auto"
) -> str:
    """
    Load utility billing data from a CSV file for calibration

    Args:
        csv_path: Path to the utility data CSV file. Can be:
                 - Absolute path: "/full/path/to/utility.csv"
                 - Filename in sample_files: "slo_utility_data.csv"
        format: CSV format type:
               - "auto": Auto-detect format from column names
               - "slo": SLO format (billing_start, billing_end, kwh, therms, etc.)
               - "simple": Simple format (date, consumption columns)

    Returns:
        JSON string with parsed utility data, summary statistics, and ready-to-use format

    Examples:
        # Load the sample SLO utility data
        load_utility_data("slo_utility_data.csv")

        # Load custom utility file
        load_utility_data("/path/to/my_utility_bills.csv", format="simple")
    """
    from energyplus_mcp_server.utils.calibration import UtilityData
    from energyplus_mcp_server.utils.path_utils import resolve_path

    try:
        logger.info("Loading utility data: %s", csv_path)

        # Resolve path (check sample_files if just filename)
        resolved_path = resolve_path(config, csv_path, file_types=['.csv'], description="utility data CSV")

        # Detect format
        import pandas as pd
        df = pd.read_csv(resolved_path)
        columns = df.columns.tolist()

        if format == "auto":
            if 'billing_start' in columns and 'kwh' in columns:
                format = "slo"
            else:
                format = "simple"

        # Load data based on format
        if format == "slo":
            elec_data, gas_data = UtilityData.from_slo_format(resolved_path)

            result = {
                "success": True,
                "file_path": resolved_path,
                "format_detected": "slo",
                "electricity": {
                    "fuel_type": elec_data.fuel_type,
                    "units": elec_data.units,
                    "n_periods": len(elec_data.bills),
                    "total_consumption": float(elec_data.get_consumption_array().sum()),
                    "monthly_values": elec_data.get_consumption_array().tolist(),
                    "date_range": f"{elec_data.bills[0].start_date} to {elec_data.bills[-1].end_date}",
                    "utility_data_dict": {
                        "fuel_type": elec_data.fuel_type,
                        "units": elec_data.units,
                        "bills": [
                            {
                                "start_date": b.start_date,
                                "end_date": b.end_date,
                                "consumption": b.consumption,
                                "demand": b.demand,
                                "cost": b.cost
                            }
                            for b in elec_data.bills
                        ]
                    }
                }
            }

            if gas_data:
                result["natural_gas"] = {
                    "fuel_type": gas_data.fuel_type,
                    "units": gas_data.units,
                    "n_periods": len(gas_data.bills),
                    "total_consumption": float(gas_data.get_consumption_array().sum()),
                    "monthly_values": gas_data.get_consumption_array().tolist(),
                    "utility_data_dict": {
                        "fuel_type": gas_data.fuel_type,
                        "units": gas_data.units,
                        "bills": [
                            {
                                "start_date": b.start_date,
                                "end_date": b.end_date,
                                "consumption": b.consumption
                            }
                            for b in gas_data.bills
                        ]
                    }
                }

        else:
            # Simple format
            util_data = UtilityData.from_csv(resolved_path)
            result = {
                "success": True,
                "file_path": resolved_path,
                "format_detected": "simple",
                "data": {
                    "fuel_type": util_data.fuel_type,
                    "units": util_data.units,
                    "n_periods": len(util_data.bills),
                    "total_consumption": float(util_data.get_consumption_array().sum()),
                    "monthly_values": util_data.get_consumption_array().tolist()
                }
            }

        return json.dumps(result, indent=2)

    except FileNotFoundError as e:
        logger.warning("Utility data file not found: %s", csv_path)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Error loading utility data: %s", e)
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    logger.info("Starting %s v%s", config.server.name, config.server.version)
    logger.info("EnergyPlus version: %s", config.energyplus.version)
    logger.info("Sample files path: %s", config.paths.sample_files_path)
    
    try:
        # Use FastMCP's built-in run method with stdio transport
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error("Server error: %s", e)
        raise
    finally:
        logger.info("Server stopped")
