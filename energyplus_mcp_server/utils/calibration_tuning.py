"""
Calibration parameter tuning utilities for EnergyPlus models

Provides structured parameter modification for model calibration,
with support for parameter bounds, multipliers, and batch updates.

EnergyPlus Model Context Protocol Server (EnergyPlus-MCP)
Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import copy

from eppy.modeleditor import IDF

logger = logging.getLogger(__name__)


class ParameterCategory(Enum):
    """Categories of calibration parameters by typical sensitivity"""
    SCHEDULES = "schedules"           # Highest sensitivity - adjust first
    INFILTRATION = "infiltration"      # High sensitivity
    INTERNAL_LOADS = "internal_loads"  # High sensitivity
    HVAC = "hvac"                      # Medium-high sensitivity
    ENVELOPE = "envelope"              # Medium sensitivity
    SETPOINTS = "setpoints"            # Lower sensitivity


@dataclass
class ParameterSpec:
    """Specification for a tunable calibration parameter"""
    name: str
    idf_object_type: str
    field_name: str
    current_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[float] = None
    units: Optional[str] = None
    category: ParameterCategory = ParameterCategory.INTERNAL_LOADS
    description: Optional[str] = None
    object_name_filter: Optional[str] = None  # Filter by object name pattern

    def validate_value(self, value: float) -> Tuple[bool, str]:
        """Validate a value against bounds"""
        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} below minimum {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} above maximum {self.max_value}"
        return True, "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "idf_object_type": self.idf_object_type,
            "field_name": self.field_name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "units": self.units,
            "category": self.category.value,
            "description": self.description
        }


# Standard calibration parameters with typical ranges
STANDARD_PARAMETERS = {
    # Infiltration parameters
    "infiltration_ach": ParameterSpec(
        name="infiltration_ach",
        idf_object_type="ZoneInfiltration:DesignFlowRate",
        field_name="Air_Changes_per_Hour",
        min_value=0.1,
        max_value=2.0,
        default_value=0.5,
        units="ACH",
        category=ParameterCategory.INFILTRATION,
        description="Zone air changes per hour from infiltration"
    ),
    "infiltration_flow_per_area": ParameterSpec(
        name="infiltration_flow_per_area",
        idf_object_type="ZoneInfiltration:DesignFlowRate",
        field_name="Flow_per_Exterior_Surface_Area",
        min_value=0.0001,
        max_value=0.001,
        default_value=0.0003,
        units="m3/s-m2",
        category=ParameterCategory.INFILTRATION,
        description="Infiltration flow rate per exterior surface area"
    ),

    # Lighting parameters
    "lighting_power_density": ParameterSpec(
        name="lighting_power_density",
        idf_object_type="Lights",
        field_name="Watts_per_Zone_Floor_Area",
        min_value=3.0,
        max_value=20.0,
        default_value=10.0,
        units="W/m2",
        category=ParameterCategory.INTERNAL_LOADS,
        description="Lighting power density"
    ),

    # Equipment parameters
    "equipment_power_density": ParameterSpec(
        name="equipment_power_density",
        idf_object_type="ElectricEquipment",
        field_name="Watts_per_Zone_Floor_Area",
        min_value=3.0,
        max_value=30.0,
        default_value=10.0,
        units="W/m2",
        category=ParameterCategory.INTERNAL_LOADS,
        description="Electric equipment power density"
    ),

    # Occupancy parameters
    "people_per_area": ParameterSpec(
        name="people_per_area",
        idf_object_type="People",
        field_name="People_per_Floor_Area",
        min_value=0.01,
        max_value=0.5,
        default_value=0.1,
        units="people/m2",
        category=ParameterCategory.INTERNAL_LOADS,
        description="Occupancy density"
    ),

    # HVAC efficiency parameters
    "cooling_cop": ParameterSpec(
        name="cooling_cop",
        idf_object_type="Coil:Cooling:DX:SingleSpeed",
        field_name="Gross_Rated_COP",
        min_value=2.5,
        max_value=5.0,
        default_value=3.5,
        units="W/W",
        category=ParameterCategory.HVAC,
        description="Cooling coil coefficient of performance"
    ),
    "heating_efficiency": ParameterSpec(
        name="heating_efficiency",
        idf_object_type="Coil:Heating:Fuel",
        field_name="Burner_Efficiency",
        min_value=0.7,
        max_value=0.98,
        default_value=0.8,
        units="fraction",
        category=ParameterCategory.HVAC,
        description="Heating coil/furnace burner efficiency"
    ),
    "fan_efficiency": ParameterSpec(
        name="fan_efficiency",
        idf_object_type="Fan:VariableVolume",
        field_name="Fan_Total_Efficiency",
        min_value=0.4,
        max_value=0.75,
        default_value=0.6,
        units="fraction",
        category=ParameterCategory.HVAC,
        description="Fan total efficiency"
    ),

    # Envelope parameters
    "window_u_factor": ParameterSpec(
        name="window_u_factor",
        idf_object_type="WindowMaterial:SimpleGlazingSystem",
        field_name="UFactor",
        min_value=0.5,
        max_value=6.0,
        default_value=2.5,
        units="W/m2-K",
        category=ParameterCategory.ENVELOPE,
        description="Window U-factor"
    ),
    "window_shgc": ParameterSpec(
        name="window_shgc",
        idf_object_type="WindowMaterial:SimpleGlazingSystem",
        field_name="Solar_Heat_Gain_Coefficient",
        min_value=0.2,
        max_value=0.85,
        default_value=0.5,
        units="fraction",
        category=ParameterCategory.ENVELOPE,
        description="Window solar heat gain coefficient"
    ),

    # Setpoint parameters
    "cooling_setpoint": ParameterSpec(
        name="cooling_setpoint",
        idf_object_type="ThermostatSetpoint:DualSetpoint",
        field_name="Cooling_Setpoint_Temperature_Schedule_Name",
        min_value=20.0,
        max_value=28.0,
        default_value=24.0,
        units="C",
        category=ParameterCategory.SETPOINTS,
        description="Cooling setpoint temperature"
    ),
    "heating_setpoint": ParameterSpec(
        name="heating_setpoint",
        idf_object_type="ThermostatSetpoint:DualSetpoint",
        field_name="Heating_Setpoint_Temperature_Schedule_Name",
        min_value=15.0,
        max_value=23.0,
        default_value=21.0,
        units="C",
        category=ParameterCategory.SETPOINTS,
        description="Heating setpoint temperature"
    ),
}


@dataclass
class ParameterModification:
    """Record of a parameter modification"""
    parameter_name: str
    object_type: str
    object_name: str
    field_name: str
    old_value: Any
    new_value: Any
    modification_type: str  # 'set', 'multiply', 'add'


class CalibrationTuner:
    """
    Manages parameter modifications for calibration

    Supports individual parameter changes, batch modifications,
    and multiplier-based adjustments.
    """

    def __init__(self, config=None):
        self.config = config
        self.modification_history: List[ParameterModification] = []

    def get_available_parameters(self) -> Dict[str, Any]:
        """Get dictionary of all standard calibration parameters"""
        params_by_category = {}
        for name, spec in STANDARD_PARAMETERS.items():
            category = spec.category.value
            if category not in params_by_category:
                params_by_category[category] = []
            params_by_category[category].append(spec.to_dict())

        return {
            "parameters_by_category": params_by_category,
            "total_parameters": len(STANDARD_PARAMETERS),
            "categories": [c.value for c in ParameterCategory]
        }

    def extract_current_values(
        self,
        idf: IDF,
        parameter_names: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract current parameter values from an IDF

        Args:
            idf: Loaded IDF object
            parameter_names: List of parameter names to extract (None = all)

        Returns:
            Dictionary with parameter values organized by name
        """
        if parameter_names is None:
            parameter_names = list(STANDARD_PARAMETERS.keys())

        results = {}

        for param_name in parameter_names:
            if param_name not in STANDARD_PARAMETERS:
                logger.warning(f"Unknown parameter: {param_name}")
                continue

            spec = STANDARD_PARAMETERS[param_name]
            objects = idf.idfobjects.get(spec.idf_object_type, [])

            values = []
            for obj in objects:
                try:
                    value = getattr(obj, spec.field_name, None)
                    if value is not None and value != '':
                        values.append({
                            "object_name": getattr(obj, 'Name', 'unnamed'),
                            "value": float(value) if isinstance(value, (int, float, str)) and value != '' else value,
                            "field": spec.field_name
                        })
                except (ValueError, AttributeError):
                    pass

            results[param_name] = {
                "spec": spec.to_dict(),
                "values": values,
                "count": len(values)
            }

        return results

    def set_parameter(
        self,
        idf: IDF,
        parameter_name: str,
        new_value: float,
        object_filter: Optional[str] = None,
        validate: bool = True
    ) -> List[ParameterModification]:
        """
        Set a calibration parameter to a specific value

        Args:
            idf: IDF object to modify
            parameter_name: Name of the parameter (from STANDARD_PARAMETERS)
            new_value: New value to set
            object_filter: Optional regex pattern to filter object names
            validate: Whether to validate value against bounds

        Returns:
            List of modifications made
        """
        if parameter_name not in STANDARD_PARAMETERS:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        spec = STANDARD_PARAMETERS[parameter_name]

        # Validate
        if validate:
            valid, msg = spec.validate_value(new_value)
            if not valid:
                raise ValueError(f"Invalid value for {parameter_name}: {msg}")

        modifications = []
        objects = idf.idfobjects.get(spec.idf_object_type, [])

        import re
        for obj in objects:
            obj_name = getattr(obj, 'Name', 'unnamed')

            # Apply filter if specified
            if object_filter and not re.search(object_filter, obj_name, re.IGNORECASE):
                continue

            try:
                old_value = getattr(obj, spec.field_name, None)
                setattr(obj, spec.field_name, new_value)

                mod = ParameterModification(
                    parameter_name=parameter_name,
                    object_type=spec.idf_object_type,
                    object_name=obj_name,
                    field_name=spec.field_name,
                    old_value=old_value,
                    new_value=new_value,
                    modification_type='set'
                )
                modifications.append(mod)
                self.modification_history.append(mod)

                logger.debug(f"Set {parameter_name} in {obj_name}: {old_value} -> {new_value}")

            except Exception as e:
                logger.warning(f"Could not set {parameter_name} in {obj_name}: {e}")

        return modifications

    def multiply_parameter(
        self,
        idf: IDF,
        parameter_name: str,
        multiplier: float,
        object_filter: Optional[str] = None,
        clamp_to_bounds: bool = True
    ) -> List[ParameterModification]:
        """
        Multiply a calibration parameter by a factor

        Args:
            idf: IDF object to modify
            parameter_name: Name of the parameter
            multiplier: Multiplication factor
            object_filter: Optional regex pattern to filter object names
            clamp_to_bounds: Whether to clamp result to parameter bounds

        Returns:
            List of modifications made
        """
        if parameter_name not in STANDARD_PARAMETERS:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        spec = STANDARD_PARAMETERS[parameter_name]
        modifications = []
        objects = idf.idfobjects.get(spec.idf_object_type, [])

        import re
        for obj in objects:
            obj_name = getattr(obj, 'Name', 'unnamed')

            if object_filter and not re.search(object_filter, obj_name, re.IGNORECASE):
                continue

            try:
                old_value = getattr(obj, spec.field_name, None)
                if old_value is None or old_value == '':
                    continue

                old_value = float(old_value)
                new_value = old_value * multiplier

                # Clamp to bounds
                if clamp_to_bounds:
                    if spec.min_value is not None:
                        new_value = max(new_value, spec.min_value)
                    if spec.max_value is not None:
                        new_value = min(new_value, spec.max_value)

                setattr(obj, spec.field_name, new_value)

                mod = ParameterModification(
                    parameter_name=parameter_name,
                    object_type=spec.idf_object_type,
                    object_name=obj_name,
                    field_name=spec.field_name,
                    old_value=old_value,
                    new_value=new_value,
                    modification_type='multiply'
                )
                modifications.append(mod)
                self.modification_history.append(mod)

                logger.debug(f"Multiplied {parameter_name} in {obj_name}: {old_value} * {multiplier} = {new_value}")

            except (ValueError, TypeError) as e:
                logger.warning(f"Could not multiply {parameter_name} in {obj_name}: {e}")

        return modifications

    def apply_parameter_set(
        self,
        idf: IDF,
        parameters: Dict[str, float],
        mode: str = "set"
    ) -> Dict[str, List[ParameterModification]]:
        """
        Apply a set of parameter values

        Args:
            idf: IDF object to modify
            parameters: Dictionary of parameter_name -> value
            mode: 'set' for absolute values, 'multiply' for multipliers

        Returns:
            Dictionary of modifications by parameter name
        """
        all_modifications = {}

        for param_name, value in parameters.items():
            if mode == "set":
                mods = self.set_parameter(idf, param_name, value)
            elif mode == "multiply":
                mods = self.multiply_parameter(idf, param_name, value)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            all_modifications[param_name] = mods

        return all_modifications

    def set_custom_parameter(
        self,
        idf: IDF,
        object_type: str,
        field_name: str,
        new_value: Any,
        object_name: Optional[str] = None
    ) -> List[ParameterModification]:
        """
        Set a custom (non-standard) parameter

        Args:
            idf: IDF object to modify
            object_type: EnergyPlus object type (e.g., "Material")
            field_name: Field name to modify
            new_value: New value to set
            object_name: Optional specific object name (None = all)

        Returns:
            List of modifications made
        """
        modifications = []
        objects = idf.idfobjects.get(object_type, [])

        for obj in objects:
            obj_name = getattr(obj, 'Name', 'unnamed')

            if object_name and obj_name != object_name:
                continue

            try:
                old_value = getattr(obj, field_name, None)
                setattr(obj, field_name, new_value)

                mod = ParameterModification(
                    parameter_name=f"{object_type}.{field_name}",
                    object_type=object_type,
                    object_name=obj_name,
                    field_name=field_name,
                    old_value=old_value,
                    new_value=new_value,
                    modification_type='set'
                )
                modifications.append(mod)
                self.modification_history.append(mod)

            except Exception as e:
                logger.warning(f"Could not set {object_type}.{field_name} in {obj_name}: {e}")

        return modifications

    def get_modification_summary(self) -> Dict[str, Any]:
        """Get summary of all modifications made"""
        by_type = {}
        by_parameter = {}

        for mod in self.modification_history:
            # By type
            if mod.modification_type not in by_type:
                by_type[mod.modification_type] = 0
            by_type[mod.modification_type] += 1

            # By parameter
            if mod.parameter_name not in by_parameter:
                by_parameter[mod.parameter_name] = []
            by_parameter[mod.parameter_name].append({
                "object": mod.object_name,
                "old": mod.old_value,
                "new": mod.new_value
            })

        return {
            "total_modifications": len(self.modification_history),
            "by_type": by_type,
            "by_parameter": by_parameter
        }

    def clear_history(self):
        """Clear modification history"""
        self.modification_history = []


class CalibrationParameterManager:
    """
    High-level interface for calibration parameter operations

    Integrates with MCP server for parameter inspection and modification.
    """

    def __init__(self, config=None):
        self.config = config
        self.tuner = CalibrationTuner(config)

    def inspect_calibration_parameters(self, idf_path: str) -> str:
        """
        Inspect current calibration parameter values in an IDF

        Args:
            idf_path: Path to the IDF file

        Returns:
            JSON string with parameter values and specifications
        """
        try:
            idf = IDF(idf_path)
            values = self.tuner.extract_current_values(idf)

            result = {
                "success": True,
                "idf_path": idf_path,
                "parameters": values,
                "available_parameters": list(STANDARD_PARAMETERS.keys())
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error inspecting parameters: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def modify_calibration_parameters(
        self,
        idf_path: str,
        modifications: Dict[str, float],
        mode: str = "set",
        output_path: Optional[str] = None
    ) -> str:
        """
        Modify calibration parameters in an IDF

        Args:
            idf_path: Path to the input IDF file
            modifications: Dict of parameter_name -> value
            mode: 'set' or 'multiply'
            output_path: Optional output path (None = auto-generate)

        Returns:
            JSON string with modification results
        """
        try:
            idf = IDF(idf_path)

            # Apply modifications
            self.tuner.clear_history()
            mods = self.tuner.apply_parameter_set(idf, modifications, mode)

            # Generate output path
            if output_path is None:
                input_path = Path(idf_path)
                output_path = str(input_path.parent / f"{input_path.stem}_tuned{input_path.suffix}")

            # Save modified IDF
            idf.saveas(output_path)

            result = {
                "success": True,
                "input_path": idf_path,
                "output_path": output_path,
                "mode": mode,
                "modifications": {
                    k: [{"object": m.object_name, "old": m.old_value, "new": m.new_value}
                        for m in v]
                    for k, v in mods.items()
                },
                "summary": self.tuner.get_modification_summary()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error modifying parameters: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def get_parameter_info(self, parameter_name: Optional[str] = None) -> str:
        """
        Get information about calibration parameters

        Args:
            parameter_name: Specific parameter name (None = all)

        Returns:
            JSON string with parameter specifications
        """
        if parameter_name:
            if parameter_name not in STANDARD_PARAMETERS:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown parameter: {parameter_name}"
                })
            spec = STANDARD_PARAMETERS[parameter_name]
            return json.dumps({
                "success": True,
                "parameter": spec.to_dict()
            }, indent=2)

        return json.dumps({
            "success": True,
            "parameters": self.tuner.get_available_parameters()
        }, indent=2)

    def create_parameter_sweep(
        self,
        parameter_name: str,
        n_values: int = 5,
        range_fraction: float = 0.2
    ) -> str:
        """
        Create a parameter sweep for sensitivity analysis

        Args:
            parameter_name: Parameter to sweep
            n_values: Number of values in sweep
            range_fraction: Fraction of range to sweep (centered on default)

        Returns:
            JSON string with sweep values
        """
        if parameter_name not in STANDARD_PARAMETERS:
            return json.dumps({
                "success": False,
                "error": f"Unknown parameter: {parameter_name}"
            })

        spec = STANDARD_PARAMETERS[parameter_name]

        if spec.min_value is None or spec.max_value is None:
            return json.dumps({
                "success": False,
                "error": f"Parameter {parameter_name} has no defined bounds"
            })

        # Calculate sweep range
        full_range = spec.max_value - spec.min_value
        sweep_range = full_range * range_fraction
        center = spec.default_value or (spec.min_value + spec.max_value) / 2

        low = max(spec.min_value, center - sweep_range / 2)
        high = min(spec.max_value, center + sweep_range / 2)

        import numpy as np
        values = np.linspace(low, high, n_values).tolist()

        return json.dumps({
            "success": True,
            "parameter": parameter_name,
            "sweep_values": values,
            "bounds": {"min": spec.min_value, "max": spec.max_value},
            "center": center,
            "units": spec.units
        }, indent=2)
