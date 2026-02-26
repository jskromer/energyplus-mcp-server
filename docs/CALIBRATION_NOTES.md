# EnergyPlus Calibration - Research Notes

**Saved:** 2025-12-19
**Status:** ✅ IMPLEMENTED - Calibration tools added to MCP server

---

## ASHRAE Level 3 Audit Data → EnergyPlus Calibration

### Standard Format: BuildingSync

[BuildingSync](https://buildingsync.net/) is the DOE-backed XML schema aligned with ASHRAE Standard 211.

| Feature | Details |
|---------|---------|
| **Schema** | XML-based, BEDES-compliant |
| **Coverage** | ASHRAE Level 1, 2, and 3 audits |
| **Integration** | DOE Audit Template, Asset Score, SEED |
| **Translator** | BuildingSync-gem converts to OpenStudio/EnergyPlus |

### Key Data Categories in BuildingSync

```
BuildingSync Schema
├── Facility
│   ├── Site (location, climate zone)
│   └── Building
│       ├── Sections (spaces, floor areas)
│       ├── Systems
│       │   ├── HVACSystems (equipment, efficiency, capacity)
│       │   ├── LightingSystems (power density, controls)
│       │   ├── PlugLoads (equipment density)
│       │   └── EnvelopeSystems (walls, roofs, windows)
│       ├── Schedules (occupancy, operations)
│       └── Contacts
├── Measures (ECMs with costs/savings)
└── Reports (utility data, benchmarking)
```

### Calibration Parameters Mapping

For auto-tuning, map **audit-collected data** to **EnergyPlus tunable parameters**:

| Audit Data (BuildingSync) | EnergyPlus Object | Tunable Parameter |
|---------------------------|-------------------|-------------------|
| Wall R-value | `Material` / `Construction` | Conductivity, thickness |
| Window U-factor, SHGC | `WindowMaterial:SimpleGlazingSystem` | U-Factor, SHGC |
| Infiltration rate | `ZoneInfiltration:DesignFlowRate` | Flow per exterior area |
| Lighting power density | `Lights` | Watts per zone floor area |
| Equipment power density | `ElectricEquipment` | Watts per zone floor area |
| Occupancy density | `People` | People per zone floor area |
| HVAC efficiency | `Coil:Cooling:DX:*`, `Boiler:*` | COP, efficiency |
| Setpoint temps | `ThermostatSetpoint:*` | Heating/cooling setpoints |
| Operating schedules | `Schedule:Compact` | Hourly fractions |

### Proposed Tools to Create

1. **Import BuildingSync XML** → Parse audit data into a calibration parameter set
2. **Generate calibration bounds** → Define min/max ranges based on audit uncertainty
3. **Apply parameters to IDF** → Modify EnergyPlus model with audit values
4. **Compare to measured data** → Calculate NMBE/CVRMSE against utility bills

### Alternative: JSON-based Calibration Format

Simpler format for direct use without full BuildingSync complexity.

---

## Key Resources

- [BuildingSync](https://buildingsync.net/)
- [DOE Audit Template](https://www.energy.gov/eere/buildings/audit-template)
- [BuildingSync GitHub Schema](https://github.com/BuildingSync/schema)
- [BuildingSync-to-OpenStudio Gem](https://github.com/BuildingSync/BuildingSync-gem)
- [ASHRAE Standard 211](https://www.ashrae.org/technical-resources/bookstore/procedures-for-commercial-building-energy-audits)

---

## Implementation Status

### Completed Tools (2025-12-19)

- [x] **Calibration Metrics Module** (`utils/calibration.py`)
  - NMBE, CV(RMSE), R² calculation per ASHRAE Guideline 14
  - Support for ASHRAE 14, IPMVP, and FEMP standards
  - Monthly and hourly calibration thresholds
  - Utility data parsing and comparison

- [x] **Parameter Tuning Module** (`utils/calibration_tuning.py`)
  - Standard calibration parameters with bounds
  - Set and multiply modification modes
  - Parameter validation against physical limits
  - Organized by sensitivity category

- [x] **Sensitivity Analysis Module** (`utils/sensitivity_analysis.py`)
  - Morris one-at-a-time (OAT) method (requires SALib)
  - Local sensitivity analysis (no dependencies)
  - Parameter ranking by influence
  - Calibration priority recommendations

- [x] **Calibration Engine** (`utils/calibration_engine.py`)
  - Orchestrates complete calibration workflow
  - Session management for iterative calibration
  - Integration with EnergyPlus simulation
  - Suggestion engine for parameter adjustments

### New MCP Tools

| Tool | Description |
|------|-------------|
| `compute_calibration_metrics` | Calculate NMBE, CV(RMSE), R² |
| `compare_to_utility_data` | Compare simulation to utility bills |
| `inspect_calibration_parameters` | Extract current parameter values |
| `modify_calibration_parameters` | Adjust parameters for calibration |
| `get_calibration_standards` | List available standards/thresholds |
| `setup_sensitivity_analysis` | Configure sensitivity study |
| `get_calibration_parameter_recommendations` | Get parameter priority order |

### Standard Parameters Supported

| Category | Parameters |
|----------|------------|
| Infiltration | ACH, flow per exterior area |
| Internal Loads | LPD (W/m²), EPD (W/m²), occupancy density |
| HVAC | Cooling COP, heating efficiency, fan efficiency |
| Envelope | Window U-factor, SHGC |
| Setpoints | Heating/cooling temperatures |

---

## Original Research Notes

(Preserved below for reference)
