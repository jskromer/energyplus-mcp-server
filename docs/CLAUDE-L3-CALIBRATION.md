# ASHRAE Level 3 Audit & EnergyPlus Calibration Workflow
## San Luis Obispo Example Building

**Purpose**: This document provides Claude Code with the methodology, data requirements, and Python tooling patterns for calibrating an EnergyPlus model against measured data from an ASHRAE Level 3 audit.

---

## 1. Project Context

### Climate Information
- **Location**: San Luis Obispo, California
- **California Climate Zone**: CZ 5 (Coastal Central California)
- **ASHRAE Climate Zone**: 3C (Marine)
- **Lat/Long**: 35.28°N, 120.66°W
- **Elevation**: ~70m (varies by site)

### Weather Data Sources
```bash
# TMY3 Data (typical meteorological year)
# Download from climate.onebuilding.org or EnergyPlus weather portal
# Nearest station: San Luis Obispo County Regional Airport (KSBP)
# WMO: 723910

# Option 1: TMY3 for baseline/design
USA_CA_San.Luis.Obispo-McChesney.Field.723910_TMY3.epw

# Option 2: TMYx (more recent 2004-2018 data)
USA_CA_San.Luis.Obispo.County.Rgnl.AP.723910_TMYx.2004-2018.epw

# Option 3: Actual Meteorological Year (AMY) for calibration
# Use NOAA ISD Lite data via diyepw package
pip install diyepw
```

**Critical for Calibration**: Use AMY (actual year) weather data matching the utility billing period, not TMY. TMY is for design; AMY is for calibration.

---

## 2. ASHRAE Level 3 Audit Data Requirements

### 2.1 Building Documentation
```
Required from audit:
├── As-built drawings (floor plans, elevations, sections)
├── Mechanical schedules
├── Electrical one-lines
├── Control sequences
├── Equipment cut sheets
└── Previous energy audits/retro-Cx reports
```

### 2.2 Utility Data (Minimum 12 months)
```python
# Utility data structure
utility_data = {
    "electricity": {
        "monthly_kwh": [...],  # 12+ months
        "monthly_kw_demand": [...],
        "billing_dates": [...],
        "rate_schedule": "PG&E E-19"
    },
    "natural_gas": {
        "monthly_therms": [...],
        "billing_dates": [...]
    }
}
```

### 2.3 Level 3-Specific Measurements
Per ASHRAE Standard 211, Level 3 requires **extended monitoring**:

| Data Type | Duration | Interval | Equipment |
|-----------|----------|----------|-----------|
| Main meter kW | 2-4 weeks min | 15-min | CT loggers or BAS trend |
| HVAC submetering | 2-4 weeks | 15-min | Portable CTs |
| Zone temperatures | 2-4 weeks | 5-15 min | Wireless loggers (HOBO) |
| Outdoor air temp/RH | Continuous | Hourly | Weather station or BAS |
| Lighting schedules | 1-2 weeks | 5-min | Occupancy/light loggers |
| Plug loads | 1-2 weeks | 15-min | Plug load meters |
| AHU runtime/status | 2-4 weeks | 1-15 min | BAS trend or CT |

### 2.4 Envelope Data
```python
envelope_audit = {
    "walls": {
        "construction": "Stucco/wood frame",
        "insulation_r": 13,  # Audit-verified or assumed
        "area_sf": 8500,
        "absorptance": 0.6  # Light colored stucco
    },
    "roof": {
        "construction": "Built-up with gravel",
        "insulation_r": 19,
        "area_sf": 12000,
        "absorptance": 0.7
    },
    "windows": {
        "u_factor": 0.57,  # Single pane aluminum
        "shgc": 0.70,
        "area_sf": 1200,
        "frame": "aluminum_no_break"
    },
    "infiltration": {
        "ach50": None,  # Blower door if available
        "assumed_ach": 0.5  # Typical for vintage
    }
}
```

### 2.5 HVAC Systems Inventory
```python
hvac_systems = {
    "cooling": [{
        "type": "Packaged_DX",
        "capacity_tons": 25,
        "eer": 10.5,
        "age_years": 15,
        "zones_served": ["Zone1", "Zone2", "Zone3"]
    }],
    "heating": [{
        "type": "Gas_Furnace",
        "capacity_mbh": 400,
        "efficiency": 0.80,
        "age_years": 15
    }],
    "ventilation": [{
        "type": "CAV",
        "cfm": 4000,
        "economizer": True,
        "doas": False
    }]
}
```

### 2.6 Internal Loads Schedule Data
```python
# From BAS trends, occupancy surveys, or nameplate surveys
schedules = {
    "occupancy": {
        "weekday": [0,0,0,0,0,0,0.1,0.5,0.9,1.0,1.0,0.8,0.5,0.9,1.0,1.0,0.8,0.3,0.1,0,0,0,0,0],
        "weekend": [0]*24,
        "peak_occupants": 150
    },
    "lighting": {
        "weekday": [0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.7,0.9,0.9,0.9,0.9,0.8,0.9,0.9,0.9,0.7,0.3,0.2,0.1,0.1,0.1,0.1,0.1],
        "lpd_w_sf": 1.2
    },
    "equipment": {
        "weekday": [0.3,0.3,0.3,0.3,0.3,0.3,0.4,0.6,0.9,0.9,0.9,0.9,0.8,0.9,0.9,0.9,0.6,0.4,0.3,0.3,0.3,0.3,0.3,0.3],
        "epd_w_sf": 1.5
    }
}
```

---

## 3. ASHRAE Guideline 14 Calibration Criteria

### Statistical Indices
```python
import numpy as np

def nmbe(measured, simulated):
    """Normalized Mean Bias Error (%)"""
    n = len(measured)
    return 100 * np.sum(measured - simulated) / ((n - 1) * np.mean(measured))

def cvrmse(measured, simulated):
    """Coefficient of Variation of RMSE (%)"""
    n = len(measured)
    rmse = np.sqrt(np.sum((measured - simulated)**2) / (n - 1))
    return 100 * rmse / np.mean(measured)

def r_squared(measured, simulated):
    """Coefficient of determination"""
    ss_res = np.sum((measured - simulated)**2)
    ss_tot = np.sum((measured - np.mean(measured))**2)
    return 1 - (ss_res / ss_tot)
```

### Acceptance Thresholds

| Standard | Data Type | NMBE | CV(RMSE) | R² |
|----------|-----------|------|----------|-----|
| ASHRAE 14 | Monthly | ±5% | ≤15% | ≥0.75 |
| ASHRAE 14 | Hourly | ±10% | ≤30% | ≥0.75 |
| IPMVP | Monthly | ±5% | ≤15% | - |
| IPMVP | Hourly | ±10% | ≤20% | - |
| FEMP | Monthly | ±5% | ≤15% | - |
| FEMP | Hourly | ±10% | ≤30% | - |

**Target for this workflow**: Monthly calibration to ASHRAE 14 (NMBE ±5%, CV(RMSE) ≤15%)

---

## 4. Calibration Workflow

### Phase 1: Baseline Model Development

```
1. Geometry from drawings/survey
   └── Use OpenStudio or manual IDF creation
   
2. Envelope from audit data
   └── Apply measured/estimated R-values, U-factors
   
3. HVAC from equipment schedules
   └── Use EnergyPlus templates for system types
   
4. Loads from nameplate surveys
   └── Apply audit-derived LPD, EPD
   
5. Schedules from BAS/logger data
   └── Create hourly fractional schedules
   
6. Weather: AMY for calibration period
   └── Match utility billing dates
```

### Phase 2: Iterative Calibration

```
Priority Order (by typical sensitivity):
1. Schedules (occupancy, HVAC, lighting)
2. Infiltration rate
3. Internal loads (plug, lighting)
4. HVAC efficiency curves
5. Envelope properties
6. Setpoints
```

### Phase 3: Validation
```
- Compare monthly totals to utility bills
- Compare hourly profiles to BAS/logger data
- Document calibration signature (scatter plots)
- Generate uncertainty band
```

---

## 5. Python Calibration Engine Structure

### Directory Structure
```
project/
├── weather/
│   ├── USA_CA_San.Luis.Obispo_TMY3.epw
│   └── SLO_2023_AMY.epw
├── models/
│   ├── baseline.idf
│   ├── calibrated.idf
│   └── measures/
├── data/
│   ├── utility_bills.csv
│   ├── bas_trends.csv
│   └── logger_data.csv
├── scripts/
│   ├── calibration_engine.py
│   ├── sensitivity_analysis.py
│   └── metrics.py
├── results/
│   └── calibration_report.md
└── CLAUDE.md  # This file
```

### Core Calibration Script Pattern
```python
#!/usr/bin/env python3
"""
EnergyPlus Calibration Engine
ASHRAE Level 3 / Guideline 14 Compliant
"""
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from eppy import modeleditor
from eppy.modeleditor import IDF

# Configuration
EPLUS_PATH = "/usr/local/EnergyPlus-24-1-0"
IDD_FILE = f"{EPLUS_PATH}/Energy+.idd"
WEATHER_FILE = "weather/SLO_2023_AMY.epw"

class CalibrationEngine:
    def __init__(self, idf_path: str):
        IDF.setiddname(IDD_FILE)
        self.idf = IDF(idf_path)
        self.base_path = Path(idf_path).parent
        
    def set_parameter(self, obj_type: str, obj_name: str, 
                      field: str, value: float):
        """Modify single IDF parameter"""
        objs = self.idf.idfobjects[obj_type]
        for obj in objs:
            if obj.Name == obj_name:
                setattr(obj, field, value)
                
    def run_simulation(self, run_dir: str = "run") -> pd.DataFrame:
        """Execute EnergyPlus and return results"""
        run_path = self.base_path / run_dir
        run_path.mkdir(exist_ok=True)
        
        temp_idf = run_path / "in.idf"
        self.idf.saveas(str(temp_idf))
        
        cmd = [
            f"{EPLUS_PATH}/energyplus",
            "-w", WEATHER_FILE,
            "-d", str(run_path),
            str(temp_idf)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # Parse results
        results_csv = run_path / "eplusout.csv"
        return pd.read_csv(results_csv) if results_csv.exists() else None
    
    def compute_metrics(self, measured: np.ndarray, 
                        simulated: np.ndarray) -> dict:
        """Calculate ASHRAE 14 calibration metrics"""
        n = len(measured)
        nmbe = 100 * np.sum(measured - simulated) / ((n-1) * np.mean(measured))
        rmse = np.sqrt(np.sum((measured - simulated)**2) / (n-1))
        cvrmse = 100 * rmse / np.mean(measured)
        
        ss_res = np.sum((measured - simulated)**2)
        ss_tot = np.sum((measured - np.mean(measured))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            "nmbe": nmbe,
            "cvrmse": cvrmse,
            "r2": r2,
            "calibrated_monthly": abs(nmbe) <= 5 and cvrmse <= 15,
            "calibrated_hourly": abs(nmbe) <= 10 and cvrmse <= 30
        }
```

### Sensitivity Analysis Pattern (Morris Method)
```python
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

def run_sensitivity(engine: CalibrationEngine, 
                    problem: dict, 
                    measured: np.ndarray) -> dict:
    """
    Morris one-at-a-time sensitivity analysis
    Identifies influential parameters before calibration
    """
    problem = {
        'num_vars': 6,
        'names': ['infiltration', 'lpd', 'epd', 
                  'cooling_eff', 'heating_eff', 'setpoint'],
        'bounds': [
            [0.1, 1.0],    # ACH
            [0.8, 1.5],    # W/sf
            [0.5, 2.0],    # W/sf
            [8, 14],       # EER
            [0.7, 0.95],   # Thermal eff
            [70, 76]       # °F
        ]
    }
    
    param_values = morris_sample.sample(problem, N=100, num_levels=4)
    
    # Run simulations for each sample
    Y = []
    for params in param_values:
        engine.set_parameter('ZoneInfiltration:DesignFlowRate', 
                           'Zone1_Infiltration', 
                           'Air_Changes_per_Hour', params[0])
        # ... set other parameters
        results = engine.run_simulation()
        monthly_kwh = results['Electricity:Facility [J](Monthly)'].values * 2.778e-7
        metrics = engine.compute_metrics(measured, monthly_kwh)
        Y.append(metrics['cvrmse'])
    
    Y = np.array(Y)
    Si = morris_analyze.analyze(problem, param_values, Y)
    
    return Si  # mu_star ranks parameter influence
```

---

## 6. Key Calibration Parameters by System

### 6.1 Envelope (High Sensitivity)
```python
envelope_params = {
    "ZoneInfiltration:DesignFlowRate": {
        "field": "Air_Changes_per_Hour",
        "range": [0.1, 1.5],
        "typical_slo": 0.35  # Mild climate, less stack effect
    },
    "WindowMaterial:SimpleGlazingSystem": {
        "U_Factor": [0.4, 1.2],  # W/m²K
        "SHGC": [0.25, 0.80]
    },
    "Material": {
        "Conductivity": "±20% of nominal"
    }
}
```

### 6.2 Internal Loads (High Sensitivity)
```python
loads_params = {
    "Lights": {
        "Watts_per_Zone_Floor_Area": [0.5, 2.0],
        "schedule_multiplier": [0.8, 1.2]
    },
    "ElectricEquipment": {
        "Watts_per_Zone_Floor_Area": [0.5, 3.0],
        "schedule_multiplier": [0.8, 1.2]
    },
    "People": {
        "Number_of_People": "±20%",
        "schedule_multiplier": [0.8, 1.2]
    }
}
```

### 6.3 HVAC (Medium-High Sensitivity)
```python
hvac_params = {
    "Coil:Cooling:DX:SingleSpeed": {
        "Gross_Rated_COP": [2.5, 4.5],
        "Gross_Rated_Sensible_Heat_Ratio": [0.65, 0.80]
    },
    "Coil:Heating:Fuel": {
        "Burner_Efficiency": [0.75, 0.95]
    },
    "Fan:VariableVolume": {
        "Fan_Total_Efficiency": [0.5, 0.7],
        "Motor_Efficiency": [0.85, 0.95]
    },
    "Controller:OutdoorAir": {
        "Economizer_Control_Type": "FixedDryBulb",
        "Economizer_Maximum_Limit_DryBulb_Temperature": [18, 24]
    }
}
```

### 6.4 Schedules (Highest Sensitivity - Adjust First)
```python
schedule_params = {
    "HVAC_operation": {
        "start_hour": [5, 8],
        "end_hour": [17, 22]
    },
    "occupancy_diversity": [0.6, 1.0],
    "weekend_operation": [0, 0.3],
    "holiday_operation": [0, 0.2]
}
```

---

## 7. Calibration Checklist

### Pre-Calibration
- [ ] Collect 12+ months utility data with billing dates
- [ ] Obtain AMY weather file matching billing period
- [ ] Complete Level 3 field measurements (2-4 weeks minimum)
- [ ] Document all HVAC equipment nameplate data
- [ ] Verify building area (gross, conditioned)
- [ ] Confirm occupancy patterns from BAS/surveys

### Model Development
- [ ] Geometry within 5% of measured areas
- [ ] Envelope R-values from audit or drawings
- [ ] Window U/SHGC from audit or age-based defaults
- [ ] HVAC capacities match nameplate
- [ ] Schedules based on BAS trends, not assumptions

### Calibration Iterations
- [ ] Run baseline simulation
- [ ] Calculate initial NMBE, CV(RMSE)
- [ ] Run sensitivity analysis to rank parameters
- [ ] Adjust schedules first (highest impact)
- [ ] Adjust internal loads
- [ ] Adjust infiltration
- [ ] Adjust HVAC curves
- [ ] Document each iteration

### Acceptance
- [ ] Monthly NMBE within ±5%
- [ ] Monthly CV(RMSE) ≤15%
- [ ] Hourly profile shapes reasonable
- [ ] All parameter values physically defensible
- [ ] Calibration signature documented

---

## 8. San Luis Obispo-Specific Considerations

### Mild Marine Climate Effects
- **Low heating loads**: Size heating equipment carefully
- **Economizer-dominated cooling**: Most hours use free cooling
- **Fog impact**: Morning marine layer affects solar gains
- **Diurnal swing**: Large day-night temperature difference
- **Minimal humidity**: Latent loads typically low

### Typical Building Stock (SLO)
- Cal Poly area: Mix of 1960s-1980s construction
- Downtown: Historic unreinforced masonry
- Highway 101 corridor: Tilt-up commercial
- Common HVAC: Packaged rooftop units, split systems

### Code Considerations
- Title 24 Climate Zone 5 requirements
- PG&E service territory
- ASHRAE 90.1-2019 for federal buildings
- CALGreen for new construction

---

## 9. References

### Standards
- ASHRAE Guideline 14-2014: Measurement of Energy, Demand, and Water Savings
- ASHRAE Standard 211-2018: Standard for Commercial Building Energy Audits
- IPMVP Volume I: Concepts and Options (EVO 2022)

### Weather Data
- climate.onebuilding.org (TMY3, TMYx files)
- NOAA ISD Lite (AMY data for diyepw)
- California Energy Commission CZ2022 weather files

### Tools
- EnergyPlus (DOE)
- OpenStudio (NREL)
- eppy (Python IDF manipulation)
- SALib (Sensitivity analysis)
- epluspar (R package for Bayesian calibration)

### Research
- Coakley, D., et al. (2014). "A review of methods to match building energy simulation models to measured data"
- Reddy, T.A. (2006). "Literature Review on Calibration of Building Energy Simulation Programs"
- New, J.R., et al. (2016). "Suitability of ASHRAE Guideline 14 Metrics for Calibration"

---

## 10. Claude Code Commands

### Quick Start
```bash
# Set up Python environment
python -m venv .venv && source .venv/bin/activate
pip install eppy pandas numpy scipy SALib matplotlib

# Download weather file
curl -O https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/CA_California/USA_CA_San.Luis.Obispo.County.Rgnl.AP.723910_TMYx.2004-2018.zip
unzip *.zip

# Run baseline simulation
python scripts/calibration_engine.py --mode baseline

# Run sensitivity analysis
python scripts/sensitivity_analysis.py --samples 100

# Run calibration
python scripts/calibration_engine.py --mode calibrate --target-cvrmse 15
```

### Useful Patterns for Claude Code
```python
# Read IDF and modify parameter
from eppy.modeleditor import IDF
IDF.setiddname('/usr/local/EnergyPlus-24-1-0/Energy+.idd')
idf = IDF('model.idf')
infiltration = idf.idfobjects['ZoneInfiltration:DesignFlowRate'][0]
infiltration.Air_Changes_per_Hour = 0.5
idf.save()

# Parse EnergyPlus output
import pandas as pd
results = pd.read_csv('eplusout.csv')
monthly_elec = results['Electricity:Facility [J](Monthly)'].values * 2.778e-7  # to kWh

# Compare to utility data
utility = pd.read_csv('utility_bills.csv')
metrics = compute_metrics(utility['kwh'].values, monthly_elec)
print(f"NMBE: {metrics['nmbe']:.1f}%, CV(RMSE): {metrics['cvrmse']:.1f}%")
```

---

*This workflow integrates ASHRAE Level 3 audit data collection with EnergyPlus model calibration following ASHRAE Guideline 14 statistical criteria. The San Luis Obispo climate context (CZ 5 / ASHRAE 3C) informs parameter ranges and typical building characteristics.*
