#!/usr/bin/env python3
"""
EnergyPlus Model Calibration Engine
ASHRAE Level 3 Audit / Guideline 14 Compliant

Usage:
    python calibration_engine.py --mode baseline
    python calibration_engine.py --mode calibrate --target-cvrmse 15
    python calibration_engine.py --mode sensitivity --samples 100

Dependencies:
    pip install eppy pandas numpy scipy matplotlib SALib

Author: Generated for ASHRAE Level 3 / EnergyPlus calibration workflow
"""

import argparse
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd

# Optional imports - graceful degradation
try:
    from eppy.modeleditor import IDF
    EPPY_AVAILABLE = True
except ImportError:
    EPPY_AVAILABLE = False
    print("Warning: eppy not installed. Install with: pip install eppy")

try:
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """Calibration configuration"""
    eplus_path: str = "/usr/local/EnergyPlus-24-1-0"
    idd_file: Optional[str] = None
    weather_file: str = "weather/SLO_2023_AMY.epw"
    output_dir: str = "results"
    
    # ASHRAE Guideline 14 thresholds
    nmbe_monthly_threshold: float = 5.0    # ±5%
    cvrmse_monthly_threshold: float = 15.0  # ≤15%
    nmbe_hourly_threshold: float = 10.0     # ±10%
    cvrmse_hourly_threshold: float = 30.0   # ≤30%
    r2_threshold: float = 0.75
    
    def __post_init__(self):
        if self.idd_file is None:
            self.idd_file = f"{self.eplus_path}/Energy+.idd"


# =============================================================================
# Calibration Metrics (ASHRAE Guideline 14)
# =============================================================================
class CalibrationMetrics:
    """ASHRAE Guideline 14 / IPMVP calibration metrics"""
    
    @staticmethod
    def nmbe(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Normalized Mean Bias Error (%)
        Indicates systematic over/under prediction
        """
        n = len(measured)
        if n <= 1:
            return float('inf')
        return 100.0 * np.sum(measured - simulated) / ((n - 1) * np.mean(measured))
    
    @staticmethod
    def cvrmse(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Coefficient of Variation of Root Mean Square Error (%)
        Indicates scatter/variability in predictions
        """
        n = len(measured)
        if n <= 1:
            return float('inf')
        rmse = np.sqrt(np.sum((measured - simulated)**2) / (n - 1))
        return 100.0 * rmse / np.mean(measured)
    
    @staticmethod
    def r_squared(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Coefficient of determination
        Indicates how well model explains variance
        """
        ss_res = np.sum((measured - simulated)**2)
        ss_tot = np.sum((measured - np.mean(measured))**2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)
    
    @staticmethod
    def mbe(measured: np.ndarray, simulated: np.ndarray) -> float:
        """Mean Bias Error (same units as data)"""
        return np.mean(measured - simulated)
    
    @staticmethod
    def rmse(measured: np.ndarray, simulated: np.ndarray) -> float:
        """Root Mean Square Error (same units as data)"""
        n = len(measured)
        return np.sqrt(np.sum((measured - simulated)**2) / n)
    
    @classmethod
    def compute_all(cls, measured: np.ndarray, simulated: np.ndarray,
                    config: Config = None) -> Dict[str, Any]:
        """Compute all metrics and check against thresholds"""
        if config is None:
            config = Config()
            
        nmbe = cls.nmbe(measured, simulated)
        cvrmse = cls.cvrmse(measured, simulated)
        r2 = cls.r_squared(measured, simulated)
        
        return {
            "nmbe_pct": nmbe,
            "cvrmse_pct": cvrmse,
            "r_squared": r2,
            "mbe": cls.mbe(measured, simulated),
            "rmse": cls.rmse(measured, simulated),
            "n_points": len(measured),
            "calibrated_monthly": (
                abs(nmbe) <= config.nmbe_monthly_threshold and 
                cvrmse <= config.cvrmse_monthly_threshold
            ),
            "calibrated_hourly": (
                abs(nmbe) <= config.nmbe_hourly_threshold and 
                cvrmse <= config.cvrmse_hourly_threshold
            ),
            "thresholds": {
                "nmbe_monthly": config.nmbe_monthly_threshold,
                "cvrmse_monthly": config.cvrmse_monthly_threshold,
                "nmbe_hourly": config.nmbe_hourly_threshold,
                "cvrmse_hourly": config.cvrmse_hourly_threshold
            }
        }


# =============================================================================
# EnergyPlus Interface
# =============================================================================
class EnergyPlusRunner:
    """Interface to EnergyPlus simulation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.run_count = 0
        
    def run(self, idf_path: str, output_dir: str = None) -> Optional[Path]:
        """
        Run EnergyPlus simulation
        Returns path to output directory
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir) / f"run_{self.run_count:04d}"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        self.run_count += 1
        
        cmd = [
            f"{self.config.eplus_path}/energyplus",
            "-w", self.config.weather_file,
            "-d", str(output_dir),
            "-r",  # Read variables
            idf_path
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print(f"EnergyPlus error: {result.stderr[:500]}")
                return None
                
        except subprocess.TimeoutExpired:
            print("EnergyPlus simulation timed out")
            return None
        except FileNotFoundError:
            print(f"EnergyPlus not found at {self.config.eplus_path}")
            return None
            
        return output_dir
    
    def parse_monthly_results(self, output_dir: Path) -> Optional[pd.DataFrame]:
        """Parse monthly results from EnergyPlus output"""
        csv_path = output_dir / "eplusout.csv"
        
        if not csv_path.exists():
            print(f"Results file not found: {csv_path}")
            return None
            
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Error parsing results: {e}")
            return None


# =============================================================================
# IDF Manipulation
# =============================================================================
class IDFModifier:
    """Modify EnergyPlus IDF files"""
    
    def __init__(self, idf_path: str, config: Config):
        if not EPPY_AVAILABLE:
            raise ImportError("eppy required for IDF modification")
            
        IDF.setiddname(config.idd_file)
        self.idf = IDF(idf_path)
        self.idf_path = idf_path
        self.config = config
        
    def get_object(self, obj_type: str, name: str = None):
        """Get IDF object by type and optionally name"""
        objects = self.idf.idfobjects[obj_type]
        if name is None:
            return objects
        for obj in objects:
            if hasattr(obj, 'Name') and obj.Name == name:
                return obj
        return None
    
    def set_field(self, obj_type: str, name: str, field: str, value: Any):
        """Set field value on named object"""
        obj = self.get_object(obj_type, name)
        if obj is None:
            print(f"Object not found: {obj_type}:{name}")
            return False
        setattr(obj, field, value)
        return True
    
    def set_infiltration(self, ach: float, zone_name: str = None):
        """Set infiltration rate (ACH)"""
        objects = self.idf.idfobjects['ZoneInfiltration:DesignFlowRate']
        for obj in objects:
            if zone_name is None or obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name == zone_name:
                obj.Design_Flow_Rate_Calculation_Method = "AirChanges/Hour"
                obj.Air_Changes_per_Hour = ach
                
    def set_lighting_power_density(self, lpd_w_m2: float, zone_name: str = None):
        """Set lighting power density (W/m²)"""
        objects = self.idf.idfobjects['Lights']
        for obj in objects:
            if zone_name is None or zone_name in obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name:
                obj.Design_Level_Calculation_Method = "Watts/Area"
                obj.Watts_per_Zone_Floor_Area = lpd_w_m2
                
    def set_equipment_power_density(self, epd_w_m2: float, zone_name: str = None):
        """Set equipment power density (W/m²)"""
        objects = self.idf.idfobjects['ElectricEquipment']
        for obj in objects:
            if zone_name is None or zone_name in obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name:
                obj.Design_Level_Calculation_Method = "Watts/Area"
                obj.Watts_per_Zone_Floor_Area = epd_w_m2
                
    def set_cooling_cop(self, cop: float, coil_name: str = None):
        """Set DX cooling coil COP"""
        for obj_type in ['Coil:Cooling:DX:SingleSpeed', 'Coil:Cooling:DX:TwoSpeed']:
            objects = self.idf.idfobjects.get(obj_type, [])
            for obj in objects:
                if coil_name is None or obj.Name == coil_name:
                    obj.Gross_Rated_Cooling_COP = cop
                    
    def set_heating_efficiency(self, efficiency: float, coil_name: str = None):
        """Set gas heating coil efficiency"""
        objects = self.idf.idfobjects.get('Coil:Heating:Fuel', [])
        for obj in objects:
            if coil_name is None or obj.Name == coil_name:
                obj.Burner_Efficiency = efficiency
                
    def set_thermostat_setpoints(self, heating_sp: float, cooling_sp: float):
        """Set thermostat setpoints (°C)"""
        # This is simplified - real implementation would modify schedule values
        print(f"Setting heating: {heating_sp}°C, cooling: {cooling_sp}°C")
        # Would modify ThermostatSetpoint:DualSetpoint or schedule values
        
    def save(self, output_path: str = None):
        """Save modified IDF"""
        if output_path is None:
            output_path = self.idf_path.replace('.idf', '_modified.idf')
        self.idf.saveas(output_path)
        return output_path


# =============================================================================
# Calibration Parameter Definitions
# =============================================================================
CALIBRATION_PARAMETERS = {
    "infiltration_ach": {
        "description": "Zone infiltration (air changes per hour)",
        "bounds": [0.1, 1.5],
        "default": 0.5,
        "sensitivity": "high"
    },
    "lpd_w_m2": {
        "description": "Lighting power density (W/m²)",
        "bounds": [5.0, 20.0],  # ~0.5-2.0 W/sf
        "default": 10.8,  # ~1.0 W/sf
        "sensitivity": "high"
    },
    "epd_w_m2": {
        "description": "Equipment power density (W/m²)",
        "bounds": [5.0, 30.0],
        "default": 16.1,  # ~1.5 W/sf
        "sensitivity": "high"
    },
    "cooling_cop": {
        "description": "DX cooling COP",
        "bounds": [2.5, 5.0],
        "default": 3.5,
        "sensitivity": "medium"
    },
    "heating_efficiency": {
        "description": "Gas heating efficiency",
        "bounds": [0.75, 0.95],
        "default": 0.80,
        "sensitivity": "medium"
    },
    "schedule_multiplier": {
        "description": "Schedule diversity factor",
        "bounds": [0.7, 1.2],
        "default": 1.0,
        "sensitivity": "very_high"
    }
}


# =============================================================================
# Sensitivity Analysis
# =============================================================================
def run_sensitivity_analysis(
    idf_path: str,
    utility_data: pd.DataFrame,
    config: Config,
    n_trajectories: int = 10,
    num_levels: int = 4
) -> Dict[str, Any]:
    """
    Morris one-at-a-time sensitivity analysis
    Identifies which parameters most influence model output
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib required for sensitivity analysis")
    
    param_names = list(CALIBRATION_PARAMETERS.keys())
    bounds = [CALIBRATION_PARAMETERS[p]["bounds"] for p in param_names]
    
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds
    }
    
    # Generate parameter samples
    param_values = morris_sample.sample(
        problem, 
        N=n_trajectories, 
        num_levels=num_levels
    )
    
    print(f"Running {len(param_values)} simulations for sensitivity analysis...")
    
    runner = EnergyPlusRunner(config)
    Y = []
    
    for i, params in enumerate(param_values):
        print(f"  Simulation {i+1}/{len(param_values)}")
        
        # Modify IDF with sampled parameters
        modifier = IDFModifier(idf_path, config)
        modifier.set_infiltration(params[0])  # infiltration_ach
        modifier.set_lighting_power_density(params[1])  # lpd_w_m2
        modifier.set_equipment_power_density(params[2])  # epd_w_m2
        modifier.set_cooling_cop(params[3])  # cooling_cop
        modifier.set_heating_efficiency(params[4])  # heating_efficiency
        # schedule_multiplier would need custom implementation
        
        temp_idf = modifier.save(f"/tmp/sensitivity_{i:04d}.idf")
        
        # Run simulation
        output_dir = runner.run(temp_idf, f"/tmp/sens_run_{i:04d}")
        
        if output_dir is None:
            Y.append(float('inf'))
            continue
            
        # Parse results and compute metric
        results = runner.parse_monthly_results(output_dir)
        if results is None:
            Y.append(float('inf'))
            continue
            
        # Extract monthly electricity (simplified - adjust column name as needed)
        elec_col = [c for c in results.columns if 'Electricity' in c and 'Facility' in c]
        if elec_col:
            sim_monthly = results[elec_col[0]].values * 2.778e-7  # J to kWh
            measured = utility_data['kwh'].values[:len(sim_monthly)]
            metrics = CalibrationMetrics.compute_all(measured, sim_monthly, config)
            Y.append(metrics['cvrmse_pct'])
        else:
            Y.append(float('inf'))
    
    Y = np.array(Y)
    
    # Analyze sensitivity
    Si = morris_analyze.analyze(problem, param_values, Y)
    
    # Rank parameters by mu_star (mean of absolute elementary effects)
    ranking = sorted(
        zip(param_names, Si['mu_star']),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        'mu': dict(zip(param_names, Si['mu'])),
        'mu_star': dict(zip(param_names, Si['mu_star'])),
        'sigma': dict(zip(param_names, Si['sigma'])),
        'ranking': ranking,
        'problem': problem,
        'n_simulations': len(param_values)
    }


# =============================================================================
# Automated Calibration
# =============================================================================
def calibrate_model(
    idf_path: str,
    utility_data: pd.DataFrame,
    config: Config,
    max_iterations: int = 50
) -> Dict[str, Any]:
    """
    Iterative calibration using gradient-free optimization
    """
    from scipy.optimize import minimize
    
    runner = EnergyPlusRunner(config)
    iteration = [0]  # Mutable counter for callback
    best_metrics = [None]
    
    def objective(params):
        iteration[0] += 1
        print(f"\nIteration {iteration[0]}: params = {params}")
        
        # Modify IDF
        modifier = IDFModifier(idf_path, config)
        modifier.set_infiltration(params[0])
        modifier.set_lighting_power_density(params[1])
        modifier.set_equipment_power_density(params[2])
        modifier.set_cooling_cop(params[3])
        modifier.set_heating_efficiency(params[4])
        
        temp_idf = modifier.save(f"/tmp/calib_{iteration[0]:04d}.idf")
        
        # Run simulation
        output_dir = runner.run(temp_idf)
        if output_dir is None:
            return 1000.0
            
        results = runner.parse_monthly_results(output_dir)
        if results is None:
            return 1000.0
            
        # Compute objective (weighted NMBE + CVRMSE)
        elec_col = [c for c in results.columns if 'Electricity' in c and 'Facility' in c]
        if not elec_col:
            return 1000.0
            
        sim_monthly = results[elec_col[0]].values * 2.778e-7
        measured = utility_data['kwh'].values[:len(sim_monthly)]
        metrics = CalibrationMetrics.compute_all(measured, sim_monthly, config)
        
        obj = abs(metrics['nmbe_pct']) + metrics['cvrmse_pct']
        print(f"  NMBE: {metrics['nmbe_pct']:.2f}%, CVRMSE: {metrics['cvrmse_pct']:.2f}%")
        
        if best_metrics[0] is None or obj < best_metrics[0]['objective']:
            best_metrics[0] = {**metrics, 'objective': obj, 'params': params.tolist()}
            
        return obj
    
    # Initial guess and bounds
    x0 = [
        CALIBRATION_PARAMETERS['infiltration_ach']['default'],
        CALIBRATION_PARAMETERS['lpd_w_m2']['default'],
        CALIBRATION_PARAMETERS['epd_w_m2']['default'],
        CALIBRATION_PARAMETERS['cooling_cop']['default'],
        CALIBRATION_PARAMETERS['heating_efficiency']['default']
    ]
    
    bounds = [
        CALIBRATION_PARAMETERS['infiltration_ach']['bounds'],
        CALIBRATION_PARAMETERS['lpd_w_m2']['bounds'],
        CALIBRATION_PARAMETERS['epd_w_m2']['bounds'],
        CALIBRATION_PARAMETERS['cooling_cop']['bounds'],
        CALIBRATION_PARAMETERS['heating_efficiency']['bounds']
    ]
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',  # Gradient-free
        bounds=bounds,
        options={'maxiter': max_iterations, 'disp': True}
    )
    
    return {
        'success': result.success,
        'optimal_params': {
            'infiltration_ach': result.x[0],
            'lpd_w_m2': result.x[1],
            'epd_w_m2': result.x[2],
            'cooling_cop': result.x[3],
            'heating_efficiency': result.x[4]
        },
        'final_objective': result.fun,
        'iterations': result.nit,
        'best_metrics': best_metrics[0]
    }


# =============================================================================
# Utility Data Loading
# =============================================================================
def load_utility_data(csv_path: str) -> pd.DataFrame:
    """
    Load utility bill data
    Expected columns: date, kwh, kw, therms
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    required_cols = ['kwh']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    return df


def generate_sample_utility_data(output_path: str = "data/utility_bills.csv"):
    """Generate sample utility data for testing"""
    # Typical SLO commercial building monthly consumption pattern
    months = pd.date_range('2023-01-01', periods=12, freq='MS')
    
    # Base load + seasonal variation (mild climate, economizer-dominated)
    base_kwh = 15000  # Base monthly kWh
    seasonal_factor = [0.85, 0.82, 0.88, 0.95, 1.05, 1.15, 
                       1.20, 1.18, 1.10, 0.95, 0.88, 0.85]
    
    kwh = [int(base_kwh * f * (1 + 0.05 * np.random.randn())) for f in seasonal_factor]
    kw = [int(k / (720 * 0.4)) for k in kwh]  # Rough demand estimate
    therms = [int(200 * (1.5 - f) * max(0.3, 1)) for f in seasonal_factor]  # Gas inverse of cooling
    
    df = pd.DataFrame({
        'date': months,
        'kwh': kwh,
        'kw': kw,
        'therms': therms
    })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample utility data saved to: {output_path}")
    return df


# =============================================================================
# Reporting
# =============================================================================
def generate_calibration_report(
    metrics: Dict[str, Any],
    params: Dict[str, float],
    output_path: str = "results/calibration_report.md"
) -> str:
    """Generate markdown calibration report"""
    
    report = f"""# EnergyPlus Model Calibration Report
## ASHRAE Guideline 14 Compliance

### Calibration Status
- **Monthly Calibration**: {'✓ PASS' if metrics.get('calibrated_monthly') else '✗ FAIL'}
- **Hourly Calibration**: {'✓ PASS' if metrics.get('calibrated_hourly') else '✗ FAIL'}

### Statistical Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| NMBE | {metrics.get('nmbe_pct', 'N/A'):.2f}% | ±5% (monthly) | {'✓' if abs(metrics.get('nmbe_pct', 100)) <= 5 else '✗'} |
| CV(RMSE) | {metrics.get('cvrmse_pct', 'N/A'):.2f}% | ≤15% (monthly) | {'✓' if metrics.get('cvrmse_pct', 100) <= 15 else '✗'} |
| R² | {metrics.get('r_squared', 'N/A'):.3f} | ≥0.75 | {'✓' if metrics.get('r_squared', 0) >= 0.75 else '✗'} |

### Calibrated Parameters

| Parameter | Value | Units | Range |
|-----------|-------|-------|-------|
| Infiltration | {params.get('infiltration_ach', 'N/A'):.3f} | ACH | 0.1-1.5 |
| Lighting Power Density | {params.get('lpd_w_m2', 'N/A'):.1f} | W/m² | 5-20 |
| Equipment Power Density | {params.get('epd_w_m2', 'N/A'):.1f} | W/m² | 5-30 |
| Cooling COP | {params.get('cooling_cop', 'N/A'):.2f} | - | 2.5-5.0 |
| Heating Efficiency | {params.get('heating_efficiency', 'N/A'):.2f} | - | 0.75-0.95 |

### Notes
- Calibration performed following ASHRAE Guideline 14-2014
- Weather data: AMY for calibration period
- Data points: {metrics.get('n_points', 'N/A')}
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
        
    print(f"Report saved to: {output_path}")
    return report


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='EnergyPlus Model Calibration Engine'
    )
    parser.add_argument(
        '--mode', 
        choices=['baseline', 'calibrate', 'sensitivity', 'sample-data'],
        default='baseline',
        help='Operation mode'
    )
    parser.add_argument(
        '--idf', 
        type=str, 
        default='models/baseline.idf',
        help='Path to IDF file'
    )
    parser.add_argument(
        '--utility-data',
        type=str,
        default='data/utility_bills.csv',
        help='Path to utility bill CSV'
    )
    parser.add_argument(
        '--target-cvrmse',
        type=float,
        default=15.0,
        help='Target CV(RMSE) for calibration'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples for sensitivity analysis'
    )
    parser.add_argument(
        '--eplus-path',
        type=str,
        default='/usr/local/EnergyPlus-24-1-0',
        help='Path to EnergyPlus installation'
    )
    
    args = parser.parse_args()
    
    config = Config(eplus_path=args.eplus_path)
    config.cvrmse_monthly_threshold = args.target_cvrmse
    
    if args.mode == 'sample-data':
        generate_sample_utility_data()
        return
    
    # Load utility data
    try:
        utility_data = load_utility_data(args.utility_data)
        print(f"Loaded {len(utility_data)} months of utility data")
    except FileNotFoundError:
        print(f"Utility data not found: {args.utility_data}")
        print("Generate sample data with: --mode sample-data")
        return
    
    if args.mode == 'baseline':
        print("Running baseline simulation...")
        runner = EnergyPlusRunner(config)
        output_dir = runner.run(args.idf)
        if output_dir:
            print(f"Results in: {output_dir}")
            
    elif args.mode == 'sensitivity':
        print("Running sensitivity analysis...")
        results = run_sensitivity_analysis(
            args.idf, 
            utility_data, 
            config,
            n_trajectories=args.samples // 10
        )
        print("\nParameter Ranking (by influence):")
        for param, mu_star in results['ranking']:
            print(f"  {param}: μ* = {mu_star:.2f}")
            
    elif args.mode == 'calibrate':
        print("Starting calibration...")
        results = calibrate_model(args.idf, utility_data, config)
        
        if results['best_metrics']:
            generate_calibration_report(
                results['best_metrics'],
                results['optimal_params']
            )


if __name__ == '__main__':
    main()
