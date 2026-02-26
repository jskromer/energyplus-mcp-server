"""
Calibration utilities for EnergyPlus model calibration

ASHRAE Guideline 14 compliant calibration metrics and utilities for
comparing simulated results against measured utility data.

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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationStandard(Enum):
    """Calibration standards with their acceptance thresholds"""
    ASHRAE_14_MONTHLY = "ashrae_14_monthly"
    ASHRAE_14_HOURLY = "ashrae_14_hourly"
    IPMVP_MONTHLY = "ipmvp_monthly"
    IPMVP_HOURLY = "ipmvp_hourly"
    FEMP_MONTHLY = "femp_monthly"
    FEMP_HOURLY = "femp_hourly"


@dataclass
class CalibrationThresholds:
    """Calibration acceptance thresholds by standard"""
    nmbe_limit: float  # Absolute value limit (%)
    cvrmse_limit: float  # Upper limit (%)
    r2_limit: Optional[float] = None  # Lower limit (optional)

    @classmethod
    def get_thresholds(cls, standard: CalibrationStandard) -> 'CalibrationThresholds':
        """Get thresholds for a specific calibration standard"""
        thresholds = {
            CalibrationStandard.ASHRAE_14_MONTHLY: cls(nmbe_limit=5.0, cvrmse_limit=15.0, r2_limit=0.75),
            CalibrationStandard.ASHRAE_14_HOURLY: cls(nmbe_limit=10.0, cvrmse_limit=30.0, r2_limit=0.75),
            CalibrationStandard.IPMVP_MONTHLY: cls(nmbe_limit=5.0, cvrmse_limit=15.0),
            CalibrationStandard.IPMVP_HOURLY: cls(nmbe_limit=10.0, cvrmse_limit=20.0),
            CalibrationStandard.FEMP_MONTHLY: cls(nmbe_limit=5.0, cvrmse_limit=15.0),
            CalibrationStandard.FEMP_HOURLY: cls(nmbe_limit=10.0, cvrmse_limit=30.0),
        }
        return thresholds[standard]


@dataclass
class CalibrationMetrics:
    """Container for calibration metric results"""
    nmbe: float  # Normalized Mean Bias Error (%)
    cvrmse: float  # Coefficient of Variation of RMSE (%)
    r_squared: float  # Coefficient of determination
    rmse: float  # Root Mean Square Error (same units as input)
    mbe: float  # Mean Bias Error (same units as input)
    mae: float  # Mean Absolute Error (same units as input)
    n_points: int  # Number of data points
    mean_measured: float  # Mean of measured values
    mean_simulated: float  # Mean of simulated values

    def is_calibrated(self, standard: CalibrationStandard) -> bool:
        """Check if metrics meet calibration standard"""
        thresholds = CalibrationThresholds.get_thresholds(standard)

        nmbe_ok = abs(self.nmbe) <= thresholds.nmbe_limit
        cvrmse_ok = self.cvrmse <= thresholds.cvrmse_limit
        r2_ok = True
        if thresholds.r2_limit is not None:
            r2_ok = self.r_squared >= thresholds.r2_limit

        return nmbe_ok and cvrmse_ok and r2_ok

    def get_status(self, standard: CalibrationStandard) -> Dict[str, Any]:
        """Get detailed calibration status against a standard"""
        thresholds = CalibrationThresholds.get_thresholds(standard)

        status = {
            "standard": standard.value,
            "is_calibrated": self.is_calibrated(standard),
            "metrics": {
                "nmbe": {
                    "value": round(self.nmbe, 2),
                    "limit": thresholds.nmbe_limit,
                    "passed": abs(self.nmbe) <= thresholds.nmbe_limit,
                    "margin": round(thresholds.nmbe_limit - abs(self.nmbe), 2)
                },
                "cvrmse": {
                    "value": round(self.cvrmse, 2),
                    "limit": thresholds.cvrmse_limit,
                    "passed": self.cvrmse <= thresholds.cvrmse_limit,
                    "margin": round(thresholds.cvrmse_limit - self.cvrmse, 2)
                }
            }
        }

        if thresholds.r2_limit is not None:
            status["metrics"]["r_squared"] = {
                "value": round(self.r_squared, 3),
                "limit": thresholds.r2_limit,
                "passed": self.r_squared >= thresholds.r2_limit,
                "margin": round(self.r_squared - thresholds.r2_limit, 3)
            }

        return status

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "nmbe_percent": round(self.nmbe, 2),
            "cvrmse_percent": round(self.cvrmse, 2),
            "r_squared": round(self.r_squared, 4),
            "rmse": round(self.rmse, 2),
            "mbe": round(self.mbe, 2),
            "mae": round(self.mae, 2),
            "n_points": self.n_points,
            "mean_measured": round(self.mean_measured, 2),
            "mean_simulated": round(self.mean_simulated, 2)
        }


class CalibrationMetricsCalculator:
    """
    Calculator for ASHRAE Guideline 14 calibration metrics

    Computes NMBE, CV(RMSE), R², and related statistics for
    comparing simulated energy data against measured data.
    """

    @staticmethod
    def compute_nmbe(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Compute Normalized Mean Bias Error (NMBE)

        NMBE indicates systematic over- or under-prediction.
        Positive = simulation under-predicts, Negative = over-predicts.

        Formula: NMBE = 100 * Σ(M - S) / ((n-1) * M̄)

        Args:
            measured: Array of measured values
            simulated: Array of simulated values

        Returns:
            NMBE as a percentage
        """
        n = len(measured)
        if n < 2:
            raise ValueError("Need at least 2 data points for NMBE calculation")

        mean_measured = np.mean(measured)
        if mean_measured == 0:
            raise ValueError("Mean of measured values cannot be zero")

        nmbe = 100.0 * np.sum(measured - simulated) / ((n - 1) * mean_measured)
        return float(nmbe)

    @staticmethod
    def compute_cvrmse(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Compute Coefficient of Variation of Root Mean Square Error (CV(RMSE))

        CV(RMSE) measures the scatter/variance of residuals.

        Formula: CV(RMSE) = 100 * RMSE / M̄
                 where RMSE = sqrt(Σ(M - S)² / (n-1))

        Args:
            measured: Array of measured values
            simulated: Array of simulated values

        Returns:
            CV(RMSE) as a percentage
        """
        n = len(measured)
        if n < 2:
            raise ValueError("Need at least 2 data points for CV(RMSE) calculation")

        mean_measured = np.mean(measured)
        if mean_measured == 0:
            raise ValueError("Mean of measured values cannot be zero")

        rmse = np.sqrt(np.sum((measured - simulated) ** 2) / (n - 1))
        cvrmse = 100.0 * rmse / mean_measured
        return float(cvrmse)

    @staticmethod
    def compute_r_squared(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Compute coefficient of determination (R²)

        R² indicates how well the simulation explains variance in measured data.

        Formula: R² = 1 - SS_res / SS_tot
                 where SS_res = Σ(M - S)²
                       SS_tot = Σ(M - M̄)²

        Args:
            measured: Array of measured values
            simulated: Array of simulated values

        Returns:
            R² value (can be negative if simulation is worse than mean)
        """
        ss_res = np.sum((measured - simulated) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)

        if ss_tot == 0:
            # All measured values are identical
            return 1.0 if ss_res == 0 else 0.0

        r_squared = 1.0 - (ss_res / ss_tot)
        return float(r_squared)

    @staticmethod
    def compute_rmse(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Compute Root Mean Square Error

        Args:
            measured: Array of measured values
            simulated: Array of simulated values

        Returns:
            RMSE in same units as input
        """
        n = len(measured)
        if n < 2:
            raise ValueError("Need at least 2 data points for RMSE calculation")
        rmse = np.sqrt(np.sum((measured - simulated) ** 2) / (n - 1))
        return float(rmse)

    @staticmethod
    def compute_mbe(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Compute Mean Bias Error

        Args:
            measured: Array of measured values
            simulated: Array of simulated values

        Returns:
            MBE in same units as input
        """
        return float(np.mean(measured - simulated))

    @staticmethod
    def compute_mae(measured: np.ndarray, simulated: np.ndarray) -> float:
        """
        Compute Mean Absolute Error

        Args:
            measured: Array of measured values
            simulated: Array of simulated values

        Returns:
            MAE in same units as input
        """
        return float(np.mean(np.abs(measured - simulated)))

    @classmethod
    def compute_all_metrics(
        cls,
        measured: Union[np.ndarray, List[float], pd.Series],
        simulated: Union[np.ndarray, List[float], pd.Series]
    ) -> CalibrationMetrics:
        """
        Compute all calibration metrics

        Args:
            measured: Measured/utility data values
            simulated: Simulated values from EnergyPlus

        Returns:
            CalibrationMetrics object with all computed metrics
        """
        # Convert to numpy arrays
        measured = np.asarray(measured, dtype=float)
        simulated = np.asarray(simulated, dtype=float)

        if len(measured) != len(simulated):
            raise ValueError(
                f"Array length mismatch: measured={len(measured)}, simulated={len(simulated)}"
            )

        if len(measured) < 2:
            raise ValueError("Need at least 2 data points for calibration metrics")

        # Check for NaN/Inf values
        if np.any(~np.isfinite(measured)) or np.any(~np.isfinite(simulated)):
            raise ValueError("Input arrays contain NaN or Inf values")

        return CalibrationMetrics(
            nmbe=cls.compute_nmbe(measured, simulated),
            cvrmse=cls.compute_cvrmse(measured, simulated),
            r_squared=cls.compute_r_squared(measured, simulated),
            rmse=cls.compute_rmse(measured, simulated),
            mbe=cls.compute_mbe(measured, simulated),
            mae=cls.compute_mae(measured, simulated),
            n_points=len(measured),
            mean_measured=float(np.mean(measured)),
            mean_simulated=float(np.mean(simulated))
        )


@dataclass
class UtilityBill:
    """Represents a single utility bill period"""
    start_date: str  # ISO format: YYYY-MM-DD
    end_date: str
    consumption: float  # Energy consumption (kWh, therms, etc.)
    demand: Optional[float] = None  # Peak demand (kW, etc.)
    cost: Optional[float] = None
    fuel_type: str = "electricity"  # electricity, natural_gas, etc.
    units: str = "kWh"


@dataclass
class UtilityData:
    """Container for utility billing data"""
    bills: List[UtilityBill]
    fuel_type: str
    units: str
    location: Optional[str] = None
    account_id: Optional[str] = None

    def to_monthly_series(self) -> pd.Series:
        """Convert bills to monthly pandas Series"""
        data = {}
        for bill in self.bills:
            # Use end date month as the billing month
            month_key = bill.end_date[:7]  # YYYY-MM
            data[month_key] = bill.consumption
        return pd.Series(data).sort_index()

    def get_consumption_array(self) -> np.ndarray:
        """Get consumption values as numpy array"""
        return np.array([bill.consumption for bill in self.bills])

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        fuel_type: str = "electricity",
        date_col: str = "date",
        consumption_col: str = "consumption",
        units: str = "kWh"
    ) -> 'UtilityData':
        """
        Load utility data from CSV file

        Expected CSV format:
        date,consumption[,demand,cost]
        2023-01-31,1500
        2023-02-28,1400
        ...
        """
        df = pd.read_csv(csv_path)

        bills = []
        for idx, row in df.iterrows():
            bill = UtilityBill(
                start_date=str(row.get('start_date', row[date_col])),
                end_date=str(row[date_col]),
                consumption=float(row[consumption_col]),
                demand=float(row['demand']) if 'demand' in row and pd.notna(row['demand']) else None,
                cost=float(row['cost']) if 'cost' in row and pd.notna(row['cost']) else None,
                fuel_type=fuel_type,
                units=units
            )
            bills.append(bill)

        return cls(bills=bills, fuel_type=fuel_type, units=units)

    @classmethod
    def from_slo_format(cls, csv_path: str) -> Tuple['UtilityData', 'UtilityData']:
        """
        Load utility data from SLO-format CSV (billing_start, billing_end, kwh, therms, etc.)

        This format matches the sample_files/slo_utility_data.csv structure:
        billing_start,billing_end,days,kwh,kw_demand,therms,cost_elec,cost_gas

        Args:
            csv_path: Path to the CSV file

        Returns:
            Tuple of (electricity_data, gas_data)
        """
        df = pd.read_csv(csv_path)

        elec_bills = []
        gas_bills = []

        for idx, row in df.iterrows():
            # Electricity bill
            elec_bills.append(UtilityBill(
                start_date=str(row['billing_start']),
                end_date=str(row['billing_end']),
                consumption=float(row['kwh']),
                demand=float(row['kw_demand']) if 'kw_demand' in row and pd.notna(row['kw_demand']) else None,
                cost=float(row['cost_elec']) if 'cost_elec' in row and pd.notna(row['cost_elec']) else None,
                fuel_type="electricity",
                units="kWh"
            ))

            # Gas bill
            if 'therms' in row and pd.notna(row['therms']) and float(row['therms']) > 0:
                gas_bills.append(UtilityBill(
                    start_date=str(row['billing_start']),
                    end_date=str(row['billing_end']),
                    consumption=float(row['therms']),
                    demand=None,
                    cost=float(row['cost_gas']) if 'cost_gas' in row and pd.notna(row['cost_gas']) else None,
                    fuel_type="natural_gas",
                    units="therms"
                ))

        elec_data = cls(bills=elec_bills, fuel_type="electricity", units="kWh")
        gas_data = cls(bills=gas_bills, fuel_type="natural_gas", units="therms") if gas_bills else None

        return elec_data, gas_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UtilityData':
        """Create UtilityData from dictionary"""
        bills = []
        for bill_data in data.get('bills', []):
            bills.append(UtilityBill(**bill_data))

        return cls(
            bills=bills,
            fuel_type=data.get('fuel_type', 'electricity'),
            units=data.get('units', 'kWh'),
            location=data.get('location'),
            account_id=data.get('account_id')
        )


class SimulationResultsParser:
    """
    Parser for EnergyPlus simulation output files

    Extracts energy consumption data from CSV or ESO files
    for comparison with utility bills.
    """

    # Conversion factors to kWh
    ENERGY_CONVERSIONS = {
        'J': 2.7778e-7,      # Joules to kWh
        'kJ': 2.7778e-4,     # kJ to kWh
        'MJ': 0.27778,       # MJ to kWh
        'GJ': 277.78,        # GJ to kWh
        'Wh': 0.001,         # Wh to kWh
        'kWh': 1.0,          # Already kWh
        'MWh': 1000.0,       # MWh to kWh
        'therm': 29.3071,    # therms to kWh
        'kBtu': 0.293071,    # kBtu to kWh
        'MMBtu': 293.071,    # MMBtu to kWh
    }

    @classmethod
    def parse_meter_csv(
        cls,
        csv_path: str,
        meter_name: str = "Electricity:Facility",
        source_units: str = "J",
        target_units: str = "kWh",
        aggregation: str = "monthly"
    ) -> pd.DataFrame:
        """
        Parse EnergyPlus meter output CSV and aggregate

        Args:
            csv_path: Path to the CSV output file
            meter_name: Name of the meter column to extract
            source_units: Units in the source file (typically J for EnergyPlus)
            target_units: Desired output units
            aggregation: Time aggregation - 'monthly', 'daily', or 'hourly'

        Returns:
            DataFrame with aggregated energy consumption
        """
        df = pd.read_csv(csv_path)

        # Find the meter column (EnergyPlus format: "MeterName [units](TimeStep)")
        meter_cols = [col for col in df.columns if meter_name in col]
        if not meter_cols:
            raise ValueError(f"Meter '{meter_name}' not found in CSV. Available: {list(df.columns)}")

        meter_col = meter_cols[0]

        # Parse the Date/Time column
        date_col = df.columns[0]  # Usually "Date/Time"
        df['datetime'] = pd.to_datetime(df[date_col], format='%m/%d %H:%M:%S', errors='coerce')

        # Add year (EnergyPlus doesn't include year in output)
        # Assume current year or extract from simulation period
        if df['datetime'].isna().any():
            # Try alternative format
            df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')

        # Convert units
        conversion = cls.ENERGY_CONVERSIONS[source_units] / cls.ENERGY_CONVERSIONS[target_units]
        df['consumption'] = df[meter_col] * conversion

        # Aggregate
        if aggregation == 'monthly':
            df['period'] = df['datetime'].dt.to_period('M')
            result = df.groupby('period')['consumption'].sum().reset_index()
            result['period'] = result['period'].astype(str)
        elif aggregation == 'daily':
            df['period'] = df['datetime'].dt.date
            result = df.groupby('period')['consumption'].sum().reset_index()
        else:  # hourly
            result = df[['datetime', 'consumption']].copy()
            result.columns = ['period', 'consumption']

        return result

    @classmethod
    def parse_annual_output(
        cls,
        output_dir: str,
        meter_name: str = "Electricity:Facility"
    ) -> Dict[str, float]:
        """
        Parse annual totals from EnergyPlus table output

        Args:
            output_dir: Directory containing simulation outputs
            meter_name: Name of the meter to extract

        Returns:
            Dictionary with annual consumption values
        """
        output_path = Path(output_dir)

        # Look for the table output HTML or CSV
        table_files = list(output_path.glob("*Table.csv")) + list(output_path.glob("*Table.htm*"))

        if not table_files:
            raise FileNotFoundError(f"No table output files found in {output_dir}")

        # Parse the first available table file
        # This is a simplified parser - full implementation would handle HTML
        table_file = table_files[0]

        if table_file.suffix == '.csv':
            df = pd.read_csv(table_file)
            # Find annual summary rows
            # EnergyPlus table format varies by report type

        raise NotImplementedError(
            "parse_annual_output is not yet implemented. "
            "Annual totals must be extracted manually from EnergyPlus table output files."
        )


class CalibrationComparator:
    """
    Compares simulated results against utility data

    Handles time period alignment and unit conversions
    for proper comparison.
    """

    def __init__(self, config=None):
        self.config = config
        self.calculator = CalibrationMetricsCalculator()

    def compare_monthly(
        self,
        utility_data: UtilityData,
        simulated_data: Union[pd.DataFrame, pd.Series, np.ndarray, List[float]],
        standard: CalibrationStandard = CalibrationStandard.ASHRAE_14_MONTHLY
    ) -> Dict[str, Any]:
        """
        Compare monthly utility bills against simulated results

        Args:
            utility_data: UtilityData object with billing data
            simulated_data: Simulated monthly consumption values
            standard: Calibration standard to evaluate against

        Returns:
            Dictionary with comparison results and calibration metrics
        """
        # Get measured values
        measured = utility_data.get_consumption_array()

        # Convert simulated data to array
        if isinstance(simulated_data, pd.DataFrame):
            simulated = simulated_data['consumption'].values
        elif isinstance(simulated_data, pd.Series):
            simulated = simulated_data.values
        else:
            simulated = np.asarray(simulated_data)

        # Validate alignment
        if len(measured) != len(simulated):
            raise ValueError(
                f"Data length mismatch: utility={len(measured)} months, "
                f"simulated={len(simulated)} months"
            )

        # Compute metrics
        metrics = self.calculator.compute_all_metrics(measured, simulated)

        # Build comparison result
        result = {
            "comparison_type": "monthly",
            "n_periods": len(measured),
            "utility_data": {
                "fuel_type": utility_data.fuel_type,
                "units": utility_data.units,
                "total": float(np.sum(measured)),
                "mean": float(np.mean(measured)),
                "std": float(np.std(measured))
            },
            "simulated_data": {
                "total": float(np.sum(simulated)),
                "mean": float(np.mean(simulated)),
                "std": float(np.std(simulated))
            },
            "difference": {
                "total": float(np.sum(measured) - np.sum(simulated)),
                "total_percent": float(100 * (np.sum(measured) - np.sum(simulated)) / np.sum(measured)),
            },
            "metrics": metrics.to_dict(),
            "calibration_status": metrics.get_status(standard),
            "monthly_comparison": []
        }

        # Add month-by-month comparison
        for i, (m, s) in enumerate(zip(measured, simulated)):
            result["monthly_comparison"].append({
                "period": i + 1,
                "measured": float(m),
                "simulated": float(s),
                "difference": float(m - s),
                "percent_diff": float(100 * (m - s) / m) if m != 0 else 0.0
            })

        return result

    def compare_from_csv(
        self,
        utility_csv: str,
        simulation_csv: str,
        meter_name: str = "Electricity:Facility",
        standard: CalibrationStandard = CalibrationStandard.ASHRAE_14_MONTHLY
    ) -> Dict[str, Any]:
        """
        Compare utility CSV against simulation CSV output

        Args:
            utility_csv: Path to utility bill CSV
            simulation_csv: Path to EnergyPlus output CSV
            meter_name: Name of meter to extract from simulation
            standard: Calibration standard to evaluate against

        Returns:
            Comparison results dictionary
        """
        # Load utility data
        utility_data = UtilityData.from_csv(utility_csv)

        # Parse simulation results
        sim_results = SimulationResultsParser.parse_meter_csv(
            simulation_csv,
            meter_name=meter_name,
            aggregation='monthly'
        )

        return self.compare_monthly(utility_data, sim_results, standard)

    def generate_calibration_report(
        self,
        comparison_result: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a markdown calibration report

        Args:
            comparison_result: Result from compare_monthly()
            output_path: Optional path to save the report

        Returns:
            Markdown formatted report string
        """
        metrics = comparison_result["metrics"]
        status = comparison_result["calibration_status"]

        report = f"""# Calibration Report

## Summary

- **Standard**: {status['standard']}
- **Calibrated**: {'✅ Yes' if status['is_calibrated'] else '❌ No'}
- **Periods Compared**: {comparison_result['n_periods']}

## Calibration Metrics

| Metric | Value | Limit | Status |
|--------|-------|-------|--------|
| NMBE | {metrics['nmbe_percent']:.2f}% | ±{status['metrics']['nmbe']['limit']}% | {'✅' if status['metrics']['nmbe']['passed'] else '❌'} |
| CV(RMSE) | {metrics['cvrmse_percent']:.2f}% | ≤{status['metrics']['cvrmse']['limit']}% | {'✅' if status['metrics']['cvrmse']['passed'] else '❌'} |
"""

        if 'r_squared' in status['metrics']:
            r2 = status['metrics']['r_squared']
            report += f"| R² | {r2['value']:.3f} | ≥{r2['limit']} | {'✅' if r2['passed'] else '❌'} |\n"

        report += f"""
## Energy Totals

| Source | Total ({comparison_result['utility_data']['units']}) |
|--------|-------|
| Measured | {comparison_result['utility_data']['total']:,.0f} |
| Simulated | {comparison_result['simulated_data']['total']:,.0f} |
| Difference | {comparison_result['difference']['total']:,.0f} ({comparison_result['difference']['total_percent']:.1f}%) |

## Additional Statistics

- **RMSE**: {metrics['rmse']:.1f} {comparison_result['utility_data']['units']}
- **MAE**: {metrics['mae']:.1f} {comparison_result['utility_data']['units']}
- **MBE**: {metrics['mbe']:.1f} {comparison_result['utility_data']['units']}

## Monthly Comparison

| Month | Measured | Simulated | Diff | % Diff |
|-------|----------|-----------|------|--------|
"""

        for month in comparison_result["monthly_comparison"]:
            report += f"| {month['period']} | {month['measured']:,.0f} | {month['simulated']:,.0f} | {month['difference']:,.0f} | {month['percent_diff']:.1f}% |\n"

        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Calibration report saved to: {output_path}")

        return report


class CalibrationManager:
    """
    Main interface for calibration operations

    Integrates with EnergyPlus MCP server for calibration workflows.
    """

    def __init__(self, config=None):
        self.config = config
        self.calculator = CalibrationMetricsCalculator()
        self.comparator = CalibrationComparator(config)

    def compute_metrics(
        self,
        measured: Union[List[float], np.ndarray],
        simulated: Union[List[float], np.ndarray],
        standard: str = "ashrae_14_monthly"
    ) -> str:
        """
        Compute calibration metrics and return JSON result

        Args:
            measured: Measured/utility data values
            simulated: Simulated values from EnergyPlus
            standard: Calibration standard name

        Returns:
            JSON string with metrics and calibration status
        """
        try:
            # Parse standard
            std = CalibrationStandard(standard)

            # Compute metrics
            metrics = self.calculator.compute_all_metrics(measured, simulated)

            result = {
                "success": True,
                "metrics": metrics.to_dict(),
                "calibration_status": metrics.get_status(std),
                "interpretation": {
                    "nmbe": self._interpret_nmbe(metrics.nmbe),
                    "cvrmse": self._interpret_cvrmse(metrics.cvrmse),
                    "r_squared": self._interpret_r_squared(metrics.r_squared)
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error computing calibration metrics: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def compare_to_utility(
        self,
        utility_data: Dict[str, Any],
        simulated_values: List[float],
        standard: str = "ashrae_14_monthly"
    ) -> str:
        """
        Compare simulation results to utility data

        Args:
            utility_data: Dictionary with utility bill data
            simulated_values: List of simulated monthly values
            standard: Calibration standard name

        Returns:
            JSON string with comparison results
        """
        try:
            std = CalibrationStandard(standard)
            util = UtilityData.from_dict(utility_data)

            result = self.comparator.compare_monthly(util, simulated_values, std)
            result["success"] = True

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error comparing to utility data: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def _interpret_nmbe(self, nmbe: float) -> str:
        """Interpret NMBE value"""
        if abs(nmbe) < 2:
            return "Excellent - minimal systematic bias"
        elif abs(nmbe) < 5:
            return "Good - within monthly calibration limits"
        elif abs(nmbe) < 10:
            return "Acceptable for hourly data - some systematic bias"
        else:
            sign = "under-predicting" if nmbe > 0 else "over-predicting"
            return f"Poor - significant systematic {sign}"

    def _interpret_cvrmse(self, cvrmse: float) -> str:
        """Interpret CV(RMSE) value"""
        if cvrmse < 10:
            return "Excellent - low scatter in predictions"
        elif cvrmse < 15:
            return "Good - within monthly calibration limits"
        elif cvrmse < 30:
            return "Acceptable for hourly data"
        else:
            return "Poor - high scatter, check model inputs"

    def _interpret_r_squared(self, r_squared: float) -> str:
        """Interpret R² value"""
        if r_squared > 0.9:
            return "Excellent - simulation explains most variance"
        elif r_squared > 0.75:
            return "Good - meets calibration threshold"
        elif r_squared > 0.5:
            return "Fair - moderate correlation"
        else:
            return "Poor - weak correlation with measured data"

    def get_available_standards(self) -> str:
        """Get list of available calibration standards"""
        standards = []
        for std in CalibrationStandard:
            thresholds = CalibrationThresholds.get_thresholds(std)
            standards.append({
                "name": std.value,
                "nmbe_limit": thresholds.nmbe_limit,
                "cvrmse_limit": thresholds.cvrmse_limit,
                "r2_limit": thresholds.r2_limit
            })
        return json.dumps({"standards": standards}, indent=2)
