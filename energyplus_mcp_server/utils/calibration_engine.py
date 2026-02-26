"""
Calibration Engine for EnergyPlus Model Calibration

Orchestrates the complete calibration workflow including:
- Parameter extraction and modification
- Sensitivity analysis
- Metrics computation
- Utility data comparison
- Iterative calibration

EnergyPlus Model Context Protocol Server (EnergyPlus-MCP)
Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.
"""

import logging
import json
import subprocess
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import tempfile

import numpy as np
import pandas as pd
from eppy.modeleditor import IDF

from .calibration import (
    CalibrationMetricsCalculator,
    CalibrationMetrics,
    CalibrationStandard,
    CalibrationComparator,
    CalibrationManager,
    UtilityData,
    SimulationResultsParser
)
from .calibration_tuning import (
    CalibrationTuner,
    CalibrationParameterManager,
    STANDARD_PARAMETERS,
    ParameterCategory
)
from .sensitivity_analysis import (
    SensitivityAnalysisManager,
    MorrisSampler,
    SALIB_AVAILABLE
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for calibration runs"""
    # Paths
    energyplus_path: str = "/usr/local/EnergyPlus-24-2-0"
    idd_path: Optional[str] = None
    weather_file: Optional[str] = None
    output_dir: str = "calibration_output"

    # Calibration targets (ASHRAE Guideline 14 monthly)
    target_nmbe: float = 5.0  # ±5%
    target_cvrmse: float = 15.0  # ≤15%
    target_r2: float = 0.75  # ≥0.75

    # Optimization settings
    max_iterations: int = 50
    convergence_threshold: float = 0.1  # Stop if improvement < 0.1%

    # Simulation settings
    simulation_timeout: int = 600  # 10 minutes

    def __post_init__(self):
        if self.idd_path is None:
            self.idd_path = f"{self.energyplus_path}/Energy+.idd"


@dataclass
class CalibrationIteration:
    """Record of a single calibration iteration"""
    iteration: int
    parameters: Dict[str, float]
    metrics: Dict[str, float]
    is_calibrated: bool
    objective: float
    idf_path: str
    output_dir: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "is_calibrated": self.is_calibrated,
            "objective": self.objective,
            "idf_path": self.idf_path,
            "output_dir": self.output_dir,
            "timestamp": self.timestamp
        }


@dataclass
class CalibrationSession:
    """Tracks a complete calibration session"""
    session_id: str
    base_idf: str
    utility_data: Dict[str, Any]
    config: CalibrationConfig
    iterations: List[CalibrationIteration] = field(default_factory=list)
    best_iteration: Optional[CalibrationIteration] = None
    status: str = "initialized"
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    def add_iteration(self, iteration: CalibrationIteration):
        """Add an iteration and update best if applicable"""
        self.iterations.append(iteration)
        if self.best_iteration is None or iteration.objective < self.best_iteration.objective:
            self.best_iteration = iteration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "base_idf": self.base_idf,
            "status": self.status,
            "n_iterations": len(self.iterations),
            "best_metrics": self.best_iteration.metrics if self.best_iteration else None,
            "best_parameters": self.best_iteration.parameters if self.best_iteration else None,
            "is_calibrated": self.best_iteration.is_calibrated if self.best_iteration else False,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class CalibrationEngine:
    """
    Main calibration engine for EnergyPlus models

    Provides complete workflow for ASHRAE Guideline 14 calibration:
    1. Load utility data and base model
    2. Run sensitivity analysis to identify key parameters
    3. Iteratively adjust parameters
    4. Validate calibration metrics
    5. Generate calibration report
    """

    def __init__(self, config: Optional[CalibrationConfig] = None, mcp_config=None):
        self.config = config or CalibrationConfig()
        self.mcp_config = mcp_config

        # Initialize components
        self.metrics_calculator = CalibrationMetricsCalculator()
        self.comparator = CalibrationComparator(mcp_config)
        self.tuner = CalibrationTuner(mcp_config)
        self.param_manager = CalibrationParameterManager(mcp_config)
        self.sensitivity_manager = SensitivityAnalysisManager(mcp_config)

        # Session tracking
        self.sessions: Dict[str, CalibrationSession] = {}

    def run_simulation(
        self,
        idf_path: str,
        weather_file: str,
        output_dir: str
    ) -> Optional[Path]:
        """
        Run EnergyPlus simulation

        Args:
            idf_path: Path to IDF file
            weather_file: Path to weather file
            output_dir: Output directory

        Returns:
            Path to output directory if successful, None otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            f"{self.config.energyplus_path}/energyplus",
            "-w", weather_file,
            "-d", str(output_path),
            "-r",  # Read variables
            idf_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.simulation_timeout
            )

            if result.returncode != 0:
                logger.error(f"EnergyPlus error: {result.stderr[:500]}")
                return None

            return output_path

        except subprocess.TimeoutExpired:
            logger.error("EnergyPlus simulation timed out")
            return None
        except FileNotFoundError:
            logger.error(f"EnergyPlus not found at {self.config.energyplus_path}")
            return None

    def compute_objective(
        self,
        measured: np.ndarray,
        simulated: np.ndarray
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Compute calibration objective function

        Args:
            measured: Measured utility values
            simulated: Simulated values

        Returns:
            Tuple of (objective, metrics_dict, is_calibrated)
        """
        metrics = self.metrics_calculator.compute_all_metrics(measured, simulated)

        # Combined objective: weighted sum of NMBE and CV(RMSE)
        objective = abs(metrics.nmbe) + metrics.cvrmse

        is_calibrated = (
            abs(metrics.nmbe) <= self.config.target_nmbe and
            metrics.cvrmse <= self.config.target_cvrmse and
            metrics.r_squared >= self.config.target_r2
        )

        return objective, metrics.to_dict(), is_calibrated

    def create_session(
        self,
        idf_path: str,
        utility_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a new calibration session

        Args:
            idf_path: Path to base IDF
            utility_data: Utility data dictionary
            session_id: Optional session ID (auto-generated if None)

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        session = CalibrationSession(
            session_id=session_id,
            base_idf=idf_path,
            utility_data=utility_data,
            config=self.config
        )

        self.sessions[session_id] = session
        logger.info(f"Created calibration session: {session_id}")

        return session_id

    def run_baseline(
        self,
        session_id: str,
        weather_file: str
    ) -> str:
        """
        Run baseline simulation for a session

        Args:
            session_id: Session ID
            weather_file: Path to weather file

        Returns:
            JSON result string
        """
        if session_id not in self.sessions:
            return json.dumps({"success": False, "error": f"Session {session_id} not found"})

        session = self.sessions[session_id]
        session.status = "running_baseline"

        try:
            # Run simulation
            output_dir = Path(self.config.output_dir) / session_id / "baseline"
            result = self.run_simulation(session.base_idf, weather_file, str(output_dir))

            if result is None:
                return json.dumps({"success": False, "error": "Baseline simulation failed"})

            # Parse results
            csv_files = list(output_dir.glob("*Meter.csv")) + list(output_dir.glob("eplusout.csv"))
            if not csv_files:
                return json.dumps({"success": False, "error": "No output CSV found"})

            # Get utility data
            utility = UtilityData.from_dict(session.utility_data)
            measured = utility.get_consumption_array()

            # Parse simulated data (simplified - would need to match time periods)
            sim_df = pd.read_csv(csv_files[0])
            # Find electricity column
            elec_cols = [c for c in sim_df.columns if 'Electricity' in c and 'Facility' in c]
            if not elec_cols:
                return json.dumps({"success": False, "error": "No electricity meter found in output"})

            simulated = sim_df[elec_cols[0]].values * 2.778e-7  # J to kWh

            # Aggregate to monthly if needed
            if len(simulated) > len(measured):
                # Hourly/timestep data - aggregate to monthly
                # This is simplified - real implementation would use date parsing
                monthly_sim = []
                points_per_month = len(simulated) // len(measured)
                for i in range(len(measured)):
                    start = i * points_per_month
                    end = (i + 1) * points_per_month
                    monthly_sim.append(np.sum(simulated[start:end]))
                simulated = np.array(monthly_sim)

            # Compute metrics
            objective, metrics, is_calibrated = self.compute_objective(measured, simulated)

            # Record iteration
            iteration = CalibrationIteration(
                iteration=0,
                parameters={},
                metrics=metrics,
                is_calibrated=is_calibrated,
                objective=objective,
                idf_path=session.base_idf,
                output_dir=str(output_dir)
            )
            session.add_iteration(iteration)
            session.status = "baseline_complete"

            return json.dumps({
                "success": True,
                "session_id": session_id,
                "baseline_metrics": metrics,
                "is_calibrated": is_calibrated,
                "objective": objective,
                "output_dir": str(output_dir),
                "next_steps": [
                    "Run sensitivity analysis to identify key parameters" if not is_calibrated else "Model already calibrated!",
                    "Adjust parameters based on sensitivity ranking",
                    "Re-run simulation and check metrics"
                ]
            }, indent=2)

        except Exception as e:
            logger.error(f"Error running baseline: {e}")
            session.status = "error"
            return json.dumps({"success": False, "error": str(e)})

    def get_session_status(self, session_id: str) -> str:
        """Get status of a calibration session"""
        if session_id not in self.sessions:
            return json.dumps({"success": False, "error": f"Session {session_id} not found"})

        session = self.sessions[session_id]
        return json.dumps({
            "success": True,
            "session": session.to_dict()
        }, indent=2)

    def suggest_parameter_adjustments(
        self,
        session_id: str,
        sensitivity_ranking: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """
        Suggest parameter adjustments based on current metrics

        Args:
            session_id: Session ID
            sensitivity_ranking: Optional ranking from sensitivity analysis

        Returns:
            JSON string with suggested adjustments
        """
        if session_id not in self.sessions:
            return json.dumps({"success": False, "error": f"Session {session_id} not found"})

        session = self.sessions[session_id]
        if not session.iterations:
            return json.dumps({"success": False, "error": "No baseline results yet"})

        latest = session.iterations[-1]
        metrics = latest.metrics

        suggestions = []

        # Analyze NMBE (bias)
        nmbe = metrics.get("nmbe_percent", 0)
        if abs(nmbe) > self.config.target_nmbe:
            if nmbe > 0:
                # Under-predicting consumption
                suggestions.append({
                    "issue": f"Model under-predicts by {nmbe:.1f}%",
                    "adjustments": [
                        "Increase equipment power density",
                        "Increase lighting power density",
                        "Increase infiltration",
                        "Check HVAC schedules for extended hours"
                    ]
                })
            else:
                # Over-predicting consumption
                suggestions.append({
                    "issue": f"Model over-predicts by {abs(nmbe):.1f}%",
                    "adjustments": [
                        "Decrease equipment power density",
                        "Decrease lighting power density",
                        "Improve envelope insulation",
                        "Increase HVAC efficiency"
                    ]
                })

        # Analyze CV(RMSE) (scatter)
        cvrmse = metrics.get("cvrmse_percent", 0)
        if cvrmse > self.config.target_cvrmse:
            suggestions.append({
                "issue": f"High variability (CV(RMSE) = {cvrmse:.1f}%)",
                "adjustments": [
                    "Review and adjust operational schedules",
                    "Check for seasonal patterns in utility data",
                    "Verify weather file matches utility billing period",
                    "Consider different parameters for heating vs cooling seasons"
                ]
            })

        # Use sensitivity ranking if available
        if sensitivity_ranking:
            top_params = sensitivity_ranking[:3]
            suggestions.append({
                "priority_parameters": [
                    f"{p[0]} (influence: {p[1]:.2f})" for p in top_params
                ],
                "note": "Focus adjustments on these high-influence parameters first"
            })

        return json.dumps({
            "success": True,
            "current_metrics": metrics,
            "target_metrics": {
                "nmbe": f"±{self.config.target_nmbe}%",
                "cvrmse": f"≤{self.config.target_cvrmse}%",
                "r2": f"≥{self.config.target_r2}"
            },
            "suggestions": suggestions
        }, indent=2)


class CalibrationOrchestrator:
    """
    High-level orchestrator for MCP server integration

    Provides simplified interface for calibration operations.
    """

    def __init__(self, config=None):
        self.config = config
        self.engine = CalibrationEngine(mcp_config=config)
        self.metrics_manager = CalibrationManager(config)
        self.param_manager = CalibrationParameterManager(config)
        self.sensitivity_manager = SensitivityAnalysisManager(config)

    def compute_calibration_metrics(
        self,
        measured: List[float],
        simulated: List[float],
        standard: str = "ashrae_14_monthly"
    ) -> str:
        """
        Compute ASHRAE Guideline 14 calibration metrics

        Args:
            measured: Measured utility values
            simulated: Simulated EnergyPlus values
            standard: Calibration standard to use

        Returns:
            JSON string with metrics and calibration status
        """
        return self.metrics_manager.compute_metrics(measured, simulated, standard)

    def compare_utility_data(
        self,
        utility_data: Dict[str, Any],
        simulated_values: List[float],
        standard: str = "ashrae_14_monthly"
    ) -> str:
        """
        Compare simulation results against utility bills

        Args:
            utility_data: Utility bill data dictionary
            simulated_values: Monthly simulated values
            standard: Calibration standard

        Returns:
            JSON string with comparison results
        """
        return self.metrics_manager.compare_to_utility(utility_data, simulated_values, standard)

    def inspect_parameters(self, idf_path: str) -> str:
        """
        Inspect current calibration parameter values

        Args:
            idf_path: Path to IDF file

        Returns:
            JSON string with parameter values
        """
        return self.param_manager.inspect_calibration_parameters(idf_path)

    def modify_parameters(
        self,
        idf_path: str,
        modifications: Dict[str, float],
        mode: str = "set",
        output_path: Optional[str] = None
    ) -> str:
        """
        Modify calibration parameters

        Args:
            idf_path: Input IDF path
            modifications: Parameter modifications
            mode: 'set' or 'multiply'
            output_path: Output path

        Returns:
            JSON string with results
        """
        return self.param_manager.modify_calibration_parameters(
            idf_path, modifications, mode, output_path
        )

    def get_calibration_standards(self) -> str:
        """Get available calibration standards and thresholds"""
        return self.metrics_manager.get_available_standards()

    def setup_sensitivity_analysis(
        self,
        idf_path: str,
        parameters: Optional[List[str]] = None,
        method: str = "local"
    ) -> str:
        """
        Set up sensitivity analysis

        Args:
            idf_path: Path to IDF
            parameters: Parameters to analyze
            method: 'morris' or 'local'

        Returns:
            JSON string with analysis setup
        """
        if method == "morris":
            return self.sensitivity_manager.setup_morris_analysis(parameters)
        else:
            return self.sensitivity_manager.setup_local_sensitivity(idf_path, parameters)

    def get_parameter_recommendations(self, category: Optional[str] = None) -> str:
        """Get recommended parameters for calibration"""
        return self.sensitivity_manager.get_recommended_parameters(category)
