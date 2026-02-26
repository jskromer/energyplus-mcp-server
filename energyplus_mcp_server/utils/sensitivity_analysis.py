"""
Sensitivity analysis utilities for EnergyPlus model calibration

Implements Morris one-at-a-time (OAT) method and parameter screening
for identifying influential parameters before calibration.

EnergyPlus Model Context Protocol Server (EnergyPlus-MCP)
Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
import json
import tempfile
import subprocess

import numpy as np
import pandas as pd

# Optional SALib import
try:
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

from eppy.modeleditor import IDF

from .calibration_tuning import STANDARD_PARAMETERS, ParameterCategory, CalibrationTuner

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis for a single parameter"""
    parameter_name: str
    mu: float  # Mean elementary effect
    mu_star: float  # Mean of absolute elementary effects
    sigma: float  # Standard deviation of elementary effects
    rank: int  # Rank by influence (1 = most influential)
    category: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "mu": round(self.mu, 4),
            "mu_star": round(self.mu_star, 4),
            "sigma": round(self.sigma, 4),
            "rank": self.rank,
            "category": self.category,
            "description": self.description,
            "interpretation": self._interpret()
        }

    def _interpret(self) -> str:
        """Interpret sensitivity results"""
        if self.mu_star > 10:
            influence = "Very high influence"
        elif self.mu_star > 5:
            influence = "High influence"
        elif self.mu_star > 1:
            influence = "Moderate influence"
        else:
            influence = "Low influence"

        # Check for non-linearity (high sigma relative to mu_star)
        if self.mu_star > 0 and self.sigma / self.mu_star > 0.5:
            nonlinearity = " with significant non-linear effects"
        else:
            nonlinearity = ""

        return f"{influence}{nonlinearity}"


@dataclass
class SensitivityAnalysisResults:
    """Container for full sensitivity analysis results"""
    results: List[SensitivityResult]
    n_simulations: int
    objective_function: str
    parameters_analyzed: List[str]
    problem_definition: Dict[str, Any]

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get parameter ranking by mu_star"""
        return [(r.parameter_name, r.mu_star) for r in sorted(self.results, key=lambda x: x.rank)]

    def get_top_parameters(self, n: int = 5) -> List[SensitivityResult]:
        """Get top N most influential parameters"""
        return sorted(self.results, key=lambda x: x.rank)[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "objective_function": self.objective_function,
            "parameters_analyzed": self.parameters_analyzed,
            "ranking": self.get_ranking(),
            "results": [r.to_dict() for r in sorted(self.results, key=lambda x: x.rank)],
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate calibration recommendations based on sensitivity"""
        recs = []
        top_params = self.get_top_parameters(3)

        recs.append(f"Focus calibration on the top {len(top_params)} most influential parameters:")
        for i, param in enumerate(top_params, 1):
            recs.append(f"  {i}. {param.parameter_name} ({param.category})")

        # Check for non-linear parameters
        nonlinear = [r for r in self.results if r.mu_star > 0 and r.sigma / r.mu_star > 0.5]
        if nonlinear:
            recs.append("Parameters with non-linear effects (may require iterative tuning):")
            for p in nonlinear[:3]:
                recs.append(f"  - {p.parameter_name}")

        return recs


class MorrisSampler:
    """
    Morris one-at-a-time (OAT) sensitivity analysis sampler

    The Morris method efficiently screens parameters to identify
    which have the greatest influence on model outputs.
    """

    def __init__(self, config=None):
        self.config = config
        if not SALIB_AVAILABLE:
            logger.warning("SALib not available - install with: pip install SALib")

    def define_problem(
        self,
        parameter_names: Optional[List[str]] = None,
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Define the sensitivity analysis problem

        Args:
            parameter_names: List of parameters to analyze (None = all standard)
            custom_bounds: Optional custom bounds for parameters

        Returns:
            SALib problem dictionary
        """
        if parameter_names is None:
            parameter_names = list(STANDARD_PARAMETERS.keys())

        names = []
        bounds = []

        for name in parameter_names:
            if name not in STANDARD_PARAMETERS:
                logger.warning(f"Unknown parameter: {name}, skipping")
                continue

            spec = STANDARD_PARAMETERS[name]
            if spec.min_value is None or spec.max_value is None:
                logger.warning(f"Parameter {name} has no bounds, skipping")
                continue

            names.append(name)

            if custom_bounds and name in custom_bounds:
                bounds.append(list(custom_bounds[name]))
            else:
                bounds.append([spec.min_value, spec.max_value])

        return {
            'num_vars': len(names),
            'names': names,
            'bounds': bounds
        }

    def generate_samples(
        self,
        problem: Dict[str, Any],
        n_trajectories: int = 10,
        num_levels: int = 4
    ) -> np.ndarray:
        """
        Generate Morris parameter samples

        Args:
            problem: SALib problem definition
            n_trajectories: Number of Morris trajectories
            num_levels: Number of grid levels

        Returns:
            2D array of parameter samples (n_samples x n_params)
        """
        if not SALIB_AVAILABLE:
            raise ImportError("SALib required for Morris sampling. Install with: pip install SALib")

        samples = morris_sample.sample(
            problem,
            N=n_trajectories,
            num_levels=num_levels
        )

        logger.info(f"Generated {len(samples)} samples for {problem['num_vars']} parameters")
        return samples

    def analyze_results(
        self,
        problem: Dict[str, Any],
        samples: np.ndarray,
        outputs: np.ndarray
    ) -> SensitivityAnalysisResults:
        """
        Analyze Morris sensitivity results

        Args:
            problem: SALib problem definition
            samples: Parameter samples used
            outputs: Model outputs for each sample

        Returns:
            SensitivityAnalysisResults object
        """
        if not SALIB_AVAILABLE:
            raise ImportError("SALib required for Morris analysis")

        # Filter out invalid outputs
        valid_mask = np.isfinite(outputs)
        if not np.all(valid_mask):
            logger.warning(f"Removing {np.sum(~valid_mask)} invalid outputs")
            # SALib requires complete trajectories, so we can't simply filter
            # Replace invalid values with mean of valid values
            outputs = np.where(valid_mask, outputs, np.nanmean(outputs))

        Si = morris_analyze.analyze(problem, samples, outputs)

        # Build results
        results = []
        for i, name in enumerate(problem['names']):
            spec = STANDARD_PARAMETERS.get(name)
            category = spec.category.value if spec else "unknown"
            description = spec.description if spec else ""

            results.append(SensitivityResult(
                parameter_name=name,
                mu=float(Si['mu'][i]),
                mu_star=float(Si['mu_star'][i]),
                sigma=float(Si['sigma'][i]),
                rank=0,  # Will be set below
                category=category,
                description=description
            ))

        # Assign ranks
        sorted_results = sorted(results, key=lambda x: x.mu_star, reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            result.rank = rank

        return SensitivityAnalysisResults(
            results=results,
            n_simulations=len(samples),
            objective_function="cvrmse",
            parameters_analyzed=problem['names'],
            problem_definition=problem
        )


class LocalSensitivityAnalyzer:
    """
    Simple local sensitivity analysis without SALib

    Performs one-at-a-time parameter variations around a base case.
    Less rigorous than Morris but doesn't require SALib.
    """

    def __init__(self, config=None):
        self.config = config
        self.tuner = CalibrationTuner(config)

    def compute_local_sensitivity(
        self,
        base_output: float,
        perturbed_outputs: Dict[str, Tuple[float, float]],  # param -> (low_output, high_output)
        perturbation_fraction: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute local sensitivity indices

        Args:
            base_output: Output at base parameter values
            perturbed_outputs: Dictionary of param -> (output_at_-10%, output_at_+10%)
            perturbation_fraction: Fraction used for perturbation

        Returns:
            Dictionary of sensitivity metrics per parameter
        """
        results = {}

        for param, (low_out, high_out) in perturbed_outputs.items():
            # Central difference approximation of derivative
            delta_output = high_out - low_out
            delta_normalized = delta_output / (2 * perturbation_fraction * base_output) if base_output != 0 else 0

            results[param] = {
                "absolute_sensitivity": abs(delta_output),
                "normalized_sensitivity": abs(delta_normalized),
                "direction": "positive" if high_out > low_out else "negative",
                "low_output": low_out,
                "high_output": high_out,
                "base_output": base_output
            }

        # Rank by absolute sensitivity
        sorted_params = sorted(results.items(), key=lambda x: x[1]["normalized_sensitivity"], reverse=True)
        for rank, (param, data) in enumerate(sorted_params, 1):
            data["rank"] = rank

        return results

    def generate_perturbation_cases(
        self,
        idf: IDF,
        parameters: List[str],
        perturbation_fraction: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate perturbation cases for local sensitivity

        Args:
            idf: Base IDF model
            parameters: Parameters to perturb
            perturbation_fraction: Fraction to perturb (e.g., 0.1 = ±10%)

        Returns:
            Dictionary with base values and perturbation specifications
        """
        cases = {"base": {}, "low": {}, "high": {}}

        current_values = self.tuner.extract_current_values(idf, parameters)

        for param in parameters:
            if param not in current_values:
                continue

            values = current_values[param].get("values", [])
            if not values:
                continue

            # Use first value as representative
            base_value = values[0].get("value")
            if base_value is None or not isinstance(base_value, (int, float)):
                continue

            spec = STANDARD_PARAMETERS.get(param)
            low_value = base_value * (1 - perturbation_fraction)
            high_value = base_value * (1 + perturbation_fraction)

            # Clamp to bounds
            if spec:
                if spec.min_value is not None:
                    low_value = max(low_value, spec.min_value)
                if spec.max_value is not None:
                    high_value = min(high_value, spec.max_value)

            cases["base"][param] = base_value
            cases["low"][param] = low_value
            cases["high"][param] = high_value

        return cases


class SensitivityAnalysisManager:
    """
    High-level interface for sensitivity analysis

    Integrates with MCP server for running sensitivity studies.
    """

    def __init__(self, config=None):
        self.config = config
        self.sampler = MorrisSampler(config)
        self.local_analyzer = LocalSensitivityAnalyzer(config)

    def check_salib_available(self) -> bool:
        """Check if SALib is installed"""
        return SALIB_AVAILABLE

    def get_recommended_parameters(self, category: Optional[str] = None) -> str:
        """
        Get recommended parameters for sensitivity analysis

        Args:
            category: Optional category filter

        Returns:
            JSON string with parameter recommendations
        """
        params_by_category = {}

        for name, spec in STANDARD_PARAMETERS.items():
            if category and spec.category.value != category:
                continue

            cat = spec.category.value
            if cat not in params_by_category:
                params_by_category[cat] = []

            params_by_category[cat].append({
                "name": name,
                "bounds": [spec.min_value, spec.max_value],
                "default": spec.default_value,
                "units": spec.units,
                "description": spec.description
            })

        # Sort categories by typical sensitivity order
        category_order = [
            "schedules",
            "infiltration",
            "internal_loads",
            "hvac",
            "envelope",
            "setpoints"
        ]

        ordered_result = {}
        for cat in category_order:
            if cat in params_by_category:
                ordered_result[cat] = params_by_category[cat]

        return json.dumps({
            "success": True,
            "parameters_by_category": ordered_result,
            "recommended_order": category_order,
            "notes": [
                "Parameters are ordered by typical sensitivity (highest first)",
                "Schedules typically have the highest impact on calibration",
                "Start with high-sensitivity parameters for efficient calibration"
            ]
        }, indent=2)

    def setup_morris_analysis(
        self,
        parameter_names: Optional[List[str]] = None,
        n_trajectories: int = 10,
        num_levels: int = 4
    ) -> str:
        """
        Set up Morris sensitivity analysis

        Args:
            parameter_names: Parameters to include (None = all)
            n_trajectories: Number of Morris trajectories
            num_levels: Grid levels for sampling

        Returns:
            JSON string with problem definition and sample count
        """
        if not SALIB_AVAILABLE:
            return json.dumps({
                "success": False,
                "error": "SALib not installed. Install with: pip install SALib",
                "alternative": "Use local sensitivity analysis instead"
            })

        try:
            problem = self.sampler.define_problem(parameter_names)
            samples = self.sampler.generate_samples(problem, n_trajectories, num_levels)

            return json.dumps({
                "success": True,
                "problem": problem,
                "n_samples": len(samples),
                "n_parameters": problem['num_vars'],
                "samples_preview": samples[:5].tolist(),
                "notes": [
                    f"Total {len(samples)} simulations required",
                    f"Each simulation tests a different parameter combination",
                    "Run simulations and collect outputs, then call analyze_morris_results"
                ]
            }, indent=2)

        except Exception as e:
            logger.error(f"Error setting up Morris analysis: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def setup_local_sensitivity(
        self,
        idf_path: str,
        parameters: Optional[List[str]] = None,
        perturbation_percent: float = 10.0
    ) -> str:
        """
        Set up local sensitivity analysis (no SALib required)

        Args:
            idf_path: Path to base IDF
            parameters: Parameters to analyze (None = all)
            perturbation_percent: Percent to perturb (e.g., 10 = ±10%)

        Returns:
            JSON string with perturbation cases
        """
        try:
            idf = IDF(idf_path)

            if parameters is None:
                parameters = list(STANDARD_PARAMETERS.keys())

            cases = self.local_analyzer.generate_perturbation_cases(
                idf,
                parameters,
                perturbation_percent / 100.0
            )

            # Count simulations needed: 1 base + 2 per parameter (low and high)
            n_params = len(cases["base"])
            n_sims = 1 + 2 * n_params

            return json.dumps({
                "success": True,
                "n_simulations_required": n_sims,
                "n_parameters": n_params,
                "perturbation_percent": perturbation_percent,
                "cases": cases,
                "simulation_order": [
                    "1. Run base case",
                    *[f"{i+2}. Run {p} LOW ({cases['low'][p]:.4g})" for i, p in enumerate(cases['base'].keys())],
                    *[f"{n_params+i+2}. Run {p} HIGH ({cases['high'][p]:.4g})" for i, p in enumerate(cases['base'].keys())]
                ]
            }, indent=2)

        except Exception as e:
            logger.error(f"Error setting up local sensitivity: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def analyze_local_results(
        self,
        base_output: float,
        parameter_outputs: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Analyze local sensitivity results

        Args:
            base_output: Output from base case
            parameter_outputs: Dict of param -> {"low": value, "high": value}

        Returns:
            JSON string with sensitivity analysis results
        """
        try:
            # Convert to tuple format
            perturbed = {
                p: (v["low"], v["high"])
                for p, v in parameter_outputs.items()
            }

            results = self.local_analyzer.compute_local_sensitivity(
                base_output,
                perturbed
            )

            # Generate ranking
            ranking = sorted(
                [(p, r["normalized_sensitivity"]) for p, r in results.items()],
                key=lambda x: x[1],
                reverse=True
            )

            return json.dumps({
                "success": True,
                "base_output": base_output,
                "results": results,
                "ranking": ranking,
                "recommendations": [
                    f"Most influential: {ranking[0][0]}" if ranking else "No parameters analyzed",
                    "Focus calibration on top-ranked parameters",
                    "Parameters with 'negative' direction decrease output when increased"
                ]
            }, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing local sensitivity: {e}")
            return json.dumps({"success": False, "error": str(e)})
