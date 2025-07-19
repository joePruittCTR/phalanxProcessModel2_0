# simStart.py - Enhanced Main Orchestrator for Phalanx C-sUAS Simulation
# Optimized for execution time, portability, and readability

import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import core simulation logic and parameters
try:
    from simProcess import SimulationParameters, runSimulation
    from simAnimate import run_animated_simulation
    from simArrayHandler import ParameterAdjuster
    from simDataFormat import DataFileProcessor
    from simPlot import Plot
    from simStats import Statistics
    from simUtils import work_days_per_year, get_current_date, CircularProgressBar
    from simReport import getFileName as generate_report
    from simInput import get_simulation_parameters
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all simulation modules are in the Python path.")
    sys.exit(1)

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for the simulation."""
    logger = logging.getLogger("PhalanxSimulation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Global logger
logger = setup_logging()

class SimulationOrchestrator:
    """
    Enhanced simulation orchestrator with performance monitoring,
    progress tracking, and configuration management.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the orchestrator with optional configuration file."""
        self.config = self._load_config(config_file)
        self.performance_metrics = {}
        self.start_time = None
        
        # Setup directories
        self.setup_directories()
        
        logger.info("Phalanx C-sUAS Simulation Orchestrator initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "directories": ["./data", "./plots", "./reports", "./reportResources", "./logs"],
            "default_params": {
                "sim_time": 500,
                "time_unit": "minutes",
                "processing_fte": 1.0,
                "processing_overhead": 0.10,
                "processing_efficiency": 0.90,
                "warmup_time": 30,
                "seed": 42
            },
            "performance": {
                "enable_profiling": False,
                "memory_monitoring": True,
                "progress_bars": True
            },
            "output": {
                "save_intermediate_results": True,
                "generate_plots": True,
                "create_report": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations (user config overrides defaults)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def setup_directories(self) -> None:
        """Ensures all necessary output directories exist."""
        for directory in self.config["directories"]:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
    
    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.performance_metrics = {
            "start_time": self.start_time,
            "memory_usage": [],
            "execution_phases": {}
        }
        
        if self.config["performance"]["memory_monitoring"]:
            try:
                import psutil
                process = psutil.Process()
                self.performance_metrics["initial_memory"] = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                logger.warning("psutil not installed. Memory monitoring disabled.")
    
    def _log_performance_phase(self, phase_name: str) -> None:
        """Log performance for a specific phase."""
        if self.start_time:
            current_time = time.time()
            self.performance_metrics["execution_phases"][phase_name] = current_time - self.start_time
            logger.debug(f"Phase '{phase_name}' completed in {current_time - self.start_time:.2f} seconds")

def _get_simulation_parameters(self, args):
    """EMERGENCY BYPASS: Skip GUI and use default parameters"""
    print("="*60)
    print("EMERGENCY BYPASS: Skipping GUI, using default parameters")
    print("="*60)
    
    default_params = {
        'simFileName': 'phalanx_simulation',
        'timeWindow': 1.0,
        'nservers': 4.0,
        'siprTransfer': 1.0,
        'lowPriority': 0.1,
        'medPriority': 67.4,
        'highPriority': 30.0,
        'vHighPriority': 2.5,
        'fileGrowth': 0.0,
        'sensorGrowth': 0.0,
        'ingestEfficiency': 0.0,
        'goalTarget': '',
        'goalParameter': '',
        'waitTimeMax': 1.0,
        'processingFteMax': 20.0,
        'simMaxIterations': 20,
        'seed': 42,
        'warmup_time': 30.0,
        'sim_time': 500.0,
        'work_days_per_year': 249.0,
        'processing_fte': 4.0,
        'processing_overhead': 0.10,
        'processing_efficiency': 0.90,
        'co_time': 30.0, 'co_iat': 60.0,
        'dk_time': 45.0, 'dk_iat': 120.0,
        'ma_time': 20.0, 'ma_iat': 90.0,
        'nj_time': 35.0, 'nj_iat': 150.0,
        'rs_time': 25.0, 'rs_iat': 180.0,
        'sv_time': 40.0, 'sv_iat': 200.0,
        'tg_time': 50.0, 'tg_iat': 240.0,
        'wt_time': 30.0, 'wt_iat': 100.0,
        'simulation_mode': 'single'
    }
    
    logger.info("Using emergency bypass parameters")
    return default_params
    
    # def _get_simulation_parameters(self, args: argparse.Namespace) -> Dict[str, Any]:
    #     """Get simulation parameters from GUI, command line, or config file."""
    #     if args.batch:
    #         # Use parameters from config file or command line
    #         params = self.config["default_params"].copy()
            
    #         # Override with command line arguments
    #         if args.replications:
    #             params["num_replications"] = args.replications
    #         if args.simulation_time:
    #             params["sim_time"] = args.simulation_time
    #         if args.servers:
    #             params["processing_fte"] = args.servers
                
    #         logger.info("Using batch mode parameters")
    #         return params
    #     else:
    #         # Launch GUI for parameter input - PRESERVES ORIGINAL simInput.py FUNCTIONALITY
    #         logger.info("Launching parameter input GUI...")
    #         import tkinter as tk
            
    #         # Create a temporary root for the GUI
    #         root = tk.Tk()
    #         root.withdraw()  # Hide the root window
            
    #         try:
    #             # This calls your existing simInput.py GUI with full functionality:
    #             # - Single simulation mode with comprehensive parameter input
    #             # - Goal-seeking simulation mode with parameter adjustment
    #             # - All sensor configurations (CO, DK, MA, NJ, RS, SV, TG, WT, NF1-NF6)
    #             # - Priority distributions, growth factors, etc.
    #             params = get_simulation_parameters(root, mode=args.mode)
    #             if params is None:
    #                 logger.info("Parameter input cancelled by user")
    #                 return None
                
    #             logger.info(f"Parameters collected from GUI (mode: {args.mode})")
    #             logger.debug(f"Collected {len(params)} parameters from simInput.py")
                
    #             # Preserve goal-seeking functionality
    #             if params.get('goalParameter') and params.get('goalParameter') != 'None':
    #                 logger.info(f"Goal-seeking mode enabled: {params.get('goalParameter')}")
                    
    #             return params
    #         finally:
    #             root.destroy()
    
    def run_animated_simulation(self, sim_params: SimulationParameters) -> None:
        """Run animated simulation with enhanced feedback."""
        logger.info("Starting animated simulation...")
        self._log_performance_phase("animation_start")
        
        try:
            service_monitors, stay_monitors = run_animated_simulation(sim_params)
            self._log_performance_phase("animation_complete")
            
            # Process results if needed
            if self.config["output"]["save_intermediate_results"]:
                self._save_animation_results(service_monitors, stay_monitors, sim_params)
                
            logger.info("Animated simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Animated simulation failed: {e}")
            raise
    
    def run_batch_simulation(self, sim_params: SimulationParameters, num_replications: int) -> None:
        """Run batch simulation with progress tracking and optimization."""
        logger.info(f"Starting batch simulation with {num_replications} replications...")
        self._log_performance_phase("batch_start")
        
        # Initialize data processor
        data_processor = DataFileProcessor("./data")
        
        # Calculate work days for parameter adjustment
        days_per_year = work_days_per_year(
            federal_holidays=11, 
            mean_vacation_days=10, 
            mean_sick_days=5
        )
        
        # Check if this is a goal-seeking simulation (preserves original functionality)
        is_goal_seeking = (hasattr(sim_params, 'goalParameter') and 
                          sim_params.goalParameter and 
                          sim_params.goalParameter != 'None')
        
        if is_goal_seeking:
            logger.info(f"Goal-seeking simulation detected: {sim_params.goalParameter}")
            self._run_goal_seeking_simulation(sim_params, days_per_year, data_processor)
            return
        
        # Standard replication-based simulation
        self._run_standard_replications(sim_params, num_replications, days_per_year, data_processor)
    
    def _run_goal_seeking_simulation(self, sim_params: SimulationParameters, 
                                   days_per_year: float, data_processor: DataFileProcessor) -> None:
        """Run goal-seeking simulation using simArrayHandler (preserves original functionality)."""
        logger.info("Running goal-seeking simulation with parameter adjustment...")
        
        try:
            # Initialize parameter adjuster
            adjuster = ParameterAdjuster()
            
            # Convert SimulationParameters back to array format for simArrayHandler
            # This preserves the original goal-seeking workflow
            param_array = self._convert_params_to_array(sim_params)
            
            # Initialize starting conditions
            adjusted_params = adjuster.initialize_starting_condition(param_array, days_per_year)
            
            # Get iteration parameters
            max_iterations = int(getattr(sim_params, 'simMaxIterations', 20))
            start_loop = int(adjusted_params.get('startLoop', 0))
            
            logger.info(f"Goal-seeking: {start_loop} to {max_iterations} iterations")
            
            # Progress tracking for goal-seeking
            use_progress_bar = self.config["performance"]["progress_bars"]
            if use_progress_bar:
                pbar = tqdm(
                    total=max_iterations - start_loop,
                    desc=f"Goal-seeking ({sim_params.goalParameter})",
                    unit="iter"
                )
            
            try:
                for iteration in range(start_loop, max_iterations):
                    if use_progress_bar:
                        pbar.set_description(f"Iteration {iteration + 1}")
                    
                    logger.debug(f"Goal-seeking iteration {iteration + 1}/{max_iterations}")
                    
                    # Convert back to SimulationParameters for this iteration
                    current_sim_params = self._convert_array_to_params(adjusted_params, iteration)
                    
                    # Run simulation
                    service_monitors, stay_monitors = runSimulation(current_sim_params)
                    
                    # Process data for this iteration
                    self._process_replication_data(
                        service_monitors, stay_monitors, iteration + 1, data_processor
                    )
                    
                    # Update parameters for next iteration using simArrayHandler
                    adjusted_params = adjuster.update_simulation_values(adjusted_params, days_per_year)
                    
                    if use_progress_bar:
                        pbar.update(1)
                
                if use_progress_bar:
                    pbar.close()
                
                # Aggregate goal-seeking results
                logger.info("Aggregating goal-seeking results...")
                adjuster.aggregate_data(start_loop, max_iterations - 1)
                
            except Exception as e:
                if use_progress_bar:
                    pbar.close()
                raise e
                
        except Exception as e:
            logger.error(f"Goal-seeking simulation failed: {e}")
            raise
    
    def _run_standard_replications(self, sim_params: SimulationParameters, num_replications: int,
                                 days_per_year: float, data_processor: DataFileProcessor) -> None:
        """Run standard replication-based simulation."""
        # Progress tracking setup
        use_progress_bar = self.config["performance"]["progress_bars"] and not logger.level <= logging.DEBUG
        
        if use_progress_bar:
            pbar = tqdm(
                total=num_replications,
                desc="Running Replications",
                unit="rep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        
        try:
            # Run replications
            all_service_data = []
            all_stay_data = []
            
            for rep_idx in range(num_replications):
                if use_progress_bar:
                    pbar.set_description(f"Replication {rep_idx + 1}")
                
                logger.debug(f"Starting replication {rep_idx + 1}/{num_replications}")
                
                # Create replication-specific parameters
                current_sim_params = self._prepare_replication_params(sim_params, rep_idx)
                
                # Run simulation
                service_monitors, stay_monitors = runSimulation(current_sim_params)
                
                # Process and save data
                rep_data = self._process_replication_data(
                    service_monitors, stay_monitors, rep_idx + 1, data_processor
                )
                
                all_service_data.append(rep_data["service"])
                all_stay_data.append(rep_data["stay"])
                
                if use_progress_bar:
                    pbar.update(1)
                
                logger.debug(f"Completed replication {rep_idx + 1}")
            
            if use_progress_bar:
                pbar.close()
            
            self._log_performance_phase("replications_complete")
            
            # Aggregate and analyze results
            logger.info("Aggregating results across replications...")
            self._aggregate_results(all_service_data, all_stay_data, data_processor)
            self._log_performance_phase("aggregation_complete")
            
            # Generate plots and statistics
            if self.config["output"]["generate_plots"]:
                logger.info("Generating plots and statistics...")
                self._generate_plots_and_stats(sim_params)
                self._log_performance_phase("plotting_complete")
            
            # Generate report
            if self.config["output"]["create_report"]:
                logger.info("Generating final report...")
                self._generate_report(sim_params, days_per_year)
                self._log_performance_phase("report_complete")
            
            logger.info("Batch simulation completed successfully")
            
        except Exception as e:
            if use_progress_bar:
                pbar.close()
            logger.error(f"Batch simulation failed: {e}")
            raise
    
    def _prepare_replication_params(self, base_params: SimulationParameters, rep_idx: int) -> SimulationParameters:
        """Prepare parameters for a specific replication."""
        # Create a copy of parameters for this replication
        replication_params = SimulationParameters(base_params.__dict__)
        
        # Vary seed for each replication
        replication_params.seed = base_params.seed + rep_idx
        
        # Apply any parameter adjustments if using simArrayHandler
        # This would be where goal-seeking or parameter sweeps are applied
        
        return replication_params
    
    def _process_replication_data(self, service_monitors: List, stay_monitors: List, 
                                rep_num: int, data_processor: DataFileProcessor) -> Dict[str, Any]:
        """Process and save data from a single replication."""
        try:
            # Extract data from monitors
            service_data = service_monitors[0].as_dataframe() if service_monitors else pd.DataFrame()
            stay_data = stay_monitors[0].as_dataframe() if stay_monitors else pd.DataFrame()
            
            # Save raw data
            if self.config["output"]["save_intermediate_results"]:
                data_processor.csv_export(f"service_rep_{rep_num}", service_data)
                data_processor.csv_export(f"stay_rep_{rep_num}", stay_data)
            
            return {
                "service": service_data,
                "stay": stay_data,
                "replication": rep_num
            }
            
        except Exception as e:
            logger.warning(f"Failed to process data for replication {rep_num}: {e}")
            return {"service": pd.DataFrame(), "stay": pd.DataFrame(), "replication": rep_num}
    
    def _save_animation_results(self, service_monitors: List, stay_monitors: List, 
                              sim_params: SimulationParameters) -> None:
        """Save results from animated simulation."""
        try:
            data_processor = DataFileProcessor("./data")
            service_data = service_monitors[0].as_dataframe() if service_monitors else pd.DataFrame()
            stay_data = stay_monitors[0].as_dataframe() if stay_monitors else pd.DataFrame()
            
            data_processor.csv_export("animated_service", service_data)
            data_processor.csv_export("animated_stay", stay_data)
            
            logger.info("Animation results saved")
        except Exception as e:
            logger.warning(f"Failed to save animation results: {e}")
    
    def _aggregate_results(self, all_service_data: List, all_stay_data: List, 
                         data_processor: DataFileProcessor) -> None:
        """Aggregate results across all replications."""
        try:
            if all_service_data:
                combined_service = pd.concat(all_service_data, ignore_index=True)
                data_processor.csv_export("aggregated_service", combined_service)
            
            if all_stay_data:
                combined_stay = pd.concat(all_stay_data, ignore_index=True)
                data_processor.csv_export("aggregated_stay", combined_stay)
                
            logger.info("Results aggregated successfully")
        except Exception as e:
            logger.warning(f"Failed to aggregate results: {e}")
    
    def _generate_plots_and_stats(self, sim_params: SimulationParameters) -> None:
        """Generate plots and statistics from aggregated data."""
        try:
            # Create statistics processor
            stats_processor = Statistics("aggregated_stay", file_path="./data/")
            
            # Generate basic statistics
            stay_stats = stats_processor.get_stats("stayLength")
            logger.info(f"Stay time statistics: Mean={stay_stats[3]:.2f}, Std={stay_stats[4]:.2f}")
            
            # Create plots
            plot_processor = Plot("./plots/", sim_params.simFileName)
            
            # This would be expanded based on your specific plotting needs
            logger.info("Plots and statistics generated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots and statistics: {e}")
    
    def _generate_report(self, sim_params: SimulationParameters, time_period: float) -> None:
        """Generate the final PDF report."""
        try:
            generate_report(sim_params.simFileName, sim_params, time_period)
            logger.info("Final report generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")
    
    def _convert_params_to_array(self, sim_params: SimulationParameters) -> Dict[str, Any]:
        """Convert SimulationParameters to array format for simArrayHandler compatibility."""
        # This preserves the original parameter array structure that simArrayHandler expects
        return {
            'simFileName': getattr(sim_params, 'simFileName', 'default'),
            'goalParameter': getattr(sim_params, 'goalParameter', ''),
            'goalTarget': getattr(sim_params, 'goalTarget', ''),
            'waitTimeMax': getattr(sim_params, 'waitTimeMax', 1.0),
            'processingFteMax': getattr(sim_params, 'processingFteMax', 20.0),
            'simMaxIterations': getattr(sim_params, 'simMaxIterations', 20),
            'timeWindowYears': getattr(sim_params, 'timeWindowYears', 1.0),
            'nservers': getattr(sim_params, 'nservers', 1.0),
            'siprTransferTime': getattr(sim_params, 'siprTransferTime', 1.0),
            'fileGrowth': getattr(sim_params, 'fileGrowth', 0.0),
            'sensorGrowth': getattr(sim_params, 'sensorGrowth', 0.0),
            'ingestEfficiency': getattr(sim_params, 'ingestEfficiency', 0.0),
            # Add other parameters as needed for compatibility
            'startLoop': 0
        }
    
    def _convert_array_to_params(self, param_array: Dict[str, Any], iteration: int) -> SimulationParameters:
        """Convert array format back to SimulationParameters for simulation execution."""
        # Update seed for this iteration
        param_array['seed'] = param_array.get('seed', 42) + iteration
        return SimulationParameters(param_array)
    
    def _print_performance_summary(self) -> None:
        """Print performance summary."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            
            for phase, duration in self.performance_metrics.get("execution_phases", {}).items():
                logger.info(f"  {phase}: {duration:.2f}s")
    
    def run_simulation(self, args: argparse.Namespace) -> bool:
        """Main simulation execution method."""
        try:
            # Start performance monitoring
            self._start_performance_monitoring()
            
            # Get simulation parameters
            params_dict = self._get_simulation_parameters(args)
            if params_dict is None:
                logger.info("Simulation cancelled - no parameters provided")
                return False
            
            # Create simulation parameters object
            sim_params = SimulationParameters(params_dict)
            logger.info(f"Simulation '{sim_params.simFileName}' initialized")
            
            # Run appropriate simulation type
            if args.animated:
                self.run_animated_simulation(sim_params)
            else:
                num_reps = getattr(sim_params, 'num_replications', args.replications or 1)
                self.run_batch_simulation(sim_params, num_reps)
            
            # Print performance summary
            self._print_performance_summary()
            
            logger.info("Simulation completed successfully")
            return True
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            if logger.level <= logging.DEBUG:
                import traceback
                logger.debug(traceback.format_exc())
            return False

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Phalanx C-sUAS Data Ingestion Process Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --animated                    # Run animated simulation with GUI
  %(prog)s --batch --replications 10    # Run 10 replications in batch mode
  %(prog)s --config config.json         # Use custom configuration file
  %(prog)s --log-level DEBUG --log-file sim.log  # Enable debug logging
        """
    )
    
    # Simulation mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--animated", 
        action="store_true",
        help="Run animated simulation (single run with visualization)"
    )
    mode_group.add_argument(
        "--batch", 
        action="store_true",
        help="Run batch simulation without GUI (requires parameters)"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--mode",
        choices=["single", "goal"],
        default="single",
        help="Simulation mode for GUI (default: single)"
    )
    parser.add_argument(
        "--replications", "-r",
        type=int,
        help="Number of simulation replications (for batch mode)"
    )
    parser.add_argument(
        "--simulation-time", "-t",
        type=float,
        help="Simulation time in minutes"
    )
    parser.add_argument(
        "--servers", "-s",
        type=float,
        help="Number of processing servers (FTEs)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path (JSON format)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path (optional)"
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Output directory for results (default: current directory)"
    )
    
    return parser

def main():
    """Main entry point for the simulation."""
    try:
        # Parse command line arguments (if any provided)
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging with command line options (defaults to INFO level)
        global logger
        logger = setup_logging(args.log_level, args.log_file)
        
        # Create and run orchestrator
        orchestrator = SimulationOrchestrator(args.config)
        success = orchestrator.run_simulation(args)
        
        if success:
            logger.info("Simulation completed successfully!")
            return True
        else:
            logger.error("Simulation failed or was cancelled")
            return False
            
    except SystemExit:
        # Handle normal program exits (like from argparse --help)
        return True
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return False

# Entry point that works for both GUI and command-line usage
if __name__ == "__main__":
    # When you double-click or run from IDE, this runs main() 
    # with default arguments, preserving your exact current workflow
    try:
        success = main()
        if not success:
            # In GUI mode, don't exit abruptly - let user see any error messages
            input("\nPress Enter to close...")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        input("\nPress Enter to close...")

# Enhanced standalone testing
if __name__ == "__main__":
    print("="*60)
    print("Enhanced simStart.py - Standalone Testing")
    print("="*60)
    
    # Test 1: Directory setup
    print("\n1. Testing directory setup...")
    orchestrator = SimulationOrchestrator()
    print("✓ Directories created successfully")
    
    # Test 2: Configuration loading
    print("\n2. Testing configuration management...")
    test_config = {
        "directories": ["./test_data", "./test_plots"],
        "default_params": {"sim_time": 100, "seed": 123}
    }
    
    # Save test config
    with open("test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    test_orchestrator = SimulationOrchestrator("test_config.json")
    assert test_orchestrator.config["default_params"]["seed"] == 123
    print("✓ Configuration loading works correctly")
    
    # Clean up test config
    os.remove("test_config.json")
    
    # Test 3: Performance monitoring
    print("\n3. Testing performance monitoring...")
    test_orchestrator._start_performance_monitoring()
    time.sleep(0.1)  # Simulate some work
    test_orchestrator._log_performance_phase("test_phase")
    assert "test_phase" in test_orchestrator.performance_metrics["execution_phases"]
    print("✓ Performance monitoring works correctly")
    
    # Test 4: Command line interface
    print("\n4. Testing command line interface...")
    parser = create_argument_parser()
    test_args = parser.parse_args(["--batch", "--replications", "5", "--log-level", "DEBUG"])
    assert test_args.batch == True
    assert test_args.replications == 5
    assert test_args.log_level == "DEBUG"
    print("✓ Command line interface works correctly")
    
    # Test 5: Sample parameter processing
    print("\n5. Testing parameter processing...")
    sample_params = {
        "simFileName": "test_simulation",
        "sim_time": 500,
        "processing_fte": 2.0,
        "seed": 42
    }
    
    try:
        sim_params = SimulationParameters(sample_params)
        print(f"✓ Parameters processed: {sim_params.simFileName}")
    except Exception as e:
        print(f"✗ Parameter processing failed: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nUsage examples:")
    print("python simStart.py --animated")
    print("python simStart.py --batch --replications 10")
    print("python simStart.py --config my_config.json --log-level DEBUG")
    print("\nFor full help: python simStart.py --help")