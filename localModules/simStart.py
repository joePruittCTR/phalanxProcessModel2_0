# simStart.py

import os
import pandas as pd # Used for data processing and saving
import numpy as np  # Used for numerical operations

# Import core simulation logic and parameters
from simProcess import SimulationParameters, runSimulation
# Import animation logic
from simAnimate import run_animated_simulation
# Import parameter adjustment logic
from simArrayHandler import ParameterAdjuster
# Import data formatting and processing
from simDataFormat import DataFileProcessor
# Import plotting utilities
from simPlot import Plot
# Import statistics calculation
from simStats import Statistics
# Import general utilities (e.g., work_days_per_year, get_current_date)
from simUtils import work_days_per_year, get_current_date
# Import report generation
from simReport import getFileName as generate_report # Alias to avoid name conflict

def setup_directories():
    """Ensures all necessary output directories exist."""
    dirs = ["./data", "./plots", "./reports", "./reportResources"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def main_simulation_run(run_animated: bool = False, num_replications: int = 1):
    """
    Orchestrates the main simulation process.
    
    Args:
        run_animated (bool): True to run the animated simulation, False for batch.
        num_replications (int): Number of times to run the simulation (for batch runs).
    """
    setup_directories()

    print("\n--- Initializing Simulation Parameters ---")
    # This dictionary would typically come from 02simInput.py
    # For now, let's use a comprehensive example.
    initial_params_dict = {
        'sim_time': 500, # Simulation time in minutes
        'time_unit': 'minutes',
        'processing_fte': 1.0,
        'processing_overhead': 0.10,
        'processing_efficiency': 0.90,
        'warmup_time': 30,
        'num_replications': num_replications,
        'seed': 42,
        
        # File type parameters (example for CO, DK, NF1, WT)
        'co_time': 8.0,
        'co_iat': 1.2, 
        'co_distribution_type': 'Exponential',

        'dk_time': 15.0,
        'dk_iat': 3.0,
        'dk_distribution_type': 'MonthlyMixedDist',
        'dk_distribution_kwargs': {'num_days': 20, 'first_peak_probability': 0.7},

        'nf1_time': 20.0,
        'nf1_iat': 0.0, # Initially inactive sensor
        'nf1_distribution_type': 'Exponential',

        'wt_time': 5.0,
        'wt_iat': 5.0,
        'wt_batch_size': 5,
        'wt_distribution_type': 'WeeklyExponential',
        
        # Growth and goal parameters (for simArrayHandler)
        'file_growth_slope': 0.0, # Will be set by parameter sweep or growth scenario
        'sensor_growth_slope': 0.0,
        'ingest_efficiency_slope': 0.0,
        'goal_parameter': 'None' # Default, would be set for parameter sweeps
    }

    # Create the initial SimulationParameters object
    base_sim_params = SimulationParameters(initial_params_dict)
    
    # Initialize data processor
    data_processor = DataFileProcessor("./data")
    
    # Calculate work days per year for simArrayHandler
    # This might be further parameterized if holidays/vacation are input
    days_per_year = work_days_per_year(federal_holidays=11, mean_vacation_days=10, mean_sick_days=5)
    print(f"Calculated work days per year: {days_per_year:.2f}")

    if run_animated:
        print("\n--- Running Animated Simulation ---")
        # For animation, typically only 1 replication makes sense
        # And usually no parameter adjustment loops.
        _service_monitors, _stay_monitors = run_animated_simulation(base_sim_params)
        print("Animated simulation finished.")
        # You might want to save data or generate a report for the animated run too
        # For simplicity, we'll skip detailed data processing for animated run in this main()
    else:
        print("\n--- Running Batch Simulation ---")
        # Example of a simple parameter sweep or multiple replications
        
        # Example: Varying FTEs from 1.0 to 3.0
        # This is where simArrayHandler would be used to adjust params for each iteration
        
        # Scenario: Run 3 replications with base parameters
        for rep_idx in range(num_replications):
            print(f"\n--- Replication {rep_idx + 1}/{num_replications} ---")
            
            # Create a mutable copy of sim_params for this replication
            # This is important if simArrayHandler modifies it in place
            current_sim_params = SimulationParameters(initial_params_dict) 
            current_sim_params.seed = base_sim_params.seed + rep_idx # Vary seed for replications

            # --- Parameter Adjustment (Example using simArrayHandler) ---
            # If you were doing a parameter sweep (e.g., varying FTEs):
            # adjuster = ParameterAdjuster()
            # current_sim_params.goal_parameter = "Ingestion FTE"
            # current_sim_params = adjuster.initialize_loop_parameter(current_sim_params, days_per_year)
            # current_sim_params.processing_fte = 1.0 + (rep_idx * 0.5) # Example incrementing FTE
            # current_sim_params = adjuster.apply_growth_factors(current_sim_params, days_per_year) # Apply growth if needed
            
            # --- Run the Simulation ---
            service_monitors, stay_monitors = runSimulation(current_sim_params)

            # --- Data Processing and Storage ---
            print("Processing simulation results...")
            
            # Extract data from monitors
            # For simplicity, just taking the first monitor from each list
            service_time_data = service_monitors[0].as_dataframe()
            stay_time_data = stay_monitors[0].as_dataframe()

            # Save raw data to CSV
            # Salabim monitors might return data in different formats.
            # Assuming .as_dataframe() gives a suitable structure.
            data_processor.csv_export(f"raw_service_rep_{rep_idx + 1}", service_time_data)
            data_processor.csv_export(f"raw_stay_rep_{rep_idx + 1}", stay_time_data)

            # Format data (as per simDataFormat.py's expectations)
            # Note: The `format_data` method in simDataFormat.py expects specific column names and structures.
            # You'll need to ensure the data from Salabim monitors is transformed into that format
            # before calling format_data if it's not directly compatible.
            # For simplicity here, I'll use dummy file names that might correspond to what format_data expects.
            # The actual integration would involve mapping Salabim monitor outputs to DataFileProcessor inputs.
            
            # Example: If you wanted to format system-wide stay data
            # You'd need to ensure 'SYS_Stay_rep_X.csv' has the right internal structure
            # data_processor.format_data(f"SYS_Stay_rep_{rep_idx + 1}", rep_idx + 1, current_sim_params.processingFte, "stay")
            # data_processor.format_data(f"SYS_Files_rep_{rep_idx + 1}", rep_idx + 1, current_sim_params.processingFte, "file")

            # --- Statistics and Plotting (Example) ---
            # You would typically aggregate data across replications before final stats/plots
            # or calculate stats per replication.
            # stats_processor = Statistics(f"SYS_Stay_rep_{rep_idx + 1}", file_path="./data/")
            # mean_stay = stats_processor.get_mean_stay()
            # print(f"Mean stay time for replication {rep_idx + 1}: {mean_stay:.2f} minutes")

        print("\n--- Aggregating Data (Example) ---")
        # After all replications, aggregate data if needed
        # For example, if you saved "SYS_Stay_rep_X.csv" and "SYS_Files_rep_X.csv"
        # data_processor.aggregate_files("SYS_Stay_rep_", 1, num_replications)
        # data_processor.aggregate_files("SYS_Files_rep_", 1, num_replications)
        # data_processor.create_master_file(["SYS_Stay_rep_Agg", "SYS_Files_rep_Agg"])
        
        print("\n--- Generating Reports ---")
        # The `generate_report` function (from _simReport) expects aggregated data to be ready
        # in the /data and /plots directories.
        # It also needs the final sim_params object and the total time period for analysis.
        # Ensure that plots (e.g., SYS_Length_of_Stay_Stair.png) are generated by _simPlot/Statistics
        # before calling generate_report.
        
        # For testing, ensure your dummy plots and data files exist in /plots and /data
        # as set up in the simReport.py __main__ block example.
        report_file_name = "Simulation_Report_" + get_current_date()
        generate_report(report_file_name, base_sim_params, base_sim_params.simTime)
        print(f"Report generation attempted for {report_file_name}.pdf")


if __name__ == '__main__':
    print("Starting Simulation Project Orchestration (simStart.py)")

    # Example Usage:
    # 1. Run a simple batch simulation with 3 replications
    # main_simulation_run(run_animated=False, num_replications=3)

    # 2. Run an animated simulation (usually 1 replication)
    main_simulation_run(run_animated=True, num_replications=1) 
    
    # Note: For animated runs, Salabim typically opens a GUI window.
    # The script will pause until you close the animation window.
    # For batch runs, it will proceed to data processing and reporting.
