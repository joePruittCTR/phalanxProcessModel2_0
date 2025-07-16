import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Corrected import path for Plot class
from simPlot import Plot 
import os

class Statistics:
    def __init__(self, file_name, file_path="./data/"):
        self.file_name = file_name
        self.file_path = file_path

    def get_stats(self, column_name):
        """
        Reads a CSV, drops specific rows, and calculates basic statistics for a given column.
        Assumes the CSV has header/footer rows to be dropped.
        """
        df = pd.read_csv(self.file_path + self.file_name + ".csv")
        
        # Robust dropping of potential header/footer rows
        if len(df) > 2: # Requires at least 3 rows to drop row 0 and the last row
            df.drop(axis=0, index=[0, df.index.max()], inplace=True)
        elif len(df) == 2: # Only drop row 0 if only 2 rows (assuming 1 header + 1 data)
            df.drop(axis=0, index=[0], inplace=True)
        # If len(df) <= 1, no drops occur, which is fine.

        # Convert column to numeric, coercing errors to NaN and then dropping NaNs
        s = pd.to_numeric(df[column_name], errors='coerce').dropna()
        
        if s.empty:
            print(f"Warning: Column '{column_name}' in '{self.file_name}.csv' is empty or contains no numeric data after cleaning.")
            return [np.nan, np.nan, np.nan, np.nan, np.nan] # Return NaNs if no data
            
        min_val = s.min()
        max_val = s.max()
        median_val = s.median()
        mean_val = s.mean()
        std_dev_val = s.std()
        return [min_val, max_val, median_val, mean_val, std_dev_val]

    def get_file_stats(self, time_window, plot_color):
        """
        Calculates file statistics over monthly intervals within a given time window.
        Assumes the CSV contains 'timeStep' and 'fileNum' columns.
        This function is likely intended for data formatted with simDataFormat.py's
        data_type="stay" which includes 'fileNum'.
        """
        df = pd.read_csv(self.file_path + self.file_name + ".csv")
        # Robust dropping
        if len(df) > 2:
            df.drop(axis=0, index=[0, df.index.max()], inplace=True)
        elif len(df) == 2:
            df.drop(axis=0, index=[0], inplace=True)
        
        # Ensure 'timeStep' and 'fileNum' are numeric
        df["timeStep"] = pd.to_numeric(df["timeStep"], errors='coerce')
        df["fileNum"] = pd.to_numeric(df["fileNum"], errors='coerce')
        df.dropna(subset=["timeStep", "fileNum"], inplace=True) # Drop rows where these are NaN

        if df.empty:
            print(f"Warning: No valid 'timeStep' or 'fileNum' data in '{self.file_name}.csv' for file stats.")
            return [0, np.nan, np.nan, np.nan, np.nan, np.nan] # Return zeros/NaNs if no data

        total_files = len(df["fileNum"])
        
        # This `month_value` (20.75) needs to be consistent with how "months" are defined in the simulation
        # (e.g., 20.75 working days per month * 8 hours/day * 60 minutes/hour = total minutes per month).
        # This is a magic number that might need to be parameterized or derived from sim_params.
        month_value = 20.75 
        
        month_files = []
        
        # Ensure time_window is large enough and not zero to avoid division by zero or infinite loops
        # Also check if the time_window can actually produce any intervals
        if time_window <= 0 or (df["timeStep"].max() < month_value and df["timeStep"].max() > 0): # Added check for max timeStep > 0
            print(f"Warning: Invalid time_window ({time_window}) or insufficient data for monthly analysis. No analysis performed.")
            return [total_files, np.nan, np.nan, np.nan, np.nan, np.nan]

        stat_months = np.ceil(time_window / month_value) # Number of intervals to analyze
        
        old_month_index = df.index.min() if not df.empty else 0 # Start from the first actual data row's index
        
        for i in range(int(stat_months)):
            month_boundary_time = month_value * (i + 1)
            
            # Find the index of the first row where timeStep is greater than or equal to month_boundary_time
            # within the remaining dataframe
            current_month_data = df[df["timeStep"] >= month_boundary_time]
            current_month_start_idx = current_month_data.index.min()
            
            if pd.isna(current_month_start_idx): # No more data points for this or future intervals
                # Count files from old_month_index to the end of the dataframe
                temp_val = df.loc[old_month_index:]["fileNum"].count()
                if temp_val > 0: # Only add if there are files in this partial interval
                    month_files.append(temp_val)
                break # No more intervals to process
            
            # Count files in the current interval (from old_month_index up to, but not including, current_month_start_idx)
            # Use .loc with explicit index range
            current_interval_df = df.loc[old_month_index : current_month_start_idx -1]
            temp_val = current_interval_df["fileNum"].count()
            
            month_files.append(temp_val)
            old_month_index = current_month_start_idx # Update start index for next interval

        # Convert to Series for robust stats calculation, handling cases where month_files might be empty
        s_month_files = pd.Series(month_files)
        s_month_files = s_month_files[s_month_files > 0] # Only consider intervals with actual files

        if s_month_files.empty:
            min_file, max_file, median_file, mean_file, std_dev_file = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            min_file = s_month_files.min()
            max_file = s_month_files.max()
            median_file = s_month_files.median()
            mean_file = s_month_files.mean()
            std_dev_file = s_month_files.std()
        
        # Ensure plot directory exists
        plot_dir = "./plots/" # This should ideally be passed in or derived from a central config
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot = Plot(plot_dir, self.file_name)
        if not s_month_files.empty: # Only plot if there's data
            plot.create_plot("box", s_month_files.tolist(), "", "Files per Month", self.file_name + "_Box", plot_color)        
        
        return [total_files, min_file, max_file, median_file, mean_file, std_dev_file]

    def get_mean_stay(self):
        """
        Calculates the mean 'stayLength' from the CSV.
        Assumes 'stayLength' column is present after dropping specific rows.
        """
        df = pd.read_csv(self.file_path + self.file_name + ".csv")
        # Robust dropping
        if len(df) > 2:
            df.drop(axis=0, index=[0, df.index.max()], inplace=True)
        elif len(df) == 2:
            df.drop(axis=0, index=[0], inplace=True)
            
        s = pd.to_numeric(df["stayLength"], errors='coerce').dropna()
        
        if s.empty:
            print(f"Warning: Column 'stayLength' in '{self.file_name}.csv' is empty or contains no numeric data after cleaning.")
            return np.nan
            
        mean_stay = s.mean()
        return mean_stay

# --- Helper functions for simReport.py ---
def createFilesStats(file_name, time_window, plot_color, file_path="./data/"):
    """Wrapper for Statistics.get_file_stats."""
    stats_processor = Statistics(file_name, file_path=file_path)
    return stats_processor.get_file_stats(time_window, plot_color)

def createQueueStats(file_name, file_path="./data/"):
    """Wrapper for Statistics.get_stats for queueLength."""
    stats_processor = Statistics(file_name, file_path=file_path)
    return stats_processor.get_stats("queueLength")

def createStayStats(file_name, file_path="./data/"):
    """Wrapper for Statistics.get_stats for stayLength."""
    stats_processor = Statistics(file_name, file_path=file_path)
    return stats_processor.get_stats("stayLength")
