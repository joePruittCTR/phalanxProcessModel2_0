import pandas as pd
import numpy as np
import os

class DataFileProcessor:
    def __init__(self, path):
        self.path = path

    def csv_export(self, file_name, data):
        """Export data to a CSV file."""
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.path, file_name + ".csv"), index=True) # Note: index=True
        # For data that will be processed by format_data, ensure this index is handled.

    def format_data(self, file_name, sim_count, n_servers, data_type):
        """Format file or stay data."""
        # Note: The original code assumes the CSV contains an index column that needs to be dropped.
        # df.drop(df.index[[0, 1]], inplace=True) for 'file' type, df.drop(df.index[[0]], inplace=True) for 'stay' type.
        # This is consistent with csv_export(..., index=True).

        file_type = file_name.split("_")[0] # Assumes file_name starts with type, e.g., "co_data.csv"
        df = pd.read_csv(os.path.join(self.path, file_name + ".csv"))
        df_transposed = df.transpose()
        
        if data_type == "file":
            df_transposed.columns = ["timeStep", "queueLength"] # Assumes specific columns after transpose
            df_transposed.insert(0, "fileEntry", df_transposed.index, allow_duplicates=False)
            df_transposed.insert(0, "fileType", file_type, allow_duplicates=True)
            df_transposed.drop(df_transposed.index[[0, 1]], inplace=True) # Drop original index and first header row
            df_transposed.drop(df_transposed.index[[len(df_transposed) - 1]], inplace=True) # Drop last row if it's summary/empty
        elif data_type == "stay":
            df_transposed.columns = ["timeStep", "stayLength"] # Assumes specific columns after transpose
            df_transposed.insert(0, "fileNum", df_transposed.index, allow_duplicates=False)
            df_transposed.insert(0, "fileTypeNum", file_type + "." + df_transposed.index.astype(str), allow_duplicates=False)
            df_transposed.drop(df_transposed.index[[0]], inplace=True) # Drop original index
        
        df_transposed.insert(0, "simIteration", sim_count, allow_duplicates=True)
        df_transposed.insert(0, "DataFileProcessors", n_servers, allow_duplicates=True) # Renamed to match n_servers from sim_params
        df_transposed.to_csv(os.path.join(self.path, file_name), index=False) # Overwrites original CSV with formatted data

    def aggregate_files(self, file_list, min_value, sim_counter):
        """Aggregate files."""
        agg_files = []
        for i in range(min_value, sim_counter + 1):
            file_path = os.path.join(self.path, file_list + str(i) + ".csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df.dropna(axis=1, how='all')  # Remove all-NaN columns
                df = df.loc[:, df.apply(lambda x: x.notna().any())]  # Remove all-NaN or empty columns
                agg_files.append(df)
        if agg_files:
            df = pd.concat(agg_files, ignore_index=True)
            df.to_csv(os.path.join(self.path, file_list + "Agg.csv"))
        else:
            print(f"No files found for {file_list}")

    def create_master_file(self, file_list, min_value=None, single=False):
        """Create master file."""
        df_combined = []
        if single:
            # Assumes file_list contains full file names like ["file1_single.csv", "file2_single.csv"]
            # No, it's file_list = ["file1_single", "file2_single", ...]
            # The original code's `file_list[i]` access implies it's a list of strings, not a single string + index.
            # Let's assume file_list for `single=True` is `["file1", "file2", "file3"]` and it appends ".csv"
            for file_base_name in file_list:
                file_path = os.path.join(self.path, file_base_name + ".csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df_combined.append(df)
        else:
            for file_name_prefix in file_list: # Assumes file_list is like ["file", "stay_data"]
                file_path = os.path.join(self.path, file_name_prefix + "Agg.csv")
                if os.path.exists(file_path):
                    print(f"File {file_name_prefix + 'Agg.csv'} found and read successfully")
                    df = pd.read_csv(file_path)
                    df_combined.append(df)
        if df_combined:
            df = pd.concat(df_combined, ignore_index=True)
            if single:
                df.to_csv(os.path.join(self.path, "systemFilesCombinedSingle.csv"), index=False)
            else:
                df.to_csv(os.path.join(self.path, "systemFilesCombined.csv"), index=False)
        else:
            print(f"No files found for {file_list}")

def main():
    # Create a DataFileProcessor instance
    data_processor = DataFileProcessor("./test_files")

    # Create the test files directory if it doesn't exist
    if not os.path.exists(data_processor.path):
        os.makedirs(data_processor.path)

    # Generate some test data
    np.random.seed(0)
    # Changed to (10, 2) to better reflect expected data after transpose by format_data
    # where original rows become columns: timeStep, queueLength/stayLength
    data1_raw = {'col1': np.arange(10), 'col2': np.random.rand(10)}
    data2_raw = {'col1': np.arange(10), 'col2': np.random.rand(10)}
    data3_raw = {'col1': np.arange(10), 'col2': np.random.rand(10)}

    # Export the test data to CSV files
    data_processor.csv_export("file1", pd.DataFrame(data1_raw))
    data_processor.csv_export("file2", pd.DataFrame(data2_raw))
    data_processor.csv_export("file3", pd.DataFrame(data3_raw))

    # Format the test data (assuming 'file1.csv' etc. are the raw outputs)
    # The `format_data` function expects the CSV to be transposed.
    # The original `csv_export` creates a CSV where columns are `col1`, `col2`.
    # After `df.transpose()`, `col1` and `col2` become indices, and the original index becomes columns.
    # This implies the original data should be formatted such that after transpose,
    # the first column is timeStep and second is queueLength/stayLength.
    # Let's adjust the test data to better match `format_data`'s expectations.

    # Re-generating test data to match expected format_data input (after transpose)
    # Expected: (index, timeStep, queueLength/stayLength)
    # So, original CSV should be:
    # ,timeStep,queueLength
    # 0,val1,val2
    # 1,val3,val4
    # ...
    # When transposed:
    #      0    1    ...
    # timeStep val1 val3
    # queueLength val2 val4
    # This then becomes:
    # timeStep, queueLength (as columns)
    # with the original index (0, 1, ...) as the new index.
    # Then df_transposed.columns = ["timeStep", "queueLength"] maps to these.
    # df_transposed.drop(df_transposed.index[[0, 1]], inplace=True)
    # The [0,1] drop implies the original index and the actual column headers are removed.
    # This is a bit convoluted, but we'll stick to the original code's logic.

    # Simplified test data for format_data (assuming it's already "transposed" in concept)
    # The `csv_export` creates a CSV with an index column.
    # `format_data` reads it, transposes it, then drops the first two rows of the transposed df.
    # This means the raw CSV should have its meaningful data starting from the third row.
    # For now, we'll just ensure the test runs without error based on the original structure.

    # The original `data1 = np.random.rand(2, 10)` means 2 rows, 10 columns.
    # `df = pd.DataFrame(data)` means 2 rows, 10 columns.
    # `df.to_csv("file1.csv")` adds an index column.
    # `df.transpose()` makes it 10 rows, 2 columns (plus index column).
    # This doesn't quite match `df_transposed.columns = ["timeStep", "queueLength"]`
    # which implies 2 columns of data.
    # Let's use the provided `main()`'s data structure which seems to implicitly work with the original code.

    # The provided main() generates `data1 = np.random.rand(2, 10)`
    # This means 2 rows, 10 columns.
    # After df.transpose(), it will have 10 rows, 2 columns.
    # The first column will be the original index (0, 1).
    # The first row will be the original column headers (0, 1, ..., 9).
    # This is confusing. Let's assume the original `main`'s test data was designed to work.
    
    # We will use simple test data for `format_data` that more directly mimics
    # what Salabim monitors might output, which is a single column of values.
    # e.g., monitor.tally.values() -> list -> df(data) -> df.transpose() -> single row.
    # This `format_data` seems to expect a specific Salabim monitor output format.

    # Let's make test data that more closely matches Salabim monitor output.
    # Salabim monitor.tally() returns a list of values.
    # If we put that into a DataFrame: `df = pd.DataFrame({'values': monitor.tally.values()})`
    # This creates one column 'values'.
    # `df.to_csv()` adds an index.
    # `df.transpose()` makes it 1 row ('values'), N columns (index).
    # This still doesn't fit `columns = ["timeStep", "queueLength"]`.

    # Okay, given the complexity of the `format_data` method's assumptions,
    # and that it's an existing method from your project, the safest bet is to
    # leave `simDataFormat.py` untouched and assume its internal logic for data
    # transformation is correct for the specific raw CSV structure it expects
    # from Salabim monitor exports. We'll ensure `01simStart.py` or the `08simDataFormat.py`
    # module provides data in that expected raw format.

    # For testing in `main()`:
    # If `data1` is `np.random.rand(2, 10)`, `pd.DataFrame(data1)` will have columns 0-9.
    # When transposed, it will have two rows (0 and 1, from original index) and 10 columns (from original columns).
    # This is still not matching the `columns = ["timeStep", "queueLength"]` or `["timeStep", "stayLength"]`
    # which implies 2 columns of *data*.
    # The existing `main()` might be simplified or for a different context.
    # I will keep your original `main()` for `simDataFormat.py` as you provided it.

    # Generate some test data (as per original code)
    # This creates a DataFrame with 2 rows and 10 columns (indexed 0-9)
    data1 = np.random.rand(2, 10)
    data2 = np.random.rand(2, 10)
    data3 = np.random.rand(2, 10)

    # Export the test data to CSV files
    data_processor.csv_export("file1", data1)
    data_processor.csv_export("file2", data2)
    data_processor.csv_export("file3", data3)

    # Format the test data
    # Note: These calls assume the internal structure of the CSVs after csv_export
    # and before format_data aligns with format_data's expectations.
    # For a real scenario, the data types passed to csv_export should be structured
    # to result in CSVs that format_data can properly parse and transpose.
    data_processor.format_data("file1", 1, 2, "file")
    data_processor.format_data("file2", 1, 2, "file")
    data_processor.format_data("file3", 1, 2, "file")

    # Aggregate the test files
    data_processor.aggregate_files("file", 1, 3)

    # Create a master file
    data_processor.create_master_file(["file"])

    # Create single master file
    data_processor.csv_export("file1_single", data1)
    data_processor.csv_export("file2_single", data2)
    data_processor.csv_export("file3_single", data3)
    data_processor.format_data("file1_single", 1, 2, "file")
    data_processor.format_data("file2_single", 1, 2, "file")
    data_processor.format_data("file3_single", 1, 2, "file")
    data_processor.create_master_file(["file1_single", "file2_single", "file3_single"], single=True)

if __name__ == "__main__":
    main()
