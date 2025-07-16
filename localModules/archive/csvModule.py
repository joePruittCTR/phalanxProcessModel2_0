# Global Imports
import pandas as pd
import os

class DataProcessor:
    def __init__(self, path):
        self.path = path

    def csv_export(self, file_name, data):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.path, file_name + ".csv"), index=True)

    def format_file_data(self, file_name, sim_count, n_servers):
        file_type = file_name.split("_")[0]
        df = pd.read_csv(os.path.join(self.path, file_name + ".csv"))
        df_transposed = df.transpose()
        df_transposed.columns = [f"Column_{i}" for i in range(len(df_transposed.columns))]
        df_transposed.insert(0, "fileEntry", df_transposed.index, allow_duplicates=False)
        df_transposed.insert(0, "fileType", file_type, allow_duplicates=True)
        df_transposed.insert(0, "simIteration", sim_count, allow_duplicates=True)
        df_transposed.insert(0, "dataProcessors", n_servers, allow_duplicates=True)
        df_transposed.drop(df_transposed.index[[0, 1]], inplace=True)
        df_transposed.drop(df_transposed.index[[len(df_transposed) - 1]], inplace=True)
        df_transposed.to_csv(os.path.join(self.path, file_name), index=False)

    def format_stay_data(self, file_name, sim_count, n_servers):
        file_type = file_name.split("_")[0]
        df = pd.read_csv(os.path.join(self.path, file_name + ".csv"))
        df_transposed = df.transpose()
        df_transposed.columns = [f"Column_{i}" for i in range(len(df_transposed.columns))]
        df_transposed.insert(0, "fileNum", df_transposed.index, allow_duplicates=False)
        df_transposed.insert(0, "fileTypeNum", file_type + "." + df_transposed.index.astype(str), allow_duplicates=False)
        df_transposed.insert(0, "simIteration", sim_count, allow_duplicates=True)
        df_transposed.insert(0, "dataProcessors", n_servers, allow_duplicates=True)
        df_transposed.drop(df_transposed.index[[0]], inplace=True)
        df_transposed.to_csv(os.path.join(self.path, file_name), index=False)

    def aggregate_files(self, file_list, min_value, sim_counter):
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

    def create_file_data_master(self, file_list, min_value):
        df_combined = []
        for file_name in file_list:
            file_path = os.path.join(self.path, file_name + "Agg.csv")
            if os.path.exists(file_path):
                print(f"File {file_name + 'Agg.csv'} found and read successfully")
                df = pd.read_csv(file_path)
                df_combined.append(df)
        if df_combined:
            df = pd.concat(df_combined, ignore_index=True)
            df.to_csv(os.path.join(self.path, "systemFilesCombined.csv"))
        else:
            print(f"No files found for {file_list}")

    def create_stay_data_master(self, file_list, min_value):
        df_combined = []
        for file_name in file_list:
            file_path = os.path.join(self.path, file_name + "Agg.csv")
            if os.path.exists(file_path):
                print(f"File {file_name + 'Agg.csv'} found and read successfully")
                df = pd.read_csv(file_path)
                df_combined.append(df)
        if df_combined:
            df = pd.concat(df_combined, ignore_index=True)
            df.to_csv(os.path.join(self.path, "systemStayCombined.csv"))
        else:
            print(f"No files found for {file_list}")

    def create_file_data_master_single(self, file_list):
        df_combined = []
        for i in range(1, len(file_list)):
            file_path = os.path.join(self.path, file_list[i] + ".csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_combined.append(df)
        if df_combined:
            df = pd.concat(df_combined, ignore_index=True)
            df.to_csv(os.path.join(self.path, "systemFilesCombinedSingle.csv"))
        else:
            print(f"No files found for {file_list}")

    def create_stay_data_master_single(self, file_list):
        df_combined = []
        for i in range(1, len(file_list)):
            file_path = os.path.join(self.path, file_list[i] + ".csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_combined.append(df)
        if df_combined:
            df = pd.concat(df_combined, ignore_index=True)
            df.to_csv(os.path.join(self.path, "systemStayCombinedSingle.csv"))
        else:
            print(f"No files found for {file_list}")
