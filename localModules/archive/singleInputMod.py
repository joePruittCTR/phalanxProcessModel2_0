import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from simUtils import work_days_per_year

class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Capabilities Development Data Ingestion Simulation - Parameter Entry")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.work_days = work_days_per_year(federal_holidays=0, mean_vacation_days=0, mean_sick_days=0, mean_extended_workdays=0, include_weekends=False)
        self.param_dict = []  # Initialize param_dict as an instance variable

        self.all_entries = {
            'simFileName': tk.StringVar(value="testFile"),
            'nservers': tk.StringVar(value="4"),
            'siprTransfer': tk.StringVar(value="1"),
            'timeWindow': tk.StringVar(value="1"),
            'lowPriority': tk.StringVar(value="0.1"),
            'medPriority': tk.StringVar(value="67.4"),
            'highPriority': tk.StringVar(value="30"),
            'vHighPriority': tk.StringVar(value="2.5"),
            'co': {'iat': tk.StringVar(value="11"), 'time': tk.StringVar(value="60")},
            'dk': {'iat': tk.StringVar(value="20"), 'time': tk.StringVar(value="60")},
            'ma': {'iat': tk.StringVar(value="1"), 'time': tk.StringVar(value="60")},
            'nj': {'iat': tk.StringVar(value="130"), 'time': tk.StringVar(value="60")},
            'rs': {'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="60")},
            'sv': {'iat': tk.StringVar(value="7"), 'time': tk.StringVar(value="30")},
            'tg': {'iat': tk.StringVar(value="1"), 'time': tk.StringVar(value="30")},
            'wt': {'iat': tk.StringVar(value="135"), 'time': tk.StringVar(value="10"), 'mean': tk.StringVar(value="32"), 'dev': tk.StringVar(value="5")},
            'nf1': {'name': tk.StringVar(value="AA"), 'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="90")},
            'nf2': {'name': tk.StringVar(value="AN"), 'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="120")},
            'nf3': {'name': tk.StringVar(value="DD"), 'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="90")},
            'nf4': {'name': tk.StringVar(value="DF"), 'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="120")},
            'nf5': {'name': tk.StringVar(value="EA"), 'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="120")},
            'nf6': {'name': tk.StringVar(value="ML"), 'iat': tk.StringVar(value="0"), 'time': tk.StringVar(value="120")},
        }

        self.notebook = ttk.Notebook(self.root, padding="3 3 12 12")
        self.notebook.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Load the logo image
        try:
            self.logo_image = Image.open("./reportResources/phalanxLogoSmall.png")
            self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        except FileNotFoundError:
            print("Error: phalanxLogoSmall.png not found in ./reportResources/")
            self.logo_photo = None  # Set to None if not found

        self.create_page1()
        self.create_page2()
        self.create_page3() # Priority
        self.create_page6() # Core Sensors
        self.create_page4() # New Sensors
        self.create_page5() # New tab for buttons

        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.bind_all_entries()

        self.create_widgets() # This line can be removed if create_widgets is empty

    def add_logo(self, page):
        if self.logo_photo:
            logo_label = tk.Label(page, image=self.logo_photo)
            logo_label.image = self.logo_photo  # Keep a reference
            logo_label.grid(row=1, column=1, sticky=tk.NS, columnspan=2, rowspan=5)
        else:
            tk.Label(page, text="Logo Not Found").grid(row=1, column=1, sticky=tk.NS, columnspan=2, rowspan=5)

    def create_page1(self):
        page1 = ttk.Frame(self.notebook, padding="3 3 12 12")
        self.notebook.add(page1, text="General Settings")

        self.add_logo(page1)

        # Instructions
        ttk.Label(page1, text="Data Entry Instructions", font=("Arial Black", 20)).grid(row=1, column=3, columnspan=4, sticky=tk.N)
        ttk.Label(page1, text="1. Adjust parameter values from default values as desired on each tab.").grid(row=2, column=3, columnspan=4, sticky=tk.W)
        ttk.Label(page1, text="2. Cycle through each tab entering the relevant information.").grid(row=3, column=3, columnspan=4, sticky=tk.W)
        ttk.Label(page1, text="3. Go to the 'Submit/Begin' tab to complete parameter entry.").grid(row=4, column=3, columnspan=4, sticky=tk.W)


    def create_page2(self):
        page2 = ttk.Frame(self.notebook, padding="3 3 12 12")
        self.notebook.add(page2, text="Resource Allocation")

        self.add_logo(page2)

        # Personnel available for data ingestion
        ttk.Label(page2, text="Number of FTE used to ingest ").grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        self.nservers_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['nservers'])
        self.nservers_entry.grid(row=1, column=4, sticky=(tk.W, tk.E))

        # File name prefix
        ttk.Label(page2, text="File name prefix:").grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        self.simFileName_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['simFileName'])
        self.simFileName_entry.grid(row=2, column=4, sticky=(tk.W, tk.E))

        # Enclave transfer time
        ttk.Label(page2, text="SIPR to NIPR transfer time (work days):").grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)
        self.siprTransfer_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['siprTransfer'])
        self.siprTransfer_entry.grid(row=3, column=4, sticky=(tk.W, tk.E))

        # Years to simulate
        ttk.Label(page2, text="Years to simulate:").grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)
        self.timeWindow_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['timeWindow'])
        self.timeWindow_entry.grid(row=4, column=4, sticky=(tk.W, tk.E))

    def create_page3(self):
        page3 = ttk.Frame(self.notebook, padding="3 3 12 12")
        self.notebook.add(page3, text="Priority Distribution")

        self.add_logo(page3)

        ttk.Label(page3, text="Data File Priority Distribution", font=("Arial Black", 10)).grid(row=1, column=3, columnspan=2, sticky=tk.W)

        # Data file priority proportions
        ttk.Label(page3, text="Low Priority (%):").grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        self.lowPriority_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['lowPriority'])
        self.lowPriority_entry.grid(row=2, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="Medium Priority (%):").grid(row=2, column=5, sticky=tk.W, padx=5, pady=5)
        self.medPriority_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['medPriority'])
        self.medPriority_entry.grid(row=2, column=6, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="High Priority (%):").grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)
        self.highPriority_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['highPriority'])
        self.highPriority_entry.grid(row=3, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="Very High Priority (%):").grid(row=3, column=5, sticky=tk.W, padx=5, pady=5)
        self.vHighPriority_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['vHighPriority'])
        self.vHighPriority_entry.grid(row=3, column=6, sticky=(tk.W, tk.E))

    def create_page6(self):
        page6 = ttk.Frame(self.notebook, padding="3 3 12 12")
        self.notebook.add(page6, text="Core Sensors")

        self.add_logo(page6)

        ttk.Label(page6, text="Core Sensors", font=("Arial Black", 10)).grid(row=1, column=3, columnspan=2, sticky=tk.W)

        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        self.sensor_entries = {}

        header_row = 2
        ttk.Label(page6, text="Sensor", width=10).grid(row=header_row, column=3, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page6, text="Files per Month", width=15).grid(row=header_row, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page6, text="Processing Time (min)", width=20).grid(row=header_row, column=5, sticky=tk.W, padx=5, pady=5)

        for i, sensor in enumerate(sensor_types):
            row = i + 3
            ttk.Label(page6, text=f"{sensor.upper()}:", width=10).grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)

            iat_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries[sensor]['iat'])
            iat_entry.grid(row=row, column=4, sticky=(tk.W, tk.E))

            time_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries[sensor]['time'])
            time_entry.grid(row=row, column=5, sticky=(tk.W, tk.E))

            self.sensor_entries[sensor] = {'iat': iat_entry, 'time': time_entry}

        # Windtalker specific entries
        row = len(sensor_types) + 3  # Position after the last core sensor
        ttk.Label(page6, text="Windtalker Batch Size:", width=20).grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)
        self.meanWt_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries['wt']['mean'])
        self.meanWt_entry.grid(row=row, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page6, text="Std Dev:", width=10).grid(row=row, column=5, sticky=tk.W, padx=5, pady=5)
        self.devWt_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries['wt']['dev'])
        self.devWt_entry.grid(row=row, column=6, sticky=(tk.W, tk.E))

    def create_page4(self):
        page4 = ttk.Frame(self.notebook, padding="3 3 12 12")
        self.notebook.add(page4, text="New Sensors")

        self.add_logo(page4)

        ttk.Label(page4, text="Potential New Sensors", font=("Arial Black", 10)).grid(row=1, column=5, columnspan=4, sticky=tk.W)

        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        self.new_sensor_entries = {}

        header_row = 2
        ttk.Label(page4, text="Sensor Code").grid(row=header_row, column=6, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page4, text="Files per Month").grid(row=header_row, column=7, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page4, text="Processing Time (min)").grid(row=header_row, column=8, sticky=tk.W, padx=5, pady=5)

        for i, sensor in enumerate(new_sensor_types):
            row = i + 3  # Start from row 3
            #col = (i % 3) * 3 + 3

            ttk.Label(page4, text=f"{sensor.upper()}:").grid(row=row, column=5, sticky=tk.W, padx=5, pady=5)

            name_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries[sensor]['name'])
            name_entry.grid(row=row, column=6, sticky=(tk.W, tk.E))

            iat_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries[sensor]['iat'])
            iat_entry.grid(row=row, column=7, sticky=(tk.W, tk.E))

            time_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries[sensor]['time'])
            time_entry.grid(row=row, column=8, sticky=(tk.W, tk.E))

            self.new_sensor_entries[sensor] = {'name': name_entry, 'iat': iat_entry, 'time': time_entry}

    def create_page5(self):
        page5 = ttk.Frame(self.notebook, padding="3 3 12 12")
        self.notebook.add(page5, text="Submit/Begin")

        self.add_logo(page5)

        ttk.Button(page5, text="Submit Parameter Entries", command=self.submit_entries).grid(row=2, column=3, sticky=tk.W)
        ttk.Button(page5, text="Begin Simulation", command=self.start_simulation).grid(row=3, column=3, sticky=tk.W)



    def create_widgets(self):
        # Create all widgets and bind them to the corresponding StringVar
        pass

    def bind_all_entries(self):
        # Bind all entries to a common validation function
        for key, value in self.all_entries.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, tk.StringVar):
                        # sub_value.trace_add("write", self.validate_entry)
                        pass
            elif isinstance(value, tk.StringVar):
                # value.trace_add("write", self.validate_entry)
                pass

    def validate_entry(self, *args):
        # Placeholder for validation logic (can be extended as needed)
        pass

    def error_box(self):
        messagebox.showerror("Priority Distribution Error", "'Data File Priority Distribution' values must sum to 100%. \nPlease change the values and submit again.")

    def start_simulation(self):
        self.root.destroy()

    def submit_entries(self):
        try:
            # Validate priority distribution
            low_priority = float(self.all_entries['lowPriority'].get())
            med_priority = float(self.all_entries['medPriority'].get())
            high_priority = float(self.all_entries['highPriority'].get())
            vhigh_priority = float(self.all_entries['vHighPriority'].get())

            if abs(low_priority + med_priority + high_priority + vhigh_priority - 100) > 1e-6:
                self.error_box()
                return

            # Save parameters to param_dict
            self.param_dict = self.save_parameters()
            self.status_label.config(text="Parameters submitted successfully!")

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            self.status_label.config(text="Error: Invalid input.")

    def save_parameters(self):
        param_dict = []

        # General settings
        param_dict.append(self.all_entries['simFileName'].get())
        param_dict.append(float(self.all_entries['nservers'].get()))

        # Helper function to calculate server time and IAT
        def calculate_times(minutes_str, files_str):
            minutes = float(minutes_str)
            files = float(files_str)
            if minutes != 0:
                server_time = (1 / ((1 / minutes) * (60 / 1) * (8 / 1)))
            else:
                server_time = 0
            param_dict.append(server_time)

            if files != 0:
                iat = 1 / ((files * 12) / self.work_days)
            else:
                iat = 0
            param_dict.append(iat)

        # Core sensors
        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg']
        for sensor in sensor_types:
            calculate_times(self.all_entries[sensor]['time'].get(), self.all_entries[sensor]['iat'].get())

        # Windtalker specific
        calculate_times(self.all_entries['wt']['time'].get(), self.all_entries['wt']['iat'].get())
        param_dict.append(float(self.all_entries['wt']['mean'].get()))
        param_dict.append(float(self.all_entries['wt']['dev'].get()))

        # New file types
        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        for sensor in new_sensor_types:
            param_dict.append(self.all_entries[sensor]['name'].get())
            calculate_times(self.all_entries[sensor]['time'].get(), self.all_entries[sensor]['iat'].get())

        # Simulation parameters
        param_dict.append(float(self.all_entries['timeWindow'].get()))
        param_dict.append(self.work_days)
        param_dict.append(float(self.all_entries['timeWindow'].get()) * self.work_days)
        param_dict.append(float(self.all_entries['lowPriority'].get()) / 100)
        param_dict.append(float(self.all_entries['medPriority'].get()) / 100)
        param_dict.append(float(self.all_entries['highPriority'].get()) / 100)
        param_dict.append(float(self.all_entries['vHighPriority'].get()) / 100)
        param_dict.append(float(self.all_entries['siprTransfer'].get()))

        return param_dict

def get_parameter_values():
    root = tk.Tk()
    gui = ParameterGUI(root)
    root.mainloop()
    return gui.param_dict  # Return the collected parameter values

def start():
    params = get_parameter_values()
    print("Collected Parameters:", params)

if __name__ == "__main__":
    start()
