import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from simUtils import work_days_per_year

class GoalParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CapDev Data Ingestion Simulation - Starting Conditions Entry")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.work_days = work_days_per_year(federal_holidays=0, mean_vacation_days=0, mean_extended_workdays=0, include_weekends=False)
        self.param_dict = []

        # Initialize all StringVar objects
        self.all_entries = {
            'simFileName': tk.StringVar(value="testFile"),
            'goalTarget': tk.StringVar(),
            'goalParameter': tk.StringVar(),
            'waitTimeMax': tk.StringVar(value="1"),
            'processingFteMax': tk.StringVar(value="20"),
            'simMaxIterations': tk.StringVar(value="20"),
            'timeWindow': tk.StringVar(value="1"),
            'nservers': tk.StringVar(value="1"),
            'siprTransferTime': tk.StringVar(value="1"),
            'fileGrowth': tk.StringVar(value="20"),
            'sensorGrowth': tk.StringVar(value="1"),
            'ingestEfficiency': tk.StringVar(value="10"),
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

        # Load the logo image
        try:
            self.logo_image = Image.open("./reportResources/phalanxLogoSmall.png")
            self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        except FileNotFoundError:
            print("Error: phalanxLogoSmall.png not found in ./reportResources/")
            self.logo_photo = None

        self.parameter_choices = [""]
        self.min_param_choices = ["Ingestion FTE"]
        self.break_point_choices = ["Growth Applied to All Sensors", "CO Files per Month", "DK Files per Month", "MA Files per Month", "NJ Files per Month",
                             "RS Files per Month", "SV Files per Month", "TG Files per Month", "WT Files per Month", "New-1 Files per Month",
                             "New-2 Files per Month", "New-3 Files per Month", "New-4 Files per Month", "New-5 Files per Month", "New-6 Files per Month"]
        self.fix_sim_choices = ["Ingestion FTE", "Growth Applied to All Sensors", "CO Files per Month", "DK Files per Month", "MA Files per Month", "NJ Files per Month",
                             "RS Files per Month", "SV Files per Month", "TG Files per Month", "WT Files per Month", "New-1 Files per Month",
                             "New-2 Files per Month", "New-3 Files per Month", "New-4 Files per Month", "New-5 Files per Month", "New-6 Files per Month"]

        self.goal_seek_list = ["", "Minimize Wait Time", "Breaking Point Analysis", "Data Processor Break Points", "Fixed Simulation Runs"]

        self.notebook = ttk.Notebook(self.root, padding="3 3 7 7")
        self.notebook.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.create_page1()  # Instructions & Filename
        self.create_page2()  # Goal Seek Parameters
        self.create_page3()  # Simulation Parameters
        self.create_page4()  # Priority Distribution
        self.create_page5()  # Core Sensors
        self.create_page6()  # Potential New Sensors
        self.create_page7()  # Submit/Begin

        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def add_logo(self, page):
        if self.logo_photo:
            logo_label = tk.Label(page, image=self.logo_photo)
            logo_label.image = self.logo_photo
            logo_label.grid(row=1, column=1, sticky=tk.NS, columnspan=2, rowspan=5)
        else:
            tk.Label(page, text="Logo Not Found").grid(row=1, column=1, sticky=tk.NS, columnspan=2, rowspan=5)

    def create_page1(self):
        page1 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page1, text="Instructions")
        self.add_logo(page1)

        i = 1
        ttk.Label(page1, text="Goal Seek Entry Instructions", font=("Arial Black", 20)).grid(row=i, column=3, columnspan=4, sticky=tk.N); i += 1
        ttk.Label(page1, text="1. Enter the file prefix to be used and select the goal.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="2. Enter the critical value for the goal metric.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="3. Confirm the upper limits for file wait time and processing FTE.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="4. Enter the maximum number of simulation iterations.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="5. Adjust remaining parameter values from default values as desired.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="6. Go to the 'Submit/Begin' tab to complete parameter entry.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1

        ttk.Label(page1, text="File name prefix:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.simFileName_entry = ttk.Entry(page1, width=7, textvariable=self.all_entries['simFileName'])
        self.simFileName_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

    def create_page2(self):
        page2 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page2, text="Goal Seek Parameters")
        self.add_logo(page2)

        i = 1
        ttk.Label(page2, text="Goal-Seek Parameters", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i += 1

        ttk.Label(page2, text="Goal Target:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.goalSeekList_entry = ttk.Combobox(page2, textvariable=self.all_entries['goalTarget'], values=self.goal_seek_list)
        self.goalSeekList_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))
        self.goalSeekList_entry.bind('<<ComboboxSelected>>', self.combo_fill)
        ttk.Button(page2, text="Target Info", command=self.info_box).grid(row=i, column=5, sticky=tk.W); i += 1

        ttk.Label(page2, text="Parameter(s) to adjust:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.goalParameter_entry = ttk.Combobox(page2, textvariable=self.all_entries['goalParameter'], values=self.parameter_choices)
        self.goalParameter_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page2, text="Wait time maximum (Days):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.waitTimeMax_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['waitTimeMax'])
        self.waitTimeMax_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page2, text="Processing FTE maximum:").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.processingFteMax_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['processingFteMax'])
        self.processingFteMax_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page2, text="Maximum simulation iterations:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.simMaxIterations_entry = ttk.Entry(page2, width=7, textvariable=self.all_entries['simMaxIterations'])
        self.simMaxIterations_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

    def create_page3(self):
        page3 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page3, text="Simulation Parameters")
        self.add_logo(page3)

        i = 1
        ttk.Label(page3, text="Simulation Parameters", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i += 1

        ttk.Label(page3, text="Years to simulate:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.timeWindow_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['timeWindow'])
        self.timeWindow_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="Number of FTE used to ingest ").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.nservers_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['nservers'])
        self.nservers_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page3, text="SIPR to NIPR transfer time (work days):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.siprTransferTime_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['siprTransferTime'])
        self.siprTransferTime_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page3, text="File Growth Coeffcient (Percent per Year):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.fileGrowth_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['fileGrowth'])
        self.fileGrowth_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="Sensor Growth Coefficient (New Sensor Types per Year):").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.sensorGrowth_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['sensorGrowth'])
        self.sensorGrowth_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page3, text="Ingest Efficiency Factor (Percent per Year):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.ingestEfficiency_entry = ttk.Entry(page3, width=7, textvariable=self.all_entries['ingestEfficiency'])
        self.ingestEfficiency_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

    def create_page4(self):
        page4 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page4, text="Priority Distribution")
        self.add_logo(page4)

        i = 1
        ttk.Label(page4, text="Data File Priority Distribution", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i += 1

        ttk.Label(page4, text="Low Priority (%):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.lowPriority_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries['lowPriority'])
        self.lowPriority_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page4, text="Medium Priority (%):").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.medPriority_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries['medPriority'])
        self.medPriority_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page4, text="High Priority (%):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.highPriority_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries['highPriority'])
        self.highPriority_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page4, text="Very High Priority (%):").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.vHighPriority_entry = ttk.Entry(page4, width=7, textvariable=self.all_entries['vHighPriority'])
        self.vHighPriority_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

    def create_page5(self):
        page5 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page5, text="Core Sensors")
        self.add_logo(page5)

        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        self.sensor_entries = {}
        i = 1

        ttk.Label(page5, text="Core Sensors", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i+=1
        header_row = i
        ttk.Label(page5, text="Sensor", width=10).grid(row=header_row, column=3, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page5, text="Files per Month", width=15).grid(row=header_row, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page5, text="Processing Time (min)", width=20).grid(row=header_row, column=5, sticky=tk.W, padx=5, pady=5)
        i+=1

        for sensor in sensor_types:
            row = i
            ttk.Label(page5, text=f"{sensor.upper()}:", width=10).grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)

            iat_entry = ttk.Entry(page5, width=7, textvariable=self.all_entries[sensor]['iat'])
            iat_entry.grid(row=row, column=4, sticky=(tk.W, tk.E))

            time_entry = ttk.Entry(page5, width=7, textvariable=self.all_entries[sensor]['time'])
            time_entry.grid(row=row, column=5, sticky=(tk.W, tk.E))

            self.sensor_entries[sensor] = {'iat': iat_entry, 'time': time_entry}
            i+=1

        # Windtalker specific entries
        row = i
        ttk.Label(page5, text="Windtalker Batch Size:", width=20).grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)
        self.meanWt_entry = ttk.Entry(page5, width=7, textvariable=self.all_entries['wt']['mean'])
        self.meanWt_entry.grid(row=row, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page5, text="Std Dev:", width=10).grid(row=row, column=5, sticky=tk.W, padx=5, pady=5)
        self.devWt_entry = ttk.Entry(page5, width=7, textvariable=self.all_entries['wt']['dev'])
        self.devWt_entry.grid(row=row, column=6, sticky=(tk.W, tk.E))

    def create_page6(self):
        page6 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page6, text="New Sensors")
        self.add_logo(page6)

        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        self.new_sensor_entries = {}
        i = 1

        ttk.Label(page6, text="Potential New Sensors", font=("Arial Black", 10)).grid(row=i, column=5, columnspan=4, sticky=tk.W); i+=1

        header_row = i
        ttk.Label(page6, text="Sensor Code", width = 10).grid(row=header_row, column=6, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page6, text="Files per Month", width = 15).grid(row=header_row, column=7, sticky=tk.W, padx=5, pady=5)
        ttk.Label(page6, text="Processing Time (min)", width = 20).grid(row=header_row, column=8, sticky=tk.W, padx=5, pady=5)
        i+=1

        for sensor in new_sensor_types:
            row = i
            ttk.Label(page6, text=f"{sensor.upper()}:", width = 10).grid(row=row, column=5, sticky=tk.W, padx=5, pady=5)

            name_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries[sensor]['name'])
            name_entry.grid(row=row, column=6, sticky=(tk.W, tk.E))

            iat_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries[sensor]['iat'])
            iat_entry.grid(row=row, column=7, sticky=(tk.W, tk.E))

            time_entry = ttk.Entry(page6, width=7, textvariable=self.all_entries[sensor]['time'])
            time_entry.grid(row=row, column=8, sticky=(tk.W, tk.E))

            self.new_sensor_entries[sensor] = {'name': name_entry, 'iat': iat_entry, 'time': time_entry}
            i+=1

    def create_page7(self):
        page7 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page7, text="Submit/Begin")
        self.add_logo(page7)

        ttk.Button(page7, text="Submit Parameter Entries", command=self.submit_entries).grid(row=2, column=3, sticky=tk.W)
        ttk.Button(page7, text="Begin Simulation", command=self.start_simulation).grid(row=3, column=3, sticky=tk.W)

    def combo_fill(self, event):
        choice = self.all_entries['goalTarget'].get()
        if choice == "Minimize Wait Time":
            self.parameter_choices = self.min_param_choices
        elif choice in ("Breaking Point Analysis", "Data Processor Break Points"):
            self.parameter_choices = self.break_point_choices
        elif choice == "Fixed Simulation Runs":
            self.parameter_choices = self.fix_sim_choices
        else:
            self.parameter_choices = [""]
        self.goalParameter_entry.config(values=self.parameter_choices)

    def info_box(self):
        messagebox.showinfo("Target Functionality", "The 'Minimize Wait Time' target iterates through FTE values to reach a target maximum wait time for a given set of initial parameters.\
Choose 'Ingestion FTE' as the parameter to adjust over time and confirm the remaining parameters. \n \n The 'Breaking Point Analysis' target keeps FTE constant and iterates through forcasted \
growth with respect to files and sensors to find the point at which the maximum wait time is exceeded. Ensure that slope values are entered for at least one of File growth or Sensor growth \
and choose which specific file type (or all files) to which the growth coeffient applies.  \n \n The 'Fixed Simulation Runs' target executes the entered 'Maximum simulation iterations with \
the given parameter values.  This option will not exit the simulation until all iteration have been executed.")

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

        # Filename Prefix
        param_dict.append(self.all_entries['simFileName'].get())

        # Goal Seek Parameters
        param_dict.append(self.all_entries['goalTarget'].get())
        param_dict.append(self.all_entries['goalParameter'].get())
        param_dict.append(float(self.all_entries['waitTimeMax'].get()))
        param_dict.append(float(self.all_entries['processingFteMax'].get()))
        param_dict.append(float(self.all_entries['simMaxIterations'].get()))

        # Simulation Parameters
        time_window_years = float(self.all_entries['timeWindow'].get())
        param_dict.append(time_window_years)
        time_window = time_window_years * self.work_days
        param_dict.append(time_window)
        param_dict.append(float(self.all_entries['nservers'].get()))
        param_dict.append(float(self.all_entries['siprTransferTime'].get()))
        file_growth = float(self.all_entries['fileGrowth'].get()) / 100
        param_dict.append(file_growth)
        sensor_growth = float(self.all_entries['sensorGrowth'].get())
        param_dict.append(sensor_growth)
        ingest_efficiency = -1 * (float(self.all_entries['ingestEfficiency'].get()) / 100)
        param_dict.append(ingest_efficiency)

        # Data File Priority Distribution
        param_dict.append(float(self.all_entries['lowPriority'].get()))
        param_dict.append(float(self.all_entries['medPriority'].get()))
        param_dict.append(float(self.all_entries['highPriority'].get()))
        param_dict.append(float(self.all_entries['vHighPriority'].get()))

        # Core Sensors
        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        for sensor in sensor_types:
            minutes = float(self.all_entries[sensor]['time'].get())
            param_dict.append(minutes)
            server_time = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0
            param_dict.append(server_time)

            files = float(self.all_entries[sensor]['iat'].get())
            param_dict.append(files)
            iat = 1 / ((files * 12) / self.work_days) if files != 0 else 0
            param_dict.append(iat)

        # Windtalker
        minutes = float(self.all_entries['wt']['time'].get())
        param_dict.append(minutes)
        wt_server_time = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0
        param_dict.append(wt_server_time)

        files = float(self.all_entries['wt']['iat'].get())
        param_dict.append(files)
        wt_iat = 1 / ((files * 12) / self.work_days) if files != 0 else 0
        param_dict.append(wt_iat)

        param_dict.append(float(self.all_entries['wt']['mean'].get()))
        param_dict.append(float(self.all_entries['wt']['dev'].get()))

        # New file types
        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        for sensor in new_sensor_types:
            param_dict.append(self.all_entries[sensor]['name'].get())
            minutes = float(self.all_entries[sensor]['time'].get())
            param_dict.append(minutes)
            server_time = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0
            param_dict.append(server_time)

            files = float(self.all_entries[sensor]['iat'].get())
            param_dict.append(files)
            iat = 1 / ((files * 12) / self.work_days) if files != 0 else 0
            param_dict.append(iat)

        start_loop = 0
        param_dict.append(start_loop)

        return param_dict

    def start_simulation(self):
        self.root.destroy()

    def error_box(self):
        messagebox.showerror("Priority Distribution Error",
                             "'Data File Priority Distribution' values must sum to 100%. \nPlease change the values and submit again.")

    def create_widgets(self):
        pass

def get_goal_parameter_values():
    root = tk.Tk()
    gui = GoalParameterGUI(root)
    root.mainloop()
    return gui.param_dict

def start():
    params = get_goal_parameter_values()
    print("Collected Parameters:", params)

if __name__ == "__main__":
    start()
