# 02simInput.py

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from simUtils import work_days_per_year # Assuming simUtils is correctly imported and available

class SimulationInputGUI:
    def __init__(self, master, mode="single"):
        # The master is the parent Tkinter window (e.g., root from 01simStart.py)
        self.master = master
        self.root = tk.Toplevel(master) # Create a Toplevel window instead of a new Tk()
        self.root.title("CapDev Data Ingestion Simulation - Input Parameters")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Make the Toplevel window modal
        self.root.transient(master)
        self.root.grab_set()

        self.work_days = work_days_per_year(federal_holidays=0, mean_vacation_days=0, mean_extended_workdays=0, include_weekends=False)
        
        # Store collected parameters as a dictionary
        self.collected_parameters = None 

        # Shared StringVar for simulation mode, initialized based on the 'mode' argument
        self.simulation_mode = tk.StringVar(value=mode)

        # Initialize all StringVar objects for single simulation
        # Using a dictionary for better organization and access by name
        self.single_entries = {
            'simFileName': tk.StringVar(value="testFile"),
            'nservers': tk.StringVar(value="4"),
            'siprTransfer': tk.StringVar(value="1"),
            'timeWindow': tk.StringVar(value="1"),
            'lowPriority': tk.StringVar(value="0.1"),
            'medPriority': tk.StringVar(value="67.4"),
            'highPriority': tk.StringVar(value="30"),
            'vHighPriority': tk.StringVar(value="2.5"),
            'co_iat': tk.StringVar(value="11"), 'co_time': tk.StringVar(value="60"),
            'dk_iat': tk.StringVar(value="20"), 'dk_time': tk.StringVar(value="60"),
            'ma_iat': tk.StringVar(value="1"), 'ma_time': tk.StringVar(value="60"),
            'nj_iat': tk.StringVar(value="130"), 'nj_time': tk.StringVar(value="60"),
            'rs_iat': tk.StringVar(value="0"), 'rs_time': tk.StringVar(value="60"),
            'sv_iat': tk.StringVar(value="7"), 'sv_time': tk.StringVar(value="30"),
            'tg_iat': tk.StringVar(value="1"), 'tg_time': tk.StringVar(value="30"),
            'wt_iat': tk.StringVar(value="135"), 'wt_time': tk.StringVar(value="10"), 'wt_mean': tk.StringVar(value="32"), 'wt_dev': tk.StringVar(value="5"),
            'nf1_name': tk.StringVar(value="AA"), 'nf1_iat': tk.StringVar(value="0"), 'nf1_time': tk.StringVar(value="90"),
            'nf2_name': tk.StringVar(value="AN"), 'nf2_iat': tk.StringVar(value="0"), 'nf2_time': tk.StringVar(value="120"),
            'nf3_name': tk.StringVar(value="DD"), 'nf3_iat': tk.StringVar(value="0"), 'nf3_time': tk.StringVar(value="90"),
            'nf4_name': tk.StringVar(value="DF"), 'nf4_iat': tk.StringVar(value="0"), 'nf4_time': tk.StringVar(value="120"),
            'nf5_name': tk.StringVar(value="EA"), 'nf5_iat': tk.StringVar(value="0"), 'nf5_time': tk.StringVar(value="120"),
            'nf6_name': tk.StringVar(value="ML"), 'nf6_iat': tk.StringVar(value="0"), 'nf6_time': tk.StringVar(value="120"),
            'fileGrowth': tk.StringVar(value="20"),
            'sensorGrowth': tk.StringVar(value="1"),
            'ingestEfficiency': tk.StringVar(value="10"),
        }

        # Initialize all StringVar objects for goal seek (keys will be consistent with single_entries where possible)
        self.goal_entries = {
            'simFileName': tk.StringVar(value="testFile"),
            'goalTarget': tk.StringVar(),
            'goalParameter': tk.StringVar(),
            'waitTimeMax': tk.StringVar(value="1"),
            'processingFteMax': tk.StringVar(value="20"),
            'simMaxIterations': tk.StringVar(value="20"),
            'timeWindow': tk.StringVar(value="1"),
            'nservers': tk.StringVar(value="1"),
            'siprTransfer': tk.StringVar(value="1"), # Consistent key name
            'fileGrowth': tk.StringVar(value="20"),
            'sensorGrowth': tk.StringVar(value="1"),
            'ingestEfficiency': tk.StringVar(value="10"),
            'lowPriority': tk.StringVar(value="0.1"),
            'medPriority': tk.StringVar(value="67.4"),
            'highPriority': tk.StringVar(value="30"),
            'vHighPriority': tk.StringVar(value="2.5"),
            'co_iat': tk.StringVar(value="11"), 'co_time': tk.StringVar(value="60"),
            'dk_iat': tk.StringVar(value="20"), 'dk_time': tk.StringVar(value="60"),
            'ma_iat': tk.StringVar(value="1"), 'ma_time': tk.StringVar(value="60"),
            'nj_iat': tk.StringVar(value="130"), 'nj_time': tk.StringVar(value="60"),
            'rs_iat': tk.StringVar(value="0"), 'rs_time': tk.StringVar(value="60"),
            'sv_iat': tk.StringVar(value="7"), 'sv_time': tk.StringVar(value="30"),
            'tg_iat': tk.StringVar(value="1"), 'tg_time': tk.StringVar(value="30"),
            'wt_iat': tk.StringVar(value="135"), 'wt_time': tk.StringVar(value="10"), 'wt_mean': tk.StringVar(value="32"), 'wt_dev': tk.StringVar(value="5"),
            'nf1_name': tk.StringVar(value="AA"), 'nf1_iat': tk.StringVar(value="0"), 'nf1_time': tk.StringVar(value="90"),
            'nf2_name': tk.StringVar(value="AN"), 'nf2_iat': tk.StringVar(value="0"), 'nf2_time': tk.StringVar(value="120"),
            'nf3_name': tk.StringVar(value="DD"), 'nf3_iat': tk.StringVar(value="0"), 'nf3_time': tk.StringVar(value="90"),
            'nf4_name': tk.StringVar(value="DF"), 'nf4_iat': tk.StringVar(value="0"), 'nf4_time': tk.StringVar(value="120"),
            'nf5_name': tk.StringVar(value="EA"), 'nf5_iat': tk.StringVar(value="0"), 'nf5_time': tk.StringVar(value="120"),
            'nf6_name': tk.StringVar(value="ML"), 'nf6_iat': tk.StringVar(value="0"), 'nf6_time': tk.StringVar(value="120"),
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

        self.create_page0()  # Simulation Mode Selection
        self.create_page1()  # Instructions & Filename
        self.create_page2()  # Goal Seek Parameters
        self.create_page3()  # Simulation Parameters
        self.create_page4()  # Priority Distribution
        self.create_page5()  # Core Sensors
        self.create_page6()  # Potential New Sensors
        self.create_page7()  # Submit/Begin

        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.update_widget_states()
        self.create_widgets() # This seems to be an empty method, can be removed if not used.

        # Center the Toplevel window relative to its master
        self.root.update_idletasks()
        x = master.winfo_x() + (master.winfo_width() // 2) - (self.root.winfo_width() // 2)
        y = master.winfo_y() + (master.winfo_height() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

    def add_logo(self, page):
        if self.logo_photo:
            logo_label = tk.Label(page, image=self.logo_photo)
            logo_label.image = self.logo_photo
            logo_label.grid(row=1, column=1, sticky=tk.NS, columnspan=2, rowspan=5)
        else:
            tk.Label(page, text="Logo Not Found").grid(row=1, column=1, sticky=tk.NS, columnspan=2, rowspan=5)

    def create_page0(self):
        page0 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page0, text="Simulation Mode")
        self.add_logo(page0)

        i = 1
        ttk.Label(page0, text="Select Simulation Mode:", font=("Arial Black", 10)).grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)

        # Radio buttons should change the mode and then update states
        self.single_radio = ttk.Radiobutton(page0, text="Single Simulation", variable=self.simulation_mode, value="single", command=self.update_widget_states)
        self.single_radio.grid(row=i+1, column=3, sticky=tk.W, padx=5, pady=5)

        self.goal_radio = ttk.Radiobutton(page0, text="Goal Seek Simulation", variable=self.simulation_mode, value="goal", command=self.update_widget_states)
        self.goal_radio.grid(row=i+2, column=3, sticky=tk.W, padx=5, pady=5)

        # Pre-select the correct tab based on initial mode
        if self.simulation_mode.get() == "goal":
            self.notebook.select(self.notebook.index("Goal Seek Parameters")) # Select the goal tab by name
        else:
            self.notebook.select(self.notebook.index("Instructions")) # Default to instructions for single mode


    def create_page1(self):
        page1 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page1, text="Instructions")
        self.add_logo(page1)

        i = 1
        ttk.Label(page1, text="Entry Instructions", font=("Arial Black", 20)).grid(row=i, column=3, columnspan=4, sticky=tk.N); i += 1
        ttk.Label(page1, text="1. Select the simulation mode on the 'Simulation Mode' tab.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="2. Enter the file prefix to be used.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="3. Complete the remaining tabs with the desired simulation parameters.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1
        ttk.Label(page1, text="4. Go to the 'Submit/Begin' tab to complete parameter entry and start the simulation.").grid(row=i, column=3, columnspan=4, sticky=tk.W); i += 1

        ttk.Label(page1, text="File name prefix:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        # Use a common StringVar for simFileName across both modes if it's truly shared
        self.simFileName_entry = ttk.Entry(page1, width=7, textvariable=self.single_entries['simFileName']) # Using single_entries for consistency
        self.simFileName_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

    def create_page2(self):
        page2 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page2, text="Goal Seek Parameters")
        self.add_logo(page2)

        i = 1
        ttk.Label(page2, text="Goal-Seek Parameters", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i += 1

        ttk.Label(page2, text="Goal Target:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.goalSeekList_entry = ttk.Combobox(page2, textvariable=self.goal_entries['goalTarget'], values=self.goal_seek_list, state=tk.DISABLED)
        self.goalSeekList_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))
        self.goalSeekList_entry.bind('<<ComboboxSelected>>', self.combo_fill)

        self.info_box_button = ttk.Button(page2, text="Target Info", command=self.info_box, state=tk.DISABLED)
        self.info_box_button.grid(row=i, column=5, sticky=tk.W); i += 1

        ttk.Label(page2, text="Parameter(s) to adjust:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.goalParameter_entry = ttk.Combobox(page2, textvariable=self.goal_entries['goalParameter'], values=self.parameter_choices, state=tk.DISABLED)
        self.goalParameter_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page2, text="Wait time maximum (Days):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.waitTimeMax_entry = ttk.Entry(page2, width=7, textvariable=self.goal_entries['waitTimeMax'], state=tk.DISABLED)
        self.waitTimeMax_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page2, text="Processing FTE maximum:").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.processingFteMax_entry = ttk.Entry(page2, width=7, textvariable=self.goal_entries['processingFteMax'], state=tk.DISABLED)
        self.processingFteMax_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page2, text="Maximum simulation iterations:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.simMaxIterations_entry = ttk.Entry(page2, width=7, textvariable=self.goal_entries['simMaxIterations'], state=tk.DISABLED)
        self.simMaxIterations_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

    def create_page3(self):
        page3 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page3, text="Simulation Parameters")
        self.add_logo(page3)

        i = 1
        ttk.Label(page3, text="Simulation Parameters", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i += 1

        ttk.Label(page3, text="Years to simulate:").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.timeWindow_entry = ttk.Entry(page3, width=7, textvariable=self.single_entries['timeWindow'], state=tk.DISABLED)
        self.timeWindow_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="Number of FTE used to ingest ").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.nservers_entry = ttk.Entry(page3, width=7, textvariable=self.single_entries['nservers'], state=tk.DISABLED)
        self.nservers_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page3, text="SIPR to NIPR transfer time (work days):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.siprTransferTime_entry = ttk.Entry(page3, width=7, textvariable=self.single_entries['siprTransfer'], state=tk.DISABLED)
        self.siprTransferTime_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page3, text="File Growth Coeffcient (Percent per Year):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.fileGrowth_entry = ttk.Entry(page3, width=7, textvariable=self.single_entries['fileGrowth'], state=tk.DISABLED)
        self.fileGrowth_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page3, text="Sensor Growth Coefficient (New Sensor Types per Year):").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.sensorGrowth_entry = ttk.Entry(page3, width=7, textvariable=self.single_entries['sensorGrowth'], state=tk.DISABLED)
        self.sensorGrowth_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page3, text="Ingest Efficiency Factor (Percent per Year):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.ingestEfficiency_entry = ttk.Entry(page3, width=7, textvariable=self.single_entries['ingestEfficiency'], state=tk.DISABLED)
        self.ingestEfficiency_entry.grid(row=i, column=4, sticky=(tk.W, tk.E)); i += 1

    def create_page4(self):
        page4 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page4, text="Priority Distribution")
        self.add_logo(page4)

        i = 1
        ttk.Label(page4, text="Data File Priority Distribution", font=("Arial Black", 10)).grid(row=i, column=3, columnspan=2, sticky=tk.W); i += 1

        ttk.Label(page4, text="Low Priority (%):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.lowPriority_entry = ttk.Entry(page4, width=7, textvariable=self.single_entries['lowPriority'], state=tk.DISABLED)
        self.lowPriority_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page4, text="Medium Priority (%):").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.medPriority_entry = ttk.Entry(page4, width=7, textvariable=self.single_entries['medPriority'], state=tk.DISABLED)
        self.medPriority_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

        ttk.Label(page4, text="High Priority (%):").grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
        self.highPriority_entry = ttk.Entry(page4, width=7, textvariable=self.single_entries['highPriority'], state=tk.DISABLED)
        self.highPriority_entry.grid(row=i, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page4, text="Very High Priority (%):").grid(row=i, column=5, sticky=tk.W, padx=5, pady=5)
        self.vHighPriority_entry = ttk.Entry(page4, width=7, textvariable=self.single_entries['vHighPriority'], state=tk.DISABLED)
        self.vHighPriority_entry.grid(row=i, column=6, sticky=(tk.W, tk.E)); i += 1

    def create_page5(self):
        page5 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page5, text="Core Sensors")
        self.add_logo(page5)

        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        self.sensor_entries_widgets = {} # Renamed to avoid confusion with self.single_entries
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

            iat_entry = ttk.Entry(page5, width=7, textvariable=self.single_entries[f'{sensor}_iat'], state=tk.DISABLED)
            iat_entry.grid(row=row, column=4, sticky=(tk.W, tk.E))

            time_entry = ttk.Entry(page5, width=7, textvariable=self.single_entries[f'{sensor}_time'], state=tk.DISABLED)
            time_entry.grid(row=row, column=5, sticky=(tk.W, tk.E))

            self.sensor_entries_widgets[sensor] = {'iat': iat_entry, 'time': time_entry}
            i+=1

        # Windtalker specific entries
        row = i
        ttk.Label(page5, text="Windtalker Batch Size:", width=20).grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)
        self.meanWt_entry = ttk.Entry(page5, width=7, textvariable=self.single_entries['wt_mean'], state=tk.DISABLED)
        self.meanWt_entry.grid(row=row, column=4, sticky=(tk.W, tk.E))

        ttk.Label(page5, text="Std Dev:", width=10).grid(row=row, column=5, sticky=tk.W, padx=5, pady=5)
        self.devWt_entry = ttk.Entry(page5, width=7, textvariable=self.single_entries['wt_dev'], state=tk.DISABLED)
        self.devWt_entry.grid(row=row, column=6, sticky=(tk.W, tk.E))

    def create_page6(self):
        page6 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page6, text="New Sensors")
        self.add_logo(page6)

        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        self.new_sensor_entries_widgets = {} # Renamed
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

            name_entry = ttk.Entry(page6, width=7, textvariable=self.single_entries[f'{sensor}_name'], state=tk.DISABLED)
            name_entry.grid(row=row, column=6, sticky=(tk.W, tk.E))

            iat_entry = ttk.Entry(page6, width=7, textvariable=self.single_entries[f'{sensor}_iat'], state=tk.DISABLED)
            iat_entry.grid(row=row, column=7, sticky=(tk.W, tk.E))

            time_entry = ttk.Entry(page6, width=7, textvariable=self.single_entries[f'{sensor}_time'], state=tk.DISABLED)
            time_entry.grid(row=row, column=8, sticky=(tk.W, tk.E))

            self.new_sensor_entries_widgets[sensor] = {'name': name_entry, 'iat': iat_entry, 'time': time_entry}
            i+=1

    def create_page7(self):
        page7 = ttk.Frame(self.notebook, padding="3 3 7 7")
        self.notebook.add(page7, text="Submit/Begin")
        self.add_logo(page7)

        ttk.Button(page7, text="Submit Parameter Entries", command=self.submit_entries).grid(row=2, column=3, sticky=tk.W)
        ttk.Button(page7, text="Begin Simulation", command=self.start_simulation).grid(row=3, column=3, sticky=tk.W)

    def combo_fill(self, event):
        choice = self.goal_entries['goalTarget'].get()
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
            if self.simulation_mode.get() == "single":
                # Validate priority distribution
                low_priority = float(self.single_entries['lowPriority'].get())
                med_priority = float(self.single_entries['medPriority'].get())
                high_priority = float(self.single_entries['highPriority'].get())
                vhigh_priority = float(self.single_entries['vHighPriority'].get())

                if abs(low_priority + med_priority + high_priority + vhigh_priority - 100) > 1e-6:
                    self.error_box()
                    return

                # Save parameters to a dictionary
                self.collected_parameters = self.save_single_parameters()
                self.status_label.config(text="Parameters submitted successfully!")
            elif self.simulation_mode.get() == "goal":
                # Validate priority distribution
                low_priority = float(self.goal_entries['lowPriority'].get())
                med_priority = float(self.goal_entries['medPriority'].get())
                high_priority = float(self.goal_entries['highPriority'].get())
                vhigh_priority = float(self.goal_entries['vHighPriority'].get())

                if abs(low_priority + med_priority + high_priority + vhigh_priority - 100) > 1e-6:
                    self.error_box()
                    return

                # Save goal parameters to a dictionary
                self.collected_parameters = self.save_goal_parameters()
                self.status_label.config(text="Goal parameters submitted successfully!")

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            self.status_label.config(text="Error: Invalid input.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.status_label.config(text=f"Error: {e}")


    def save_single_parameters(self):
        params = {} # Use a dictionary

        # Filename Prefix
        params['simFileName'] = self.single_entries['simFileName'].get()

        # Simulation Parameters
        time_window_years = float(self.single_entries['timeWindow'].get())
        params['timeWindowYears'] = time_window_years
        params['timeWindowDays'] = time_window_years * self.work_days
        params['nservers'] = float(self.single_entries['nservers'].get())
        params['siprTransfer'] = float(self.single_entries['siprTransfer'].get())
        params['fileGrowth'] = float(self.single_entries['fileGrowth'].get()) / 100
        params['sensorGrowth'] = float(self.single_entries['sensorGrowth'].get())
        params['ingestEfficiency'] = -1 * (float(self.single_entries['ingestEfficiency'].get()) / 100)

        # Data File Priority Distribution
        params['lowPriority'] = float(self.single_entries['lowPriority'].get())
        params['medPriority'] = float(self.single_entries['medPriority'].get())
        params['highPriority'] = float(self.single_entries['highPriority'].get())
        params['vHighPriority'] = float(self.single_entries['vHighPriority'].get())

        # Core Sensors
        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        for sensor in sensor_types:
            minutes = float(self.single_entries[f'{sensor}_time'].get())
            params[f'{sensor}_time'] = minutes
            params[f'{sensor}_server_time'] = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0

            files = float(self.single_entries[f'{sensor}_iat'].get())
            params[f'{sensor}_files'] = files
            params[f'{sensor}_iat_calc'] = 1 / ((files * 12) / self.work_days) if files != 0 else 0

        # Windtalker specific
        params['wt_mean'] = float(self.single_entries['wt_mean'].get())
        params['wt_dev'] = float(self.single_entries['wt_dev'].get())

        # New file types
        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        for sensor in new_sensor_types:
            params[f'{sensor}_name'] = self.single_entries[f'{sensor}_name'].get()
            minutes = float(self.single_entries[f'{sensor}_time'].get())
            params[f'{sensor}_time'] = minutes
            params[f'{sensor}_server_time'] = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0

            files = float(self.single_entries[f'{sensor}_iat'].get())
            params[f'{sensor}_files'] = files
            params[f'{sensor}_iat_calc'] = 1 / ((files * 12) / self.work_days) if files != 0 else 0

        params['startLoop'] = 0 # This seems like a fixed value

        return params

    def save_goal_parameters(self):
        params = {} # Use a dictionary

        # Filename Prefix
        params['simFileName'] = self.goal_entries['simFileName'].get()

        # Goal Seek Parameters
        params['goalTarget'] = self.goal_entries['goalTarget'].get()
        params['goalParameter'] = self.goal_entries['goalParameter'].get()
        params['waitTimeMax'] = float(self.goal_entries['waitTimeMax'].get())
        params['processingFteMax'] = float(self.goal_entries['processingFteMax'].get())
        params['simMaxIterations'] = float(self.goal_entries['simMaxIterations'].get())

        # Simulation Parameters (ensure consistency with single_entries keys)
        time_window_years = float(self.goal_entries['timeWindow'].get())
        params['timeWindowYears'] = time_window_years
        params['timeWindowDays'] = time_window_years * self.work_days
        params['nservers'] = float(self.goal_entries['nservers'].get())
        params['siprTransfer'] = float(self.goal_entries['siprTransfer'].get())
        params['fileGrowth'] = float(self.goal_entries['fileGrowth'].get()) / 100
        params['sensorGrowth'] = float(self.goal_entries['sensorGrowth'].get())
        params['ingestEfficiency'] = -1 * (float(self.goal_entries['ingestEfficiency'].get()) / 100)

        # Data File Priority Distribution
        params['lowPriority'] = float(self.goal_entries['lowPriority'].get())
        params['medPriority'] = float(self.goal_entries['medPriority'].get())
        params['highPriority'] = float(self.goal_entries['highPriority'].get())
        params['vHighPriority'] = float(self.goal_entries['vHighPriority'].get())

        # Core Sensors (ensure consistency with single_entries keys)
        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        for sensor in sensor_types:
            minutes = float(self.goal_entries[f'{sensor}_time'].get())
            params[f'{sensor}_time'] = minutes
            params[f'{sensor}_server_time'] = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0

            files = float(self.goal_entries[f'{sensor}_iat'].get())
            params[f'{sensor}_files'] = files
            params[f'{sensor}_iat_calc'] = 1 / ((files * 12) / self.work_days) if files != 0 else 0

        # Windtalker specific (ensure consistency with single_entries keys)
        params['wt_mean'] = float(self.goal_entries['wt_mean'].get())
        params['wt_dev'] = float(self.goal_entries['wt_dev'].get())

        # New file types (ensure consistency with single_entries keys)
        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        for sensor in new_sensor_types:
            params[f'{sensor}_name'] = self.goal_entries[f'{sensor}_name'].get()
            minutes = float(self.goal_entries[f'{sensor}_time'].get())
            params[f'{sensor}_time'] = minutes
            params[f'{sensor}_server_time'] = (1 / ((1 / minutes) * (60 / 1) * (8 / 1))) if minutes != 0 else 0

            # This line was using self.goal_entries['iat'].get() which is wrong; it should be sensor-specific.
            # Assuming you meant self.goal_entries[f'{sensor}_iat'].get() as per other sensors.
            files = float(self.goal_entries[f'{sensor}_iat'].get()) 
            params[f'{sensor}_files'] = files
            params[f'{sensor}_iat_calc'] = 1 / ((files * 12) / self.work_days) if files != 0 else 0

        params['startLoop'] = 0 # This seems like a fixed value

        return params


    def update_widget_states(self):
        mode = self.simulation_mode.get()
        
        # Determine states based on mode
        goal_state = tk.NORMAL if mode == "goal" else tk.DISABLED
        single_state = tk.NORMAL if mode == "single" else tk.DISABLED

        # Iterate through all entry widgets and set their state based on the current mode
        # This is more robust than hardcoding lists of widgets
        for key, var in self.single_entries.items():
            # Skip keys that are part of nested structures (e.g., 'co_iat', 'co_time')
            # and handle them separately below or ensure they are directly mapped to entries
            if '_iat' in key or '_time' in key or '_mean' in key or '_dev' in key or '_name' in key:
                continue
            
            # Find the corresponding widget. This requires a mapping or consistent naming.
            # For simplicity, I'll assume a direct mapping from single_entries keys to widget names
            # You might need to adjust this if your widget names are not directly derived from keys
            widget_name = key + '_entry' # Example: simFileName_entry for simFileName
            if hasattr(self, widget_name):
                getattr(self, widget_name).config(state=single_state)
            
        for key, var in self.goal_entries.items():
            if '_iat' in key or '_time' in key or '_mean' in key or '_dev' in key or '_name' in key:
                continue
            widget_name = key + '_entry'
            if hasattr(self, widget_name):
                getattr(self, widget_name).config(state=goal_state)

        # Explicitly set states for widgets that are not directly mapped or are special
        self.simFileName_entry.config(state=tk.NORMAL) # Filename is always active

        # Goal Seek Parameter widgets
        self.goalSeekList_entry.config(state=goal_state)
        self.goalParameter_entry.config(state=goal_state)
        self.waitTimeMax_entry.config(state=goal_state)
        self.processingFteMax_entry.config(state=goal_state)
        self.simMaxIterations_entry.config(state=goal_state)
        self.info_box_button.config(state=goal_state)

        # Simulation Parameter widgets (assuming these are shared across modes, but state changes)
        self.timeWindow_entry.config(state=single_state)
        self.nservers_entry.config(state=single_state)
        self.siprTransferTime_entry.config(state=single_state)
        self.fileGrowth_entry.config(state=single_state)
        self.sensorGrowth_entry.config(state=single_state)
        self.ingestEfficiency_entry.config(state=single_state)

        # Priority Distribution widgets
        self.lowPriority_entry.config(state=single_state)
        self.medPriority_entry.config(state=single_state)
        self.highPriority_entry.config(state=single_state)
        self.vHighPriority_entry.config(state=single_state)

        # Core Sensor widgets
        sensor_types = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        for sensor in sensor_types:
            if sensor in self.sensor_entries_widgets:
                self.sensor_entries_widgets[sensor]['iat'].config(state=single_state)
                self.sensor_entries_widgets[sensor]['time'].config(state=single_state)
        
        # Windtalker specific
        self.meanWt_entry.config(state=single_state)
        self.devWt_entry.config(state=single_state)

        # New Sensor widgets
        new_sensor_types = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        for sensor in new_sensor_types:
            if sensor in self.new_sensor_entries_widgets:
                self.new_sensor_entries_widgets[sensor]['name'].config(state=single_state)
                self.new_sensor_entries_widgets[sensor]['iat'].config(state=single_state)
                self.new_sensor_entries_widgets[sensor]['time'].config(state=single_state)

    def create_widgets(self):
        # This method seems to be empty. If it's not used, it can be removed.
        pass

    def start_simulation(self):
        # This method is called by the "Begin Simulation" button.
        # It should trigger the submission of parameters and then close the window.
        self.submit_entries() # Ensure parameters are saved before closing
        if self.collected_parameters is not None: # Only close if submission was successful
            self.root.destroy()

    def error_box(self):
        messagebox.showerror("Priority Distribution Error",
                             "'Data File Priority Distribution' values must sum to 100%. \nPlease change the values and submit again.")

# Public function to be called from 01simStart.py
def get_simulation_parameters(master_window, mode="single"):
    """
    Launches the simulation input GUI and returns the collected parameters.
    Args:
        master_window: The parent Tkinter window (e.g., root from 01simStart.py).
        mode (str): "single" or "goal" to set the initial simulation mode.
    Returns:
        dict: A dictionary of collected parameter values, or None if cancelled or error.
    """
    gui = SimulationInputGUI(master_window, mode)
    master_window.wait_window(gui.root) # Wait for the Toplevel window to close
    return gui.collected_parameters

# Standalone test block for 02simInput.py
if __name__ == "__main__":
    test_root = tk.Tk()
    test_root.withdraw() # Hide the main root window for standalone testing

    print("Running 02simInput.py in standalone test mode (single)...")
    params_single = get_simulation_parameters(test_root, mode="single")
    if params_single:
        print("Collected Single Parameters:")
        for key, value in params_single.items():
            print(f"  {key}: {value}")
    else:
        print("Single parameter collection cancelled or failed.")

    # You could add another test for "goal" mode here if desired
    # print("\nRunning 02simInput.py in standalone test mode (goal)...")
    # params_goal = get_simulation_parameters(test_root, mode="goal")
    # if params_goal:
    #     print("Collected Goal Parameters:")
    #     for key, value in params_goal.items():
    #         print(f"  {key}: {value}")
    # else:
    #     print("Goal parameter collection cancelled or failed.")

    test_root.destroy()
