# simInput.py - Enhanced Parameter Input System
# Optimized for maintainability, user experience, and performance

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from PIL import Image, ImageTk
import traceback

# Configure logging for this module
logger = logging.getLogger("PhalanxSimulation.Input")

try:
    from simUtils import work_days_per_year
except ImportError:
    logger.warning("simUtils not available. Using fallback work_days_per_year calculation.")
    def work_days_per_year(**kwargs):
        return 249  # Standard work days fallback

@dataclass
class SensorConfig:
    """Configuration for a sensor type with validation rules."""
    name: str
    display_name: str
    default_iat: float
    default_time: float
    iat_range: Tuple[float, float] = (0.0, 1000.0)
    time_range: Tuple[float, float] = (1.0, 300.0)
    description: str = ""
    has_custom_name: bool = False
    has_mean_dev: bool = False

@dataclass
class ParameterTemplate:
    """Template for common parameter sets."""
    name: str
    description: str
    parameters: Dict[str, Any]

class EnhancedParameterValidator:
    """Enhanced parameter validation with detailed error reporting."""
    
    @staticmethod
    def validate_numeric_range(value: str, min_val: float, max_val: float, param_name: str) -> Tuple[bool, str, float]:
        """Validate numeric input within range."""
        try:
            num_val = float(value)
            if min_val <= num_val <= max_val:
                return True, "", num_val
            else:
                return False, f"{param_name} must be between {min_val} and {max_val}", num_val
        except ValueError:
            return False, f"{param_name} must be a valid number", 0.0
    
    @staticmethod
    def validate_priority_distribution(low: str, med: str, high: str, vhigh: str) -> Tuple[bool, str]:
        """Validate priority distribution sums to 100%."""
        try:
            values = [float(x) for x in [low, med, high, vhigh]]
            total = sum(values)
            if abs(total - 100.0) < 0.01:  # Allow small floating point errors
                return True, ""
            else:
                return False, f"Priority distribution values must sum to 100%. Current sum: {total:.2f}%"
        except ValueError:
            return False, "All priority distribution values must be valid numbers"
    
    @staticmethod
    def validate_file_name(filename: str) -> Tuple[bool, str]:
        """Validate simulation file name."""
        if not filename or not filename.strip():
            return False, "File name cannot be empty"
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
        for char in invalid_chars:
            if char in filename:
                return False, f"File name cannot contain: {' '.join(invalid_chars)}"
        
        return True, ""

class ParameterTemplateManager:
    """Manages parameter templates for quick setup."""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> List[ParameterTemplate]:
        """Load default parameter templates."""
        return [
            ParameterTemplate(
                name="Light Load Scenario",
                description="Minimal sensor activity for testing",
                parameters={
                    'nservers': '2',
                    'co_iat': '5', 'dk_iat': '10', 'ma_iat': '2',
                    'nj_iat': '50', 'rs_iat': '5', 'sv_iat': '8',
                    'tg_iat': '3', 'wt_iat': '60'
                }
            ),
            ParameterTemplate(
                name="Standard Operations",
                description="Typical operational parameters",
                parameters={
                    'nservers': '4',
                    'co_iat': '11', 'dk_iat': '20', 'ma_iat': '1',
                    'nj_iat': '130', 'rs_iat': '1', 'sv_iat': '2',
                    'tg_iat': '1', 'wt_iat': '130'
                }
            ),
            ParameterTemplate(
                name="High Load Scenario",
                description="Maximum expected sensor activity",
                parameters={
                    'nservers': '6',
                    'co_iat': '20', 'dk_iat': '30', 'ma_iat': '5',
                    'nj_iat': '200', 'rs_iat': '10', 'sv_iat': '15',
                    'tg_iat': '8', 'wt_iat': '200'
                }
            ),
            ParameterTemplate(
                name="Future Expansion",
                description="Projected parameters with new sensors active",
                parameters={
                    'nservers': '8',
                    'nf1_iat': '5', 'nf2_iat': '3', 'nf3_iat': '4', 'nf4_iat': '6'
                }
            )
        ]
    
    def apply_template(self, template_name: str, entries: Dict[str, tk.StringVar]) -> None:
        """Apply a template to the current parameter entries."""
        template = next((t for t in self.templates if t.name == template_name), None)
        if not template:
            return
        
        for param_name, value in template.parameters.items():
            if param_name in entries:
                entries[param_name].set(value)
    
    def save_current_as_template(self, name: str, description: str, 
                               entries: Dict[str, tk.StringVar]) -> None:
        """Save current parameters as a new template."""
        parameters = {key: var.get() for key, var in entries.items()}
        new_template = ParameterTemplate(name, description, parameters)
        self.templates.append(new_template)

class EnhancedSimulationInputGUI:
    """Enhanced simulation input GUI with improved UX and maintainability."""
    
    def __init__(self, master, mode="single"):
        self.master = master
        self.root = tk.Toplevel(master)
        self.root.title("Phalanx C-sUAS Simulation - Enhanced Parameter Input")
        self.root.geometry("900x700")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Make window modal
        self.root.transient(master)
        self.root.grab_set()
        
        # Initialize components
        self.work_days = work_days_per_year(federal_holidays=0, mean_vacation_days=0, 
                                           mean_extended_workdays=0, include_weekends=False)
        self.validator = EnhancedParameterValidator()
        self.template_manager = ParameterTemplateManager()
        self.collected_parameters = None
        
        # Configuration for sensors
        self.sensor_configs = self._define_sensor_configurations()
        
        # Parameter storage
        self.simulation_mode = tk.StringVar(value=mode)
        self.entries = {}
        self.widgets = {}
        
        # UI setup
        self._load_logo()
        self._create_ui()
        self._setup_validation()
        self._center_window()

        # Force window to be visible and interactive (fixes hanging issue)
        self.root.deiconify()        # Ensure window is not iconified
        self.root.lift()             # Bring window to front  
        self.root.focus_force()      # Force keyboard focus
        self.root.attributes('-topmost', True)  # Temporarily keep on top
        self.root.update()           # Process all pending events
        self.root.update_idletasks() # Update display immediately
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        logger.debug("GUI window visibility fix applied")
        
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)        
        logger.info(f"Enhanced parameter input GUI initialized (mode: {mode})")
    
    def _define_sensor_configurations(self) -> Dict[str, SensorConfig]:
        """Define all sensor configurations in one place for easy maintenance."""
        return {
            # Core sensors
            'co': SensorConfig("co", "COCOM", 11, 60, description="COCOM sensor data files"),
            'dk': SensorConfig("dk", "DK", 20, 60, description="DK sensor data files"),
            'ma': SensorConfig("ma", "MA", 1, 60, description="MA sensor data files"),
            'nj': SensorConfig("nj", "NJ", 130, 60, description="NJ sensor data files"),
            'rs': SensorConfig("rs", "RS", 1, 60, description="RS sensor data files"),
            'sv': SensorConfig("sv", "SV", 2, 30, description="SV sensor data files"),
            'tg': SensorConfig("tg", "TG", 1, 30, description="TG sensor data files"),
            'wt': SensorConfig("wt", "Windtalker", 130, 10, description="Windtalker sensor data files", 
                             has_mean_dev=True),
            
            # New/Future sensors
            'nf1': SensorConfig("nf1", "New Sensor 1", 0, 90, description="Future sensor type 1", 
                              has_custom_name=True),
            'nf2': SensorConfig("nf2", "New Sensor 2", 0, 120, description="Future sensor type 2", 
                              has_custom_name=True),
            'nf3': SensorConfig("nf3", "New Sensor 3", 0, 120, description="Future sensor type 3", 
                              has_custom_name=True),
            'nf4': SensorConfig("nf4", "New Sensor 4", 0, 120, description="Future sensor type 4", 
                              has_custom_name=True),
            'nf5': SensorConfig("nf5", "New Sensor 5", 0, 90, description="Future sensor type 5", 
                              has_custom_name=True),
            'nf6': SensorConfig("nf6", "New Sensor 6", 0, 90, description="Future sensor type 6", 
                              has_custom_name=True),
        }
    
    def _load_logo(self) -> None:
        """Load logo image if available."""
        self.logo_photo = None
        logo_paths = ["./reportResources/phalanxLogoSmall.png", "./phalanxLogoSmall.png"]
        
        for logo_path in logo_paths:
            try:
                if Path(logo_path).exists():
                    image = Image.open(logo_path)
                    self.logo_photo = ImageTk.PhotoImage(image)
                    logger.debug(f"Logo loaded from {logo_path}")
                    break
            except Exception as e:
                logger.debug(f"Failed to load logo from {logo_path}: {e}")
    
    def _create_ui(self) -> None:
        """Create the main UI structure."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title and logo
        self._create_header(main_frame)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create pages
        self._create_mode_selection_page()
        self._create_basic_parameters_page()
        self._create_sensors_page()
        self._create_advanced_page()
        self._create_templates_page()
        self._create_submit_page()
        
        # Status bar
        self._create_status_bar(main_frame)
    
    def _create_header(self, parent) -> None:
        """Create header with logo and title."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        # Logo
        if self.logo_photo:
            logo_label = tk.Label(header_frame, image=self.logo_photo)
            logo_label.grid(row=0, column=0, rowspan=2, padx=(0, 15))
        
        # Title
        title_label = ttk.Label(header_frame, text="Phalanx C-sUAS Simulation", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=1, sticky=tk.W)
        
        subtitle_label = ttk.Label(header_frame, text="Enhanced Parameter Input System", 
                                  font=("Arial", 11))
        subtitle_label.grid(row=1, column=1, sticky=tk.W)
    
    def _create_mode_selection_page(self) -> None:
        """Create simulation mode selection page."""
        page = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(page, text="Simulation Mode")
        
        # Mode selection
        mode_frame = ttk.LabelFrame(page, text="Select Simulation Mode", padding="20")
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Radiobutton(mode_frame, text="Single Simulation", 
                       variable=self.simulation_mode, value="single",
                       command=self._on_mode_change).pack(anchor=tk.W, pady=5)
        
        ttk.Label(mode_frame, text="Run a single simulation with specified parameters",
                 foreground="gray").pack(anchor=tk.W, padx=(20, 0))
        
        ttk.Radiobutton(mode_frame, text="Goal-Seeking Simulation", 
                       variable=self.simulation_mode, value="goal",
                       command=self._on_mode_change).pack(anchor=tk.W, pady=(10, 5))
        
        ttk.Label(mode_frame, text="Automatically adjust parameters to achieve target goals",
                 foreground="gray").pack(anchor=tk.W, padx=(20, 0))
        
        # Instructions
        instructions_frame = ttk.LabelFrame(page, text="Instructions", padding="20")
        instructions_frame.pack(fill=tk.BOTH, expand=True)
        
        instructions = [
            "1. Select your desired simulation mode above",
            "2. Configure basic parameters in the 'Basic Parameters' tab",
            "3. Set sensor parameters in the 'Sensor Configuration' tab",
            "4. Adjust advanced settings if needed in the 'Advanced' tab",
            "5. Use templates for quick parameter setup in the 'Templates' tab",
            "6. Review and submit your parameters in the 'Submit' tab"
        ]
        
        for instruction in instructions:
            ttk.Label(instructions_frame, text=instruction).pack(anchor=tk.W, pady=2)
    
    def _create_basic_parameters_page(self) -> None:
        """Create basic parameters page."""
        page = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(page, text="Basic Parameters")
        
        # Create scrollable frame
        canvas = tk.Canvas(page)
        scrollbar = ttk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Basic parameters
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Simulation Settings", padding="15")
        basic_frame.pack(fill=tk.X, pady=(0, 15))
        
        self._add_parameter_field(basic_frame, "simFileName", "Simulation File Name", "testFile",
                                tooltip="Base name for output files")
        self._add_parameter_field(basic_frame, "timeWindow", "Time Window (Years)", "1",
                                tooltip="Simulation duration in years")
        self._add_parameter_field(basic_frame, "nservers", "Processing Servers (FTEs)", "4",
                                tooltip="Number of full-time equivalent processors")
        self._add_parameter_field(basic_frame, "siprTransfer", "SIPR Transfer Time", "1",
                                tooltip="Time for SIPR to NIPR data transfer")
        
        # Priority distribution
        priority_frame = ttk.LabelFrame(scrollable_frame, text="Data File Priority Distribution (%)", padding="15")
        priority_frame.pack(fill=tk.X, pady=(0, 15))
        
        self._add_parameter_field(priority_frame, "lowPriority", "Low Priority", "0.1")
        self._add_parameter_field(priority_frame, "medPriority", "Medium Priority", "67.4")
        self._add_parameter_field(priority_frame, "highPriority", "High Priority", "30.0")
        self._add_parameter_field(priority_frame, "vHighPriority", "Very High Priority", "2.5")
        
        # Growth parameters (for goal-seeking)
        self.growth_frame = ttk.LabelFrame(scrollable_frame, text="Growth Parameters", padding="15")
        self.growth_frame.pack(fill=tk.X)
        
        self._add_parameter_field(self.growth_frame, "fileGrowth", "File Growth Rate", "0.0",
                                tooltip="Annual growth rate for file volumes")
        self._add_parameter_field(self.growth_frame, "sensorGrowth", "Sensor Growth Rate", "0.0",
                                tooltip="Annual growth rate for sensor count")
        self._add_parameter_field(self.growth_frame, "ingestEfficiency", "Efficiency Improvement", "0.0",
                                tooltip="Annual efficiency improvement rate")
    
    def _create_sensors_page(self) -> None:
        """Create sensors configuration page."""
        page = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(page, text="Sensor Configuration")
        
        # Create scrollable frame
        canvas = tk.Canvas(page)
        scrollbar = ttk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Core sensors
        core_frame = ttk.LabelFrame(scrollable_frame, text="Core Sensors", padding="15")
        core_frame.pack(fill=tk.X, pady=(0, 15))
        
        core_sensors = ['co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt']
        for sensor in core_sensors:
            self._create_sensor_group(core_frame, sensor)
        
        # Future sensors
        future_frame = ttk.LabelFrame(scrollable_frame, text="Future/New Sensors", padding="15")
        future_frame.pack(fill=tk.X)
        
        future_sensors = ['nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6']
        for sensor in future_sensors:
            self._create_sensor_group(future_frame, sensor)
    
    def _create_sensor_group(self, parent, sensor_key: str) -> None:
        """Create a group of widgets for a sensor."""
        config = self.sensor_configs[sensor_key]
        
        # Main frame for this sensor
        sensor_frame = ttk.Frame(parent)
        sensor_frame.pack(fill=tk.X, pady=5)
        
        # Title
        title_frame = ttk.Frame(sensor_frame)
        title_frame.pack(fill=tk.X)
        ttk.Label(title_frame, text=config.display_name, 
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        if config.description:
            ttk.Label(title_frame, text=f"({config.description})", 
                     foreground="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Parameter fields
        params_frame = ttk.Frame(sensor_frame)
        params_frame.pack(fill=tk.X, padx=(20, 0))
        
        # Custom name field for new sensors
        if config.has_custom_name:
            self._add_parameter_field(params_frame, f"{sensor_key}_name", "Custom Name", 
                                    config.display_name, width=20)
        
        # IAT and processing time
        self._add_parameter_field(params_frame, f"{sensor_key}_iat", "Files per Month", 
                                str(config.default_iat), width=10)
        self._add_parameter_field(params_frame, f"{sensor_key}_time", "Processing Time (min)", 
                                str(config.default_time), width=10)
        
        # Special fields for Windtalker
        if config.has_mean_dev:
            self._add_parameter_field(params_frame, "wt_mean", "Mean Value", "30", width=10)
            self._add_parameter_field(params_frame, "wt_dev", "Deviation", "20.1", width=10)
    
    def _create_advanced_page(self) -> None:
        """Create advanced parameters page."""
        page = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(page, text="Advanced")
        
        # Goal-seeking parameters
        self.goal_frame = ttk.LabelFrame(page, text="Goal-Seeking Parameters", padding="15")
        self.goal_frame.pack(fill=tk.X, pady=(0, 15))
        
        self._add_parameter_field(self.goal_frame, "goalParameter", "Goal Parameter", "None")
        self._add_parameter_field(self.goal_frame, "waitTimeMax", "Maximum Wait Time", "1.0")
        self._add_parameter_field(self.goal_frame, "processingFteMax", "Maximum FTEs", "20.0")
        self._add_parameter_field(self.goal_frame, "simMaxIterations", "Maximum Iterations", "20")
        
        # Export/Import settings
        export_frame = ttk.LabelFrame(page, text="Parameter Management", padding="15")
        export_frame.pack(fill=tk.X)
        
        button_frame = ttk.Frame(export_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Export Parameters", 
                  command=self._export_parameters).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Import Parameters", 
                  command=self._import_parameters).pack(side=tk.LEFT)
    
    def _create_templates_page(self) -> None:
        """Create parameter templates page."""
        page = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(page, text="Templates")
        
        # Template selection
        selection_frame = ttk.LabelFrame(page, text="Parameter Templates", padding="15")
        selection_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.template_var = tk.StringVar()
        template_combo = ttk.Combobox(selection_frame, textvariable=self.template_var,
                                     values=[t.name for t in self.template_manager.templates],
                                     state="readonly", width=30)
        template_combo.pack(side=tk.LEFT, padx=(0, 10))
        template_combo.bind('<<ComboboxSelected>>', self._on_template_selected)
        
        ttk.Button(selection_frame, text="Apply Template", 
                  command=self._apply_template).pack(side=tk.LEFT, padx=(0, 10))
        
        # Template description
        self.template_desc = tk.Text(page, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.template_desc.pack(fill=tk.X, pady=(0, 15))
        
        # Save current as template
        save_frame = ttk.LabelFrame(page, text="Save Current Parameters", padding="15")
        save_frame.pack(fill=tk.X)
        
        ttk.Label(save_frame, text="Template Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.new_template_name = tk.StringVar()
        ttk.Entry(save_frame, textvariable=self.new_template_name, width=30).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(save_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.new_template_desc = tk.Text(save_frame, height=3, width=40)
        self.new_template_desc.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Button(save_frame, text="Save as Template", 
                  command=self._save_template).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=10)
    
    def _create_submit_page(self) -> None:
        """Create submit/review page."""
        page = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(page, text="Submit")
        
        # Validation results
        validation_frame = ttk.LabelFrame(page, text="Parameter Validation", padding="15")
        validation_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.validation_text = tk.Text(validation_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.validation_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ttk.Frame(page)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Validate Parameters", 
                  command=self._validate_all_parameters).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Begin Simulation", 
                  command=self._submit_parameters).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", 
                  command=self._cancel).pack(side=tk.RIGHT)
    
    def _create_status_bar(self, parent) -> None:
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _add_parameter_field(self, parent, key: str, label: str, default: str = "", 
                           tooltip: str = "", width: int = 15) -> None:
        """Add a parameter input field with validation."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        # Label
        label_widget = ttk.Label(frame, text=label + ":", width=20)
        label_widget.pack(side=tk.LEFT)
        
        # Entry
        var = tk.StringVar(value=default)
        entry = ttk.Entry(frame, textvariable=var, width=width)
        entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Store references
        self.entries[key] = var
        self.widgets[key] = entry
        
        # Tooltip
        if tooltip:
            self._create_tooltip(entry, tooltip)
    
    def _create_tooltip(self, widget, text: str) -> None:
        """Create tooltip for widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="lightyellow", 
                           relief=tk.SOLID, borderwidth=1, font=("Arial", 8))
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
    
    def _setup_validation(self) -> None:
        """Setup real-time validation."""
        for key, var in self.entries.items():
            var.trace_add("write", lambda *args, k=key: self._validate_field(k))
    
    def _validate_field(self, key: str) -> None:
        """Validate a single field."""
        try:
            value = self.entries[key].get()
            widget = self.widgets[key]
            
            # Reset style
            widget.configure(style="TEntry")
            
            # Perform validation based on field type
            if key.endswith('_iat') or key.endswith('_time'):
                sensor_key = key.split('_')[0]
                if sensor_key in self.sensor_configs:
                    config = self.sensor_configs[sensor_key]
                    if key.endswith('_iat'):
                        valid, msg, _ = self.validator.validate_numeric_range(
                            value, *config.iat_range, f"{config.display_name} Files per Month"
                        )
                    else:
                        valid, msg, _ = self.validator.validate_numeric_range(
                            value, *config.time_range, f"{config.display_name} Processing Time"
                        )
                    
                    if not valid:
                        widget.configure(style="Error.TEntry")
                        self.status_var.set(msg)
                        return
            
            # Clear status if validation passes
            self.status_var.set("Ready")
            
        except Exception as e:
            logger.debug(f"Validation error for {key}: {e}")
    
    def _validate_all_parameters(self) -> None:
        """Validate all parameters and show results."""
        errors = []
        warnings = []
        
        # Validate file name
        filename = self.entries['simFileName'].get()
        valid, msg = self.validator.validate_file_name(filename)
        if not valid:
            errors.append(f"File Name: {msg}")
        
        # Validate priority distribution
        if 'lowPriority' in self.entries:
            valid, msg = self.validator.validate_priority_distribution(
                self.entries['lowPriority'].get(),
                self.entries['medPriority'].get(), 
                self.entries['highPriority'].get(),
                self.entries['vHighPriority'].get()
            )
            if not valid:
                errors.append(f"Priority Distribution: {msg}")
        
        # Validate sensor parameters
        for sensor_key, config in self.sensor_configs.items():
            iat_key = f"{sensor_key}_iat"
            time_key = f"{sensor_key}_time"
            
            if iat_key in self.entries:
                iat_value = self.entries[iat_key].get()
                valid, msg, _ = self.validator.validate_numeric_range(
                    iat_value, *config.iat_range, f"{config.display_name} Files per Month"
                )
                if not valid:
                    errors.append(f"{config.display_name}: {msg}")
                elif float(iat_value) == 0:
                    warnings.append(f"{config.display_name}: Sensor is disabled (0 files per month)")
            
            if time_key in self.entries:
                time_value = self.entries[time_key].get()
                valid, msg, _ = self.validator.validate_numeric_range(
                    time_value, *config.time_range, f"{config.display_name} Processing Time"
                )
                if not valid:
                    errors.append(f"{config.display_name}: {msg}")
        
        # Display results
        self.validation_text.configure(state=tk.NORMAL)
        self.validation_text.delete(1.0, tk.END)
        
        if not errors and not warnings:
            self.validation_text.insert(tk.END, "✓ All parameters are valid!\n", "success")
        
        if errors:
            self.validation_text.insert(tk.END, "❌ ERRORS:\n", "error")
            for error in errors:
                self.validation_text.insert(tk.END, f"  • {error}\n")
            self.validation_text.insert(tk.END, "\n")
        
        if warnings:
            self.validation_text.insert(tk.END, "⚠️ WARNINGS:\n", "warning")
            for warning in warnings:
                self.validation_text.insert(tk.END, f"  • {warning}\n")
        
        self.validation_text.configure(state=tk.DISABLED)
        
        # Configure text tags for styling
        self.validation_text.tag_configure("success", foreground="green")
        self.validation_text.tag_configure("error", foreground="red")
        self.validation_text.tag_configure("warning", foreground="orange")
    
    def _on_mode_change(self) -> None:
        """Handle simulation mode changes."""
        mode = self.simulation_mode.get()
        
        # Show/hide mode-specific elements
        if mode == "goal":
            self.goal_frame.configure(style="TLabelframe")
            self.growth_frame.configure(style="TLabelframe")
        else:
            # You might want to disable these for single mode
            pass
        
        logger.debug(f"Mode changed to: {mode}")
    
    def _on_template_selected(self, event=None) -> None:
        """Handle template selection."""
        template_name = self.template_var.get()
        template = next((t for t in self.template_manager.templates if t.name == template_name), None)
        
        if template:
            self.template_desc.configure(state=tk.NORMAL)
            self.template_desc.delete(1.0, tk.END)
            self.template_desc.insert(tk.END, template.description)
            self.template_desc.configure(state=tk.DISABLED)
    
    def _apply_template(self) -> None:
        """Apply selected template."""
        template_name = self.template_var.get()
        if template_name:
            self.template_manager.apply_template(template_name, self.entries)
            self.status_var.set(f"Applied template: {template_name}")
    
    def _save_template(self) -> None:
        """Save current parameters as template."""
        name = self.new_template_name.get().strip()
        description = self.new_template_desc.get(1.0, tk.END).strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter a template name")
            return
        
        self.template_manager.save_current_as_template(name, description, self.entries)
        self.status_var.set(f"Saved template: {name}")
        
        # Update template list
        template_combo = None
        for widget in self.notebook.winfo_children():
            # Find the combobox in templates page
            pass  # Implementation would update the combobox values
    
    def _export_parameters(self) -> None:
        """Export current parameters to JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Parameters"
        )
        
        if filename:
            try:
                params = {key: var.get() for key, var in self.entries.items()}
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=2)
                self.status_var.set(f"Parameters exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export parameters: {e}")
    
    def _import_parameters(self) -> None:
        """Import parameters from JSON file."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Parameters"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                for key, value in params.items():
                    if key in self.entries:
                        self.entries[key].set(str(value))
                
                self.status_var.set(f"Parameters imported from {filename}")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import parameters: {e}")
    
    def _submit_parameters(self) -> None:
        """Submit parameters and close GUI."""
        # Validate first
        self._validate_all_parameters()
        
        # Check for errors (simple check)
        validation_content = self.validation_text.get(1.0, tk.END)
        if "❌ ERRORS:" in validation_content:
            messagebox.showerror("Validation Error", 
                               "Please fix all validation errors before proceeding.")
            return
        
        try:
            # Collect all parameters
            params = {}
            for key, var in self.entries.items():
                value = var.get().strip()
                
                # Convert numeric values
                if key in ['timeWindow', 'nservers', 'siprTransfer', 'waitTimeMax', 
                          'processingFteMax', 'simMaxIterations'] or \
                   key.endswith('_iat') or key.endswith('_time') or \
                   'Priority' in key or 'Growth' in key or key in ['wt_mean', 'wt_dev']:
                    try:
                        params[key] = float(value) if value else 0.0
                    except ValueError:
                        params[key] = 0.0
                else:
                    params[key] = value
            
            # Add mode and work days
            params['simulation_mode'] = self.simulation_mode.get()
            params['work_days_per_year'] = self.work_days
            
            self.collected_parameters = params
            logger.info(f"Parameters collected successfully ({len(params)} parameters)")
            
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error collecting parameters: {e}")
            messagebox.showerror("Error", f"Failed to collect parameters: {e}")
    
    def _cancel(self) -> None:
        """Cancel parameter input."""
        self.collected_parameters = None
        self.root.destroy()
    
    def _center_window(self) -> None:
        """Center the window on screen."""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

# Configure custom styles
def setup_styles():
    """Setup custom styles for enhanced appearance."""
    style = ttk.Style()
    style.configure("Error.TEntry", fieldbackground="lightcoral")

# Public interface functions
def get_simulation_parameters(master_window, mode="single"):
    """
    Enhanced public interface for getting simulation parameters.
    
    Args:
        master_window: Parent Tkinter window
        mode: "single" or "goal" simulation mode
        
    Returns:
        Dict of parameters or None if cancelled
    """
    setup_styles()
    
    gui = EnhancedSimulationInputGUI(master_window, mode)
    master_window.wait_window(gui.root)
    
    return gui.collected_parameters

# Enhanced standalone testing
if __name__ == "__main__":
    print("="*70)
    print("Enhanced simInput.py - Standalone Testing")
    print("="*70)
    
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    def test_parameter_validation():
        """Test parameter validation."""
        print("\n1. Testing parameter validation...")
        validator = EnhancedParameterValidator()
        
        # Test numeric validation
        valid, msg, val = validator.validate_numeric_range("5.0", 0, 10, "Test Param")
        assert valid == True
        assert val == 5.0
        
        valid, msg, val = validator.validate_numeric_range("15", 0, 10, "Test Param")
        assert valid == False
        
        # Test priority distribution
        valid, msg = validator.validate_priority_distribution("10", "70", "15", "5")
        assert valid == True
        
        valid, msg = validator.validate_priority_distribution("10", "70", "15", "10")
        assert valid == False
        
        print("✓ Parameter validation tests passed")
    
    def test_sensor_configuration():
        """Test sensor configuration system."""
        print("\n2. Testing sensor configuration...")
        gui_mock = EnhancedSimulationInputGUI.__new__(EnhancedSimulationInputGUI)
        configs = gui_mock._define_sensor_configurations()
        
        assert 'co' in configs
        assert 'nf1' in configs
        assert configs['co'].display_name == "COCOM"
        assert configs['nf1'].has_custom_name == True
        
        print("✓ Sensor configuration tests passed")
    
    def test_template_manager():
        """Test template management."""
        print("\n3. Testing template manager...")
        manager = ParameterTemplateManager()
        
        assert len(manager.templates) > 0
        assert any(t.name == "Standard Operations" for t in manager.templates)
        
        print("✓ Template manager tests passed")
    
    def test_gui_creation():
        """Test GUI creation."""
        print("\n4. Testing GUI creation...")
        try:
            test_root = tk.Tk()
            test_root.withdraw()
            
            gui = EnhancedSimulationInputGUI(test_root, mode="single")
            assert gui.simulation_mode.get() == "single"
            assert len(gui.entries) > 0
            
            gui.root.destroy()
            test_root.destroy()
            print("✓ GUI creation tests passed")
        except Exception as e:
            print(f"✗ GUI creation test failed: {e}")
    
    # Run tests
    test_parameter_validation()
    test_sensor_configuration()
    test_template_manager()
    test_gui_creation()
    
    print("\n" + "="*70)
    print("All enhanced simInput.py tests completed!")
    print("="*70)
    print("\nNew features added:")
    print("• Enhanced parameter validation with real-time feedback")
    print("• Parameter templates for quick setup")
    print("• Import/export functionality")
    print("• Improved tooltips and user guidance")
    print("• Better error handling and status reporting")
    print("• Streamlined sensor configuration")
    print("• Enhanced visual design")
    
    print("\nTo test the full GUI:")
    print("python simInput.py")