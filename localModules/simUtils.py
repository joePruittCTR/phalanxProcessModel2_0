# simUtils.py - Enhanced Comprehensive Utility Library
# Centralized utilities for the Phalanx C-sUAS Simulation System

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import tkinter as tk
from tkinter import ttk

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure module logger
logger = logging.getLogger("PhalanxSimulation.Utils")

# ================================================================================================
# CONFIGURATION MANAGEMENT
# ================================================================================================

@dataclass
class SimulationConfig:
    """Configuration container for simulation settings."""
    base_directory: str = "./"
    data_directory: str = "./data"
    plots_directory: str = "./plots"
    reports_directory: str = "./reports"
    resources_directory: str = "./reportResources"
    logs_directory: str = "./logs"
    
    # Performance settings
    enable_progress_bars: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_profiling: bool = False
    max_memory_usage_mb: float = 2048.0
    
    # File settings
    auto_create_directories: bool = True
    backup_existing_files: bool = True
    max_backup_files: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'base_directory': self.base_directory,
            'data_directory': self.data_directory,
            'plots_directory': self.plots_directory,
            'reports_directory': self.reports_directory,
            'resources_directory': self.resources_directory,
            'logs_directory': self.logs_directory,
            'enable_progress_bars': self.enable_progress_bars,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'enable_performance_profiling': self.enable_performance_profiling,
            'max_memory_usage_mb': self.max_memory_usage_mb,
            'auto_create_directories': self.auto_create_directories,
            'backup_existing_files': self.backup_existing_files,
            'max_backup_files': self.max_backup_files
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'SimulationConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return cls()
    
    def save_to_file(self, config_path: str) -> bool:
        """Save configuration to JSON file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False

# Global configuration instance
_config = SimulationConfig()

def get_config() -> SimulationConfig:
    """Get the global simulation configuration."""
    return _config

def set_config(config: SimulationConfig) -> None:
    """Set the global simulation configuration."""
    global _config
    _config = config

# ================================================================================================
# PROGRESS TRACKING AND USER FEEDBACK
# ================================================================================================

class EnhancedProgressBar:
    """Enhanced progress bar with multiple backend support."""
    
    def __init__(self, total: int, description: str = "", 
                 use_tqdm: bool = None, use_gui: bool = False):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        
        # Determine which progress bar to use
        if use_tqdm is None:
            use_tqdm = TQDM_AVAILABLE and _config.enable_progress_bars
        
        self.use_tqdm = use_tqdm and not use_gui
        self.use_gui = use_gui
        
        # Initialize appropriate progress bar
        if self.use_tqdm:
            self.pbar = tqdm(
                total=total,
                desc=description,
                unit="item",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        elif self.use_gui:
            self.gui_bar = None  # Will be set by GUI if needed
        else:
            # Fallback to simple console output
            self.pbar = None
            print(f"Starting: {description} (0/{total})")
    
    def update(self, amount: int = 1, description: str = None) -> None:
        """Update progress bar."""
        self.current += amount
        
        if description:
            self.description = description
        
        if self.use_tqdm and self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(amount)
        elif self.use_gui and self.gui_bar:
            progress_percent = (self.current / self.total) * 100
            self.gui_bar.set_progress(progress_percent)
        else:
            # Fallback console output
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%) "
                      f"ETA: {eta:.1f}s", end='', flush=True)
    
    def set_gui_bar(self, gui_bar) -> None:
        """Set GUI progress bar widget."""
        self.gui_bar = gui_bar
    
    def close(self) -> None:
        """Close progress bar."""
        if self.use_tqdm and self.pbar:
            self.pbar.close()
        elif not self.use_gui:
            print()  # New line for console output
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Original CircularProgressBar class (preserved for backward compatibility)
class CircularProgressBar(tk.Canvas):
    """A circular progress bar widget for Tkinter (original implementation preserved)."""
    
    def __init__(self, parent, width=80, height=80, bg="white", fg="blue", progress=0):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.radius = min(self.center_x, self.center_y) - 10
        self.bg = bg
        self.fg = fg
        self._progress = progress
        
        self.create_oval(self.center_x - self.radius, self.center_y - self.radius,
                         self.center_x + self.radius, self.center_y + self.radius,
                         outline="gray", width=2)
        self.progress_arc = self.create_arc(self.center_x - self.radius, self.center_y - self.radius,
                                            self.center_x + self.radius, self.center_y + self.radius,
                                            start=90, extent=0, style="arc", outline=self.fg, width=3)
        self.label_text = tk.StringVar()
        self.label = tk.Label(self, textvariable=self.label_text, bg=self.bg)
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        self.update_progress_label()
        self.update_arc()

    def set_progress(self, progress):
        """Set the progress value and update the widget."""
        self._progress = max(0, min(progress, 100))
        if self._progress < 50:
            self.itemconfigure(self.progress_arc, outline="red")
        elif self._progress < 100:
            self.itemconfigure(self.progress_arc, outline="orange")
        else:
            self.itemconfigure(self.progress_arc, outline="green")
        self.update_arc()
        self.update_progress_label()

    def update_arc(self):
        """Update the progress arc based on the current progress value."""
        angle = 360 * (self._progress / 100)
        self.itemconfigure(self.progress_arc, extent=-angle)

    def update_progress_label(self):
        """Update the progress label text."""
        self.label_text.set(f"{self._progress:.0f}%")

# ================================================================================================
# PERFORMANCE MONITORING
# ================================================================================================

class PerformanceMonitor:
    """Performance monitoring and profiling utilities."""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
        self.memory_samples = []
        self.cpu_samples = []
        
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.checkpoints = {}
        self.memory_samples = []
        self.cpu_samples = []
        
        if PSUTIL_AVAILABLE and _config.enable_memory_monitoring:
            process = psutil.Process()
            self.memory_samples.append({
                'time': 0,
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent()
            })
    
    def checkpoint(self, name: str) -> float:
        """Add a performance checkpoint."""
        if self.start_time is None:
            self.start_monitoring()
        
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        
        # Sample memory if available
        if PSUTIL_AVAILABLE and _config.enable_memory_monitoring:
            try:
                process = psutil.Process()
                self.memory_samples.append({
                    'time': elapsed,
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent()
                })
            except Exception:
                pass  # Ignore sampling errors
        
        logger.debug(f"Performance checkpoint '{name}': {elapsed:.2f}s")
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if self.start_time is None:
            return {}
        
        total_time = time.time() - self.start_time
        summary = {
            'total_time': total_time,
            'checkpoints': self.checkpoints.copy()
        }
        
        if self.memory_samples:
            memory_values = [s['memory_mb'] for s in self.memory_samples]
            summary['memory'] = {
                'peak_mb': max(memory_values),
                'average_mb': sum(memory_values) / len(memory_values),
                'samples': len(memory_values)
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print performance summary."""
        summary = self.get_summary()
        if not summary:
            return
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Time: {summary['total_time']:.2f} seconds")
        
        if summary['checkpoints']:
            print("\nCheckpoints:")
            for name, time_val in summary['checkpoints'].items():
                print(f"  {name}: {time_val:.2f}s")
        
        if 'memory' in summary:
            mem = summary['memory']
            print(f"\nMemory Usage:")
            print(f"  Peak: {mem['peak_mb']:.1f} MB")
            print(f"  Average: {mem['average_mb']:.1f} MB")
        
        print("="*50)

# Global performance monitor
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _performance_monitor

@contextmanager
def performance_context(name: str):
    """Context manager for timing code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"Performance context '{name}': {elapsed:.2f}s")

# ================================================================================================
# FILE AND DIRECTORY UTILITIES
# ================================================================================================

def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
    return dir_path

def setup_simulation_directories(config: SimulationConfig = None) -> Dict[str, Path]:
    """Setup all simulation directories based on configuration."""
    if config is None:
        config = get_config()
    
    directories = {
        'base': ensure_directory_exists(config.base_directory),
        'data': ensure_directory_exists(config.data_directory),
        'plots': ensure_directory_exists(config.plots_directory),
        'reports': ensure_directory_exists(config.reports_directory),
        'resources': ensure_directory_exists(config.resources_directory),
        'logs': ensure_directory_exists(config.logs_directory)
    }
    
    logger.info(f"Simulation directories initialized: {len(directories)} directories")
    return directories

def backup_file_if_exists(file_path: Union[str, Path], max_backups: int = None) -> Optional[Path]:
    """Backup a file if it exists, maintaining version history."""
    if max_backups is None:
        max_backups = get_config().max_backup_files
    
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".{timestamp}{file_path.suffix}.bak")
    
    try:
        # Copy file to backup
        import shutil
        shutil.copy2(file_path, backup_path)
        
        # Clean up old backups
        cleanup_old_backups(file_path, max_backups)
        
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Failed to create backup for {file_path}: {e}")
        return None

def cleanup_old_backups(original_file: Path, max_backups: int) -> None:
    """Clean up old backup files, keeping only the most recent."""
    if max_backups <= 0:
        return
    
    # Find all backup files
    backup_pattern = f"{original_file.stem}.*.{original_file.suffix}.bak"
    backup_files = list(original_file.parent.glob(backup_pattern))
    
    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove excess backups
    for old_backup in backup_files[max_backups:]:
        try:
            old_backup.unlink()
            logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            logger.warning(f"Failed to remove old backup {old_backup}: {e}")

def safe_file_write(file_path: Union[str, Path], content: str, 
                   backup_existing: bool = None) -> bool:
    """Safely write content to file with optional backup."""
    if backup_existing is None:
        backup_existing = get_config().backup_existing_files
    
    file_path = Path(file_path)
    
    try:
        # Backup existing file if requested
        if backup_existing:
            backup_file_if_exists(file_path)
        
        # Ensure directory exists
        ensure_directory_exists(file_path.parent)
        
        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.debug(f"Successfully wrote file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return False

# ================================================================================================
# DATE AND TIME UTILITIES
# ================================================================================================

def get_current_date() -> str:
    """Return the current date in YYYY-MM-DD format (original function preserved)."""
    return datetime.today().strftime("%Y-%m-%d")

def get_current_timestamp() -> str:
    """Return current timestamp in ISO format."""
    return datetime.now().isoformat()

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def work_days_per_year(federal_holidays: int = 0, mean_vacation_days: int = 0, 
                      mean_sick_days: int = 0, mean_extended_workdays: int = 0, 
                      include_weekends: bool = False) -> float:
    """
    Estimate the number of work days per year (original function preserved and enhanced).
    
    Args:
        federal_holidays: Number of federal holidays
        mean_vacation_days: Average vacation days taken
        mean_sick_days: Average sick days taken  
        mean_extended_workdays: Average extended work days
        include_weekends: Whether to include weekends as work days
        
    Returns:
        Number of work days per year
    """
    days_per_year = 365
    weeks_per_year = 52
    weekend_days_per_year = 2 * weeks_per_year if not include_weekends else 0
    
    work_days = (days_per_year - weekend_days_per_year - federal_holidays - 
                mean_vacation_days - mean_sick_days + mean_extended_workdays)
    
    # Ensure reasonable bounds
    work_days = max(200, min(365, work_days))
    
    logger.debug(f"Calculated work days per year: {work_days}")
    return work_days

# ================================================================================================
# IMAGE AND UI UTILITIES
# ================================================================================================

def resize_image(img, new_width: int, new_height: int):
    """
    Resize a Tkinter PhotoImage to the specified dimensions (original function preserved).
    
    Args:
        img: Tkinter PhotoImage object
        new_width: Target width in pixels
        new_height: Target height in pixels
        
    Returns:
        Resized PhotoImage object
    """
    old_width = img.width()
    old_height = img.height()
    new_photo_image = tk.PhotoImage(width=new_width, height=new_height)
    
    for x in range(new_width):
        for y in range(new_height):
            x_old = int(x * old_width / new_width)
            y_old = int(y * old_height / new_height)
            rgb = '#%02x%02x%02x' % img.get(x_old, y_old)
            new_photo_image.put(rgb, (x, y))
    
    return new_photo_image

def load_image_safely(image_path: Union[str, Path], 
                     fallback_size: Tuple[int, int] = (100, 100)) -> Optional[tk.PhotoImage]:
    """Safely load an image with fallback to colored rectangle."""
    try:
        from PIL import Image, ImageTk
        image = Image.open(image_path)
        return ImageTk.PhotoImage(image)
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        
        # Create fallback colored rectangle
        try:
            fallback = tk.PhotoImage(width=fallback_size[0], height=fallback_size[1])
            fallback.put("gray", to=(0, 0, fallback_size[0], fallback_size[1]))
            return fallback
        except Exception:
            return None

def center_window(window, width: int = None, height: int = None) -> None:
    """Center a Tkinter window on the screen."""
    window.update_idletasks()
    
    if width is None:
        width = window.winfo_width()
    if height is None:
        height = window.winfo_height()
    
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    window.geometry(f"{width}x{height}+{x}+{y}")

# ================================================================================================
# MATHEMATICAL AND STATISTICAL UTILITIES
# ================================================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with fallback for division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between minimum and maximum bounds."""
    return max(min_val, min(max_val, value))

def interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Linear interpolation between two points."""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not data:
        return {}
    
    if NUMPY_AVAILABLE:
        arr = np.array(data)
        return {
            'count': len(data),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75))
        }
    else:
        # Fallback implementation without numpy
        sorted_data = sorted(data)
        n = len(data)
        
        mean_val = sum(data) / n
        variance = sum((x - mean_val) ** 2 for x in data) / n
        std_val = variance ** 0.5
        
        return {
            'count': n,
            'mean': mean_val,
            'median': sorted_data[n // 2],
            'std': std_val,
            'min': min(data),
            'max': max(data),
            'q25': sorted_data[n // 4],
            'q75': sorted_data[3 * n // 4]
        }

# ================================================================================================
# VALIDATION AND ERROR HANDLING
# ================================================================================================

def validate_numeric_input(value: Any, min_val: float = None, max_val: float = None,
                          param_name: str = "parameter") -> Tuple[bool, str, float]:
    """
    Validate numeric input with optional range checking.
    
    Returns:
        Tuple of (is_valid, error_message, converted_value)
    """
    try:
        num_val = float(value)
        
        if min_val is not None and num_val < min_val:
            return False, f"{param_name} must be >= {min_val}", num_val
        
        if max_val is not None and num_val > max_val:
            return False, f"{param_name} must be <= {max_val}", num_val
        
        return True, "", num_val
        
    except (ValueError, TypeError):
        return False, f"{param_name} must be a valid number", 0.0

def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Tuple[bool, str]:
    """
    Validate a file path.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            return False, f"File does not exist: {path}"
        
        if not must_exist and path.exists() and not path.is_file():
            return False, f"Path exists but is not a file: {path}"
        
        # Check if directory is writable (for new files)
        if not must_exist:
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return False, f"Cannot create directory {parent_dir}: {e}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid file path: {e}"

@contextmanager
def error_context(operation_name: str, raise_on_error: bool = False):
    """Context manager for consistent error handling."""
    try:
        logger.debug(f"Starting operation: {operation_name}")
        yield
        logger.debug(f"Completed operation: {operation_name}")
    except Exception as e:
        error_msg = f"Error in {operation_name}: {e}"
        logger.error(error_msg)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Traceback for {operation_name}:\n{traceback.format_exc()}")
        
        if raise_on_error:
            raise
        else:
            return None

# ================================================================================================
# LOGGING AND DEBUG UTILITIES
# ================================================================================================

def setup_enhanced_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                          include_performance: bool = False) -> logging.Logger:
    """Setup enhanced logging configuration."""
    # Get or create logger
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
        ensure_directory_exists(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Performance logging (if enabled)
    if include_performance and _config.enable_performance_profiling:
        perf_handler = logging.FileHandler(f"performance_{get_current_date()}.log")
        perf_handler.setFormatter(detailed_formatter)
        perf_handler.setLevel(logging.DEBUG)
        logger.addHandler(perf_handler)
    
    return logger

def log_system_info() -> None:
    """Log system information for debugging."""
    logger.info("="*50)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    if PSUTIL_AVAILABLE:
        logger.info(f"CPU count: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # Log available packages
    packages = []
    if NUMPY_AVAILABLE:
        packages.append(f"numpy {np.__version__}")
    if PANDAS_AVAILABLE:
        packages.append(f"pandas {pd.__version__}")
    if TQDM_AVAILABLE:
        packages.append("tqdm")
    if PSUTIL_AVAILABLE:
        packages.append("psutil")
    
    logger.info(f"Available packages: {', '.join(packages) if packages else 'None'}")
    logger.info("="*50)

# ================================================================================================
# MAIN TESTING AND EXAMPLES
# ================================================================================================

def main():
    """Enhanced standalone testing and examples."""
    print("="*70)
    print("Enhanced simUtils.py - Comprehensive Testing")
    print("="*70)
    
    # Setup logging for testing
    logger = setup_enhanced_logging("INFO", "test_simutils.log")
    log_system_info()
    
    # Test 1: Configuration Management
    print("\n1. Testing Configuration Management...")
    config = SimulationConfig()
    config.enable_progress_bars = True
    config.max_memory_usage_mb = 1024.0
    
    # Save and load config
    config.save_to_file("test_config.json")
    loaded_config = SimulationConfig.load_from_file("test_config.json")
    assert loaded_config.max_memory_usage_mb == 1024.0
    print("✓ Configuration management works correctly")
    
    # Cleanup
    Path("test_config.json").unlink(missing_ok=True)
    
    # Test 2: Progress Bar System
    print("\n2. Testing Progress Bar System...")
    with EnhancedProgressBar(100, "Test Progress", use_tqdm=False) as pbar:
        for i in range(0, 101, 20):
            pbar.update(20, f"Step {i//20 + 1}")
            time.sleep(0.1)
    print("✓ Progress bar system works correctly")
    
    # Test 3: Performance Monitoring
    print("\n3. Testing Performance Monitoring...")
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    
    time.sleep(0.1)
    monitor.checkpoint("Test Checkpoint 1")
    
    time.sleep(0.1)
    monitor.checkpoint("Test Checkpoint 2")
    
    summary = monitor.get_summary()
    assert "Test Checkpoint 1" in summary['checkpoints']
    assert "Test Checkpoint 2" in summary['checkpoints']
    print("✓ Performance monitoring works correctly")
    
    # Test 4: File Operations
    print("\n4. Testing File Operations...")
    test_dir = ensure_directory_exists("./test_files")
    assert test_dir.exists()
    
    # Test safe file write
    test_content = "Test content for simUtils"
    success = safe_file_write(test_dir / "test.txt", test_content)
    assert success
    assert (test_dir / "test.txt").exists()
    
    # Test backup
    backup_path = backup_file_if_exists(test_dir / "test.txt")
    assert backup_path is not None
    assert backup_path.exists()
    
    print("✓ File operations work correctly")
    
    # Cleanup test files
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    # Test 5: Mathematical Utilities
    print("\n5. Testing Mathematical Utilities...")
    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats = calculate_statistics(test_data)
    
    assert stats['mean'] == 5.5
    assert stats['count'] == 10
    assert stats['min'] == 1
    assert stats['max'] == 10
    
    # Test validation
    valid, msg, val = validate_numeric_input("5.5", 0, 10, "test")
    assert valid == True
    assert val == 5.5
    
    valid, msg, val = validate_numeric_input("15", 0, 10, "test")
    assert valid == False
    
    print("✓ Mathematical utilities work correctly")
    
    # Test 6: Original Functions (Backward Compatibility)
    print("\n6. Testing Original Functions...")
    
    # Test original CircularProgressBar
    root = tk.Tk()
    root.withdraw()
    progress_bar = CircularProgressBar(root, width=200, height=200)
    progress_bar.set_progress(75)
    assert progress_bar._progress == 75
    root.destroy()
    
    # Test original utility functions
    current_date = get_current_date()
    assert len(current_date) == 10  # YYYY-MM-DD format
    
    work_days = work_days_per_year(federal_holidays=11, mean_vacation_days=10)
    assert 200 <= work_days <= 365
    
    print("✓ Original functions preserved and working")
    
    # Test 7: Date and Time Utilities
    print("\n7. Testing Date and Time Utilities...")
    
    timestamp = get_current_timestamp()
    assert "T" in timestamp  # ISO format check
    
    duration_str = format_duration(125.5)
    assert "minutes" in duration_str
    
    print("✓ Date and time utilities work correctly")
    
    # Performance Summary
    print("\n8. Performance Summary...")
    monitor.checkpoint("All Tests Complete")
    monitor.print_summary()
    
    # Cleanup
    Path("test_simutils.log").unlink(missing_ok=True)
    
    print("\n" + "="*70)
    print("All enhanced simUtils.py tests completed successfully!")
    print("="*70)
    print("\nKey enhancements added:")
    print("• Configuration management system")
    print("• Enhanced progress tracking (tqdm + GUI)")
    print("• Performance monitoring and profiling")
    print("• Comprehensive file operations with backup")
    print("• Mathematical and statistical utilities")
    print("• Enhanced validation and error handling")
    print("• Improved logging and debugging tools")
    print("• All original functions preserved for compatibility")
    
    print("\nThe module is now ready to serve as the central utility hub!")

if __name__ == "__main__":
    main()