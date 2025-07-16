# Global Imports
import tkinter as tk
from datetime import datetime
import pandas as pd
import numpy as np

# Local Imports
from localModules.archive.plottingModule import systemBoxPlot

class FileStatistics:
    def __init__(self, file_name, time_window, plot_color):
        self.file_name = file_name
        self.time_window = time_window
        self.plot_color = plot_color
        self.file_path = "./data/"

    def get_file_stats(self):
        df = pd.read_csv(self.file_path + self.file_name + ".csv")
        df.drop(axis=0, index=[0, max(df.index)], inplace=True)
        total_files = len(df["fileNum"])
        old_month_index = 1
        month_value = 20.75
        month_files = []
        stat_months = np.ceil(self.time_window / month_value)
        i_max = stat_months.astype(int)
        for i in range(0, i_max):
            month_number = month_value * (i + 1)
            df["diff"] = abs(df["timeStep"] - month_number)
            if df["diff"].empty:
                temp_val = 0
                month_files.append(temp_val)
            else:
                month_index = df["diff"].idxmin()
                temp_val = float(df.iloc[old_month_index:month_index]["fileNum"].count())
                month_files.append(temp_val)
                old_month_index = month_index
        min_file = pd.Series(month_files).min()
        max_file = pd.Series(month_files).max()
        median_file = pd.Series(month_files).median()
        mean_file = pd.Series(month_files).mean()
        std_dev_file = pd.Series(month_files).std()
        systemBoxPlot(month_files, "Files", "Files_per_Month", self.plot_color, self.file_name + "_Box")
        return [total_files, min_file, max_file, median_file, mean_file, std_dev_file]

class QueueStatistics:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_path = "./data/"

    def get_queue_stats(self):
        df = pd.read_csv(self.file_path + self.file_name + ".csv")
        df.drop(axis=0, index=[0, 1, max(df.index)], inplace=True)
        s = pd.Series(df["queueLength"])
        min_queue = s.min()
        max_queue = s.max()
        median_queue = s.median()
        mean_queue = s.mean()
        std_dev_queue = s.std()
        return [min_queue, max_queue, median_queue, mean_queue, std_dev_queue]

class StayStatistics:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_path = "./data/"

    def get_stay_stats(self):
        df = pd.read_csv(self.file_path + self.file_name + ".csv")
        df.drop(axis=0, index=[0, max(df.index)], inplace=True)
        s = pd.Series(df["stayLength"])
        min_stay = s.min()
        max_stay = s.max()
        median_stay = s.median()
        mean_stay = s.mean()
        std_dev_stay = s.std()
        return [min_stay, max_stay, median_stay, mean_stay, std_dev_stay]

class SimulationStatistics:
    def __init__(self, sim_count):
        self.sim_count = sim_count
        self.file_path = "./data/"

    def get_mean_stay(self):
        df = pd.read_csv(self.file_path + "SYS_Stay" + str(self.sim_count) + ".csv")
        df.drop(axis=0, index=[0, max(df.index)], inplace=True)
        s = pd.to_numeric(df["stayLength"], errors='coerce')
        mean_stay = s.mean()
        return mean_stay

class CircularProgressBar(tk.Canvas):
    def __init__(self, parent, width=80, height=80, bg="white", fg="blue", progress=0, *args, **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0, *args, **kwargs)
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
    def set_progress(self, progress):
         self._progress = max(0, min(progress, 100))
         self.update_arc()
         self.update_progress_label()
    def update_arc(self):
        angle = 360 * (self._progress / 100)
        self.itemconfigure(self.progress_arc, extent=-angle)
    def update_progress_label(self):
         self.label_text.set(f"{self._progress}%")

def get_current_date():
    x = datetime.today()
    return x.strftime("%Y-%m-%d")

def resizeImage(img, newWidth, newHeight):
    oldWidth = img.width()
    oldHeight = img.height()
    newPhotoImage = tk.PhotoImage(width=newWidth, height=newHeight)
    for x in range(newWidth):
        for y in range(newHeight):
            xOld = int(x * oldWidth / newWidth)
            yOld = int(y * oldHeight / newHeight)
            rgb = '#%02x%02x%02x' % img.get(xOld, yOld)
            newPhotoImage.put(rgb, (x, y))
    return newPhotoImage

def workDaysPerYear():
    # Estimate work days per year; assumes no vacation or sick days
    # and 11 federal holidays per year
    global workDays
    daysPerYear = 365
    weeksPerYear = 52
    weekendDaysPerYear = 2 * weeksPerYear
    federalHolidays = 11
    workDays = daysPerYear - weekendDaysPerYear - federalHolidays
    return workDays

def main():
    file_stats = FileStatistics("SYS_Files", 12, "blue")
    file_stats_list = file_stats.get_file_stats()
    print(file_stats_list)

    queue_stats = QueueStatistics("SYS_Queue")
    queue_stats_list = queue_stats.get_queue_stats()
    print(queue_stats_list)

    stay_stats = StayStatistics("SYS_Stay")
    stay_stats_list = stay_stats.get_stay_stats()
    print(stay_stats_list)

    sim_stats = SimulationStatistics(1)
    mean_stay = sim_stats.get_mean_stay()
    print(mean_stay)

    current_date = get_current_date()
    print("Current Date: ", current_date)

    work_days = workDaysPerYear()
    print("Work Days per Year: ", work_days)

    root = tk.Tk()
    root.title("Circular Progress Bar Example")
    progress_bar = CircularProgressBar(root, width=100, height=100, fg="green")
    progress_bar.pack(pady=20)
    
    def update_progress():
        current_progress = update_progress.counter % 101
        progress_bar.set_progress(current_progress)
        update_progress.counter += 1
        root.after(50, update_progress)
    update_progress.counter = 0
    update_progress()
    root.mainloop()

if __name__ == "__main__":
    main()