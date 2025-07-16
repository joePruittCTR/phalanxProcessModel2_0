# Global Imports
import matplotlib.pyplot as plt
import numpy as np
import random
#filePath = "./capDevProcessModel/plots/"
filePath = "./plots/"

class Plot:
    def __init__(self, file_path, sim_file_name):
        self.file_path = file_path
        self.sim_file_name = sim_file_name

    def save_plot(self, plot_title):
        plt.savefig(self.file_path + self.sim_file_name + "_" + plot_title + ".png")
        plt.close()

class ScatterPlot(Plot):
    def __init__(self, file_path, sim_file_name, file_queue_x, file_queue_y, plot_xlabel, plot_ylabel, plot_title, plot_color):
        super().__init__(file_path, sim_file_name)
        self.file_queue_x = file_queue_x
        self.file_queue_y = file_queue_y
        self.plot_xlabel = plot_xlabel
        self.plot_ylabel = plot_ylabel
        self.plot_title = plot_title
        self.plot_color = plot_color

    def create_plot(self):
        fig, ax = plt.subplots(facecolor="gray")
        ax.set_facecolor("silver")
        ax.set_xlabel(self.plot_xlabel, color="whitesmoke")
        ax.set_ylabel(self.plot_ylabel, color="whitesmoke")
        ax.set_title(self.plot_title, color="snow")
        ax.scatter(self.file_queue_x, self.file_queue_y, color=self.plot_color)
        ax.tick_params(labelcolor="white")
        self.save_plot(self.plot_title)

class HistPlot(Plot):
    def __init__(self, file_path, sim_file_name, file_queue, plot_xlabel, plot_ylabel, plot_title, plot_color):
        super().__init__(file_path, sim_file_name)
        self.file_queue = file_queue
        self.plot_xlabel = plot_xlabel
        self.plot_ylabel = plot_ylabel
        self.plot_title = plot_title
        self.plot_color = plot_color

    def create_plot(self):
        fig, ax = plt.subplots(facecolor="gray")
        ax.set_facecolor("silver")
        ax.set_xlabel(self.plot_xlabel, color="whitesmoke")
        ax.set_ylabel(self.plot_ylabel, color="whitesmoke")
        ax.set_title(self.plot_title, color="snow")
        ax.hist(self.file_queue, color=self.plot_color, density=True, histtype="stepfilled", align="left")
        ax.tick_params(labelcolor="white")
        self.save_plot(self.plot_title)

class BoxPlot(Plot):
    def __init__(self, file_path, sim_file_name, file_queue, plot_ylabel, plot_title, plot_color):
        super().__init__(file_path, sim_file_name)
        self.file_queue = file_queue
        self.plot_ylabel = plot_ylabel
        self.plot_title = plot_title
        self.plot_color = plot_color

    def create_plot(self):
        fig, ax = plt.subplots(facecolor="gray")
        ax.set_facecolor("silver")
        ax.set_xticks([0, 1])
        ax.set_ylabel(self.plot_ylabel, color="whitesmoke")
        ax.set_title(self.plot_title, color="snow")
        q1, median, q3 = np.percentile(self.file_queue, [25, 50, 75])
        ax.text(1.1, q1, f"Q1: {q1:.2f}", color=self.plot_color)
        ax.text(1.1, median, f"Median: {median:.2f}", color=self.plot_color)
        ax.text(1.1, q3, f"Q3: {q3:.2f}", color=self.plot_color)
        ax.boxplot(self.file_queue, patch_artist=True, manage_ticks=True, showfliers=True, notch=False, positions=[1],
                   flierprops={"color": self.plot_color}, medianprops={"color": "black"},
                   boxprops={"facecolor": self.plot_color, "edgecolor": self.plot_color},
                   whiskerprops={"color": self.plot_color}, capprops={"color": self.plot_color})
        ax.tick_params(labelcolor="white")
        self.save_plot(self.plot_title)

class StairPlot(Plot):
    def __init__(self, file_path, sim_file_name, file_queue, plot_xlabel, plot_ylabel, plot_title, plot_color):
        super().__init__(file_path, sim_file_name)
        self.file_queue = file_queue
        self.plot_xlabel = plot_xlabel
        self.plot_ylabel = plot_ylabel
        self.plot_title = plot_title
        self.plot_color = plot_color

    def create_plot(self):
        fig, ax = plt.subplots(facecolor="gray")
        ax.set_facecolor("silver")
        ax.set_xlabel(self.plot_xlabel, color="whitesmoke")
        ax.set_ylabel(self.plot_ylabel, color="whitesmoke")
        ax.set_title(self.plot_title, color="snow")
        ax.stairs(self.file_queue, color=self.plot_color)
        ax.tick_params(labelcolor="white")
        self.save_plot(self.plot_title)

class LinePlot(Plot):
    def __init__(self, file_path, sim_file_name, file_queue, plot_xlabel, plot_ylabel, plot_title, plot_color):
        super().__init__(file_path, sim_file_name)
        self.file_queue = file_queue
        self.plot_xlabel = plot_xlabel
        self.plot_ylabel = plot_ylabel
        self.plot_title = plot_title
        self.plot_color = plot_color

    def create_plot(self):
        fig, ax = plt.subplots(facecolor="gray")
        ax.set_facecolor("silver")
        ax.set_xlabel(self.plot_xlabel, color="whitesmoke")
        ax.set_ylabel(self.plot_ylabel, color="whitesmoke")
        ax.set_title(self.plot_title, color="snow")
        ax.plot(self.file_queue, color=self.plot_color)
        ax.tick_params(labelcolor="white")
        self.save_plot(self.plot_title)

class MultiLinePlot(Plot):
    def __init__(self, file_path, sim_file_name, file_queue, plot_xlabel, plot_ylabel, plot_title, plot_color):
        super().__init__(file_path, sim_file_name)
        self.file_queue = file_queue
        self.plot_xlabel = plot_xlabel
        self.plot_ylabel = plot_ylabel
        self.plot_title = plot_title
        self.plot_color = plot_color

    def create_plot(self):
        fig, ax = plt.subplots(facecolor="gray")
        ax.set_facecolor("silver")
        ax.set_xlabel(self.plot_xlabel, color="whitesmoke")
        ax.set_ylabel(self.plot_ylabel, color="whitesmoke")
        ax.set_title(self.plot_title, color="snow")
        j = len(self.plot_color)
        for i in range(0, j):
            ax.plot(self.file_queue[[i]], color=self.plot_color[i])
        ax.tick_params(labelcolor="white")
        self.save_plot(self.plot_title)

if __name__ == "__main__":
    sim_file_name = "testPlots"
    file_path = "./capDevProcessModel/plots/"
    file_queue_x = [random.randint(1, 100) for _ in range(10)]
    file_queue_y = [random.randint(1, 100) for _ in range(10)]
    scatter_plot = ScatterPlot(file_path, sim_file_name, file_queue_x, file_queue_y, "Time", "Files", "Random Numbers_Scatter", "red")
    scatter_plot.create_plot()

    file_queue = [random.randint(1, 100) for _ in range(10)]
    hist_plot = HistPlot(file_path, sim_file_name, file_queue, "Time", "Files", "Random Numbers_Hist", "red")
    hist_plot.create_plot()

    file_queue_data = [np.random.rand(50) * 100]
    flier_high = [np.random.rand(10) * 100 + 100]
    flier_low = [np.random.rand(10) * -100]
    file_queue = file_queue_data + flier_high + flier_low
    box_plot = BoxPlot(file_path, sim_file_name, file_queue_data, "Files", "Random Numbers_Box", "red")
    box_plot.create_plot()

    file_queue = [random.randint(1, 100) for _ in range(10)]
    stair_plot = StairPlot(file_path, sim_file_name, file_queue, "Time", "Files", "Random Numbers_Stair", "red")
    stair_plot.create_plot()

    file_queue = [random.randint(1, 100) for _ in range(10)]
    line_plot = LinePlot(file_path, sim_file_name, file_queue, "Time", "Files", "Random Numbers_Line", "red")
    line_plot.create_plot()