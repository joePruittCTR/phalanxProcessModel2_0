import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, file_path, sim_file_name):
        self.file_path = file_path
        self.sim_file_name = sim_file_name

    def set_plot_properties(self, ax, plot_xlabel, plot_ylabel, plot_title):
        ax.set_facecolor("silver")
        ax.set_xlabel(plot_xlabel, color="whitesmoke")
        ax.set_ylabel(plot_ylabel, color="whitesmoke")
        ax.set_title(plot_title, color="snow")
        ax.tick_params(labelcolor="white") # Sets tick label color

    def save_plot(self, plot_title):
        plt.savefig(self.file_path + self.sim_file_name + "_" + plot_title.replace(" ", "_") + ".png") # Replaced spaces for filename safety
        plt.close()

    def create_plot(self, plot_type, file_queue, plot_xlabel="", plot_ylabel="", plot_title="", plot_color=""):
        fig, ax = plt.subplots(facecolor="gray")
        self.set_plot_properties(ax, plot_xlabel, plot_ylabel, plot_title)

        if plot_type == "scatter":
            # file_queue is expected to be a list of two lists: [[x_values], [y_values]]
            if len(file_queue) == 2 and isinstance(file_queue[0], list) and isinstance(file_queue[1], list):
                ax.scatter(file_queue[0], file_queue[1], color=plot_color)
            else:
                print(f"Warning: Scatter plot expects file_queue in format [[x_values], [y_values]], got {type(file_queue)}")
        elif plot_type == "hist":
            ax.hist(file_queue, color=plot_color, density=True, histtype="stepfilled", align="left")
        elif plot_type == "box":
            # file_queue is expected to be a single list of numerical data
            if isinstance(file_queue, list) or isinstance(file_queue, np.ndarray):
                q1, median, q3 = np.percentile(file_queue, [25, 50, 75])
                ax.text(1.1, q1, f"Q1: {q1:.2f}", color=plot_color)
                ax.text(1.1, median, f"Median: {median:.2f}", color=plot_color)
                ax.text(1.1, q3, f"Q3: {q3:.2f}", color=plot_color)
                ax.boxplot(file_queue, patch_artist=True, manage_ticks=True, showfliers=True, notch=False, positions=[1],
                           flierprops={"markerfacecolor": plot_color, "markeredgecolor": plot_color}, # Fixed flierprops
                           medianprops={"color": "black"},
                           boxprops={"facecolor": plot_color, "edgecolor": plot_color},
                           whiskerprops={"color": plot_color}, capprops={"color": plot_color})
            else:
                print(f"Warning: Box plot expects a list or numpy array, got {type(file_queue)}")
        elif plot_type == "stair":
            ax.stairs(file_queue, color=plot_color)
        elif plot_type == "line":
            ax.plot(file_queue, color=plot_color)
        elif plot_type == "multiline":
            # file_queue is expected to be a list of lists, and plot_color a list of colors
            if isinstance(file_queue, list) and isinstance(plot_color, list) and len(file_queue) == len(plot_color):
                for i in range(len(plot_color)):
                    ax.plot(file_queue[i], color=plot_color[i])
            else:
                print(f"Warning: Multiline plot expects file_queue as list of lists and plot_color as list of colors of same length.")
        else:
            print(f"Error: Unknown plot type '{plot_type}'")

        self.save_plot(plot_title)

def main():
    # Example usage:
    import os
    current_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(current_dir, "plots") # Use os.path.join for cross-platform compatibility
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot = Plot(plot_dir + os.sep, "sim_file_name") # Add os.sep for path separator

    # Test cases with more appropriate data structures
    print("Creating plots in:", plot_dir)
    plot.create_plot("scatter", [[1, 2, 3], [4, 5, 6]], "X-axis", "Y-axis", "Scatter Plot Test", "blue")
    plot.create_plot("hist", [1, 1, 2, 3, 3, 3, 4, 5, 5], "Value", "Density", "Histogram Test", "red")
    plot.create_plot("box", [1, 2, 3, 4, 5, 10, 0], "", "Value", "Box Plot Test", "green") # Added outliers
    plot.create_plot("stair", [1, 2, 3, 4, 5], "Step", "Value", "Stair Plot Test", "purple")
    plot.create_plot("line", [1, 2, 3, 4, 5], "Time", "Value", "Line Plot Test", "orange")
    plot.create_plot("multiline", [[1, 2, 3], [4, 5, 6]], "Time", "Value", "Multi-Line Plot Test", ["blue", "red"])
    print("Plots created successfully (check 'plots' directory).")

if __name__ == "__main__":
    main()
