import tkinter as tk
import math

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

if __name__ == "__main__":
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