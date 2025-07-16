import tkinter as tk
from datetime import datetime

class CircularProgressBar(tk.Canvas):
    """A circular progress bar widget for Tkinter."""
    
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
        self.label_text.set(f"{self._progress}%")

def get_current_date():
    """Return the current date in YYYY-MM-DD format."""
    return datetime.today().strftime("%Y-%m-%d")

def resize_image(img, new_width, new_height):
    """Resize a Tkinter PhotoImage to the specified dimensions."""
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

def work_days_per_year(federal_holidays=0, mean_vacation_days=0, mean_sick_days=0, mean_extended_workdays=0, include_weekends=False):
    """Estimate the number of work days per year."""
    days_per_year = 365
    weeks_per_year = 52
    weekend_days_per_year = 2 * weeks_per_year if not include_weekends else 0
    return days_per_year - weekend_days_per_year - federal_holidays - mean_vacation_days - mean_sick_days + mean_extended_workdays

def main():
    # Test the CircularProgressBar class
    root = tk.Tk()
    progress_bar = CircularProgressBar(root, width=200, height=200)
    progress_bar.pack()
    for i in range(101):
        progress_bar.set_progress(i)
        root.update()
        root.after(10)

    # Test the get_current_date function
    print("Current date:", get_current_date())

    # Test the resize_image function
    # Note: This requires a Tkinter PhotoImage object, which is not created here for simplicity.
    # img = tk.PhotoImage(file="image.png")
    # resized_img = resize_image(img, 100, 100)

    # Test the work_days_per_year function
    print("Work days per year (default):", work_days_per_year())
    print("Work days per year (with parameters):", work_days_per_year(federal_holidays=11, mean_vacation_days=10, mean_sick_days=5, mean_extended_workdays=2, include_weekends=False))

    root.mainloop()

if __name__ == "__main__":
    main()
