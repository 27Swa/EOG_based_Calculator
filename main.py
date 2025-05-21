import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal
import pandas as pd
import sys
from tkinter import ttk
import handlingfunctions as hd
# === EOG Classifier Configuration ===
MODEL_PATH = "Morphological Feature model.joblib"
SAMPLE_RATE = 176  # Matches test script
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 20
ORDER = 2

label_map = {
    0: 'up',
    1: 'down',
    2: 'right',
    3: 'left',
    4: 'blink'
}

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    messagebox.showerror("Error", f"Model file '{MODEL_PATH}' not found. Please ensure the model file is in the correct location.")
    sys.exit(1)

def load_and_process_file(filepath):
    with open(filepath, 'r') as f:
        raw = f.read().strip().replace('\n', ',')
    signal = np.array([float(x) for x in raw.split(',') if x.strip()])
    return hd.butter_bandpass_filter(signal, LOW_CUTOFF, HIGH_CUTOFF, SAMPLE_RATE, ORDER)

class EOGCalculatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EOG Calculator")
        self.expression = ""
        self.selector_pos = (4, 4)
        self.buttons_map = {}
        self.processing = False
        self.movement_history = []
        self.max_history = 5  # Keep last 5 movements
        
        # Movement history display
        self.history_var = tk.StringVar(value="Movement History: ")
        self.history_label = None
        self.setup_ui()

    def setup_ui(self):
        self.root.minsize(700, 600)
        self.root.configure(bg='#2c3e50')
        
        for i in range(10):  # 9 for main grid + 1 for history
            self.root.grid_rowconfigure(i, weight=1)
            self.root.grid_columnconfigure(i, weight=1)

        # Movement history label
        self.history_label = tk.Label(self.root, textvariable=self.history_var,
                                    font=("Helvetica", 12),
                                    fg="#ffffff", bg='#2c3e50', pady=5)
        self.history_label.grid(row=9, column=0, columnspan=9,
                              sticky="w", padx=20, pady=(0, 10))

        self.status_label = tk.Label(self.root, text="Ready", font=("Helvetica", 20, "bold"),
                                   fg="#ffffff", bg='#2c3e50', pady=10)
        self.status_label.grid(row=8, column=0, columnspan=3, sticky="w", padx=20)

        # Display with dark theme
        self.display = tk.Entry(self.root, font=("Helvetica", 28), bd=0,
                              relief=tk.FLAT, justify='right',
                              bg='#34495e', fg='white', insertbackground='white')
        self.display.grid(row=0, column=0, columnspan=9, pady=20, 
                         padx=20, sticky="nsew", ipady=15)
        
        # Add a frame for buttons with padding
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.grid(row=1, column=0, rowspan=7, columnspan=9, 
                         padx=20, pady=10, sticky="nsew")
        for i in range(9):
            button_frame.grid_rowconfigure(i, weight=1)
            button_frame.grid_columnconfigure(i, weight=1)
        
        self.layout_buttons(button_frame)

        # Sample button with dark theme
        sample_btn = tk.Button(self.root, text="Add Sample", 
                             font=("Helvetica", 14, "bold"),
                             bg="#2ecc71", fg="white", 
                             relief=tk.FLAT,
                             command=self.run_prediction_pipeline)
        sample_btn.grid(row=8, column=3, columnspan=3, 
                       sticky="nsew", padx=20, pady=(0, 20))
        
        # Bind hover effects for sample button
        sample_btn.bind('<Enter>', lambda e: e.widget.configure(bg='#27ae60'))
        sample_btn.bind('<Leave>', lambda e: e.widget.configure(bg='#2ecc71'))

        self.update_selector()

    def layout_buttons(self, parent):
        button_layout = {
            '.': (4, 4), 'C': (1, 4), 'E': (2, 5),
            '8': (2, 3), '9': (2, 4),
            '1': (3, 2), '2': (4, 2), '3': (5, 2), '0': (4, 1),
            '4': (3, 6), '5': (4, 6), '6': (5, 6), '7': (4, 7),
            '/': (6, 3), '+': (6, 4), '-': (6, 5), '*': (7, 4)
        }

        for val, (r, c) in button_layout.items():
            bg_color = {
                'E': '#e74c3c',
                'C': '#f39c12',
                '+': '#7f8c8d',
                '-': '#7f8c8d',
                '*': '#7f8c8d',
                '/': '#7f8c8d'
            }.get(val, '#95a5a6')

            btn = tk.Button(parent, text="●" if val == "." else val,
                          font=("Helvetica", 18, "bold"),
                          width=3, height=1,
                          bg=bg_color, fg='#2c3e50',
                          relief=tk.FLAT,
                          state='disabled')
            
            btn._command = lambda v=val: self.on_click(v)
            btn.grid(row=r, column=c, padx=5, pady=5)
            self.buttons_map[(r, c)] = btn
            
            self.setup_button_hover(btn, val, r, c)

    def setup_button_hover(self, btn, val, r, c):
        def on_enter(e):
            if val == 'E': e.widget.configure(bg='#c0392b')
            elif val == 'C': e.widget.configure(bg='#d35400')
            elif val in ['+', '-', '*', '/']: e.widget.configure(bg='#6c7a7a')
            else: e.widget.configure(bg='#7f8c8d')

        def on_leave(e):
            if (r, c) == self.selector_pos:
                e.widget.configure(bg='#3498db')
            else:
                if val == 'E': e.widget.configure(bg='#e74c3c')
                elif val == 'C': e.widget.configure(bg='#f39c12')
                elif val in ['+', '-', '*', '/']: e.widget.configure(bg='#7f8c8d')
                else: e.widget.configure(bg='#95a5a6')

        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)

    def update_selector(self):
        for pos, btn in self.buttons_map.items():
            btn.config(bg='#3498db' if pos == self.selector_pos else self.get_default_color(btn.cget('text')))

    def get_default_color(self, val):
        return {
            'E': '#e74c3c',
            'C': '#f39c12',
            '+': '#7f8c8d',
            '-': '#7f8c8d',
            '*': '#7f8c8d',
            '/': '#7f8c8d'
        }.get(val, '#95a5a6')

    def update_movement_history(self, movement):
        """Update movement history and display"""
        # If movement is blink, add the automatic center movement
        if movement == "blink":
            self.movement_history.append("blink → center")
        else:
            self.movement_history.append(movement)
            
        if len(self.movement_history) > self.max_history:
            self.movement_history.pop(0)
        
        history_text = "Movement History: " + " → ".join(self.movement_history)
        self.history_var.set(history_text)

    def move_selector(self, direction):
        r, c = self.selector_pos
        positions = list(self.buttons_map.keys())
        
        # Find valid positions in the requested direction
        valid_positions = []
        if direction == "up":
            # First try direct up
            valid_positions = [(row, col) for (row, col) in positions if row < r and col == c]
            # If no direct up position, try diagonal
            if not valid_positions:
                valid_positions = [(row, col) for (row, col) in positions if row < r and abs(col - c) <= 1]
        elif direction == "down":
            # First try direct down
            valid_positions = [(row, col) for (row, col) in positions if row > r and col == c]
            # If no direct down position, try diagonal
            if not valid_positions:
                valid_positions = [(row, col) for (row, col) in positions if row > r and abs(col - c) <= 1]
        elif direction == "left":
            # First try direct left
            valid_positions = [(row, col) for (row, col) in positions if col < c and row == r]
            # If no direct left position, try diagonal
            if not valid_positions:
                valid_positions = [(row, col) for (row, col) in positions if col < c and abs(row - r) <= 1]
        elif direction == "right":
            # First try direct right
            valid_positions = [(row, col) for (row, col) in positions if col > c and row == r]
            # If no direct right position, try diagonal
            if not valid_positions:
                valid_positions = [(row, col) for (row, col) in positions if col > c and abs(row - r) <= 1]
        
        if valid_positions:
            # Find the nearest position
            if direction in ["up", "down"]:
                new_pos = min(valid_positions, key=lambda pos: abs(pos[0] - r))
            else:  # left or right
                new_pos = min(valid_positions, key=lambda pos: abs(pos[1] - c))
            
            self.selector_pos = new_pos
            self.update_selector()
            self.update_status(f"Moved {direction}", "blue")
            self.update_movement_history(direction)
        else:
            self.update_status(f"Cannot move {direction}", "orange")

    def trigger_selection(self):
        if self.selector_pos in self.buttons_map:
            btn = self.buttons_map[self.selector_pos]
            if hasattr(btn, '_command'):
                btn._command()
            # Update history with blink (which includes centering)
            self.update_movement_history("blink")
            # Return to center
            self.selector_pos = (4, 4)
            self.update_selector()

    def on_click(self, char):
        if char == "C":
            self.expression = ""
            self.display.delete(0, tk.END)
        elif char == "E":
            self.root.quit()
        else:
            self.expression += str(char)
            
            self.display.delete(0, tk.END)
            self.display.insert(tk.END, self.expression)
            
            if len(self.expression) >= 3:
                last_three = self.expression[-3:]
                if (last_three[0].isdigit() and 
                    last_three[1] in ['+', '-', '*', '/'] and 
                    last_three[2].isdigit()):
                    try:
                        result = str(eval(self.expression))
                        self.display.delete(0, tk.END)
                        self.display.insert(tk.END, result)
                        self.expression = result
                    except:
                        messagebox.showerror("Error", "Invalid Expression")

    def update_status(self, message, color="white"):
        colors = {
            "blue": "#3498db",
            "green": "#2ecc71",
            "red": "#ff6b6b",
            "orange": "#f39c12",
            "white": "#ffffff"
        }
        self.status_label.config(text=message, fg=colors.get(color, "#ffffff"))
        self.root.update()

    def run_prediction_pipeline(self):
        if self.processing:
            self.update_status("Already processing a sample...", "orange")
            return

        try:
            self.processing = True
            self.update_status("Select horizontal and vertical signal files...", "blue")
            
            files = filedialog.askopenfilenames(
                title="Select Both Signal Files",
                filetypes=[("Text files", "*.txt")],
                multiple=True
            )
            
            if not files or len(files) != 2:
                raise ValueError("Please select exactly two signal files (horizontal and vertical)")

            h_signal = hd.validate_signal_file(files[0])
            v_signal = hd.validate_signal_file(files[1])

            h_filtered = hd.butter_bandpass_filter(h_signal, LOW_CUTOFF, HIGH_CUTOFF, SAMPLE_RATE, ORDER)
            v_filtered = hd.butter_bandpass_filter(v_signal, LOW_CUTOFF, HIGH_CUTOFF, SAMPLE_RATE, ORDER)

            h_features = hd.extract_morphological_features(h_filtered.reshape(1, -1))
            v_features = hd.extract_morphological_features(v_filtered.reshape(1, -1))

            selected_features = hd.features_selection(h_features,v_features)

            label = hd.prediction(selected_features)

            self.update_status(f"Predicted: {label}", "green")

            if label == "blink":
                self.trigger_selection()
            else:
                self.move_selector(label)

        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            messagebox.showerror("Error", str(e))
        finally:
            self.processing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = EOGCalculatorUI(root)
    root.mainloop()
