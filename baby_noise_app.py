#!/usr/bin/env python3
# Baby-Noise Generator App
# GUI application for the Baby-Noise Generator

import os
import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Import our noise generator module
from noise_generator import (
    NoiseConfig, StreamingNoiseGenerator, NoiseGenerator, 
    load_preset, auto_select_backend, SAMPLE_RATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("baby-noise-app")

# Constants
APP_TITLE = "Baby-Noise Generator"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Documents/BabyNoise")
BUFFER_SIZE = 2048  # Audio buffer size for streaming
UPDATE_INTERVAL = 50  # ms between UI updates


class BabyNoiseApp:
    """GUI application for the Baby-Noise Generator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(800, 600)
        
        # Set up variables
        self.streaming = False
        self.recording = False
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=20)
        self.generator = None
        self.output_path = None
        
        # Load presets
        self.preset_path = os.path.join(os.path.dirname(__file__), "presets.yaml")
        with open(self.preset_path, 'r') as f:
            self.presets = yaml.safe_load(f)['presets']
        
        # Set up UI variables
        self.preset_var = tk.StringVar(value="default")
        self.seed_var = tk.StringVar(value="auto")
        self.white_var = tk.DoubleVar(value=0.4)
        self.pink_var = tk.DoubleVar(value=0.4)
        self.brown_var = tk.DoubleVar(value=0.2)
        self.warmth_var = tk.DoubleVar(value=0.5)  # For UI slider (translates to white-pink-brown mix)
        self.rms_var = tk.DoubleVar(value=-63.0)
        self.duration_var = tk.IntVar(value=600)  # 10 minutes
        self.lfo_var = tk.DoubleVar(value=0.1)
        self.lfo_enabled_var = tk.BooleanVar(value=True)
        
        # Current metrics
        self.current_rms = -100.0
        self.current_peak = -100.0
        self.rms_history = []
        
        # Create UI
        self.create_ui()
        
        # Initialize audio
        self.initialize_audio()
        
        # Load default preset
        self.load_preset("default")
        
        # Update UI every 50ms
        self.root.after(UPDATE_INTERVAL, self.update_ui)
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top section (controls)
        control_frame = ttk.LabelFrame(main_frame, text="Noise Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Preset selection
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT, padx=5)
        preset_combobox = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                       values=list(self.presets.keys()))
        preset_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        preset_combobox.bind("<<ComboboxSelected>>", self.on_preset_change)
        
        # Seed control
        seed_frame = ttk.Frame(control_frame)
        seed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(seed_frame, textvariable=self.seed_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(seed_frame, text="Randomize", command=self.randomize_seed).pack(side=tk.LEFT, padx=5)
        
        # Noise color blend controls
        color_frame = ttk.LabelFrame(control_frame, text="Noise Color")
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Warmth slider (maps to color mix)
        warmth_frame = ttk.Frame(color_frame)
        warmth_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(warmth_frame, text="Warmer").pack(side=tk.LEFT, padx=5)
        warmth_slider = ttk.Scale(warmth_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                 variable=self.warmth_var, command=self.on_warmth_change)
        warmth_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(warmth_frame, text="Brighter").pack(side=tk.LEFT, padx=5)
        
        # Manual color mix frame
        manual_color_frame = ttk.Frame(color_frame)
        manual_color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(manual_color_frame, text="White:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        white_slider = ttk.Scale(manual_color_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                variable=self.white_var, command=self.on_color_change)
        white_slider.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        
        ttk.Label(manual_color_frame, text="Pink:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        pink_slider = ttk.Scale(manual_color_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                               variable=self.pink_var, command=self.on_color_change)
        pink_slider.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        
        ttk.Label(manual_color_frame, text="Brown:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        brown_slider = ttk.Scale(manual_color_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                variable=self.brown_var, command=self.on_color_change)
        brown_slider.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        
        manual_color_frame.columnconfigure(1, weight=1)
        
        # Volume controls
        volume_frame = ttk.LabelFrame(control_frame, text="Volume")
        volume_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(volume_frame, text="Level (dB SPL):").pack(side=tk.LEFT, padx=5)
        rms_slider = ttk.Scale(volume_frame, from_=-70, to=-55, orient=tk.HORIZONTAL, 
                              variable=self.rms_var)
        rms_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Current RMS display
        self.rms_label = ttk.Label(volume_frame, text="Current: -- dB")
        self.rms_label.pack(side=tk.LEFT, padx=5)
        
        # LFO controls
        lfo_frame = ttk.LabelFrame(control_frame, text="Modulation")
        lfo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(lfo_frame, text="Enable gentle modulation", 
                       variable=self.lfo_enabled_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(lfo_frame, text="Rate (Hz):").pack(side=tk.LEFT, padx=5)
        lfo_slider = ttk.Scale(lfo_frame, from_=0.05, to=0.2, orient=tk.HORIZONTAL, 
                              variable=self.lfo_var)
        lfo_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Playback controls
        playback_frame = ttk.LabelFrame(main_frame, text="Playback")
        playback_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Play/Stop buttons
        button_frame = ttk.Frame(playback_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = ttk.Button(button_frame, text="▶ Play", 
                                     command=self.toggle_playback, width=15)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Render controls
        render_frame = ttk.LabelFrame(main_frame, text="Render")
        render_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Duration and filename
        duration_frame = ttk.Frame(render_frame)
        duration_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(duration_frame, text="Duration:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(duration_frame, textvariable=self.duration_var, 
                    values=[300, 600, 1800, 3600, 7200, 36000]).pack(side=tk.LEFT, padx=5)
        ttk.Label(duration_frame, text="seconds").pack(side=tk.LEFT)
        
        # Render button
        self.render_button = ttk.Button(render_frame, text="Render to File...", 
                                       command=self.render_to_file)
        self.render_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        # Visualization area
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.setup_plots()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=5, pady=2)
    
    def setup_plots(self):
        """Set up the visualization plots"""
        self.fig.clear()
        
        # Two plots: spectrum and level meter
        self.spectrum_ax = self.fig.add_subplot(121)
        self.level_ax = self.fig.add_subplot(122)
        
        # Spectrum plot
        self.spectrum_ax.set_title("Noise Spectrum")
        self.spectrum_ax.set_xlabel("Frequency (Hz)")
        self.spectrum_ax.set_ylabel("Amplitude (dB)")
        self.spectrum_ax.set_xscale("log")
        self.spectrum_ax.set_xlim(20, 20000)
        self.spectrum_ax.set_ylim(-30, 0)
        self.spectrum_ax.grid(True)
        
        # Plot spectrum lines (will be updated later)
        x = np.logspace(np.log10(20), np.log10(20000), 100)
        self.white_line, = self.spectrum_ax.plot(x, np.zeros_like(x), label="White")
        self.pink_line, = self.spectrum_ax.plot(x, np.zeros_like(x), label="Pink")
        self.brown_line, = self.spectrum_ax.plot(x, np.zeros_like(x), label="Brown")
        self.mix_line, = self.spectrum_ax.plot(x, np.zeros_like(x), 'k--', linewidth=2, label="Mix")
        self.spectrum_ax.legend()
        
        # Level meter
        self.level_ax.set_title("Level Meter")
        self.level_ax.set_xlabel("Time (s)")
        self.level_ax.set_ylabel("Level (dB)")
        self.level_ax.set_ylim(-70, -50)
        self.level_ax.grid(True)
        
        # Create level history line
        self.level_line, = self.level_ax.plot([], [], 'g-')
        
        # Target level line
        self.target_line, = self.level_ax.plot([], [], 'r--')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def initialize_audio(self):
        """Initialize audio playback"""
        # Check available audio devices
        devices = sd.query_devices()
        logger.info(f"Found {len(devices)} audio devices")
        
        # Use default output device
        self.output_device = sd.default.device[1]
        device_info = sd.query_devices(self.output_device)
        logger.info(f"Using output device: {device_info['name']}")
    
    def audio_callback(self, outdata, frames, time, status):
        """Audio callback for sounddevice"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            if self.audio_queue.empty():
                # Generate more audio if queue is empty
                chunk = self.generator.get_next_chunk(BUFFER_SIZE)
                self.audio_queue.put(chunk)
            
            # Get data from queue
            data = self.audio_queue.get_nowait()
            
            # Update metrics
            self.current_rms = 20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-10)
            self.current_peak = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
            self.rms_history.append(self.current_rms)
            if len(self.rms_history) > 100:
                self.rms_history = self.rms_history[-100:]
            
            # Fill output buffer
            if len(data) < len(outdata):
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
            else:
                outdata[:, 0] = data[:len(outdata)]
                
        except queue.Empty:
            # If queue is empty, fill with zeros
            outdata.fill(0)
            logger.warning("Audio buffer underrun")
    
    def start_streaming(self):
        """Start audio streaming"""
        if self.streaming:
            return
        
        # Create configuration for streaming
        config = self.create_config()
        config.use_gpu = False  # Force CPU for streaming
        
        # Create generator
        self.generator = StreamingNoiseGenerator(config)
        
        # Clear audio queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Pre-fill queue with a few chunks
        for _ in range(5):
            chunk = self.generator.get_next_chunk(BUFFER_SIZE)
            self.audio_queue.put(chunk)
        
        # Start audio stream
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=BUFFER_SIZE
        )
        self.stream.start()
        
        self.streaming = True
        self.play_button.configure(text="⏹ Stop")
        self.status_var.set("Playing...")
    
    def stop_streaming(self):
        """Stop audio streaming"""
        if not self.streaming:
            return
        
        # Stop stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.streaming = False
        self.play_button.configure(text="▶ Play")
        self.status_var.set("Stopped")
        
        # Clear audio queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
    
    def toggle_playback(self):
        """Toggle audio playback"""
        if self.streaming:
            self.stop_streaming()
        else:
            self.start_streaming()
    
    def create_config(self):
        """Create a configuration from current UI state"""
        # Parse seed
        seed_str = self.seed_var.get()
        if seed_str == "auto":
            seed = int(time.time())
        elif seed_str == "random":
            seed = np.random.randint(0, 2**32 - 1)
        else:
            try:
                seed = int(seed_str)
            except ValueError:
                seed = int(time.time())
                self.seed_var.set(str(seed))
        
        # Get color mix
        color_mix = {
            'white': self.white_var.get(),
            'pink': self.pink_var.get(),
            'brown': self.brown_var.get()
        }
        
        # Normalize to sum to 1.0
        total = sum(color_mix.values())
        if total > 0:
            color_mix = {k: v / total for k, v in color_mix.items()}
        else:
            color_mix = {'white': 0.4, 'pink': 0.4, 'brown': 0.2}
        
        # Get LFO rate if enabled
        lfo_rate = self.lfo_var.get() if self.lfo_enabled_var.get() else None
        
        # Create config
        config = NoiseConfig(
            seed=seed,
            duration=self.duration_var.get(),
            color_mix=color_mix,
            rms_target=self.rms_var.get(),
            peak_ceiling=-1.0,
            lfo_rate=lfo_rate,
            sample_rate=SAMPLE_RATE,
            use_gpu=auto_select_backend()
        )
        
        return config
    
    def render_to_file(self):
        """Render noise to a file"""
        if self.streaming:
            messagebox.showwarning("Cannot Render", "Please stop playback before rendering.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        
        # Default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"baby_noise_{timestamp}.wav"
        default_path = os.path.join(DEFAULT_OUTPUT_DIR, default_filename)
        
        # Ask for output path
        output_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_OUTPUT_DIR,
            initialfile=default_filename,
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("FLAC files", "*.flac"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Create config
        config = self.create_config()
        
        # Update status
        self.status_var.set(f"Rendering to {os.path.basename(output_path)}...")
        self.render_button.configure(state=tk.DISABLED)
        self.root.update()
        
        # Create generator
        generator = NoiseGenerator(config)
        
        # Start rendering in a separate thread
        threading.Thread(
            target=self._render_thread,
            args=(generator, output_path),
            daemon=True
        ).start()
    
    def _render_thread(self, generator, output_path):
        """Background thread for rendering"""
        try:
            # Generate to file
            generator.generate_to_file(output_path)
            
            # Update UI from main thread
            self.root.after(0, lambda: self._render_complete(output_path))
        except Exception as e:
            logger.error(f"Error rendering: {e}")
            self.root.after(0, lambda: self._render_error(str(e)))
    
    def _render_complete(self, output_path):
        """Called when rendering is complete"""
        self.render_button.configure(state=tk.NORMAL)
        self.status_var.set(f"Rendered to {os.path.basename(output_path)}")
        
        # Ask if user wants to open the directory
        if messagebox.askyesno("Render Complete", 
                              f"Noise file saved to {output_path}\n\nOpen containing folder?"):
            self._open_directory(os.path.dirname(output_path))
    
    def _render_error(self, error_message):
        """Called when rendering fails"""
        self.render_button.configure(state=tk.NORMAL)
        self.status_var.set(f"Error: {error_message}")
        messagebox.showerror("Render Error", f"Failed to render noise: {error_message}")
    
    def _open_directory(self, path):
        """Open directory in file explorer"""
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    
    def load_preset(self, preset_name):
        """Load a preset from the presets dictionary"""
        if preset_name not in self.presets:
            preset_name = "default"
        
        preset = self.presets[preset_name]
        
        # Update UI variables
        color_mix = preset.get('color_mix', {'white': 0.4, 'pink': 0.4, 'brown': 0.2})
        self.white_var.set(color_mix.get('white', 0.4))
        self.pink_var.set(color_mix.get('pink', 0.4))
        self.brown_var.set(color_mix.get('brown', 0.2))
        
        # Update warmth slider based on mix
        # Higher warmth = more brown, less white
        warmth = (color_mix.get('brown', 0.0) * 2 + color_mix.get('pink', 0.0)) / 2
        self.warmth_var.set(warmth)
        
        # Other parameters
        self.rms_var.set(preset.get('rms_target', -63.0))
        
        lfo_rate = preset.get('lfo_rate')
        if lfo_rate is not None:
            self.lfo_var.set(lfo_rate)
            self.lfo_enabled_var.set(True)
        else:
            self.lfo_enabled_var.set(False)
        
        # Update visualization
        self.update_visualization()
    
    def on_preset_change(self, event=None):
        """Handle preset selection change"""
        self.load_preset(self.preset_var.get())
    
    def on_warmth_change(self, event=None):
        """Handle warmth slider change"""
        warmth = self.warmth_var.get()
        
        # Map warmth to color mix:
        # 0.0 = all white
        # 0.5 = equal mix of white and pink
        # 1.0 = mostly brown with some pink
        
        if warmth < 0.5:
            # 0.0-0.5: White to equal white/pink
            t = warmth * 2  # 0-1
            white = 1.0 - 0.6 * t
            pink = 0.6 * t
            brown = 0.0
        else:
            # 0.5-1.0: Equal white/pink to mostly brown
            t = (warmth - 0.5) * 2  # 0-1
            white = 0.4 - 0.3 * t
            pink = 0.4 - 0.1 * t
            brown = 0.2 + 0.4 * t
        
        # Update sliders without triggering their callbacks
        self.white_var.set(white)
        self.pink_var.set(pink)
        self.brown_var.set(brown)
        
        # Update visualization
        self.update_visualization()
    
    def on_color_change(self, event=None):
        """Handle manual color sliders change"""
        # Normalize to sum to 1.0
        total = self.white_var.get() + self.pink_var.get() + self.brown_var.get()
        
        if total > 0:
            # Update warmth based on mix (inverse of on_warmth_change calculation)
            warmth = (self.brown_var.get() * 2 + self.pink_var.get()) / 2
            self.warmth_var.set(warmth)
        
        # Update visualization
        self.update_visualization()
    
    def randomize_seed(self):
        """Generate a random seed"""
        seed = np.random.randint(0, 2**32 - 1)
        self.seed_var.set(str(seed))
    
    def update_visualization(self):
        """Update the visualization plots"""
        # Update spectral lines
        x = np.logspace(np.log10(20), np.log10(20000), 100)
        
        # White noise (flat)
        white_y = np.zeros_like(x) - 3.0
        
        # Pink noise (-3 dB/octave)
        pink_y = -10 * np.log10(x / 20) - 3.0
        
        # Brown noise (-6 dB/octave)
        brown_y = -20 * np.log10(x / 20) - 3.0
        
        # Apply high-pass for brown
        brown_y[x < 20] = -30
        
        # Mix according to color mix
        total = self.white_var.get() + self.pink_var.get() + self.brown_var.get()
        if total > 0:
            mix_y = (self.white_var.get() * np.power(10, white_y/10) + 
                    self.pink_var.get() * np.power(10, pink_y/10) + 
                    self.brown_var.get() * np.power(10, brown_y/10)) / total
            mix_y = 10 * np.log10(mix_y)
        else:
            mix_y = white_y
        
        # Update lines
        self.white_line.set_ydata(white_y)
        self.pink_line.set_ydata(pink_y)
        self.brown_line.set_ydata(brown_y)
        self.mix_line.set_ydata(mix_y)
        
        # Update target level in level meter
        x_range = np.linspace(0, 10, 100)
        self.target_line.set_data(x_range, np.ones_like(x_range) * self.rms_var.get())
        
        # Redraw canvas
        self.canvas.draw()
    
    def update_ui(self):
        """Update UI elements (called periodically)"""
        # Update RMS display
        if self.streaming:
            self.rms_label.configure(text=f"Current: {self.current_rms:.1f} dB")
            
            # Update level meter
            if self.rms_history:
                x = np.linspace(0, 10, len(self.rms_history))
                self.level_line.set_data(x, self.rms_history)
                self.level_ax.set_xlim(0, 10)
                self.canvas.draw()
        
        # Schedule next update
        self.root.after(UPDATE_INTERVAL, self.update_ui)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Baby-Noise Generator GUI")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    # Create the root window
    root = tk.Tk()
    
    # Create the app
    app = BabyNoiseApp(root)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()