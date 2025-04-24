#!/usr/bin/env python3
# Baby-Noise Generator App v2.0 - GPU Optimized
# Exclusively optimized for GPU acceleration with no CPU fallback

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
import soundfile as sf

# Import CuPy - required for this GPU-only version
try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal
    from cupyx.scipy import fft as cufft
except ImportError:
    raise ImportError(
        "CuPy is required for this GPU-only version. "
        "Install with: pip install cupy-cuda12x"
    )

# Constants
SAMPLE_RATE = 44100
DEFAULT_PEAK_CEILING = -3.0  # dBFS
BROWN_LEAKY_ALPHA = 0.999    # Leaky integrator coefficient
FFT_BLOCK_SIZE = 2**18       # ~1.5 seconds at 44.1 kHz (large block optimization)
BLOCK_OVERLAP = 4096         # For smooth transitions between blocks
STREAM_BUFFER_SIZE = 65536   # Increased buffer size for GPU efficiency
APP_TITLE = "Baby-Noise Generator v2.0 - GPU Optimized"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Documents/BabyNoise")
BUFFER_SIZE = 4096  # Increased audio buffer size for better GPU utilization
UPDATE_INTERVAL = 50  # ms between UI updates

# Adaptive progress throttling based on render duration
def get_progress_throttle(duration):
    """Get appropriate throttle interval based on render duration"""
    if duration < 300:  # < 5 minutes
        return 0.05  # 50ms for short renders
    elif duration < 1800:  # < 30 minutes
        return 0.1  # 100ms for medium renders
    else:  # >= 30 minutes
        return 0.3  # 300ms for long renders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("baby-noise-app")

# Output profiles with specific settings
NOISE_PROFILES = {
    "baby-safe": {
        "rms_target": -63.0,        # Default RMS level (AAP guideline)
        "lufs_threshold": -27.0,    # LUFS threshold (approx AAP guideline)
        "peak_ceiling": -3.0,       # True peak ceiling
        "pre_emphasis": False,      # No pre-emphasis needed
        "description": "AAP-compliant safe levels"
    },
    "youtube-pub": {
        "rms_target": -20.0,        # Higher RMS for YouTube
        "lufs_threshold": -16.0,    # LUFS threshold for YouTube
        "peak_ceiling": -2.0,       # Less headroom needed
        "pre_emphasis": True,       # Add pre-emphasis for codec resilience
        "description": "Optimized for YouTube publishing"
    }
}

# Cache for device info - initialized once and reused
_device_info_cache = None

# Utility functions
def get_device_info():
    """Get basic information about the CUDA device (cached)"""
    global _device_info_cache
    if _device_info_cache is None:
        device_id = cp.cuda.device.get_device_id()
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        
        # Properly handle name that could be bytes or string
        name = device_props["name"]
        if isinstance(name, (bytes, bytearray)):
            name = name.decode()
        
        _device_info_cache = {
            "name": name,
            "compute_capability": f"{device_props['major']}.{device_props['minor']}",
            "total_memory": device_props["totalGlobalMem"] / (1024**3),  # GB
            "multiprocessors": device_props["multiProcessorCount"],
            "max_threads_per_block": device_props["maxThreadsPerBlock"],
            "mem_clock_rate": device_props["memoryClockRate"] / 1000,  # MHz
        }
    
    return _device_info_cache

# GPU Memory Optimization - Dynamically adjust block size based on available memory
def optimize_block_size():
    """Determine optimal FFT block size based on GPU memory"""
    device_info = get_device_info()
    total_mem_gb = device_info["total_memory"]
    
    # Scale block size based on available memory
    # For devices with less than 4GB, use smaller blocks
    if total_mem_gb < 4.0:
        return 2**17  # Half the default size
    elif total_mem_gb > 8.0:
        return 2**19  # Double the default size for high-memory GPUs
    else:
        return FFT_BLOCK_SIZE  # Default size

def create_pink_filter(n_taps, sample_rate):
    """Create FIR filter for pink noise using frequency sampling method"""
    # Create frequency response for pink noise (-10 dB/decade falloff)
    nyquist = sample_rate / 2
    freqs = cp.linspace(0, nyquist, n_taps//2 + 1)
    # Avoid division by zero
    freqs[0] = freqs[1]
    
    # Create pink spectrum (1/f)
    response = 1.0 / cp.sqrt(freqs)
    
    # Normalize
    response = response / cp.max(response)
    
    # Convert to filter taps using FFT
    # Make sure we get exactly n_taps total
    if n_taps % 2 == 0:  # Even number of taps
        full_response = cp.concatenate([response, response[-2:0:-1]])
    else:  # Odd number of taps
        full_response = cp.concatenate([response, response[-2::-1]])
    
    # Ensure exact length
    if len(full_response) != n_taps:
        # Pad or truncate to exact length
        if len(full_response) < n_taps:
            full_response = cp.pad(full_response, (0, n_taps - len(full_response)))
        else:
            full_response = full_response[:n_taps]
    
    filter_taps = cp.real(cp.fft.ifft(full_response))
    
    # Window the filter to reduce Gibbs phenomenon
    window = cp.hamming(len(filter_taps))
    filter_taps = filter_taps * window
    
    # Normalize for unity gain at DC
    filter_taps = filter_taps / cp.sum(filter_taps)
    
    return filter_taps

def normalize_color_mix(color_mix):
    """Normalize color mix to sum to 1.0"""
    total = sum(color_mix.values())
    if total > 0:
        return {k: v / total for k, v in color_mix.items()}
    else:
        # Default if all zeros
        return {'white': 0.4, 'pink': 0.4, 'brown': 0.2}

# Noise configuration dataclass
class NoiseConfig:
    def __init__(self, 
                 seed=None, 
                 duration=600,
                 color_mix=None, 
                 rms_target=-63.0, 
                 peak_ceiling=-3.0,
                 lfo_rate=None, 
                 sample_rate=SAMPLE_RATE, 
                 channels=1,
                 profile="baby-safe"):
        """Initialize noise configuration"""
        self.seed = seed if seed is not None else int(time.time())
        self.duration = duration  # seconds
        self.color_mix = color_mix or {'white': 0.4, 'pink': 0.4, 'brown': 0.2}
        self.rms_target = rms_target  # dBFS
        self.peak_ceiling = peak_ceiling  # dBFS
        self.lfo_rate = lfo_rate  # Hz, None for no modulation
        self.sample_rate = sample_rate
        self.use_gpu = True  # Always use GPU in this version
        self.channels = channels  # 1=mono, 2=stereo
        self.profile = profile  # Output profile name

    def validate(self):
        """Validate configuration"""
        # Normalize color mix to sum to 1.0
        self.color_mix = normalize_color_mix(self.color_mix)
        
        # Apply profile settings if needed
        if self.profile in NOISE_PROFILES:
            profile_settings = NOISE_PROFILES[self.profile]
            # Only override if not explicitly set
            if not hasattr(self, '_custom_rms'):
                self.rms_target = profile_settings.get("rms_target", self.rms_target)
            if not hasattr(self, '_custom_peak'):
                self.peak_ceiling = profile_settings.get("peak_ceiling", self.peak_ceiling)

    def set_rms_target(self, value):
        """Set custom RMS target"""
        self._custom_rms = True
        self.rms_target = value
    
    def set_peak_ceiling(self, value):
        """Set custom peak ceiling"""
        self._custom_peak = True
        self.peak_ceiling = value

# GPU-accelerated noise generator
class NoiseGenerator:
    """GPU-accelerated noise generator for rendering to file"""
    
    def __init__(self, config):
        """Initialize generator with configuration"""
        self.config = config
        self.config.validate()
        self.optimal_block_size = optimize_block_size()
        self.total_samples = int(self.config.duration * self.config.sample_rate)
        self.channels = self.config.channels
        
        # Initialize filters and state
        self._init_gpu()
        self._init_filters()
        
        # Progress tracking
        self.progress_callback = None
        self.is_cancelled = False
    
    def _init_gpu(self):
        """Initialize GPU context and PRNG"""
        # Create the PRNG with specified seed
        self.rng = cp.random.RandomState(seed=self.config.seed)
        
        # Get device info for logging
        device_info = get_device_info()
        logger.info(f"Using GPU: {device_info['name']} with {device_info['total_memory']:.2f} GB memory")
        logger.info(f"Compute capability: {device_info['compute_capability']}")
        
        # Warm up GPU by allocating a small array
        warmup = cp.zeros((1024, 1024), dtype=cp.float32)
        del warmup
    
    def _init_filters(self):
        """Initialize filters for pink and brown noise"""
        # Create filter coefficients for pink noise
        # Pink noise approximation using 8th order filter
        # -10 dB/decade = -3 dB/octave slope
        self._pink_filter_taps = create_pink_filter(4097, self.config.sample_rate)
        
        # Precompute highpass filter coefficients for brown noise
        cutoff = 20.0 / (self.config.sample_rate / 2)
        # Use second-order sections form for better numerical stability
        self._brown_hp_sos = cusignal.butter(2, cutoff, 'high', output='sos')
        
        # For stereo processing
        self.decorrelation_phases = None
        if self.channels == 2:
            # Create phase shift array for stereo decorrelation
            # Use different phase shifts across the spectrum for natural stereo image
            n_freqs = self.optimal_block_size // 2 + 1
            self.decorrelation_phases = cp.zeros(n_freqs, dtype=cp.complex64)
            
            # Create decorrelation with natural frequency-dependent phase differences
            # Higher frequencies get more decorrelation
            phases = cp.linspace(0, cp.pi/4, n_freqs)  # 0 to 45 degrees
            # Apply quadratic curve to phase differences (more in mids and highs)
            phases = phases**2 / (cp.pi/4)
            self.decorrelation_phases = cp.exp(1j * phases)
    
    def _generate_white_noise_block(self, block_size):
        """Generate white noise block on GPU"""
        if self.channels == 1:
            # Mono output
            return self.rng.normal(0, 1, block_size).astype(cp.float32)
        else:
            # Stereo output with slight decorrelation
            noise_left = self.rng.normal(0, 1, block_size).astype(cp.float32)
            noise_right = self.rng.normal(0, 1, block_size).astype(cp.float32)
            return cp.vstack((noise_left, noise_right))
    
    def _apply_pink_filter(self, white_noise):
        """Apply pink filter to white noise on GPU"""
        if self.channels == 1:
            # Mono processing
            return cusignal.fftconvolve(white_noise, self._pink_filter_taps, mode='same')
        else:
            # Stereo processing
            left = white_noise[0]
            right = white_noise[1]
            
            # Process each channel
            pink_left = cusignal.fftconvolve(left, self._pink_filter_taps, mode='same')
            pink_right = cusignal.fftconvolve(right, self._pink_filter_taps, mode='same')
            
            return cp.vstack((pink_left, pink_right))
    
    def _generate_brown_mono(self, white_noise):
        """Generate brown noise from white noise using vectorized IIR filtering"""
        # Brown noise is an IIR filter: y[n] = alpha * y[n-1] + scale * x[n]
        alpha = BROWN_LEAKY_ALPHA
        scale = cp.sqrt(1.0 - alpha*alpha)  # Normalization factor
        
        # Apply the filter (much faster than a Python loop)
        # b=[scale], a=[1, -alpha] implements y[n]=alpha*y[n-1]+scale*x[n]
        brown = cusignal.lfilter([scale], [1, -alpha], white_noise)
        
        # Apply high-pass filter to remove DC offset using second-order sections
        brown = cusignal.sosfilt(self._brown_hp_sos, brown)
        
        return brown
    
    def _generate_brown_noise(self, white_noise):
        """Generate brown noise from white noise using leaky integration"""
        if self.channels == 1:
            # Mono processing
            return self._generate_brown_mono(white_noise)
        else:
            # Stereo processing - apply to each channel separately
            left = white_noise[0]
            right = white_noise[1]
            
            # Process each channel separately using the mono generator
            brown_left = self._generate_brown_mono(left)
            brown_right = self._generate_brown_mono(right)
            
            return cp.vstack((brown_left, brown_right))
    
    def _apply_stereo_decorrelation(self, noise_block):
        """Apply stereo decorrelation in frequency domain for realistic stereo field"""
        if self.channels == 1:
            return noise_block
        
        # Get left and right channels
        left = noise_block[0]
        right = noise_block[1]
        
        # Process right channel for decorrelation (leave left untouched as reference)
        # Convert to frequency domain
        right_fft = cufft.rfft(right)
        
        # Apply phase shift for decorrelation
        right_fft = right_fft * self.decorrelation_phases
        
        # Convert back to time domain
        right_decorrelated = cufft.irfft(right_fft, n=len(right))
        
        # Recombine channels
        return cp.vstack((left, right_decorrelated))
    
    def _apply_gain_and_limiting(self, noise_block, target_rms, peak_ceiling):
        """Apply gain adjustment and limiting"""
        # Handle mono vs stereo
        if self.channels == 1:
            # Calculate current RMS
            current_rms = cp.sqrt(cp.mean(noise_block**2))
            
            # Calculate gain needed
            target_linear = 10 ** (target_rms / 20.0)
            gain = target_linear / (current_rms + 1e-10)
            
            # Apply gain
            noise_block = noise_block * gain
            
            # Apply limiting if needed
            peak = cp.max(cp.abs(noise_block))
            peak_threshold = 10 ** (peak_ceiling / 20.0)
            
            if peak > peak_threshold:
                limiting_gain = peak_threshold / peak
                noise_block = noise_block * limiting_gain
                gain_db = float(20 * cp.log10(limiting_gain))
                logger.info(f"Applied limiting: {gain_db:.1f} dB")
        else:
            # For stereo, process combined energy
            # Calculate RMS across both channels
            combined_rms = cp.sqrt(cp.mean((noise_block[0]**2 + noise_block[1]**2) / 2))
            
            # Calculate gain needed
            target_linear = 10 ** (target_rms / 20.0)
            gain = target_linear / (combined_rms + 1e-10)
            
            # Apply gain to both channels
            noise_block = noise_block * gain
            
            # Apply limiting if needed (check both channels)
            peak_left = cp.max(cp.abs(noise_block[0]))
            peak_right = cp.max(cp.abs(noise_block[1]))
            peak = cp.maximum(peak_left, peak_right)
            peak_threshold = 10 ** (peak_ceiling / 20.0)
            
            if peak > peak_threshold:
                limiting_gain = peak_threshold / peak
                noise_block = noise_block * limiting_gain
                gain_db = float(20 * cp.log10(limiting_gain))
                logger.info(f"Applied limiting: {gain_db:.1f} dB")
        
        return noise_block
    
    def _apply_true_peak_limiting(self, noise_block, peak_ceiling):
        """Apply true-peak limiting with 4x oversampling"""
        # True peak detection using 4x oversampling
        oversampling_factor = 4
        
        if self.channels == 1:
            # Mono processing
            # Upsample for true-peak detection
            upsampled = cusignal.resample(noise_block, len(noise_block) * oversampling_factor)
            
            # Find true peak
            true_peak = cp.max(cp.abs(upsampled))
            true_peak_db = float(20 * cp.log10(true_peak + 1e-10))
            
            # Apply limiting if needed
            peak_threshold = 10 ** (peak_ceiling / 20.0)
            if true_peak > peak_threshold:
                limiting_gain = peak_threshold / true_peak
                noise_block = noise_block * limiting_gain
                gain_db = float(20 * cp.log10(limiting_gain))
                logger.info(f"Applied true-peak limiting: {gain_db:.1f} dB")
        else:
            # Stereo processing
            # Process each channel
            upsampled_left = cusignal.resample(noise_block[0], len(noise_block[0]) * oversampling_factor)
            upsampled_right = cusignal.resample(noise_block[1], len(noise_block[1]) * oversampling_factor)
            
            # Find true peak across both channels
            true_peak_left = cp.max(cp.abs(upsampled_left))
            true_peak_right = cp.max(cp.abs(upsampled_right))
            true_peak = cp.maximum(true_peak_left, true_peak_right)
            true_peak_db = float(20 * cp.log10(true_peak + 1e-10))
            
            # Apply limiting if needed
            peak_threshold = 10 ** (peak_ceiling / 20.0)
            if true_peak > peak_threshold:
                limiting_gain = peak_threshold / true_peak
                noise_block = noise_block * limiting_gain
                gain_db = float(20 * cp.log10(limiting_gain))
                logger.info(f"Applied true-peak limiting: {gain_db:.1f} dB")
        
        return noise_block
    
    def _apply_pre_emphasis(self, noise_block):
        """Apply pre-emphasis filter for better codec performance on YouTube"""
        if not NOISE_PROFILES.get(self.config.profile, {}).get("pre_emphasis", False):
            return noise_block
        
        # Pre-emphasis filter (boost above 5kHz for YouTube codec)
        nyquist = self.config.sample_rate / 2
        cutoff = 5000 / nyquist
        
        # Design shelving filter
        b, a = cusignal.butter(2, cutoff, 'high', analog=False)
        
        # Apply gain to shelving filter
        b = b * 1.5  # +3.5 dB boost
        
        if self.channels == 1:
            # Apply filter
            emphasized = cusignal.lfilter(b, a, noise_block)
            
            # Re-normalize to maintain RMS
            original_rms = cp.sqrt(cp.mean(noise_block**2))
            emphasized_rms = cp.sqrt(cp.mean(emphasized**2))
            gain_factor = original_rms / (emphasized_rms + 1e-10)
            
            return emphasized * gain_factor
        else:
            # Process each channel
            emphasized_left = cusignal.lfilter(b, a, noise_block[0])
            emphasized_right = cusignal.lfilter(b, a, noise_block[1])
            
            # Re-normalize to maintain RMS
            original_rms = cp.sqrt(cp.mean((noise_block[0]**2 + noise_block[1]**2) / 2))
            emphasized_rms = cp.sqrt(cp.mean((emphasized_left**2 + emphasized_right**2) / 2))
            gain_factor = original_rms / (emphasized_rms + 1e-10)
            
            return cp.vstack((emphasized_left * gain_factor, emphasized_right * gain_factor))
    
    def _apply_lfo_modulation(self, noise_block, block_start_idx):
        """Apply LFO modulation if enabled"""
        if self.config.lfo_rate is None:
            return noise_block
        
        # Create LFO modulation envelope
        block_len = noise_block.shape[-1]
        block_time = block_len / self.config.sample_rate
        
        # Generate time indices for this block
        t_start = block_start_idx / self.config.sample_rate
        t = cp.linspace(t_start, t_start + block_time, block_len, endpoint=False)
        
        # Create sinusoidal modulation (±1dB)
        modulation_depth = 10**(1.0/20) - 10**(-1.0/20)  # ±1dB in linear scale
        modulation = 1.0 + modulation_depth/2 * cp.sin(2 * cp.pi * self.config.lfo_rate * t)
        
        # Apply modulation
        if self.channels == 1:
            return noise_block * modulation
        else:
            # Apply same modulation to both channels
            return cp.vstack((noise_block[0] * modulation, noise_block[1] * modulation))
    
    def _blend_noise_colors(self, white, pink, brown, color_mix):
        """Blend different noise colors on GPU according to color mix"""
        white_gain = cp.sqrt(color_mix.get('white', 0))
        pink_gain = cp.sqrt(color_mix.get('pink', 0))
        brown_gain = cp.sqrt(color_mix.get('brown', 0))
        
        # Normalize to ensure consistent levels regardless of mix
        total_power = white_gain**2 + pink_gain**2 + brown_gain**2
        normalization = cp.sqrt(1.0 / (total_power + 1e-10))
        
        white_gain *= normalization
        pink_gain *= normalization
        brown_gain *= normalization
        
        if self.channels == 1:
            # Mix mono signals
            return (white * white_gain + 
                    pink * pink_gain + 
                    brown * brown_gain)
        else:
            # Mix stereo signals (channel-wise)
            left_mix = (white[0] * white_gain + 
                       pink[0] * pink_gain + 
                       brown[0] * brown_gain)
            
            right_mix = (white[1] * white_gain + 
                        pink[1] * pink_gain + 
                        brown[1] * brown_gain)
            
            return cp.vstack((left_mix, right_mix))
    
    def generate_block(self, block_size, block_start_idx=0):
        """Generate a block of noise with the specified configuration"""
        # Generate white noise base
        white_noise = self._generate_white_noise_block(block_size)
        
        # Generate pink noise from white noise
        pink_noise = self._apply_pink_filter(white_noise)
        
        # Generate brown noise from white noise
        brown_noise = self._generate_brown_noise(white_noise)
        
        # Blend according to color mix
        mixed_noise = self._blend_noise_colors(
            white_noise, pink_noise, brown_noise, self.config.color_mix
        )
        
        # Apply stereo decorrelation if needed
        if self.channels == 2:
            mixed_noise = self._apply_stereo_decorrelation(mixed_noise)
        
        # Apply LFO modulation if enabled
        if self.config.lfo_rate is not None:
            mixed_noise = self._apply_lfo_modulation(mixed_noise, block_start_idx)
        
        # Apply gain and limiting
        mixed_noise = self._apply_gain_and_limiting(
            mixed_noise, self.config.rms_target, self.config.peak_ceiling
        )
        
        # Apply true-peak limiting
        mixed_noise = self._apply_true_peak_limiting(mixed_noise, self.config.peak_ceiling)
        
        # Apply pre-emphasis if enabled in profile
        mixed_noise = self._apply_pre_emphasis(mixed_noise)
        
        return mixed_noise
    
    def generate_to_file(self, output_path, progress_callback=None):
        """Generate noise and save to file with progress tracking"""
        self.progress_callback = progress_callback
        self.is_cancelled = False
        
        start_time = time.time()
        last_progress_update = time.time()
        last_progress_value = 0.0
        
        # Determine throttle interval based on duration
        progress_throttle = get_progress_throttle(self.config.duration)
        
        # Determine format based on extension
        _, ext = os.path.splitext(output_path)
        if ext.lower() == '.flac':
            output_format = 'FLAC'
            subtype = None
        else:
            output_format = 'WAV'
            subtype = 'PCM_24'  # Always 24-bit
        
        # Create output file
        with sf.SoundFile(
            output_path, 
            mode='w', 
            samplerate=self.config.sample_rate,
            channels=self.channels,
            format=output_format,
            subtype=subtype
        ) as f:
            # Process in blocks to manage memory
            samples_remaining = self.total_samples
            samples_written = 0
            block_size = self.optimal_block_size
            
            # Large overlap for smooth transitions
            overlap = BLOCK_OVERLAP
            overlap_buffer = None
            
            while samples_remaining > 0 and not self.is_cancelled:
                # Determine block size for this iteration
                current_block_size = min(block_size, samples_remaining + overlap)
                
                # Generate block
                noise_block = self.generate_block(current_block_size, samples_written)
                
                # Move from GPU to CPU
                if self.channels == 1:
                    output_data = cp.asnumpy(noise_block)
                else:
                    # Transpose for soundfile's expected format
                    output_data = cp.asnumpy(noise_block.T)
                
                # Apply overlap from previous block if available
                if overlap_buffer is not None:
                    # Crossfade with previous block's overlap region
                    fade_in = np.linspace(0, 1, overlap)
                    fade_out = np.linspace(1, 0, overlap)
                    
                    if self.channels == 1:
                        output_data[:overlap] = (
                            output_data[:overlap] * fade_in + 
                            overlap_buffer * fade_out
                        )
                    else:
                        output_data[:overlap, :] = (
                            output_data[:overlap, :] * fade_in[:, np.newaxis] + 
                            overlap_buffer * fade_out[:, np.newaxis]
                        )
                
                # Save overlap buffer for next iteration
                if samples_remaining > overlap:
                    if self.channels == 1:
                        overlap_buffer = output_data[-overlap:].copy()
                    else:
                        overlap_buffer = output_data[-overlap:, :].copy()
                
                # Write to file (excluding overlap for next block)
                write_length = min(len(output_data) - overlap, samples_remaining)
                f.write(output_data[:write_length])
                
                # Update progress
                samples_written += write_length
                samples_remaining -= write_length
                
                # Report progress with adaptive throttling
                current_time = time.time()
                progress_percent = (samples_written / self.total_samples) * 100
                
                # Check if enough time has passed OR progress has changed significantly
                progress_change = abs(progress_percent - last_progress_value)
                if (self.progress_callback and 
                    ((current_time - last_progress_update >= progress_throttle) or 
                     (progress_change >= 1.0))):  # Update at least every 1%
                    self.progress_callback(progress_percent)
                    last_progress_update = current_time
                    last_progress_value = progress_percent
        
        # Calculate processing metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Measure output file statistics
        result = {
            "processing_time": processing_time,
            "samples_generated": samples_written,
            "real_time_factor": self.config.duration / processing_time,
        }
        
        logger.info(f"Generated {self.config.duration:.1f}s of noise in {processing_time:.1f}s "
                    f"(speed: {result['real_time_factor']:.1f}x real-time)")
        
        # Measure true peak and RMS on disk to verify
        try:
            audio_data, _ = sf.read(output_path)
            if self.channels == 2:
                # Calculate metrics on both channels
                peak_left = np.max(np.abs(audio_data[:, 0]))
                peak_right = np.max(np.abs(audio_data[:, 1]))
                peak_db = 20 * np.log10(max(peak_left, peak_right) + 1e-10)
                
                # Calculate LUFS using a simplified method
                # For accurate LUFS, use pyloudnorm or similar library
                rms_left = np.sqrt(np.mean(audio_data[:, 0]**2))
                rms_right = np.sqrt(np.mean(audio_data[:, 1]**2))
                rms_db = 20 * np.log10((rms_left + rms_right) / 2 + 1e-10)
                
                # Approximate LUFS from RMS
                lufs = rms_db + 3.0  # Simple approximation
            else:
                # Calculate metrics for mono
                peak_db = 20 * np.log10(np.max(np.abs(audio_data)) + 1e-10)
                rms_db = 20 * np.log10(np.sqrt(np.mean(audio_data**2)) + 1e-10)
                lufs = rms_db + 3.0  # Simple approximation
            
            result["peak_db"] = peak_db
            result["rms_db"] = rms_db
            result["integrated_lufs"] = lufs
            
            logger.info(f"Output file metrics: {peak_db:.1f} dBFS peak, {lufs:.1f} LUFS")
        except Exception as e:
            logger.error(f"Error measuring output file: {e}")
            result["error"] = f"Generated successfully, but encountered an error measuring the output: {str(e)}"
        
        return result
    
    def cancel_generation(self):
        """Cancel ongoing generation"""
        self.is_cancelled = True


# Streaming noise generator for real-time playback
class StreamingNoiseGenerator:
    """Streaming noise generator for real-time playback with GPU acceleration"""
    
    def __init__(self, config):
        """Initialize streaming generator"""
        self.config = config
        self.config.validate()
        
        # Initialize GPU context and filters
        self._init_gpu()
        self._init_filters()
        
        # State for streaming
        self.position = 0
        self.stream_buffer_size = STREAM_BUFFER_SIZE
        self.buffer_queue = queue.Queue(maxsize=3)
        self.running = False
        self.thread = None
        
        # Start background processing
        self._start_background_thread()
    
    def _init_gpu(self):
        """Initialize GPU context and PRNG"""
        # Create the PRNG with specified seed
        self.rng = cp.random.RandomState(seed=self.config.seed)
        
        # Warm up GPU by allocating a small array
        warmup = cp.zeros((1024, 1024), dtype=cp.float32)
        del warmup
    
    def _init_filters(self):
        """Initialize filters for pink and brown noise"""
        # Pink noise filter (shorter for real-time)
        self._pink_filter_taps = create_pink_filter(2049, self.config.sample_rate)
        
        # Precompute highpass filter coefficients for brown noise
        cutoff = 20.0 / (self.config.sample_rate / 2)
        # Use second-order sections form for better numerical stability
        self._brown_hp_sos = cusignal.butter(2, cutoff, 'high', output='sos')
        
        # For stereo processing
        self.decorrelation_phases = None
        if self.config.channels == 2:
            # Create phase shift array for stereo decorrelation
            n_freqs = self.stream_buffer_size // 2 + 1
            self.decorrelation_phases = cp.zeros(n_freqs, dtype=cp.complex64)
            
            # Create decorrelation with natural frequency-dependent phase differences
            phases = cp.linspace(0, cp.pi/4, n_freqs)  # 0 to 45 degrees
            phases = phases**2 / (cp.pi/4)  # Apply quadratic curve
            self.decorrelation_phases = cp.exp(1j * phases)
    
    def _generate_brown_mono(self, white_noise):
        """Generate brown noise from white noise using vectorized IIR filtering"""
        # Brown noise is an IIR filter: y[n] = alpha * y[n-1] + scale * x[n]
        alpha = BROWN_LEAKY_ALPHA
        scale = cp.sqrt(1.0 - alpha*alpha)  # Normalization factor
        
        # Apply the filter (much faster than a Python loop)
        # b=[scale], a=[1, -alpha] implements y[n]=alpha*y[n-1]+scale*x[n]
        brown = cusignal.lfilter([scale], [1, -alpha], white_noise)
        
        # Apply high-pass filter to remove DC offset using second-order sections
        brown = cusignal.sosfilt(self._brown_hp_sos, brown)
        
        return brown
    
    def _start_background_thread(self):
        """Start background thread for buffer generation"""
        self.running = True
        self.thread = threading.Thread(target=self._buffer_generation_thread, daemon=True)
        self.thread.start()
    
    def _buffer_generation_thread(self):
        """Background thread to pre-generate buffers with improved error handling"""
        while self.running:
            try:
                if self.buffer_queue.qsize() < 2:
                    # Generate a new buffer
                    buffer_size = self.stream_buffer_size
                    try:
                        noise_block = self._generate_buffer(buffer_size)
                        self.buffer_queue.put(noise_block, block=False)
                    except Exception as e:
                        # Log the error
                        logger.error(f"Error generating buffer: {e}")
                        # Generate silence as fallback
                        if self.config.channels == 1:
                            silence = np.zeros(buffer_size, dtype=np.float32)
                        else:
                            silence = np.zeros((buffer_size, 2), dtype=np.float32)
                        
                        # Try to put silence in the queue
                        try:
                            self.buffer_queue.put(silence, block=False)
                        except queue.Full:
                            pass  # Queue is full, just continue
                else:
                    # Sleep to avoid busy waiting
                    time.sleep(0.01)
            except Exception as e:
                # Catch any other errors in the thread
                logger.error(f"Error in buffer generation thread: {e}")
                # Sleep briefly to avoid CPU thrashing if there's a persistent error
                time.sleep(0.1)
    
    def _generate_buffer(self, buffer_size):
        """Generate a buffer of noise"""
        # Generate base white noise
        if self.config.channels == 1:
            white_noise = self.rng.normal(0, 1, buffer_size).astype(cp.float32)
        else:
            # Stereo with decorrelation
            white_left = self.rng.normal(0, 1, buffer_size).astype(cp.float32)
            white_right = self.rng.normal(0, 1, buffer_size).astype(cp.float32)
            white_noise = cp.vstack((white_left, white_right))
        
        # Generate pink noise
        if self.config.channels == 1:
            pink_noise = cusignal.fftconvolve(white_noise, self._pink_filter_taps, mode='same')
        else:
            pink_left = cusignal.fftconvolve(white_noise[0], self._pink_filter_taps, mode='same')
            pink_right = cusignal.fftconvolve(white_noise[1], self._pink_filter_taps, mode='same')
            pink_noise = cp.vstack((pink_left, pink_right))
        
        # Generate brown noise using the vectorized approach
        if self.config.channels == 1:
            brown_noise = self._generate_brown_mono(white_noise)
        else:
            # Process each channel separately using the mono generator
            brown_left = self._generate_brown_mono(white_noise[0])
            brown_right = self._generate_brown_mono(white_noise[1])
            brown_noise = cp.vstack((brown_left, brown_right))
        
        # Blend according to color mix
        white_gain = cp.sqrt(self.config.color_mix.get('white', 0))
        pink_gain = cp.sqrt(self.config.color_mix.get('pink', 0))
        brown_gain = cp.sqrt(self.config.color_mix.get('brown', 0))
        
        # Normalize
        total_power = white_gain**2 + pink_gain**2 + brown_gain**2
        normalization = cp.sqrt(1.0 / (total_power + 1e-10))
        
        white_gain *= normalization
        pink_gain *= normalization
        brown_gain *= normalization
        
        # Blend noise types
        if self.config.channels == 1:
            mixed_noise = (white_noise * white_gain + 
                          pink_noise * pink_gain + 
                          brown_noise * brown_gain)
        else:
            left_mix = (white_noise[0] * white_gain + 
                       pink_noise[0] * pink_gain + 
                       brown_noise[0] * brown_gain)
            
            right_mix = (white_noise[1] * white_gain + 
                        pink_noise[1] * pink_gain + 
                        brown_noise[1] * brown_gain)
            
            mixed_noise = cp.vstack((left_mix, right_mix))
        
        # Apply stereo decorrelation if needed
        if self.config.channels == 2:
            # Apply in frequency domain
            right_fft = cufft.rfft(mixed_noise[1])
            right_fft = right_fft * self.decorrelation_phases
            right_decorrelated = cufft.irfft(right_fft, n=len(mixed_noise[1]))
            mixed_noise = cp.vstack((mixed_noise[0], right_decorrelated))
        
        # Apply LFO modulation if enabled
        if self.config.lfo_rate is not None:
            t = cp.linspace(
                self.position / self.config.sample_rate,
                (self.position + buffer_size) / self.config.sample_rate,
                buffer_size, endpoint=False
            )
            
            # Create sinusoidal modulation (±1dB)
            modulation_depth = 10**(1.0/20) - 10**(-1.0/20)
            modulation = 1.0 + modulation_depth/2 * cp.sin(2 * cp.pi * self.config.lfo_rate * t)
            
            if self.config.channels == 1:
                mixed_noise = mixed_noise * modulation
            else:
                # Apply to both channels
                mixed_noise = cp.vstack((mixed_noise[0] * modulation, mixed_noise[1] * modulation))
        
        # Apply gain and limiting
        if self.config.channels == 1:
            # Calculate current RMS
            current_rms = cp.sqrt(cp.mean(mixed_noise**2))
            
            # Calculate gain needed
            target_linear = 10 ** (self.config.rms_target / 20.0)
            gain = target_linear / (current_rms + 1e-10)
            
            # Apply gain
            mixed_noise = mixed_noise * gain
            
            # Apply peak limiting
            peak = cp.max(cp.abs(mixed_noise))
            peak_threshold = 10 ** (self.config.peak_ceiling / 20.0)
            
            if peak > peak_threshold:
                limiting_gain = peak_threshold / peak
                mixed_noise = mixed_noise * limiting_gain
        else:
            # Stereo processing
            # Calculate RMS across both channels
            combined_rms = cp.sqrt(cp.mean((mixed_noise[0]**2 + mixed_noise[1]**2) / 2))
            
            # Calculate gain needed
            target_linear = 10 ** (self.config.rms_target / 20.0)
            gain = target_linear / (combined_rms + 1e-10)
            
            # Apply gain to both channels
            mixed_noise = mixed_noise * gain
            
            # Apply limiting if needed
            peak_left = cp.max(cp.abs(mixed_noise[0]))
            peak_right = cp.max(cp.abs(mixed_noise[1]))
            peak = cp.maximum(peak_left, peak_right)
            peak_threshold = 10 ** (self.config.peak_ceiling / 20.0)
            
            if peak > peak_threshold:
                limiting_gain = peak_threshold / peak
                mixed_noise = mixed_noise * limiting_gain
        
        # Apply pre-emphasis if enabled
        if NOISE_PROFILES.get(self.config.profile, {}).get("pre_emphasis", False):
            # Design shelving filter
            nyquist = self.config.sample_rate / 2
            cutoff = 5000 / nyquist
            b, a = cusignal.butter(2, cutoff, 'high', analog=False)
            b = b * 1.5  # +3.5 dB boost
            
            if self.config.channels == 1:
                # Apply filter
                mixed_noise = cusignal.lfilter(b, a, mixed_noise)
            else:
                # Process each channel
                mixed_noise = cp.vstack((
                    cusignal.lfilter(b, a, mixed_noise[0]),
                    cusignal.lfilter(b, a, mixed_noise[1])
                ))
        
        # Update position
        self.position += buffer_size
        
        # Transfer to CPU
        if self.config.channels == 1:
            return cp.asnumpy(mixed_noise)
        else:
            # Transpose for output format
            return cp.asnumpy(mixed_noise.T)
    
    def get_next_chunk(self, chunk_size):
        """Get next chunk of audio for streaming"""
        # Try to get a pre-generated buffer
        try:
            if not self.buffer_queue.empty():
                buffer = self.buffer_queue.get(block=False)
                
                # Return appropriate sized chunk
                if len(buffer) >= chunk_size:
                    if self.config.channels == 1:
                        return buffer[:chunk_size]
                    else:
                        return buffer[:chunk_size, :]
                else:
                    # Buffer too small, pad with zeros
                    if self.config.channels == 1:
                        result = np.zeros(chunk_size, dtype=np.float32)
                        result[:len(buffer)] = buffer
                    else:
                        result = np.zeros((chunk_size, 2), dtype=np.float32)
                        result[:len(buffer), :] = buffer
                    return result
            else:
                # Generate directly (should be rare if background thread is working)
                logger.warning("Buffer queue empty, generating directly")
                noise_block = self._generate_buffer(chunk_size)
                return noise_block
        except Exception as e:
            logger.error(f"Error getting next chunk: {e}")
            # Return silence on error
            if self.config.channels == 1:
                return np.zeros(chunk_size, dtype=np.float32)
            else:
                return np.zeros((chunk_size, 2), dtype=np.float32)
    
    def shutdown(self):
        """Shutdown the streaming generator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# Helper functions
def load_preset(preset_name):
    """Load a preset from the presets.yaml file"""
    preset_path = os.path.join(os.path.dirname(__file__), "presets.yaml")
    
    with open(preset_path, 'r') as f:
        presets_data = yaml.safe_load(f)
    
    presets = presets_data["presets"]
    
    if preset_name not in presets:
        preset_name = "default"
    
    return presets[preset_name]


def warmth_to_color_mix(warmth):
    """Convert warmth parameter (0-100) to color mix dict"""
    warmth_frac = warmth / 100.0
    
    if warmth_frac < 0.33:
        # 0-33%: Mostly white to equal white/pink
        t = warmth_frac * 3  # 0-1
        white = 1.0 - 0.5 * t
        pink = 0.5 * t
        brown = 0.0
    elif warmth_frac < 0.67:
        # 33-67%: Equal white/pink to equal pink/brown
        t = (warmth_frac - 0.33) * 3  # 0-1
        white = 0.5 - 0.5 * t
        pink = 0.5
        brown = 0.0 + 0.5 * t
    else:
        # 67-100%: Equal pink/brown to mostly brown
        t = (warmth_frac - 0.67) * 3  # 0-1
        white = 0.0
        pink = 0.5 - 0.4 * t
        brown = 0.5 + 0.4 * t
    
    # Use centralized normalization
    return normalize_color_mix({
        'white': white,
        'pink': pink,
        'brown': brown
    })


def generate_stereo_noise(config):
    """Generate stereo noise with the given configuration"""
    # Force stereo mode
    config.channels = 2
    config.validate()
    
    # Create generator and generate
    generator = NoiseGenerator(config)
    
    # Determine the total number of samples
    total_samples = int(config.duration * config.sample_rate)
    
    # Generate in one large block
    noise_block = generator.generate_block(total_samples)
    
    # Transfer to CPU
    return cp.asnumpy(noise_block.T)  # Transpose for correct stereo format


def measure_true_peak(audio_data, sample_rate=SAMPLE_RATE):
    """Measure true peak level with 4x oversampling"""
    # Convert to GPU
    if audio_data.ndim == 1:
        # Mono
        gpu_data = cp.asarray(audio_data)
        
        # Upsample
        oversampled = cusignal.resample(gpu_data, len(gpu_data) * 4)
        
        # Find peak
        true_peak = float(cp.max(cp.abs(oversampled)))
        true_peak_db = 20 * np.log10(true_peak + 1e-10)
        
        return true_peak_db
    else:
        # Multi-channel
        left = cp.asarray(audio_data[:, 0])
        right = cp.asarray(audio_data[:, 1])
        
        # Upsample
        left_over = cusignal.resample(left, len(left) * 4)
        right_over = cusignal.resample(right, len(right) * 4)
        
        # Find peak across channels
        peak_left = float(cp.max(cp.abs(left_over)))
        peak_right = float(cp.max(cp.abs(right_over)))
        true_peak = max(peak_left, peak_right)
        true_peak_db = 20 * np.log10(true_peak + 1e-10)
        
        return true_peak_db


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
        self.warmth_var = tk.DoubleVar(value=50)  # 0-100 scale for UI
        self.rms_var = tk.DoubleVar(value=-63.0)
        self.duration_var = tk.IntVar(value=600)  # 10 minutes
        self.lfo_var = tk.DoubleVar(value=0.1)
        self.lfo_enabled_var = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0.0)  # Progress for rendering
        self.profile_var = tk.StringVar(value="baby-safe")
        self.channels_var = tk.IntVar(value=1)  # 1=mono, 2=stereo
        self.output_format_var = tk.StringVar(value="WAV (24-bit)")
        
        # Check for GPU
        try:
            device_info = get_device_info()
            logger.info(f"Using GPU: {device_info['name']} with {device_info['total_memory']:.2f} GB memory")
        except Exception as e:
            logger.error(f"No CUDA GPU detected: {e}")
            messagebox.showerror(
                "GPU Required", 
                "This version requires a CUDA-compatible GPU.\n"
                "No compatible GPU was detected.\n\n"
                "Please ensure you have a CUDA-compatible NVIDIA GPU and the correct drivers installed."
            )
            self.root.destroy()
            return
        
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
        self._create_main_frame()
        self._create_control_section()
        self._create_playback_section()
        self._create_render_section()
        self._create_visualization()
        self._create_status_bar()
    
    def _create_main_frame(self):
        """Create the main application frame"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
    
    def _create_control_section(self):
        """Create the noise control section"""
        # Create top section (controls)
        control_frame = ttk.LabelFrame(self.main_frame, text="Noise Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Preset selection
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT, padx=5)
        preset_combobox = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                       values=list(self.presets.keys()))
        preset_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        preset_combobox.bind("<<ComboboxSelected>>", self.on_preset_change)
        
        # Profile selection (new)
        profile_frame = ttk.Frame(control_frame)
        profile_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(profile_frame, text="Profile:").pack(side=tk.LEFT, padx=5)
        profile_combobox = ttk.Combobox(profile_frame, textvariable=self.profile_var,
                                       values=["baby-safe", "youtube-pub"])
        profile_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        profile_combobox.bind("<<ComboboxSelected>>", self.on_profile_change)
        
        # Add profile info label
        self.profile_info = ttk.Label(profile_frame, text="AAP-compliant safe levels", font=("", 8, "italic"))
        self.profile_info.pack(side=tk.LEFT, padx=5)
        
        # Seed control
        seed_frame = ttk.Frame(control_frame)
        seed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(seed_frame, textvariable=self.seed_var).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(seed_frame, text="Randomize", command=self.randomize_seed).pack(side=tk.LEFT, padx=5)
        
        # Channel selection (new)
        channel_frame = ttk.Frame(control_frame)
        channel_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(channel_frame, text="Channels:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(channel_frame, text="Mono", variable=self.channels_var, value=1).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(channel_frame, text="Stereo", variable=self.channels_var, value=2).pack(side=tk.LEFT, padx=5)
        
        # Noise color blend controls
        color_frame = ttk.LabelFrame(control_frame, text="Noise Color")
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Warmth slider (maps to color mix)
        warmth_frame = ttk.Frame(color_frame)
        warmth_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(warmth_frame, text="Brighter").pack(side=tk.LEFT, padx=5)
        warmth_slider = ttk.Scale(warmth_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                 variable=self.warmth_var, command=self.on_warmth_change)
        warmth_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(warmth_frame, text="Warmer").pack(side=tk.LEFT, padx=5)
        
        # Add warmth percentage label
        self.warmth_label = ttk.Label(warmth_frame, text="50%")
        self.warmth_label.pack(side=tk.LEFT, padx=5)
        
        # Manual color mix frame (advanced users)
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
                              variable=self.rms_var, command=self.on_rms_change)
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
    
    def _create_playback_section(self):
        """Create the playback controls section"""
        playback_frame = ttk.LabelFrame(self.main_frame, text="Playback")
        playback_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Play/Stop buttons
        button_frame = ttk.Frame(playback_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = ttk.Button(button_frame, text="▶ Play", 
                                     command=self.toggle_playback, width=15)
        self.play_button.pack(side=tk.LEFT, padx=5)
    
    def _create_render_section(self):
        """Create the render controls section"""
        render_frame = ttk.LabelFrame(self.main_frame, text="Render")
        render_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Duration and filename
        duration_frame = ttk.Frame(render_frame)
        duration_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(duration_frame, text="Duration:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(duration_frame, textvariable=self.duration_var, 
                    values=[300, 600, 1800, 3600, 7200, 36000]).pack(side=tk.LEFT, padx=5)
        ttk.Label(duration_frame, text="seconds").pack(side=tk.LEFT)
        
        # Format selection
        format_frame = ttk.Frame(render_frame)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(format_frame, textvariable=self.output_format_var, 
                    values=["WAV (24-bit)", "FLAC"]).pack(side=tk.LEFT, padx=5)
        
        # Add GPU info label
        device_info = get_device_info()
        gpu_info = ttk.Label(format_frame, 
                            text=f"Using: {device_info['name']} ({device_info['total_memory']:.1f} GB)", 
                            font=("", 8, "italic"))
        gpu_info.pack(side=tk.LEFT, padx=5)
        
        # Render button and progress bar
        render_button_frame = ttk.Frame(render_frame)
        render_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.render_button = ttk.Button(render_button_frame, text="Render to File...", 
                                       command=self.render_to_file)
        self.render_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Progress bar for rendering
        self.progress_bar = ttk.Progressbar(render_frame, variable=self.progress_var, 
                                           length=200, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def _create_visualization(self):
        """Create the visualization section"""
        # Separator
        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        # Visualization area
        viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.setup_plots()
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_var = tk.StringVar(value="Ready - GPU Accelerated")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, 
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
        
        # Safety threshold line
        # Get threshold based on profile
        profile = NOISE_PROFILES.get(self.profile_var.get(), NOISE_PROFILES["baby-safe"])
        safety_threshold = -60.0  # Default threshold
        self.safety_line, = self.level_ax.plot([], [], 'r:', linewidth=1.5)
        x_range = np.linspace(0, 10, 100)
        self.safety_line.set_data(x_range, np.ones_like(x_range) * safety_threshold)
        
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
                
                # Reshape for sounddevice if stereo
                if len(chunk.shape) > 1 and chunk.shape[1] == 2:
                    # Already in stereo format
                    pass
                elif self.channels_var.get() == 2:
                    # Convert mono to stereo
                    chunk = np.column_stack((chunk, chunk))
                
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
                if data.ndim == 1:  # Mono
                    outdata[:len(data), 0] = data
                    outdata[len(data):, 0] = 0
                else:  # Stereo/multi-channel
                    outdata[:len(data)] = data
                    outdata[len(data):] = 0
            else:
                if data.ndim == 1:  # Mono
                    outdata[:, 0] = data[:len(outdata)]
                else:  # Stereo/multi-channel
                    outdata[:] = data[:len(outdata)]
                
        except queue.Empty:
            # If queue is empty, fill with zeros
            outdata.fill(0)
            logger.warning("Audio buffer underrun")
    
    def start_streaming(self):
        """Start audio streaming"""
        if self.streaming:
            return
        
        # Create configuration for streaming
        config = self.create_config()[0]
        
        # Create generator
        self.generator = StreamingNoiseGenerator(config)
        
        # Clear audio queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Pre-fill queue with a few chunks
        for _ in range(5):
            chunk = self.generator.get_next_chunk(BUFFER_SIZE)
            
            # Reshape for sounddevice if stereo
            if self.channels_var.get() == 2 and len(chunk.shape) == 1:
                # Convert mono to stereo
                chunk = np.column_stack((chunk, chunk))
                
            self.audio_queue.put(chunk)
        
        # Start audio stream
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=self.channels_var.get(),
            callback=self.audio_callback,
            blocksize=BUFFER_SIZE
        )
        self.stream.start()
        
        self.streaming = True
        self.play_button.configure(text="⏹ Stop")
        self.status_var.set("Playing... (GPU Accelerated)")
    
    def stop_streaming(self):
        """Stop audio streaming"""
        if not self.streaming:
            return
        
        # Stop stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Shut down generator thread
        if self.generator:
            self.generator.shutdown()
            self.generator = None
        
        self.streaming = False
        self.play_button.configure(text="▶ Play")
        self.status_var.set("Ready - GPU Accelerated")
        
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
        
        # Normalize using the utility function
        color_mix = normalize_color_mix(color_mix)
        
        # Get LFO rate if enabled
        lfo_rate = self.lfo_var.get() if self.lfo_enabled_var.get() else None
        
        # Determine output format 
        format_str = self.output_format_var.get()
        if format_str == "FLAC":
            output_format = "FLAC"
        else:  # WAV (24-bit)
            output_format = "PCM_24"
        
        # Determine peak ceiling based on profile
        profile = self.profile_var.get()
        profile_settings = NOISE_PROFILES.get(profile, NOISE_PROFILES["baby-safe"])
        peak_ceiling = profile_settings.get("peak_ceiling", -3.0)
        
        # Create config
        config = NoiseConfig(
            seed=seed,
            duration=self.duration_var.get(),
            color_mix=color_mix,
            rms_target=self.rms_var.get(),
            peak_ceiling=peak_ceiling,
            lfo_rate=lfo_rate,
            sample_rate=SAMPLE_RATE,
            channels=self.channels_var.get(),
            profile=profile
        )
        
        return config, output_format
    
    def update_progress(self, progress_percentage):
        """Update the progress bar during rendering"""
        self.progress_var.set(progress_percentage)
        self.root.update_idletasks()
    
    def render_to_file(self):
        """Render noise to a file"""
        if self.streaming:
            messagebox.showwarning("Cannot Render", "Please stop playback before rendering.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        
        # Get output format
        format_str = self.output_format_var.get()
        if format_str == "FLAC":
            extension = ".flac"
        else:  # WAV (24-bit)
            extension = ".wav"
        
        # Channel info for filename
        channels = "mono" if self.channels_var.get() == 1 else "stereo"
        
        # Default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"baby_noise_{channels}_24bit_{timestamp}{extension}"
        default_path = os.path.join(DEFAULT_OUTPUT_DIR, default_filename)
        
        # Ask for output path
        output_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_OUTPUT_DIR,
            initialfile=default_filename,
            defaultextension=extension,
            filetypes=[
                ("WAV files", "*.wav"), 
                ("FLAC files", "*.flac"), 
                ("All files", "*.*")
            ]
        )
        
        if not output_path:
            return
        
        # Create config
        config, _ = self.create_config()
        
        # Update status and disable render button
        self.status_var.set(f"Rendering to {os.path.basename(output_path)}... (GPU Accelerated)")
        self.render_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
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
            # Update status (on main thread) with correct extension
            self.root.after(0, lambda: self.status_var.set(
                f"Rendering to {os.path.basename(output_path)}... (GPU Accelerated)"
            ))
            
            # Generate to file
            result = generator.generate_to_file(output_path, self.update_progress)
            
            # Update UI from main thread
            self.root.after(0, lambda: self._render_complete(output_path, result))
        except Exception as e:
            logger.error(f"Error rendering: {e}")
            self.root.after(0, lambda: self._render_error(str(e)))
    
    def _render_complete(self, output_path, result=None):
        """Called when rendering is complete"""
        self.render_button.configure(state=tk.NORMAL)
        
        # Check for error in result
        if result and "error" in result:
            self.status_var.set(f"Warning: {result['error']}")
            messagebox.showwarning("Render Warning", result["error"])
        
        # Show real-time factor in status
        if result:
            real_time_factor = result.get('real_time_factor', 0)
            self.status_var.set(
                f"Rendered to {os.path.basename(output_path)} "
                f"({result.get('integrated_lufs', 0):.1f} LUFS, {result.get('peak_db', 0):.1f} dBFS peak, "
                f"{real_time_factor:.1f}x real-time)"
            )
        else:
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
    
    def on_rms_change(self, event=None):
        """Handle RMS slider change"""
        # Get profile safety threshold
        profile = NOISE_PROFILES.get(self.profile_var.get(), NOISE_PROFILES["baby-safe"])
        safety_threshold_lufs = profile.get("lufs_threshold", -27.0)
        
        # Convert LUFS threshold to approximate RMS for UI warning
        # Simple approximation: RMS ~= LUFS - 3dB for typical noise
        safety_threshold_rms = safety_threshold_lufs - 3.0 if safety_threshold_lufs > -60 else -60.0
        
        # Check if current RMS exceeds safety threshold for baby-safe profile
        if self.profile_var.get() == "baby-safe" and self.rms_var.get() > safety_threshold_rms:
            # Highlight the slider in red to indicate warning
            self.rms_label.configure(foreground="red")
            # Show warning if significantly over threshold
            if self.rms_var.get() > safety_threshold_rms + 2.0:
                messagebox.showwarning(
                    "High Volume Warning", 
                    f"The selected RMS level of {self.rms_var.get():.1f} dBFS exceeds the "
                    f"AAP recommended level of {safety_threshold_rms:.1f} dBFS (50 dB SPL) "
                    f"for infant hearing safety."
                )
        else:
            # Reset to normal color
            self.rms_label.configure(foreground="")
        
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
        
        # Brown noise (-5.8 dB/octave to match BROWN_LEAKY_ALPHA=0.999)
        # Factor (1-α) gives ~-5.8 dB/oct instead of -6 dB/oct
        brown_y = -20 * np.log10(x / 20) * (1-0.999) / (1-0.995) - 3.0
        
        # Apply high-pass for brown
        brown_y[x < 20] = -30
        
        # Mix according to color mix
        color_mix = normalize_color_mix({
            'white': self.white_var.get(),
            'pink': self.pink_var.get(),
            'brown': self.brown_var.get()
        })
        
        mix_y = (color_mix['white'] * np.power(10, white_y/10) + 
                color_mix['pink'] * np.power(10, pink_y/10) + 
                color_mix['brown'] * np.power(10, brown_y/10))
        mix_y = 10 * np.log10(mix_y)
        
        # Update lines
        self.white_line.set_ydata(white_y)
        self.pink_line.set_ydata(pink_y)
        self.brown_line.set_ydata(brown_y)
        self.mix_line.set_ydata(mix_y)
        
        # Update target level in level meter
        x_range = np.linspace(0, 10, 100)
        self.target_line.set_data(x_range, np.ones_like(x_range) * self.rms_var.get())
        
        # Update safety threshold line based on profile
        profile = NOISE_PROFILES.get(self.profile_var.get(), NOISE_PROFILES["baby-safe"])
        if self.profile_var.get() == "baby-safe":
            safety_threshold = -60.0  # AAP recommended max (approx)
        else:
            # For youtube-pub, use a more relaxed threshold
            safety_threshold = -30.0  # Just a visual reference point
        
        self.safety_line.set_data(x_range, np.ones_like(x_range) * safety_threshold)
        
        # Redraw canvas
        self.canvas.draw()
    
    def update_ui(self):
        """Update UI elements (called periodically)"""
        # Update RMS display
        if self.streaming:
            self.rms_label.configure(text=f"Current: {self.current_rms:.1f} dB")
            
            # Update label color based on safety threshold
            profile = NOISE_PROFILES.get(self.profile_var.get(), NOISE_PROFILES["baby-safe"])
            if self.profile_var.get() == "baby-safe" and self.current_rms > -60.0:
                self.rms_label.configure(foreground="red")
            else:
                self.rms_label.configure(foreground="")
            
            # Update level meter
            if self.rms_history:
                x = np.linspace(0, 10, len(self.rms_history))
                self.level_line.set_data(x, self.rms_history)
                self.level_ax.set_xlim(0, 10)
                self.canvas.draw()
        
        # Schedule next update
        self.root.after(UPDATE_INTERVAL, self.update_ui)
        
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
        # Calculate warmth value (0-100) from color mix
        white_ratio = color_mix.get('white', 0.0)
        pink_ratio = color_mix.get('pink', 0.0)
        brown_ratio = color_mix.get('brown', 0.0)
        
        # Simple warmth calculation:
        # 0 = all white, 50 = equal white/pink, 100 = all brown
        warmth = 0
        if white_ratio < 0.01 and brown_ratio > 0.99:
            warmth = 100  # All brown
        elif white_ratio > 0.99 and brown_ratio < 0.01:
            warmth = 0    # All white
        else:
            # Weighted calculation
            warmth = int((0.2 * pink_ratio + 0.8 * brown_ratio) * 100)
        
        self.warmth_var.set(warmth)
        self.warmth_label.configure(text=f"{warmth}%")
        
        # Other parameters
        self.rms_var.set(preset.get('rms_target', -63.0))
        
        # Get current profile safety threshold
        profile = NOISE_PROFILES.get(self.profile_var.get(), NOISE_PROFILES["baby-safe"])
        safety_threshold_lufs = profile.get("lufs_threshold", -27.0)
        safety_threshold_rms = safety_threshold_lufs - 3.0 if safety_threshold_lufs > -60 else -60.0
        
        # Check if RMS exceeds safety threshold and warn for baby-safe profile
        if self.profile_var.get() == "baby-safe" and self.rms_var.get() > safety_threshold_rms:
            self.rms_label.configure(foreground="red")
            messagebox.showwarning(
                "High Volume Preset", 
                f"The selected preset '{preset_name}' has an RMS level of {self.rms_var.get():.1f} dBFS, "
                f"which exceeds the AAP recommended level of {safety_threshold_rms:.1f} dBFS (50 dB SPL) "
                f"for infant hearing safety."
            )
        else:
            self.rms_label.configure(foreground="")
        
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
    
    def on_profile_change(self, event=None):
        """Handle profile selection change"""
        profile = self.profile_var.get()
        profile_settings = NOISE_PROFILES.get(profile, NOISE_PROFILES["baby-safe"])
        
        # Update profile info text
        self.profile_info.configure(text=profile_settings.get("description", ""))
        
        # Adjust RMS target based on profile
        if profile == "baby-safe":
            # Set baby-safe default level
            if self.rms_var.get() > -60.0:  # If current level is unsafe
                self.rms_var.set(-63.0)  # Set to default safe level
        else:  # youtube-pub
            # Set YouTube-appropriate levels
            if self.rms_var.get() < -30.0:  # If current level is too quiet for YouTube
                self.rms_var.set(-20.0)  # ~-16 LUFS for YouTube
        
        # Update visualization (including safety thresholds)
        self.update_visualization()
    
    def on_warmth_change(self, event=None):
        """Handle warmth slider change"""
        # Update percentage label
        warmth = int(self.warmth_var.get())
        self.warmth_label.configure(text=f"{warmth}%")
        
        # Convert warmth to color mix using the utility function
        color_mix = warmth_to_color_mix(warmth)
        
        # Update sliders without triggering their callbacks
        self.white_var.set(color_mix['white'])
        self.pink_var.set(color_mix['pink'])
        self.brown_var.set(color_mix['brown'])
        
        # Update visualization
        self.update_visualization()
    
    def on_color_change(self, event=None):
        """Handle manual color sliders change"""
        # Calculate warmth based on color mix
        color_mix = normalize_color_mix({
            'white': self.white_var.get(),
            'pink': self.pink_var.get(),
            'brown': self.brown_var.get()
        })
        
        white_ratio = color_mix['white']
        pink_ratio = color_mix['pink']
        brown_ratio = color_mix['brown']
        
        # Calculate warmth (0-100)
        if white_ratio > 0.99 and brown_ratio < 0.01:
            warmth = 0  # All white
        elif white_ratio < 0.01 and brown_ratio > 0.99:
            warmth = 100  # All brown
        else:
            warmth = int((0.2 * pink_ratio + 0.8 * brown_ratio) * 100)
            
        # Update warmth slider and label without triggering callback
        self.warmth_var.set(warmth)
        self.warmth_label.configure(text=f"{warmth}%")
        
        # Update visualization
        self.update_visualization()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Baby-Noise Generator GUI (GPU-Optimized)")
    parser.add_argument("--profile", choices=["baby-safe", "youtube-pub"], default="baby-safe",
                       help="Default output profile")
    args = parser.parse_args()
    
    # Check for GPU availability before starting
    try:
        device_info = get_device_info()
        logger.info(f"Using GPU: {device_info['name']} with {device_info['total_memory']:.2f} GB memory")
    except Exception as e:
        logger.error(f"No CUDA GPU detected: {e}")
        print(f"ERROR: No CUDA GPU detected: {e}")
        print("This GPU-optimized version requires a CUDA-compatible GPU.")
        print("Please ensure you have a CUDA-compatible NVIDIA GPU and the correct drivers installed.")
        return 1
    
    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    # Create the root window
    root = tk.Tk()
    
    # Create the app
    app = BabyNoiseApp(root)
    
    # Set default profile from command line
    app.profile_var.set(args.profile)
    app.on_profile_change()
    
    # Start the main loop
    root.mainloop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())