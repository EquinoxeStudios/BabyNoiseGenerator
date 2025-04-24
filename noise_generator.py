#!/usr/bin/env python3
# Baby-Noise Generator App v2.0 - GPU Optimized (Headless Version)
# Exclusively optimized for GPU acceleration with no CPU fallback
# Headless version for cloud/colab environments with no GUI requirements

import os
import sys
import time
import threading
import queue
import numpy as np
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
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
APP_TITLE = "Baby-Noise Generator v2.0 - GPU Optimized (Headless)"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/BabyNoise")

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

# Noise configuration dataclass
class NoiseConfig:
    def __init__(self, 
                 seed=None, 
                 duration=600,
                 color_mix=None,
                 warmth=None, 
                 rms_target=-63.0, 
                 peak_ceiling=-3.0,
                 lfo_rate=None, 
                 sample_rate=SAMPLE_RATE, 
                 channels=1,
                 profile="baby-safe"):
        """Initialize noise configuration"""
        self.seed = seed if seed is not None else int(time.time())
        self.duration = duration  # seconds
        
        # Handle warmth parameter if provided
        if warmth is not None:
            self.color_mix = warmth_to_color_mix(warmth)
        else:
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
                else:
                    # Print progress to console
                    if current_time - last_progress_update >= progress_throttle or progress_change >= 5.0:
                        logger.info(f"Progress: {progress_percent:.1f}%")
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

def print_progress(progress):
    """Simple progress callback that prints to console"""
    print(f"\rProgress: {progress:.1f}%", end='')
    if progress >= 100:
        print()  # Add newline when done

def load_preset(preset_name):
    """Load a preset from the presets.yaml file"""
    preset_path = os.path.join(os.path.dirname(__file__), "presets.yaml")
    
    # Default presets if file not found
    default_presets = {
        "default": {
            "color_mix": {'white': 0.4, 'pink': 0.4, 'brown': 0.2},
            "rms_target": -63.0,
            "lfo_rate": 0.1
        },
        "newborn_deep": {
            "color_mix": {'white': 0.2, 'pink': 0.3, 'brown': 0.5},
            "rms_target": -65.0,
            "lfo_rate": 0.05
        },
        "youtube": {
            "color_mix": {'white': 0.4, 'pink': 0.4, 'brown': 0.2},
            "rms_target": -20.0,
            "lfo_rate": 0.1
        }
    }
    
    try:
        with open(preset_path, 'r') as f:
            presets_data = yaml.safe_load(f)
        
        presets = presets_data["presets"]
    except (FileNotFoundError, KeyError):
        logger.warning(f"Presets file not found or invalid: {preset_path}")
        presets = default_presets
    
    if preset_name not in presets:
        logger.warning(f"Preset '{preset_name}' not found, using default")
        preset_name = "default"
    
    return presets[preset_name]

def main():
    """Main entry point for the headless Baby-Noise Generator"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Baby-Noise Generator v2.0 - GPU Accelerated (Headless)")
    
    # Key parameters
    parser.add_argument("--output", type=str, help="Output file path (WAV or FLAC)", default="baby_noise.wav")
    parser.add_argument("--duration", type=int, help="Duration in seconds", default=600)
    parser.add_argument("--channels", type=int, choices=[1, 2], help="Number of channels (1=mono, 2=stereo)", default=1)
    parser.add_argument("--profile", type=str, choices=["baby-safe", "youtube-pub"], 
                        help="Output profile", default="baby-safe")
    
    # Noise characteristics
    parser.add_argument("--warmth", type=float, help="Warmth parameter (0-100)", default=50)
    parser.add_argument("--white", type=float, help="White noise amount (0-1)", default=None)
    parser.add_argument("--pink", type=float, help="Pink noise amount (0-1)", default=None)
    parser.add_argument("--brown", type=float, help="Brown noise amount (0-1)", default=None)
    
    # Advanced parameters
    parser.add_argument("--seed", type=str, help="Random seed (int or 'random')", default="auto")
    parser.add_argument("--lfo", type=float, help="LFO rate in Hz (0 to disable)", default=None)
    parser.add_argument("--preset", type=str, help="Load a preset configuration", default=None)
    parser.add_argument("--rms", type=float, help="Target RMS level in dBFS", default=None)
    
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
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle preset loading if specified
    preset_config = {}
    if args.preset:
        preset_config = load_preset(args.preset)
        logger.info(f"Loaded preset: {args.preset}")
    
    # Parse seed
    if args.seed == "auto":
        seed = int(time.time())
    elif args.seed == "random":
        seed = np.random.randint(0, 2**32 - 1)
    else:
        try:
            seed = int(args.seed)
        except ValueError:
            seed = int(time.time())
            logger.warning(f"Invalid seed '{args.seed}', using time-based seed: {seed}")
    
    # Determine color mix - prioritize explicit mix over warmth
    color_mix = None
    if args.white is not None and args.pink is not None and args.brown is not None:
        color_mix = {'white': args.white, 'pink': args.pink, 'brown': args.brown}
        logger.info(f"Using explicit color mix: white={args.white}, pink={args.pink}, brown={args.brown}")
    elif args.preset and "color_mix" in preset_config:
        color_mix = preset_config["color_mix"]
        logger.info(f"Using preset color mix: {color_mix}")
    
    # Create configuration
    config = NoiseConfig(
        seed=seed,
        duration=args.duration,
        color_mix=color_mix,
        warmth=args.warmth if color_mix is None else None,
        profile=args.profile,
        channels=args.channels,
        lfo_rate=args.lfo if args.lfo is not None else preset_config.get("lfo_rate")
    )
    
    # Override with explicit RMS if provided
    if args.rms is not None:
        config.set_rms_target(args.rms)
    elif args.preset and "rms_target" in preset_config:
        config.set_rms_target(preset_config["rms_target"])
    
    # Print configuration summary
    logger.info(f"Generating {args.duration}s of {'stereo' if args.channels==2 else 'mono'} noise...")
    logger.info(f"Output: {args.output}")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Warmth: {args.warmth}%")
    logger.info(f"Seed: {seed}")
    
    # Create generator
    generator = NoiseGenerator(config)
    
    # Generate to file with progress tracking in console
    try:
        result = generator.generate_to_file(args.output, print_progress)
        
        # Print results
        print("\nGeneration complete!")
        print(f"Output file: {os.path.abspath(args.output)}")
        print(f"Duration: {args.duration}s ({args.duration/60:.1f} minutes)")
        print(f"Processing time: {result['processing_time']:.1f}s")
        print(f"Real-time factor: {result['real_time_factor']:.1f}x")
        
        if "integrated_lufs" in result:
            print(f"LUFS: {result['integrated_lufs']:.1f}")
            print(f"Peak: {result['peak_db']:.1f} dBFS")
        
    except KeyboardInterrupt:
        print("\nGeneration cancelled by user")
        generator.cancel_generation()
        return 1
    except Exception as e:
        logger.error(f"Error generating noise: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())