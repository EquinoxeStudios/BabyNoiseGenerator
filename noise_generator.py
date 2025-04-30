#!/usr/bin/env python3
# Baby-Noise Generator App v2.0.6 - Enhanced Organic DSP Edition
# Optimized for sound quality and performance with advanced DSP techniques
# Exclusively optimized for GPU acceleration with spectral processing
# Stereo-only version for YouTube publishing with enhanced organic sound

import os
import sys
import time
import numpy as np
import argparse
import yaml
import logging
import json
import contextlib

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
FFT_BLOCK_SIZE = 2**18       # ~1.5 seconds at 44.1 kHz (large block optimization)
BLOCK_OVERLAP = 8192         # For smooth transitions between blocks (increased for better blending)
LOOP_CROSSFADE = 2**15       # Specific crossfade size for loop points (32768 samples ≈ 0.75s)
APP_TITLE = "Baby-Noise Generator v2.0.6 - Enhanced Organic DSP (Headless)"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/BabyNoise")
MIN_DURATION = 1             # Minimum allowed duration in seconds
MAX_MEMORY_LIMIT_FRACTION = 0.8  # Maximum fraction of GPU memory to use

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

# Output profile for YouTube publishing (only profile now)
YOUTUBE_PROFILE = {
    "rms_target": -20.0,        # Higher RMS for YouTube
    "lufs_threshold": -16.0,    # LUFS threshold for YouTube
    "peak_ceiling": -2.0,       # Less headroom needed
    "pre_emphasis": True,       # Add pre-emphasis for codec resilience
    "description": "Optimized for YouTube publishing"
}

# Cache for device info - initialized once and reused
_device_info_cache = None

# Context manager for CUDA streams
@contextlib.contextmanager
def cuda_stream():
    """Context manager for CUDA streams that ensures proper cleanup"""
    stream = cp.cuda.Stream()
    try:
        yield stream
    finally:
        # Synchronize the stream when done (no destroy needed)
        stream.synchronize()
        # CuPy streams don't have a destroy method - they're managed by Python's GC

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
def optimize_block_size(fraction=0.8):
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

def setup_memory_pool():
    """Configure CuPy memory pool to reduce fragmentation"""
    mem_pool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    
    # Get total GPU memory
    device_info = get_device_info()
    total_memory = device_info["total_memory"] * 1024**3  # Convert GB to bytes
    
    # Set memory pool limit to a fraction of total memory
    mem_pool.set_limit(size=int(total_memory * MAX_MEMORY_LIMIT_FRACTION))
    
    # Free all blocks to start with a clean state
    mem_pool.free_all_blocks()
    pinned_pool.free_all_blocks()
    
    logger.info(f"Memory pool configured with limit: {mem_pool.get_limit()/1024**3:.2f} GB")
    return mem_pool, pinned_pool

def get_available_memory():
    """Get available memory on GPU in bytes"""
    mem_pool = cp.get_default_memory_pool()
    free_bytes = cp.cuda.runtime.memGetInfo()[0]
    used_bytes = mem_pool.used_bytes()
    return free_bytes, used_bytes

def determine_precision(duration):
    """Determine optimal precision based on render duration"""
    # For very long renders, consider using lower precision to save memory
    if duration > 7200:  # > 2 hours
        # Check if the GPU supports mixed precision
        device_info = get_device_info()
        compute_capability = float(device_info["compute_capability"])
        
        # Only use FP16 on newer GPUs with compute capability >= 7.0 (Volta and newer)
        if compute_capability >= 7.0:
            logger.info("Using mixed precision (FP16) for long render")
            return cp.float16
    
    # Default to full precision
    return cp.float32

def normalize_color_mix(color_mix):
    """Normalize color mix to sum to 1.0"""
    total = sum(color_mix.values())
    if total > 0:
        return {k: v / total for k, v in color_mix.items()}
    else:
        # Default if all zeros
        return {'white': 0.4, 'pink': 0.4, 'brown': 0.2}

def warmth_to_color_mix(warmth):
    """Enhanced warmth parameter (0-100) to color mix conversion with psychoacoustic curve"""
    warmth_frac = warmth / 100.0
    
    # Use a power curve for more natural perception
    # Human perception of "warmth" is roughly logarithmic
    warmth_curve = warmth_frac ** 1.5  # More intuitive control
    
    if warmth_curve < 0.33:
        # 0-33%: Mostly white to equal white/pink
        t = warmth_curve * 3  # 0-1
        white = 1.0 - 0.6 * t
        pink = 0.6 * t
        brown = 0.0
    elif warmth_curve < 0.67:
        # 33-67%: Equal white/pink to equal pink/brown
        t = (warmth_curve - 0.33) * 3  # 0-1
        white = 0.4 - 0.4 * t
        pink = 0.6
        brown = 0.0 + 0.6 * t
    else:
        # 67-100%: Equal pink/brown to mostly brown
        t = (warmth_curve - 0.67) * 3  # 0-1
        white = 0.0
        pink = 0.6 - 0.5 * t
        brown = 0.6 + 0.4 * t
    
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
                 rms_target=None, 
                 peak_ceiling=None,
                 lfo_rate=None, 
                 sample_rate=SAMPLE_RATE, 
                 natural_modulation=True,  # Enable subtle natural modulation
                 haas_effect=True,         # Enable Haas effect for stereo enhancement
                 enhanced_stereo=True):    # Enable enhanced stereo decorrelation
        """Initialize noise configuration"""
        self.seed = seed if seed is not None else int(time.time())
        self.duration = duration  # seconds
        
        # Handle warmth parameter if provided
        if warmth is not None:
            self.warmth = warmth
            self.color_mix = warmth_to_color_mix(warmth)
        else:
            self.color_mix = color_mix or {'white': 0.4, 'pink': 0.4, 'brown': 0.2}
            
        # Use YouTube profile values by default if not explicitly set
        self.rms_target = rms_target if rms_target is not None else YOUTUBE_PROFILE["rms_target"]
        self.peak_ceiling = peak_ceiling if peak_ceiling is not None else YOUTUBE_PROFILE["peak_ceiling"]
        self.lufs_threshold = YOUTUBE_PROFILE["lufs_threshold"]
        self.pre_emphasis = YOUTUBE_PROFILE["pre_emphasis"]
            
        self.lfo_rate = lfo_rate  # Hz, None for no modulation
        self.sample_rate = sample_rate
        self.use_gpu = True  # Always use GPU in this version
        
        # Stereo enhancement options
        self.natural_modulation = natural_modulation
        self.haas_effect = haas_effect
        self.enhanced_stereo = enhanced_stereo

    def validate(self):
        """Validate configuration"""
        # Ensure duration is positive
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        
        # Clamp warmth to 0-100 range if it exists
        if hasattr(self, 'warmth') and self.warmth is not None:
            self.warmth = max(0, min(100, self.warmth))
        
        # Ensure color_mix values are within [0,1]
        if self.color_mix:
            self.color_mix = {k: max(0, min(1, v)) for k, v in self.color_mix.items()}
            
        # Check RMS target in sensible range (-100 to 0 dBFS)
        self.rms_target = max(-100, min(0, self.rms_target))
        
        # Check peak ceiling in sensible range (-100 to 0 dBFS)
        self.peak_ceiling = max(-100, min(0, self.peak_ceiling))
        
        # Clamp LFO rate to non-negative values
        if self.lfo_rate is not None:
            self.lfo_rate = max(0, self.lfo_rate)
            # Treat zero as None (disabled)
            if self.lfo_rate == 0:
                self.lfo_rate = None
        
        # Normalize color mix to sum to 1.0
        self.color_mix = normalize_color_mix(self.color_mix)

    def set_rms_target(self, value):
        """Set custom RMS target"""
        self.rms_target = value
    
    def set_peak_ceiling(self, value):
        """Set custom peak ceiling"""
        self.peak_ceiling = value

# GPU-accelerated noise generator
class NoiseGenerator:
    """GPU-accelerated noise generator for rendering to file"""
    
    def __init__(self, config):
        """Initialize generator with configuration"""
        self.config = config
        self.config.validate()
        
        # Configure memory pool
        self.mem_pool, self.pinned_pool = setup_memory_pool()
        
        # Determine precision based on render duration
        self.precision = determine_precision(self.config.duration)
        logger.info(f"Using {self.precision} precision")
        
        # Initialize block size
        self.optimal_block_size = optimize_block_size()
        self.total_samples = int(self.config.duration * self.config.sample_rate)
        
        # Initialize PRNG and spectral filters
        self._init_gpu()
        self._init_filters()
        
        # Precompute crossfade windows on GPU
        self._fade_in = cp.linspace(0, 1, BLOCK_OVERLAP, dtype=self.precision)
        self._fade_out = cp.linspace(1, 0, BLOCK_OVERLAP, dtype=self.precision)
        
        # Progress tracking
        self.progress_callback = None
        self.is_cancelled = False
        
        # Pre-allocate buffers for largest possible block
        self._buffer_size = self.optimal_block_size
        # Always stereo
        self._white_buffer = cp.zeros((2, self._buffer_size), dtype=self.precision)
        self._pink_buffer = cp.zeros((2, self._buffer_size), dtype=self.precision)
        self._brown_buffer = cp.zeros((2, self._buffer_size), dtype=self.precision)
        self._mixed_buffer = cp.zeros((2, self._buffer_size), dtype=self.precision)
            
    def _init_gpu(self):
        """Initialize GPU context and PRNG"""
        # Create the PRNG with specified seed
        self.rng = cp.random.RandomState(seed=self.config.seed)
        
        # Get device info for logging
        device_info = get_device_info()
        logger.info(f"Using GPU: {device_info['name']} with {device_info['total_memory']:.2f} GB memory")
        logger.info(f"Compute capability: {device_info['compute_capability']}")
        
        # Warm up GPU by allocating a small array
        warmup = cp.zeros((1024, 1024), dtype=self.precision)
        del warmup
    
    def _init_filters(self):
        """Initialize spectral filters for noise generation and processing"""
        # Create the frequency-dependent phase shifts for stereo decorrelation
        # Get block size and frequency resolution
        block_size = self.optimal_block_size
        n_freqs = block_size // 2 + 1
        freq_resolution = self.config.sample_rate / block_size
        
        # Initialize and directly calculate decorrelation phases
        if self.config.enhanced_stereo:
            # Enhanced frequency-dependent decorrelation
            phases = cp.zeros(n_freqs, dtype=self.precision)
            
            # Calculate frequency bin indices correctly using frequency resolution
            low_freq_idx = int(300 / freq_resolution)
            mid_freq_idx = int(1500 / freq_resolution)
            
            # Progressive phase shift based on frequency bands
            # Less decorrelation in bass for better mono compatibility
            phases[:low_freq_idx] = cp.linspace(0, cp.pi/8, low_freq_idx)         # 0-22.5 degrees
            phases[low_freq_idx:mid_freq_idx] = cp.linspace(cp.pi/8, cp.pi/4,     # 22.5-45 degrees
                                                        mid_freq_idx-low_freq_idx)
            phases[mid_freq_idx:] = cp.linspace(cp.pi/4, cp.pi/2.5,                # 45-72 degrees
                                            n_freqs-mid_freq_idx)
        else:
            # Original simple decorrelation (linear 0-45 degrees)
            phases = cp.linspace(0, cp.pi/4, n_freqs)  # 0 to 45 degrees
            # Apply quadratic curve to phase differences (more in mids and highs)
            phases = phases**2 / (cp.pi/4)
            
        # Convert phase shifts to complex exponentials for FFT multiplication
        self.decorrelation_phases = cp.exp(1j * phases)
        
        # Precompute spectral shaping filters for pink and brown noise
        self._precompute_pink_noise_shaping(n_freqs, freq_resolution)
        self._precompute_brown_noise_shaping(n_freqs, freq_resolution)
        
    def _precompute_pink_noise_shaping(self, n_freqs, freq_resolution):
        """Precompute pink noise spectral shaping for -3dB/octave response"""
        # Create frequency bins
        freqs = cp.arange(n_freqs, dtype=self.precision) * freq_resolution
        
        # Create pink noise characteristic: -3dB/octave = 1/f
        # Start with ones for proper handling of DC component
        pink_curve = cp.ones(n_freqs, dtype=self.precision)
        
        # Apply 1/f curve for frequencies > 0 (skip bin 0 which is DC)
        non_dc_mask = freqs > 0
        pink_curve[non_dc_mask] = 1.0 / cp.sqrt(freqs[non_dc_mask])
        
        # Normalize to maintain energy
        pink_curve = pink_curve / cp.sqrt(cp.mean(pink_curve[1:]**2))
        
        # Store the precomputed filter
        self.pink_spectral_filter = pink_curve
        
    def _precompute_brown_noise_shaping(self, n_freqs, freq_resolution):
        """Precompute brown noise spectral shaping coefficients
        
        This creates a single frequency-domain filter that exactly reproduces:
        - -6dB/octave slope (1/f²) for brown noise characteristic
        - Low shelf boost from the original biquad filter (+3dB @ 75Hz)
        - High-pass filter effect from the original SOS filter (20Hz)
        
        By calculating the exact frequency response of the original IIR filters,
        we maintain the precise characteristics while gaining the performance 
        benefits of spectral processing.
        """
        # Create frequency bins (normalized angular frequency ω)
        # ω ranges from 0 to π, corresponding to 0 to Nyquist frequency
        omega = cp.linspace(0, cp.pi, n_freqs, dtype=self.precision)
        
        # Create brown noise characteristic: -6dB/octave = 1/f²
        # Handle DC (ω=0) carefully to avoid division by zero
        brown_curve = cp.ones(n_freqs, dtype=self.precision)
        non_dc = omega > 1e-6  # Avoid division by near-zero
        brown_curve[non_dc] = 1.0 / (omega[non_dc] ** 2)
        
        # Set DC component to a small value instead of infinity
        brown_curve[0] = brown_curve[1] * 10  # Finite value for DC
        
        # Calculate exact frequency response of high-pass Butterworth filter (20Hz)
        hp_cutoff = 20.0 / (self.config.sample_rate / 2)  # Normalized cutoff
        
        # Use SOS form of the filter for better numerical stability
        sos = cusignal.butter(2, hp_cutoff, 'high', output='sos')
        
        # Calculate frequency response for each SOS section and combine
        hp_response = cp.ones(n_freqs, dtype=cp.complex128)
        
        for section in sos:
            b0, b1, b2, a0, a1, a2 = section
            
            # Normalize a0 to 1 (it should already be 1 for Butterworth)
            b0, b1, b2 = b0/a0, b1/a0, b2/a0
            a0, a1, a2 = 1.0, a1/a0, a2/a0
            
            # Calculate frequency response for this section
            # H(ω) = (b0 + b1*e^(-jω) + b2*e^(-j2ω)) / (1 + a1*e^(-jω) + a2*e^(-j2ω))
            numerator = b0 + b1 * cp.exp(-1j * omega) + b2 * cp.exp(-2j * omega)
            denominator = 1.0 + a1 * cp.exp(-1j * omega) + a2 * cp.exp(-2j * omega)
            
            # Multiply by this section's response
            hp_response *= numerator / denominator
        
        # Calculate exact frequency response of low-shelf filter (+3dB @ 75Hz)
        shelf_cutoff = 75.0 / (self.config.sample_rate / 2)  # Normalized cutoff
        gain_db = 3.0
        
        # Calculate biquad coefficients for low shelf
        # Using the standard equation for a low-shelf filter at +3dB
        gain_linear = 10 ** (gain_db / 20.0)
        shelf_A = cp.sqrt(gain_linear)
        
        # Bilinear transform parameter
        tan_half_omega_c = cp.tan(cp.pi * shelf_cutoff / 2)
        
        # Biquad coefficients for low shelf at +3dB
        # These are derived from the Audio EQ Cookbook by Robert Bristow-Johnson
        shelf_b0 = 1.0 + shelf_A * tan_half_omega_c
        shelf_b1 = 2.0 * (shelf_A - 1.0) * tan_half_omega_c
        shelf_b2 = (shelf_A - 1.0) * tan_half_omega_c - shelf_A
        shelf_a0 = 1.0 + tan_half_omega_c / shelf_A
        shelf_a1 = 2.0 * (1.0 - shelf_A) * tan_half_omega_c / shelf_A 
        shelf_a2 = (shelf_A - 1.0) * tan_half_omega_c / shelf_A - 1.0
        
        # Calculate frequency response of low shelf
        # H(ω) = (shelf_b0 + shelf_b1*e^(-jω) + shelf_b2*e^(-j2ω)) / (shelf_a0 + shelf_a1*e^(-jω) + shelf_a2*e^(-j2ω))
        shelf_numerator = shelf_b0 + shelf_b1 * cp.exp(-1j * omega) + shelf_b2 * cp.exp(-2j * omega)
        shelf_denominator = shelf_a0 + shelf_a1 * cp.exp(-1j * omega) + shelf_a2 * cp.exp(-2j * omega)
        shelf_response = shelf_numerator / shelf_denominator
        
        # Combine all filter responses
        # For frequency domain, we multiply the complex responses
        combined_response = brown_curve * cp.abs(hp_response) * cp.abs(shelf_response)
        
        # Ensure DC is exactly zero to prevent offset
        combined_response[0] = 0.0
        
        # Normalize to maintain energy
        combined_response = combined_response / cp.sqrt(cp.mean(combined_response[1:]**2))
        
        # Store the precomputed filter
        self.brown_spectral_filter = combined_response
    
    def _generate_white_noise_block(self, block_size):
        """Generate stereo white noise block on GPU"""
        # Generate independent samples for each channel
        self._white_buffer[0, :block_size] = self.rng.normal(0, 1, block_size).astype(self.precision)
        self._white_buffer[1, :block_size] = self.rng.normal(0, 1, block_size).astype(self.precision)
        return self._white_buffer[:, :block_size]
    
    def _apply_pink_filter(self, white_noise, block_size):
        """Apply pink filter to white noise using spectral method
        
        This is a highly optimized implementation that directly applies the
        characteristic -3dB/octave slope in the frequency domain.
        """
        # Process left and right channels separately
        left = white_noise[0, :block_size]
        right = white_noise[1, :block_size]
        
        # Transform to frequency domain
        left_fft = cufft.rfft(left)
        right_fft = cufft.rfft(right)
        
        # Apply the precomputed spectral shaping
        left_fft = left_fft * self.pink_spectral_filter[:len(left_fft)]
        right_fft = right_fft * self.pink_spectral_filter[:len(right_fft)]
        
        # Transform back to time domain
        self._pink_buffer[0, :block_size] = cufft.irfft(left_fft, n=block_size)
        self._pink_buffer[1, :block_size] = cufft.irfft(right_fft, n=block_size)
        
        return self._pink_buffer[:, :block_size]
    
    def _generate_brown_noise(self, white_noise, block_size):
        """Generate enhanced brown noise from white noise using spectral shaping (stereo)
        
        This optimized implementation applies brown noise characteristics directly
        in the frequency domain, combining:
        1. -6dB/octave slope (for brown noise characteristic)
        2. Low-shelf boost at 75Hz (for enhanced low-end)
        3. High-pass filtering at 20Hz (to remove DC offset)
        
        This is more efficient than sequential time-domain filtering.
        """
        # Process left and right channels separately
        left = white_noise[0, :block_size]
        right = white_noise[1, :block_size]
        
        # Transform to frequency domain
        left_fft = cufft.rfft(left)
        right_fft = cufft.rfft(right)
        
        # Apply the precomputed spectral shaping
        left_fft = left_fft * self.brown_spectral_filter[:len(left_fft)]
        right_fft = right_fft * self.brown_spectral_filter[:len(right_fft)]
        
        # Transform back to time domain
        self._brown_buffer[0, :block_size] = cufft.irfft(left_fft, n=block_size)
        self._brown_buffer[1, :block_size] = cufft.irfft(right_fft, n=block_size)
        
        return self._brown_buffer[:, :block_size]
    
    def _apply_stereo_decorrelation(self, noise_block, block_size):
        """Apply enhanced stereo decorrelation in frequency domain for realistic stereo field"""
        # Get left and right channels
        left = noise_block[0, :block_size]
        right = noise_block[1, :block_size]
        
        # Process right channel for decorrelation (leave left untouched as reference)
        # Convert to frequency domain
        right_fft = cufft.rfft(right)
        
        # Apply phase shift for decorrelation
        right_fft = right_fft * self.decorrelation_phases[:len(right_fft)]
        
        # Convert back to time domain
        right_decorrelated = cufft.irfft(right_fft, n=len(right))
        
        # Recombine channels
        noise_block[1, :block_size] = right_decorrelated
        
        return noise_block
    
    def _apply_haas_effect(self, noise_block, block_size):
        """Apply subtle Haas effect for enhanced stereo width"""
        if not self.config.haas_effect:
            return noise_block
        
        # Create a subtle delay (5-15ms) for one channel
        delay_samples = int(0.008 * self.config.sample_rate)  # 8ms delay
        
        # Only apply to a portion of the spectrum to maintain phase coherence in bass
        
        # Convert right channel to frequency domain
        right_orig = noise_block[1, :block_size]
        right_fft = cufft.rfft(right_orig)
        
        # Calculate frequency bin for cutoff (apply Haas only above ~150Hz)
        n_freqs = len(right_fft)
        freq_resolution = self.config.sample_rate / block_size
        cutoff_bin = int(150 / freq_resolution)
        
        # Create delayed version
        right_delayed = cp.zeros(block_size, dtype=self.precision)
        right_delayed[delay_samples:] = right_orig[:block_size-delay_samples]
        right_delayed_fft = cufft.rfft(right_delayed)
        
        # Create hybrid: original in low frequencies, delayed in mid/high frequencies
        hybrid_fft = cp.copy(right_fft)
        hybrid_fft[cutoff_bin:] = right_delayed_fft[cutoff_bin:] * 0.7 + right_fft[cutoff_bin:] * 0.3
        
        # Convert back to time domain
        hybrid_right = cufft.irfft(hybrid_fft, n=block_size)
        
        # Update the right channel
        noise_block[1, :block_size] = hybrid_right
        
        return noise_block
    
    # New enhanced functions for organic sound

    def _create_complex_modulator(self, t, base_freq, complexity=3, depth=0.02):
        """Create a complex modulation source from multiple sine waves
        
        Parameters:
            t: Time array
            base_freq: Base frequency in Hz
            complexity: Number of summed sine waves
            depth: Overall modulation depth
            
        Returns complex modulation signal with values centered around 1.0
        """
        # Start with a baseline of 1.0
        modulator = cp.ones_like(t, dtype=self.precision)
        
        # Use prime-number-based frequency ratios for non-repeating patterns
        prime_ratios = [1.0, 1.7, 2.3, 3.1, 3.7, 4.1]
        
        # Scale the depth based on complexity (more components = lower individual amplitude)
        component_depth = depth / cp.sqrt(complexity)
        
        # Sum multiple sine waves with different frequencies, phases and amplitudes
        for i in range(complexity):
            # Generate frequency with slight tuning variations for each component
            # Use prime number ratios to avoid simple harmonic patterns
            freq = base_freq * prime_ratios[i % len(prime_ratios)]
            
            # Randomize phase for each component based on seed
            phase_offset = cp.pi * 2 * (((self.config.seed + i * 1000) % 10000) / 10000.0)
            
            # Generate component, decreasing amplitude for higher components
            amplitude = component_depth / (1.0 + i * 0.5)
            component = amplitude * cp.sin(2 * cp.pi * freq * t + phase_offset)
            
            # Add to the modulator
            modulator += component
        
        return modulator

    def _create_filtered_noise_lfo(self, t, cutoff_freq=0.2, depth=0.02, seed_offset=0):
        """Create a filtered noise LFO for organic modulation
        
        Parameters:
            t: Time array
            cutoff_freq: LPF cutoff frequency in Hz
            depth: Modulation depth
            seed_offset: Offset to add to seed for different generators
            
        Returns: Slowly varying filtered noise centered around 1.0
        """
        # Create a separate random generator with a derived seed
        lfo_rng = cp.random.RandomState(seed=self.config.seed + seed_offset)
        
        # Generate white noise slightly longer than needed
        # (for filter startup transients)
        padding = int(self.config.sample_rate / cutoff_freq)
        noise_len = len(t) + padding
        noise = lfo_rng.normal(0, 1, noise_len).astype(self.precision)
        
        # Design a low-pass filter for very slow modulations
        nyquist = self.config.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Use Butterworth filter design
        b, a = cusignal.butter(2, normalized_cutoff, 'low')
        
        # Apply the filter to get a slowly varying signal
        filtered = cusignal.lfilter(b, a, noise)[padding:]  # Skip padding/transient
        
        # Normalize and scale the filtered noise
        filtered = filtered / (cp.std(filtered) * 3 + 1e-10)  # Normalize to roughly ±0.33 range
        
        # Center around 1.0 with specified depth
        return 1.0 + filtered * depth
    
    def _apply_natural_modulation(self, noise_block, block_start_idx, block_size):
        """Add subtle multi-band modulation for more organic sound"""
        if not self.config.natural_modulation:
            return noise_block
                
        # Create time indices for this block
        t_start = block_start_idx / self.config.sample_rate
        t = cp.linspace(t_start, t_start + block_size/self.config.sample_rate, block_size, endpoint=False)
        
        # Create complex modulation sources
        # Slow modulation for lows (0.05-0.07 Hz)
        low_mod = self._create_filtered_noise_lfo(t, cutoff_freq=0.05, depth=0.03, seed_offset=1)
        
        # Medium modulation for mids (0.1-0.2 Hz) 
        mid_mod = self._create_complex_modulator(t, base_freq=0.13, complexity=3, depth=0.025)
        
        # Faster but subtle modulation for highs (0.2-0.3 Hz)
        high_mod = self._create_complex_modulator(t, base_freq=0.27, complexity=4, depth=0.02)
        
        # Process left channel with spectral modulation
        left_fft = cufft.rfft(noise_block[0, :block_size])
        
        # Get frequency bin indices using correct frequency resolution
        n_bins = len(left_fft)
        freq_resolution = self.config.sample_rate / block_size
        low_idx = int(300 / freq_resolution)
        mid_idx = int(1500 / freq_resolution)
        
        # Create smooth transition envelopes between bands
        # Create smooth envelopes with crossfades between bands
        low_env = cp.ones(low_idx, dtype=self.precision)
        low_transition = min(int(low_idx * 0.1), low_idx)  # 10% transition, with safety check
        if low_transition > 0:
            low_env[-low_transition:] = cp.linspace(1.0, 0.0, low_transition, dtype=self.precision)
        
        mid_env = cp.zeros(mid_idx - low_idx, dtype=self.precision)
        mid_transition_low = min(int((mid_idx - low_idx) * 0.1), mid_idx - low_idx)
        mid_transition_high = min(int((mid_idx - low_idx) * 0.1), mid_idx - low_idx - mid_transition_low)
        
        if mid_transition_low > 0:
            mid_env[:mid_transition_low] = cp.linspace(0.0, 1.0, mid_transition_low, dtype=self.precision)
        if mid_transition_high > 0:
            mid_env[-(mid_transition_high):] = cp.linspace(1.0, 0.0, mid_transition_high, dtype=self.precision)
        # Fill the middle part with ones
        if mid_transition_low < len(mid_env) - mid_transition_high:
            mid_env[mid_transition_low:len(mid_env)-mid_transition_high] = 1.0
        
        high_env = cp.zeros(n_bins - mid_idx, dtype=self.precision)
        high_transition = min(int((n_bins - mid_idx) * 0.1), n_bins - mid_idx)
        if high_transition > 0:
            high_env[:high_transition] = cp.linspace(0.0, 1.0, high_transition, dtype=self.precision)
        high_env[high_transition:] = 1.0
        
        # Apply modulation to left channel frequency bands with complex modulators
        # Low frequencies: use filtered noise LFO for organic variation
        mod_factor_low = low_mod[0] - 1.0  # Extract modulation factor (centered at 0)
        left_fft[:low_idx] = left_fft[:low_idx] * (1.0 + mod_factor_low * low_env)
        
        # Mid frequencies: use complex sine modulator
        mod_factor_mid = mid_mod[0] - 1.0
        left_fft[low_idx:mid_idx] = left_fft[low_idx:mid_idx] * (1.0 + mod_factor_mid * mid_env)
        
        # High frequencies: use more complex modulator with higher component count
        mod_factor_high = high_mod[0] - 1.0
        left_fft[mid_idx:] = left_fft[mid_idx:] * (1.0 + mod_factor_high * high_env)
        
        # Convert back to time domain
        noise_block[0, :block_size] = cufft.irfft(left_fft, n=block_size)
        
        # Process right channel with phase offset for enhanced decorrelation
        right_fft = cufft.rfft(noise_block[1, :block_size])
        
        # Apply modulation to right channel with phase offset for enhanced stereo effect
        # Dynamically modulate the phase offset itself for added organic quality
        phase_offset = cp.pi / 4  # 45 degree base offset
        phase_variation = cp.pi / 12  # ±15 degree variation
        current_phase = phase_offset + phase_variation * cp.sin(2 * cp.pi * 0.037 * t_start)
        
        # Apply the same modulators but with phase offset to right channel
        right_fft[:low_idx] = right_fft[:low_idx] * (1.0 + mod_factor_low * cp.cos(current_phase) * low_env)
        right_fft[low_idx:mid_idx] = right_fft[low_idx:mid_idx] * (1.0 + mod_factor_mid * cp.cos(current_phase + cp.pi/6) * mid_env)
        right_fft[mid_idx:] = right_fft[mid_idx:] * (1.0 + mod_factor_high * cp.cos(current_phase + cp.pi/3) * high_env)
        
        # Convert back to time domain
        noise_block[1, :block_size] = cufft.irfft(right_fft, n=block_size)
        
        return noise_block

    def _apply_dynamic_stereo_parameters(self, noise_block, block_start_idx, block_size):
        """Apply dynamic, slowly evolving stereo parameters"""
        
        # Only apply if both stereo enhancement and natural modulation are enabled
        if not (self.config.enhanced_stereo and self.config.natural_modulation):
            return noise_block
        
        # Create slowly varying modulation for stereo parameters
        t_start = block_start_idx / self.config.sample_rate
        
        # Generate modulator for decorrelation amount (very slow: 0.023 Hz)
        decorr_mod = 1.0 + 0.15 * cp.sin(2 * cp.pi * 0.023 * t_start)
        
        # Generate modulator for Haas delay time (also very slow: 0.017 Hz)
        # Add a different phase offset so it doesn't correlate with decorrelation modulation
        delay_mod = 1.0 + 0.2 * cp.sin(2 * cp.pi * 0.017 * t_start + 0.5)
        
        # Apply dynamic decorrelation if enhanced stereo is enabled
        if self.config.enhanced_stereo:
            # Get the right channel and convert to frequency domain
            right = noise_block[1, :block_size]
            right_fft = cufft.rfft(right)
            
            # Create frequency-dependent phase shifts with time-varying strength
            n_freqs = len(right_fft)
            freq_resolution = self.config.sample_rate / block_size
            
            # Calculate frequency band boundaries
            low_freq_idx = int(300 / freq_resolution)
            mid_freq_idx = int(1500 / freq_resolution)
            
            # Create dynamic phases with modulated strength
            phases = cp.zeros(n_freqs, dtype=self.precision)
            
            # Scale phase offsets by the decorrelation modulator
            phases[:low_freq_idx] = cp.linspace(0, cp.pi/8, low_freq_idx) * decorr_mod
            phases[low_freq_idx:mid_freq_idx] = cp.linspace(cp.pi/8, cp.pi/4, mid_freq_idx-low_freq_idx) * decorr_mod
            phases[mid_freq_idx:] = cp.linspace(cp.pi/4, cp.pi/2.5, n_freqs-mid_freq_idx) * decorr_mod
            
            # Convert to complex exponentials for FFT multiplication
            dynamic_decorr_phases = cp.exp(1j * phases)
            
            # Apply the dynamic phase shift
            right_fft = right_fft * dynamic_decorr_phases
            
            # Convert back to time domain
            noise_block[1, :block_size] = cufft.irfft(right_fft, n=block_size)
        
        # Apply dynamic Haas effect if enabled
        if self.config.haas_effect:
            # Base delay is 8ms
            base_delay = 0.008  # seconds
            delay_variation = 0.002  # ±2ms variation
            
            # Calculate current delay time with modulation
            current_delay = base_delay + delay_variation * (delay_mod - 1.0)
            
            # Convert to samples
            delay_samples = int(current_delay * self.config.sample_rate)
            
            # Get right channel for processing
            right_orig = noise_block[1, :block_size]
            
            # Create delayed version of right channel
            right_delayed = cp.zeros(block_size, dtype=self.precision)
            right_delayed[delay_samples:] = right_orig[:block_size-delay_samples]
            
            # Convert both to frequency domain
            right_fft = cufft.rfft(right_orig)
            right_delayed_fft = cufft.rfft(right_delayed)
            
            # Apply frequency-dependent mixing of original and delayed signals
            # Apply original in low frequencies, delayed in mid/high frequencies
            n_freqs = len(right_fft)
            cutoff_bin = int(150 / (self.config.sample_rate / block_size))
            
            # Create crossfade between original and delayed signal
            xfade_width = int(cutoff_bin * 0.5)  # 50% of cutoff for smooth transition
            
            # Create transition curve (from 0 to 1)
            transition = cp.zeros(n_freqs, dtype=self.precision)
            transition[:cutoff_bin-xfade_width] = 0.0  # Below crossfade: all original
            transition[cutoff_bin+xfade_width:] = 0.7  # Above crossfade: 70% delayed, 30% original
            
            # Smooth crossfade in the transition region
            if xfade_width > 0:
                transition[cutoff_bin-xfade_width:cutoff_bin+xfade_width] = cp.linspace(
                    0.0, 0.7, 2*xfade_width, dtype=self.precision)
            
            # Create the hybrid signal
            hybrid_fft = right_fft * (1.0 - transition) + right_delayed_fft * transition
            
            # Convert back to time domain
            noise_block[1, :block_size] = cufft.irfft(hybrid_fft, n=block_size)
        
        return noise_block

    def _apply_micro_pitch_variations(self, noise_block, block_start_idx, block_size):
        """Apply subtle micro-pitch variations for added naturalness"""
        # Only apply if natural modulation is enabled
        if not self.config.natural_modulation:
            return noise_block
            
        # Create extremely slow modulation sources for pitch variations
        t_start = block_start_idx / self.config.sample_rate
        
        # Create different pitch variation rates for different frequency bands
        # These are extremely slow to avoid obvious wobbling effects
        low_pitch_mod = 0.005 * cp.sin(2 * cp.pi * 0.011 * t_start)           # ±0.5% variation
        mid_pitch_mod = 0.003 * cp.sin(2 * cp.pi * 0.019 * t_start + 0.7)     # ±0.3% variation
        high_pitch_mod = 0.002 * cp.sin(2 * cp.pi * 0.031 * t_start + 1.4)    # ±0.2% variation
        
        # Process each channel
        for ch in range(2):
            # Convert to frequency domain
            channel_fft = cufft.rfft(noise_block[ch, :block_size])
            
            # Get frequency bin indices
            n_bins = len(channel_fft)
            freq_resolution = self.config.sample_rate / block_size
            low_idx = int(300 / freq_resolution)
            mid_idx = int(1500 / freq_resolution)
            
            # Create arrays to hold the resampled spectrum
            resampled_fft = cp.zeros_like(channel_fft, dtype=cp.complex128)
            
            # Process each frequency band separately with different pitch variations
            # For low frequencies: apply low_pitch_mod
            # For this implementation, we'll use a simplified approach using bin shifting
            
            # Low frequencies
            for i in range(1, low_idx):  # Skip DC (bin 0)
                # Calculate fractional bin shift
                bin_shift = i * low_pitch_mod  # Proportional to frequency
                bin_int = int(i + bin_shift)
                bin_frac = (i + bin_shift) - bin_int
                
                # Ensure we stay within valid bin range
                if 0 <= bin_int < n_bins-1:
                    # Linear interpolation between bins for fractional shifts
                    resampled_fft[i] = channel_fft[bin_int] * (1 - bin_frac) + channel_fft[bin_int+1] * bin_frac
                    
            # Mid frequencies
            for i in range(low_idx, mid_idx):
                bin_shift = i * mid_pitch_mod
                bin_int = int(i + bin_shift)
                bin_frac = (i + bin_shift) - bin_int
                
                if 0 <= bin_int < n_bins-1:
                    resampled_fft[i] = channel_fft[bin_int] * (1 - bin_frac) + channel_fft[bin_int+1] * bin_frac
                    
            # High frequencies
            for i in range(mid_idx, n_bins):
                bin_shift = i * high_pitch_mod
                bin_int = int(i + bin_shift)
                bin_frac = (i + bin_shift) - bin_int
                
                if 0 <= bin_int < n_bins-1:
                    resampled_fft[i] = channel_fft[bin_int] * (1 - bin_frac) + channel_fft[bin_int+1] * bin_frac
            
            # Preserve DC component
            resampled_fft[0] = channel_fft[0]
            
            # Convert back to time domain
            noise_block[ch, :block_size] = cufft.irfft(resampled_fft, n=block_size)
        
        return noise_block
        
    def _apply_lfo_modulation(self, noise_block, block_start_idx, block_size):
        """Apply LFO modulation if enabled"""
        if not self.config.lfo_rate:
            return noise_block
        
        # Create LFO modulation envelope
        block_time = block_size / self.config.sample_rate
        
        # Generate time indices for this block
        t_start = block_start_idx / self.config.sample_rate
        t = cp.linspace(t_start, t_start + block_time, block_size, endpoint=False)
        
        # Create sinusoidal modulation (±1dB)
        modulation_depth = 10**(1.0/20) - 10**(-1.0/20)  # ±1dB in linear scale
        modulation = 1.0 + modulation_depth/2 * cp.sin(2 * cp.pi * self.config.lfo_rate * t)
        
        # Apply same modulation to both channels
        noise_block[0, :block_size] = noise_block[0, :block_size] * modulation
        noise_block[1, :block_size] = noise_block[1, :block_size] * modulation
        
        return noise_block
    
    def _blend_noise_colors(self, white, pink, brown, color_mix, block_size):
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
        
        # Mix stereo signals (channel-wise)
        self._mixed_buffer[0, :block_size] = (
            white[0, :block_size] * white_gain + 
            pink[0, :block_size] * pink_gain + 
            brown[0, :block_size] * brown_gain
        )
        
        self._mixed_buffer[1, :block_size] = (
            white[1, :block_size] * white_gain + 
            pink[1, :block_size] * pink_gain + 
            brown[1, :block_size] * brown_gain
        )
        
        return self._mixed_buffer[:, :block_size]
    
    def _apply_gain_and_limiting(self, noise_block, target_rms, peak_ceiling, block_size):
        """Multi-stage gain and limiting for better sound quality using efficient linear calculations"""
        # First stage: Apply target gain
        # For stereo, process combined energy
        # Calculate RMS across both channels
        combined_rms = cp.sqrt(cp.mean(
            (noise_block[0, :block_size]**2 + noise_block[1, :block_size]**2) / 2
        ))
        
        # Calculate gain needed
        target_linear = 10 ** (target_rms / 20.0)
        gain = target_linear / (combined_rms + 1e-10)
        
        # Apply gain to both channels
        noise_block[:, :block_size] = noise_block[:, :block_size] * gain
        
        # Second stage: Soft knee limiter (more musical)
        # Threshold in linear domain
        threshold_db = peak_ceiling - 6  # 6dB below ceiling
        threshold = 10 ** (threshold_db / 20.0)
        
        # Pre-calculate knee parameters in linear domain
        ratio = 4.0  # 4:1 compression
        ratio_inverse = 1.0 / ratio
        
        # Knee width in dB and linear
        knee_width_db = 6.0  # 6dB knee width
        knee_start = threshold * (10 ** (-knee_width_db / 40.0))  # -3dB (half knee width) from threshold
        knee_end = threshold * (10 ** (knee_width_db / 40.0))     # +3dB (half knee width) from threshold
        
        # Pre-calculate constants for the knee's quadratic curve
        # These formulas match the dB curve but operate in linear domain
        knee_range = knee_end - knee_start
        knee_factor_a = (ratio_inverse - 1.0) / (2.0 * knee_range)
        knee_factor_b = (2.0 * knee_start * (1.0 - ratio_inverse)) / (2.0 * knee_range)
        
        # Process each channel separately
        for ch in range(2):
            # Calculate input levels (absolute value)
            input_level = cp.abs(noise_block[ch, :block_size])
            
            # Initialize gain reduction array (1.0 = no reduction)
            gain_reduction = cp.ones(block_size, dtype=self.precision)
            
            # Create masks for the different regions
            below_knee = input_level <= knee_start
            above_knee = input_level >= knee_end
            in_knee = ~(below_knee | above_knee)
            
            # Calculate gain reduction for samples in knee region (quadratic curve)
            if cp.any(in_knee):
                # Apply quadratic curve in linear domain
                x = input_level[in_knee]
                gain_reduction[in_knee] = 1.0 / (1.0 + knee_factor_a * (x - knee_start)**2 + knee_factor_b * (x - knee_start))
            
            # Calculate gain reduction for samples above knee (full compression)
            if cp.any(above_knee):
                # Apply fixed ratio compression in linear domain
                x = input_level[above_knee]
                # Linear equation equivalent to the dB calculation
                gain_reduction[above_knee] = (threshold * ((knee_end / threshold)**(1.0-ratio_inverse))) / (x ** (1.0 - ratio_inverse))
            
            # Apply gain reduction
            noise_block[ch, :block_size] = noise_block[ch, :block_size] * gain_reduction
        
        # Third stage: Hard limiting for true peaks (simple and fast)
        peak_threshold = 10 ** (peak_ceiling / 20.0)
        
        # Check both stereo channels
        peak_left = cp.max(cp.abs(noise_block[0, :block_size]))
        peak_right = cp.max(cp.abs(noise_block[1, :block_size]))
        peak = cp.maximum(peak_left, peak_right)
        
        # Only apply hard limiting if needed
        if peak > peak_threshold:
            limiting_gain = peak_threshold / peak
            noise_block[:, :block_size] = noise_block[:, :block_size] * limiting_gain
            gain_db = float(20 * cp.log10(limiting_gain))
            # Only log significant limiting
            if limiting_gain < 0.8:  # More than ~2dB reduction
                logger.info(f"Applied limiting: {gain_db:.1f} dB")
            else:
                logger.debug(f"Applied limiting: {gain_db:.1f} dB")
        
        return noise_block
    
    def _apply_true_peak_limiting(self, noise_block, peak_ceiling, block_size):
        """Apply true-peak limiting with 4x oversampling using optimized FFT-based approach"""
        # True peak detection using 4x oversampling
        oversampling_factor = 4
        
        # Quick peak check first to avoid unnecessary processing
        peak_left = cp.max(cp.abs(noise_block[0, :block_size]))
        peak_right = cp.max(cp.abs(noise_block[1, :block_size]))
        peak = cp.maximum(peak_left, peak_right)
            
        peak_threshold = 10 ** (peak_ceiling / 20.0)
        
        # Only do expensive true-peak calculation if we're close to the threshold
        if peak > peak_threshold * 0.8:  # Within ~2dB of threshold
            # Custom FFT-based 4x oversampling - more efficient than cusignal.resample
            
            # Process each channel with direct FFT zero-padding technique
            def fft_upsample(channel):
                # Step 1: Transform to frequency domain
                X = cufft.rfft(channel)
                
                # Step 2: Create zero-padded spectrum (frequency domain)
                n_orig = len(X)  # Number of original frequency bins
                n_padded = n_orig + (oversampling_factor - 1) * (len(channel) // 2)
                
                # Create padded spectrum with zeros
                X_padded = cp.zeros(n_padded, dtype=cp.complex128)
                
                # Copy original spectrum to beginning of padded spectrum
                X_padded[:n_orig] = X
                
                # Step 3: Inverse FFT to get oversampled signal
                # The length parameter ensures we get exactly 4x the original length
                return cufft.irfft(X_padded, n=block_size * oversampling_factor)
            
            # Apply to both channels
            upsampled_left = fft_upsample(noise_block[0, :block_size])
            upsampled_right = fft_upsample(noise_block[1, :block_size])
            
            # Find true peak across both channels
            true_peak_left = cp.max(cp.abs(upsampled_left))
            true_peak_right = cp.max(cp.abs(upsampled_right))
            true_peak = cp.maximum(true_peak_left, true_peak_right)
            true_peak_db = float(20 * cp.log10(true_peak + 1e-10))
            
            # Apply limiting if needed
            if true_peak > peak_threshold:
                limiting_gain = peak_threshold / true_peak
                noise_block[:, :block_size] = noise_block[:, :block_size] * limiting_gain
                gain_db = float(20 * cp.log10(limiting_gain))
                # Only log significant limiting
                if limiting_gain < 0.9:  # More than ~1dB reduction
                    logger.info(f"Applied true-peak limiting: {gain_db:.1f} dB")
                else:
                    logger.debug(f"Applied true-peak limiting: {gain_db:.1f} dB")
        
        return noise_block
    
    def _apply_pre_emphasis(self, noise_block, block_size):
        """Apply pre-emphasis filter for better codec performance on YouTube"""
        if not self.config.pre_emphasis:
            return noise_block
        
        # Pre-emphasis filter (boost above 5kHz for YouTube codec)
        nyquist = self.config.sample_rate / 2
        cutoff = 5000 / nyquist
        
        # Instead of caching this filter, just create it on demand as it's rarely called
        # Design shelving filter
        b, a = cusignal.butter(2, cutoff, 'high', analog=False)
        
        # Apply gain to shelving filter - ensure b is a CuPy array
        b = cp.array(b) * 1.5  # +3.5 dB boost
        
        # Process each channel
        emphasized_left = cusignal.lfilter(b, a, noise_block[0, :block_size])
        emphasized_right = cusignal.lfilter(b, a, noise_block[1, :block_size])
        
        # Re-normalize to maintain RMS
        original_rms = cp.sqrt(cp.mean(
            (noise_block[0, :block_size]**2 + noise_block[1, :block_size]**2) / 2
        ))
        emphasized_rms = cp.sqrt(cp.mean(
            (emphasized_left**2 + emphasized_right**2) / 2
        ))
        gain_factor = original_rms / (emphasized_rms + 1e-10)
        
        noise_block[0, :block_size] = emphasized_left * gain_factor
        noise_block[1, :block_size] = emphasized_right * gain_factor
        
        return noise_block
    
    def generate_block(self, block_size, block_start_idx=0):
        """Generate a block of noise with the specified configuration"""
        # Generate white noise base
        white_noise = self._generate_white_noise_block(block_size)
        
        # Determine which noise types to generate based on color mix
        # This optimization skips processing for noise types with negligible contribution
        color_mix = self.config.color_mix
        generate_pink = color_mix.get('pink', 0) > 0.001  # Skip if < 0.1% contribution
        generate_brown = color_mix.get('brown', 0) > 0.001  # Skip if < 0.1% contribution
        
        # Generate pink noise only if needed
        if generate_pink:
            pink_noise = self._apply_pink_filter(white_noise, block_size)
        else:
            # Use zeros buffer if pink noise not needed
            pink_noise = cp.zeros_like(white_noise[:, :block_size])
        
        # Generate brown noise only if needed
        if generate_brown:
            brown_noise = self._generate_brown_noise(white_noise, block_size)
        else:
            # Use zeros buffer if brown noise not needed
            brown_noise = cp.zeros_like(white_noise[:, :block_size])
        
        # Blend according to color mix
        mixed_noise = self._blend_noise_colors(
            white_noise, pink_noise, brown_noise, self.config.color_mix, block_size
        )
        
        # Apply stereo decorrelation
        mixed_noise = self._apply_stereo_decorrelation(mixed_noise, block_size)
        
        # Apply Haas effect for enhanced stereo imaging if enabled
        if self.config.haas_effect:
            mixed_noise = self._apply_haas_effect(mixed_noise, block_size)
        
        # NEW! Apply dynamic stereo parameters for evolving stereo field
        mixed_noise = self._apply_dynamic_stereo_parameters(mixed_noise, block_start_idx, block_size)
            
        # Apply natural modulation for more organic sound if enabled
        if self.config.natural_modulation:
            mixed_noise = self._apply_natural_modulation(mixed_noise, block_start_idx, block_size)
        
        # NEW! Apply subtle micro-pitch variations for added naturalness
        mixed_noise = self._apply_micro_pitch_variations(mixed_noise, block_start_idx, block_size)
        
        # Apply LFO modulation if enabled
        mixed_noise = self._apply_lfo_modulation(mixed_noise, block_start_idx, block_size)
        
        # Apply multi-stage gain and limiting
        mixed_noise = self._apply_gain_and_limiting(
            mixed_noise, self.config.rms_target, self.config.peak_ceiling, block_size
        )
        
        # Apply true-peak limiting
        mixed_noise = self._apply_true_peak_limiting(mixed_noise, self.config.peak_ceiling, block_size)
        
        # Apply pre-emphasis if enabled in profile
        mixed_noise = self._apply_pre_emphasis(mixed_noise, block_size)
        
        return mixed_noise
    
    def _adjust_block_size(self, current_size, free_memory, used_memory):
        """Adaptively adjust block size based on memory usage"""
        # Baseline - don't go below this for efficiency
        min_block_size = 2**16  # ~65ms @ 44.1kHz
        
        # Available memory in MB
        available_mb = free_memory / (1024 * 1024)
        used_mb = used_memory / (1024 * 1024)
        
        # Thresholds for adjustment
        low_mem_threshold_mb = 200  # MB
        high_mem_threshold_mb = 1000  # MB
        
        # If memory is running low, reduce block size
        if available_mb < low_mem_threshold_mb:
            # Severe memory pressure - reduce by half
            adjusted_size = max(min_block_size, current_size // 2)
            logger.info(f"Memory pressure detected ({available_mb:.1f}MB free), "
                        f"reducing block size to {adjusted_size}")
            return adjusted_size
        
        # If plenty of memory and currently using small blocks, increase
        if available_mb > high_mem_threshold_mb and current_size < FFT_BLOCK_SIZE:
            # Good memory availability - increase cautiously
            adjusted_size = min(FFT_BLOCK_SIZE * 2, current_size * 2)
            logger.info(f"Memory available ({available_mb:.1f}MB free), "
                        f"increasing block size to {adjusted_size}")
            return adjusted_size
        
        # Otherwise keep current size
        return current_size
    
    def generate_to_file(self, output_path, progress_callback=None, make_loop=True):
        """Generate noise and save to file with progress tracking"""
        import soundfile as sf
        
        self.progress_callback = progress_callback
        self.is_cancelled = False
        
        # Verify GPU is working before proceeding
        try:
            # Allocate a small test array to verify GPU is working
            test_array = cp.zeros((1024, 1024), dtype=self.precision)
            del test_array
        except cp.cuda.memory.OutOfMemoryError:
            error_msg = "GPU out of memory. Try a smaller render duration."
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        except Exception as e:
            error_msg = f"GPU initialization error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        
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
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Create output file - always stereo (2 channels)
            with sf.SoundFile(
                output_path, 
                mode='w', 
                samplerate=self.config.sample_rate,
                channels=2,
                format=output_format,
                subtype=subtype
            ) as f:
                # Process in blocks to manage memory
                samples_remaining = self.total_samples
                samples_written = 0
                block_size = self.optimal_block_size
                
                # Large overlap for smooth transitions - stored on GPU for efficiency
                overlap_buffer_gpu = None
                
                # Memory check counters
                memory_check_interval = 5  # Check memory every N blocks
                block_counter = 0
                
                while samples_remaining > 0 and not self.is_cancelled:
                    # Check memory periodically and adjust block size if needed
                    block_counter += 1
                    if block_counter % memory_check_interval == 0:
                        free_memory, used_memory = get_available_memory()
                        block_size = self._adjust_block_size(block_size, free_memory, used_memory)
                    
                    # Determine block size for this iteration
                    current_block_size = min(block_size, samples_remaining + BLOCK_OVERLAP)
                    
                    # Generate block using context manager for CUDA streams
                    with cuda_stream() as compute_stream:
                        with compute_stream:
                            noise_block = self.generate_block(current_block_size, samples_written)
                    
                    # Calculate safe overlap size
                    overlap = min(BLOCK_OVERLAP, current_block_size - 1, samples_remaining)
                    
                    # Transpose noise_block for soundfile's expected format (channels as columns)
                    # Keep on GPU for crossfade operations
                    transposed_block = cp.transpose(noise_block[:, :current_block_size])
                    
                    # Apply overlap from previous block if available - all operations on GPU
                    if overlap_buffer_gpu is not None:
                        # Reshape fade arrays for broadcasting correctly on GPU
                        fade_in_shaped = self._fade_in[:overlap].reshape(-1, 1)
                        fade_out_shaped = self._fade_out[:overlap].reshape(-1, 1)
                        
                        # Perform crossfade directly on GPU
                        transposed_block[:overlap, :] = (
                            transposed_block[:overlap, :] * fade_in_shaped + 
                            overlap_buffer_gpu[:overlap, :] * fade_out_shaped
                        )
                    
                    # Save overlap buffer for next iteration (on GPU)
                    if samples_remaining > overlap:
                        overlap_buffer_gpu = transposed_block[-overlap:, :].copy()
                    
                    # Now move the processed data to CPU only when needed for file writing
                    with cuda_stream() as transfer_stream:
                        with transfer_stream:
                            # Only transfer the part we need to write (exclude future overlap)
                            write_length = min(len(transposed_block) - overlap, samples_remaining)
                            output_data = cp.asnumpy(transposed_block[:write_length])
                    
                    # Write to file (already excluding overlap for next block)
                    f.write(output_data)
                    
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
                        # Print progress to console less frequently
                        if current_time - last_progress_update >= progress_throttle or progress_change >= 5.0:
                            logger.info(f"Progress: {progress_percent:.1f}%")
                            last_progress_update = current_time
                            last_progress_value = progress_percent
                            
                    # Free memory periodically
                    if block_counter % (memory_check_interval * 2) == 0:
                        # Release unused memory back to the pool
                        self.mem_pool.free_all_blocks()
                        self.pinned_pool.free_all_blocks()
                        
            # Apply seamless looping as a separate post-processing step if requested
            if make_loop and self.config.duration >= 5.0:  # Only for files longer than 5 seconds
                # Calculate appropriate crossfade duration based on file length
                # Use 2-5% of total duration with limits
                crossfade_duration = min(2.0, max(0.5, self.config.duration * 0.03))
                
                # Import the necessary modules for post-processing
                import numpy as np
                from scipy import signal
                
                # Create the seamless loop
                logger.info(f"Applying seamless loop with {crossfade_duration:.2f}s crossfade...")
                
                # Define the post-processing function for creating loops
                def create_seamless_loop(file_path, crossfade_duration=1.0):
                    """Create a seamless loop by crossfading the beginning and end of an audio file."""
                    import tempfile
                    import os
                    
                    try:
                        # Get file info
                        info = sf.info(file_path)
                        sample_rate = info.samplerate
                        channels = info.channels
                        total_frames = info.frames
                        
                        # Calculate crossfade samples (ensure it's even for better FFT)
                        crossfade_samples = int(crossfade_duration * sample_rate)
                        if crossfade_samples % 2 == 1:
                            crossfade_samples += 1
                            
                        # Check if file is long enough for crossfade
                        if total_frames <= crossfade_samples * 2:
                            logger.warning(f"File too short for {crossfade_duration}s crossfade, skipping loop creation")
                            return False
                            
                        # Load the entire file
                        data, sr = sf.read(file_path)
                        
                        # Extract beginning and end sections
                        beginning = data[:crossfade_samples].copy()
                        ending = data[-crossfade_samples:].copy()
                        
                        # Create temporary file for processing
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # Perform phase alignment between beginning and end
                        # For each channel if stereo
                        if channels > 1:
                            # Process each channel
                            for ch in range(channels):
                                # Find correlation between beginning and ending
                                corr = signal.correlate(ending[:, ch], beginning[:, ch], mode='same')
                                lag = np.argmax(corr) - (len(corr) // 2)
                                
                                # Adjust lag to be within reasonable bounds
                                if abs(lag) > crossfade_samples // 4:
                                    lag = 0  # If alignment seems poor, don't shift
                                
                                # Apply shift based on correlation
                                if lag > 0:
                                    # Shift ending forward
                                    ending[lag:, ch] = ending[:-lag, ch]
                                elif lag < 0:
                                    # Shift beginning forward
                                    beginning[-lag:, ch] = beginning[:lag, ch]
                        else:
                            # Same process for mono
                            corr = signal.correlate(ending, beginning, mode='same')
                            lag = np.argmax(corr) - (len(corr) // 2)
                            if abs(lag) <= crossfade_samples // 4:
                                if lag > 0:
                                    ending[lag:] = ending[:-lag]
                                elif lag < 0:
                                    beginning[-lag:] = beginning[:lag]
                        
                        # Create equal power crossfade windows (sine-squared and cosine-squared)
                        t = np.linspace(0, np.pi/2, crossfade_samples)
                        fade_in = np.sin(t) ** 2
                        fade_out = np.cos(t) ** 2
                        
                        # Apply crossfade
                        if channels > 1:
                            # Reshape for broadcasting with multichannel audio
                            fade_in = fade_in.reshape(-1, 1)
                            fade_out = fade_out.reshape(-1, 1)
                        
                        # Create crossfaded section
                        crossfaded = beginning * fade_in + ending * fade_out
                        
                        # Create final looped audio: everything except the last crossfade_samples, 
                        # then append the crossfaded section
                        final_data = np.concatenate([data[:-crossfade_samples], crossfaded])
                        
                        # Write to temporary file first (to ensure it's valid)
                        sf.write(temp_path, final_data, sr)
                        
                        # If successful, replace original file
                        os.replace(temp_path, file_path)
                        
                        logger.info(f"Created seamless loop with {crossfade_duration}s crossfade and phase alignment")
                        return True
                        
                    except Exception as e:
                        logger.error(f"Failed to create seamless loop: {str(e)}")
                        # Clean up temp file if it exists
                        if 'temp_path' in locals() and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass
                        return False
                
                # Apply the seamless loop
                create_seamless_loop(output_path, crossfade_duration)
        
        except Exception as e:
            logger.error(f"Error during file generation: {str(e)}")
            return {
                "error": f"Failed to generate file: {str(e)}",
                "success": False
            }
            
        # Calculate processing metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Measure output file statistics
        result = {
            "processing_time": processing_time,
            "samples_generated": samples_written,
            "real_time_factor": self.config.duration / processing_time,
            "success": True
        }
        
        logger.info(f"Generated {self.config.duration:.1f}s of stereo noise in {processing_time:.1f}s "
                    f"(speed: {result['real_time_factor']:.1f}x real-time)")
        
        # Measure true peak and RMS on disk to verify
        try:
            audio_data, sample_rate = sf.read(output_path)
            
            # Calculate metrics on both channels
            peak_left = np.max(np.abs(audio_data[:, 0]))
            peak_right = np.max(np.abs(audio_data[:, 1]))
            peak_db = 20 * np.log10(max(peak_left, peak_right) + 1e-10)
            
            # Calculate RMS
            rms_left = np.sqrt(np.mean(audio_data[:, 0]**2))
            rms_right = np.sqrt(np.mean(audio_data[:, 1]**2))
            rms_db = 20 * np.log10((rms_left + rms_right) / 2 + 1e-10)
            
            # Accurate LUFS measurement using pyloudnorm if available
            try:
                import pyloudnorm as pyln
                
                # Create BS.1770-4 meter
                meter = pyln.Meter(sample_rate)  # defaults to BS.1770-4
                
                # Measure integrated loudness (stereo)
                lufs = meter.integrated_loudness(audio_data)
                
                logger.info(f"Measured accurate LUFS using pyloudnorm (BS.1770-4)")
            except ImportError:
                # Fall back to approximate LUFS calculation
                lufs = rms_db + 3.0  # Simple approximation
                logger.info(f"Using approximate LUFS calculation (pyloudnorm not installed)")
            
            # Store measurements in results
            result["peak_db"] = peak_db
            result["rms_db"] = rms_db
            result["integrated_lufs"] = lufs
            
            logger.info(f"Output file metrics: {peak_db:.1f} dBFS peak, {lufs:.1f} LUFS")
            
            # Log if looping was applied
            if make_loop and self.config.duration >= 5.0:
                result["looped"] = True
                logger.info("File has been processed for seamless looping")
            else:
                result["looped"] = False
                
        except Exception as e:
            logger.error(f"Error measuring output file: {e}")
            result["warning"] = f"Generated successfully, but encountered an error measuring the output: {str(e)}"
        
        return result
        
    def cancel_generation(self):
        """Cancel ongoing generation"""
        self.is_cancelled = True
        
    def __del__(self):
        """Clean up resources when generator is destroyed"""
        try:
            # Free all memory
            self.mem_pool.free_all_blocks()
            self.pinned_pool.free_all_blocks()
        except Exception:
            pass


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
            "rms_target": -20.0,  # YouTube standard level
            "lfo_rate": 0.1
        },
        "white_only": {
            "color_mix": {'white': 1.0, 'pink': 0.0, 'brown': 0.0},
            "rms_target": -20.0,  # YouTube standard level
            "lfo_rate": 0.1
        },
        "pink_only": {
            "color_mix": {'white': 0.0, 'pink': 1.0, 'brown': 0.0},
            "rms_target": -20.0,  # YouTube standard level
            "lfo_rate": 0.1
        },
        "brown_only": {
            "color_mix": {'white': 0.0, 'pink': 0.0, 'brown': 1.0},
            "rms_target": -20.0,  # YouTube standard level
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
    parser = argparse.ArgumentParser(description="Baby-Noise Generator v2.0.6 - Enhanced Organic DSP (Headless)")
    
    # Key parameters - removed channels parameter, now always stereo
    parser.add_argument("--output", type=str, help="Output file path (WAV or FLAC)", default="baby_noise.wav")
    parser.add_argument("--duration", type=int, help="Duration in seconds", default=600)
    
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
    parser.add_argument("--peak", type=float, help="True-peak ceiling (dBFS)", default=None)
    
    # Enhanced audio quality options - stereo-specific options kept
    parser.add_argument("--natural-mod", action="store_true", help="Enable natural modulation for more organic sound")
    parser.add_argument("--no-natural-mod", action="store_false", dest="natural_mod", help="Disable natural modulation")
    parser.add_argument("--haas", action="store_true", help="Enable Haas effect for enhanced stereo width")
    parser.add_argument("--no-haas", action="store_false", dest="haas", help="Disable Haas effect") 
    parser.add_argument("--enhanced-stereo", action="store_true", help="Enable enhanced stereo decorrelation")
    parser.add_argument("--no-enhanced-stereo", action="store_false", dest="enhanced_stereo", help="Use basic stereo decorrelation")
    
    # Looping options
    parser.add_argument("--loop", action="store_true", help="Create seamless looping file (optimized for YouTube)", default=True)
    parser.add_argument("--no-loop", action="store_false", dest="loop", help="Don't create special loop crossfade")
    
    # Set defaults for enhanced options
    parser.set_defaults(natural_mod=True, haas=True, enhanced_stereo=True)
    
    # Output options
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
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
    
    # Validate duration
    if args.duration <= 0:
        print(f"ERROR: Duration must be positive, got {args.duration}")
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
    
    # Create configuration with enhanced stereo options
    config = NoiseConfig(
        seed=seed,
        duration=args.duration,
        color_mix=color_mix,
        warmth=args.warmth if color_mix is None else None,
        rms_target=args.rms if args.rms is not None else preset_config.get("rms_target"),
        peak_ceiling=args.peak,
        lfo_rate=args.lfo if args.lfo is not None else preset_config.get("lfo_rate"),
        natural_modulation=args.natural_mod,
        haas_effect=args.haas,
        enhanced_stereo=args.enhanced_stereo
    )
    
    # Print configuration summary
    logger.info(f"Generating {args.duration}s of stereo noise...")
    logger.info(f"Output: {args.output}")
    logger.info(f"YouTube-optimized profile: {YOUTUBE_PROFILE['lufs_threshold']} LUFS")
    if color_mix is None:
        logger.info(f"Warmth: {args.warmth}%")
    else:
        logger.info(f"Color mix: white={color_mix['white']:.2f}, pink={color_mix['pink']:.2f}, brown={color_mix['brown']:.2f}")
    logger.info(f"Seed: {seed}")
    
    # Log stereo enhancement options
    logger.info(f"Enhanced stereo: {'ON' if args.enhanced_stereo else 'OFF'}")
    logger.info(f"Haas effect: {'ON' if args.haas else 'OFF'}")
    logger.info(f"Natural modulation: {'ON' if args.natural_mod else 'OFF'}")
    logger.info(f"Seamless looping: {'ON' if args.loop else 'OFF'}")
    
    # Check for pyloudnorm availability for accurate LUFS measurement
    try:
        import pyloudnorm
        logger.info("pyloudnorm detected - will use accurate BS.1770-4 LUFS measurement")
    except ImportError:
        logger.info("pyloudnorm not detected - will use approximate LUFS calculation")
        logger.info("For accurate loudness measurement, install pyloudnorm: pip install pyloudnorm")
    
    # Create generator
    generator = NoiseGenerator(config)
    
    # Generate to file with progress tracking in console
    try:
        result = generator.generate_to_file(args.output, print_progress, make_loop=args.loop)
        
        # Check if generation was successful
        if not result.get("success", False):
            print(f"\nError: {result.get('error', 'Unknown error')}")
            return 1
            
        # Print results
        if args.json:
            # Output as JSON
            print(json.dumps(result))
        else:
            # Human-readable output
            print("\nGeneration complete!")
            print(f"Output file: {os.path.abspath(args.output)}")
            print(f"Duration: {args.duration}s ({args.duration/60:.1f} minutes)")
            print(f"Processing time: {result['processing_time']:.1f}s")
            print(f"Real-time factor: {result['real_time_factor']:.1f}x")
            
            if "integrated_lufs" in result:
                print(f"LUFS: {result['integrated_lufs']:.1f}")
                print(f"Peak: {result['peak_db']:.1f} dBFS")
                
            if "warning" in result:
                print(f"Warning: {result['warning']}")
        
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