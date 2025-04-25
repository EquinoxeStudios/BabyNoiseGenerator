#!/usr/bin/env python3
# Baby-Noise Generator App v2.0.2 - Enhanced DSP Edition (Headless Version)
# Optimized for sound quality and performance with advanced DSP techniques
# Exclusively optimized for GPU acceleration with no CPU fallback
# Stereo-only version for YouTube publishing

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
BLOCK_OVERLAP = 4096         # For smooth transitions between blocks
BROWN_LEAKY_ALPHA = 0.999    # Leaky integrator coefficient
APP_TITLE = "Baby-Noise Generator v2.0.2 - Enhanced DSP (Headless)"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/BabyNoise")
MIN_DURATION = 1             # Minimum allowed duration in seconds
MAX_MEMORY_LIMIT_FRACTION = 0.8  # Maximum fraction of GPU memory to use

# Enhanced filter cache with categories for different filter types
_filter_cache = {
    'pink_fir': {},          # FIR filters for pink noise
    'pink_iir': {},          # IIR filters for pink noise
    'brown_hp': {},          # High-pass filters for brown noise 
    'brown_shelf': {},       # Shelving filters for brown noise enhancement
    'emphasis': {},          # Pre-emphasis filters
    'decorrelation': {}      # Decorrelation phase arrays
}

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

def create_pink_filter(n_taps, sample_rate):
    """Create FIR filter for pink noise using frequency sampling method with caching"""
    # Check cache first
    cache_key = (n_taps, sample_rate)
    if cache_key in _filter_cache['pink_fir']:
        return _filter_cache['pink_fir'][cache_key]
    
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
    # Make sure we get exactly n_taps total with proper symmetry for IFFT
    if n_taps % 2 == 0:  # Even number of taps
        # For even N: [R₀, R₁, ..., Rₙ₋₁, Rₙ, Rₙ₋₁, ..., R₁]
        full_response = cp.concatenate([response, response[-2::-1]])
    else:  # Odd number of taps
        # For odd N: [R₀, R₁, ..., Rₙ₋₁, Rₙ₋₁, ..., R₁]
        full_response = cp.concatenate([response, response[-2:0:-1]])
    
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
    
    # Cache the result
    _filter_cache['pink_fir'][cache_key] = filter_taps
    
    return filter_taps

def create_pink_iir_coeffs(sample_rate):
    """Create enhanced IIR filter coefficients for pink noise (8th order)"""
    # Check cache first
    if sample_rate in _filter_cache['pink_iir']:
        return _filter_cache['pink_iir'][sample_rate]
    
    # Enhanced pink noise IIR coefficients (8th order approximation)
    # Better low-frequency accuracy while maintaining efficiency
    b = cp.array([0.049922035, -0.095993537, 0.050612699, -0.004408786, 
                  0.002142, -0.000534, 0.000089, -0.000011], dtype=cp.float32)
    a = cp.array([1.0, -2.494956002, 2.017265875, -0.522189400,
                  0.0598, -0.00482, 0.000235, -0.0000043], dtype=cp.float32)
    
    # Cache the result
    _filter_cache['pink_iir'][sample_rate] = (b, a)
    
    return b, a

def create_brown_filters(sample_rate):
    """Create filters for enhanced brown noise"""
    # Check cache first
    if sample_rate in _filter_cache['brown_hp']:
        return _filter_cache['brown_hp'][sample_rate]
    
    # High-pass filter to remove DC offset (20 Hz cutoff)
    cutoff_hp = 20.0 / (sample_rate / 2)
    sos_hp = cusignal.butter(2, cutoff_hp, 'high', output='sos')
    
    # Low shelf filter to enhance low-end body (75 Hz, +3dB)
    # This would normally use cusignal.butter with 'lowshelf', but we'll implement manually for now
    cutoff_shelf = 75.0 / (sample_rate / 2)
    gain_db = 3.0
    
    # Approximate shelf filter with 2nd order Butterworth cascade
    # Save both in cache
    _filter_cache['brown_hp'][sample_rate] = sos_hp
    
    return sos_hp

def create_shelf_filter(sample_rate, cutoff, gain_db):
    """Create a shelf filter for frequency enhancement"""
    # Check cache first
    cache_key = (sample_rate, cutoff, gain_db)
    if cache_key in _filter_cache['brown_shelf']:
        return _filter_cache['brown_shelf'][cache_key]
    
    # Convert cutoff to normalized frequency
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    # Use second-order butterworth filter approximation
    # This is a simple approximation of a shelf filter
    if gain_db > 0:
        # Low shelf boost (for brown noise enhancement)
        b, a = cusignal.butter(2, normalized_cutoff, 'low')
        gain_linear = 10 ** (gain_db / 20.0)
        b = b * gain_linear
    else:
        # High shelf cut (not used here but included for completeness)
        b, a = cusignal.butter(2, normalized_cutoff, 'high')
        gain_linear = 10 ** (-gain_db / 20.0)
        b = b * gain_linear
    
    # Cache the result
    _filter_cache['brown_shelf'][cache_key] = (b, a)
    
    return b, a

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
                 rms_target=-63.0, 
                 peak_ceiling=-3.0,
                 lfo_rate=None, 
                 sample_rate=SAMPLE_RATE, 
                 profile="baby-safe",
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
            
        self.rms_target = rms_target  # dBFS
        self.peak_ceiling = peak_ceiling  # dBFS
        self.lfo_rate = lfo_rate  # Hz, None for no modulation
        self.sample_rate = sample_rate
        self.use_gpu = True  # Always use GPU in this version
        self.profile = profile  # Output profile name
        
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
        
        # Configure memory pool
        self.mem_pool, self.pinned_pool = setup_memory_pool()
        
        # Determine precision based on render duration
        self.precision = determine_precision(self.config.duration)
        logger.info(f"Using {self.precision} precision")
        
        # Initialize block size
        self.optimal_block_size = optimize_block_size()
        self.total_samples = int(self.config.duration * self.config.sample_rate)
        
        # Initialize filters and state
        self._init_gpu()
        self._init_filters()
        
        # Precompute crossfade windows
        self._fade_in = np.linspace(0, 1, BLOCK_OVERLAP)
        self._fade_out = np.linspace(1, 0, BLOCK_OVERLAP)
        
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
        """Initialize enhanced filters for pink and brown noise"""
        # Create filter coefficients for pink noise - FIR filter for long blocks
        self._pink_filter_taps = create_pink_filter(4097, self.config.sample_rate)
        
        # Enhanced pink noise IIR coefficients (8th order) for short blocks
        self._pink_b, self._pink_a = create_pink_iir_coeffs(self.config.sample_rate)
        
        # Initialize IIR filter state for better precision - for stereo
        # Ensure proper dimensions for the filter order
        filter_order = max(len(self._pink_a), len(self._pink_b)) - 1
        self._pink_zi_left = cp.zeros(filter_order, dtype=self.precision)
        self._pink_zi_right = cp.zeros(filter_order, dtype=self.precision)
        
        # Enhanced brown noise filters
        # Second-order sections form for better numerical stability in high-pass
        self._brown_hp_sos = create_brown_filters(self.config.sample_rate)
        
        # Create shelf filter for enhanced brown noise low end
        self._brown_shelf_b, self._brown_shelf_a = create_shelf_filter(
            self.config.sample_rate, 75.0, 3.0)  # 75Hz, +3dB boost
        
        # Initialize leaky integrator coefficients for brown noise
        alpha = BROWN_LEAKY_ALPHA
        scale = cp.sqrt(1.0 - alpha*alpha)  # Normalization factor
        
        # Create filter coefficients as CuPy arrays with correct precision
        self._brown_b = cp.array([scale], dtype=self.precision)
        self._brown_a = cp.array([1.0, -alpha], dtype=self.precision)
        
        # Initialize brown noise filter states for both channels
        brown_order = max(len(self._brown_a), len(self._brown_b)) - 1
        self._brown_zi_left = cp.zeros(brown_order, dtype=self.precision)
        self._brown_zi_right = cp.zeros(brown_order, dtype=self.precision)
        
        # For SOS filters, state is per section, with 2 values per section
        self._brown_hp_zi_left = cp.zeros((self._brown_hp_sos.shape[0], 2), dtype=self.precision)
        self._brown_hp_zi_right = cp.zeros((self._brown_hp_sos.shape[0], 2), dtype=self.precision)
        
        self._brown_shelf_zi_left = cp.zeros(max(len(self._brown_shelf_a), len(self._brown_shelf_b))-1, dtype=self.precision)
        self._brown_shelf_zi_right = cp.zeros(max(len(self._brown_shelf_a), len(self._brown_shelf_b))-1, dtype=self.precision)
        
        # Create frequency-dependent phase shifts for stereo decorrelation
        # Get block size and frequency resolution
        block_size = self.optimal_block_size
        n_freqs = block_size // 2 + 1
        freq_resolution = self.config.sample_rate / block_size
        
        # Initialize phases array
        self.decorrelation_phases = cp.zeros(n_freqs, dtype=cp.complex64)
        
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
    
    def _generate_white_noise_block(self, block_size):
        """Generate stereo white noise block on GPU"""
        # Generate independent samples for each channel
        self._white_buffer[0, :block_size] = self.rng.normal(0, 1, block_size).astype(self.precision)
        self._white_buffer[1, :block_size] = self.rng.normal(0, 1, block_size).astype(self.precision)
        return self._white_buffer[:, :block_size]
    
    def _apply_pink_filter(self, white_noise, block_size):
        """Apply pink filter to white noise on GPU"""
        # For small blocks or short renders, use the faster IIR filter
        if block_size < 32768:
            return self._apply_pink_filter_iir(white_noise, block_size)
            
        # Process stereo channels
        left = white_noise[0, :block_size]
        right = white_noise[1, :block_size]
        
        # Process each channel
        self._pink_buffer[0, :block_size] = cusignal.fftconvolve(
            left, self._pink_filter_taps, mode='same'
        )
        self._pink_buffer[1, :block_size] = cusignal.fftconvolve(
            right, self._pink_filter_taps, mode='same'
        )
        
        return self._pink_buffer[:, :block_size]
    
    def _apply_pink_filter_iir(self, white_noise, block_size):
        """Apply enhanced pink filter using 8th order IIR approximation (faster for short blocks)"""
        # Process left channel
        left = white_noise[0, :block_size]
        right = white_noise[1, :block_size]
        
        # Process each channel
        left_filtered, self._pink_zi_left = cusignal.lfilter(
            self._pink_b, self._pink_a, left, zi=self._pink_zi_left
        )
        right_filtered, self._pink_zi_right = cusignal.lfilter(
            self._pink_b, self._pink_a, right, zi=self._pink_zi_right
        )
        
        self._pink_buffer[0, :block_size] = left_filtered
        self._pink_buffer[1, :block_size] = right_filtered
        
        return self._pink_buffer[:, :block_size]
    
    def _generate_brown_noise(self, white_noise, block_size):
        """Generate enhanced brown noise from white noise (stereo)"""
        # Process left channel
        left = white_noise[0, :block_size]
        right = white_noise[1, :block_size]
        
        # Process left channel
        # First-order leaky integrator
        left_brown, self._brown_zi_left = cusignal.lfilter(
            self._brown_b, self._brown_a, left, zi=self._brown_zi_left)
        
        # Apply low-shelf filter for enhanced low-end
        left_brown, self._brown_shelf_zi_left = cusignal.lfilter(
            self._brown_shelf_b, self._brown_shelf_a, left_brown, zi=self._brown_shelf_zi_left)
        
        # Apply high-pass to remove DC offset
        left_brown, self._brown_hp_zi_left = cusignal.sosfilt(
            self._brown_hp_sos, left_brown, zi=self._brown_hp_zi_left)
        
        # Process right channel
        # First-order leaky integrator
        right_brown, self._brown_zi_right = cusignal.lfilter(
            self._brown_b, self._brown_a, right, zi=self._brown_zi_right)
        
        # Apply low-shelf filter for enhanced low-end
        right_brown, self._brown_shelf_zi_right = cusignal.lfilter(
            self._brown_shelf_b, self._brown_shelf_a, right_brown, zi=self._brown_shelf_zi_right)
        
        # Apply high-pass to remove DC offset
        right_brown, self._brown_hp_zi_right = cusignal.sosfilt(
            self._brown_hp_sos, right_brown, zi=self._brown_hp_zi_right)
        
        # Store results
        self._brown_buffer[0, :block_size] = left_brown
        self._brown_buffer[1, :block_size] = right_brown
        
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
    
    def _apply_natural_modulation(self, noise_block, block_start_idx, block_size):
        """Add subtle multi-band modulation for more organic sound"""
        if not self.config.natural_modulation:
            return noise_block
            
        # Create different modulation rates for different frequency bands
        block_time = block_size / self.config.sample_rate
        
        # Generate time indices for this block
        t_start = block_start_idx / self.config.sample_rate
        t = cp.linspace(t_start, t_start + block_time, block_size, endpoint=False)
        
        # Different modulation rates for different frequency bands
        # These subtle modulations create a more natural, less static sound
        
        # Slow modulation for lows (0.07 Hz)
        low_mod = 1.0 + 0.02 * cp.sin(2 * cp.pi * 0.07 * t)
        
        # Medium modulation for mids (0.13 Hz)
        mid_mod = 1.0 + 0.015 * cp.sin(2 * cp.pi * 0.13 * t)
        
        # Faster modulation for highs (0.27 Hz)
        high_mod = 1.0 + 0.01 * cp.sin(2 * cp.pi * 0.27 * t)
        
        # Process left channel with spectral modulation
        left_fft = cufft.rfft(noise_block[0, :block_size])
        
        # Get frequency bin indices using correct frequency resolution
        n_bins = len(left_fft)
        freq_resolution = self.config.sample_rate / block_size
        low_idx = int(300 / freq_resolution)
        mid_idx = int(1500 / freq_resolution)
        
        # Apply modulation to left channel
        low_env = cp.linspace(1.0, 0.8, low_idx)
        left_fft[:low_idx] = left_fft[:low_idx] * (1.0 + 0.03 * cp.sin(2 * cp.pi * 0.05 * t_start) * low_env)
        
        mid_env = cp.concatenate([cp.linspace(0.8, 1.0, min(10, low_idx)), 
                                cp.ones(mid_idx - low_idx - min(10, low_idx))])
        left_fft[low_idx:mid_idx] = left_fft[low_idx:mid_idx] * (1.0 + 0.02 * cp.sin(2 * cp.pi * 0.13 * t_start) * mid_env)
        
        high_env = cp.ones(n_bins - mid_idx)
        left_fft[mid_idx:] = left_fft[mid_idx:] * (1.0 + 0.01 * cp.sin(2 * cp.pi * 0.21 * t_start) * high_env)
        
        # Convert back to time domain
        noise_block[0, :block_size] = cufft.irfft(left_fft, n=block_size)
        
        # Process right channel with phase offset for enhanced decorrelation
        right_fft = cufft.rfft(noise_block[1, :block_size])
        
        # Apply modulation to right channel with phase offset
        phase_offset = cp.pi / 4  # 45 degree offset
        right_fft[:low_idx] = right_fft[:low_idx] * (1.0 + 0.03 * cp.sin(2 * cp.pi * 0.05 * t_start + phase_offset) * low_env)
        right_fft[low_idx:mid_idx] = right_fft[low_idx:mid_idx] * (1.0 + 0.02 * cp.sin(2 * cp.pi * 0.13 * t_start + phase_offset) * mid_env)
        right_fft[mid_idx:] = right_fft[mid_idx:] * (1.0 + 0.01 * cp.sin(2 * cp.pi * 0.21 * t_start + phase_offset) * high_env)
        
        # Convert back to time domain
        noise_block[1, :block_size] = cufft.irfft(right_fft, n=block_size)
        
        return noise_block
    
    def _apply_gain_and_limiting(self, noise_block, target_rms, peak_ceiling, block_size):
        """Multi-stage gain and limiting for better sound quality"""
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
        # Threshold in dB and linear domains
        threshold_db = peak_ceiling - 6  # 6dB below ceiling
        threshold = 10 ** (threshold_db / 20.0)
        ratio = 4.0  # 4:1 compression
        knee_width_db = 6.0  # 6dB knee width
        knee_width = knee_width_db / 20.0  # Convert to scaling factor
        
        # Apply soft knee compression with logarithmic calculation for improved numerical stability
        # Process each channel separately
        for ch in range(2):
            # Calculate gain reduction for each sample
            input_level = cp.abs(noise_block[ch, :block_size])
            
            # Initialize gain reduction array
            gain_reduction = cp.ones(block_size, dtype=self.precision)
            
            # Find samples exceeding threshold
            over_threshold = input_level > threshold
            
            if cp.any(over_threshold):
                # Convert input level to dB for more stable calculation
                input_level_db = 20.0 * cp.log10(input_level + 1e-10)
                threshold_with_knee_db = threshold_db - (knee_width_db / 2.0)
                
                # Samples in knee region or above
                above_knee_threshold = input_level_db >= threshold_with_knee_db
                
                if cp.any(above_knee_threshold):
                    # Calculate how far into the knee or above the samples are
                    amount_over_db = input_level_db[above_knee_threshold] - threshold_with_knee_db
                    
                    # Knee region: softer compression curve
                    in_knee = amount_over_db <= knee_width_db
                    if cp.any(in_knee):
                        # Quadratic formula for smooth knee
                        knee_factor = amount_over_db[in_knee] / knee_width_db
                        knee_gain_db = amount_over_db[in_knee] * ((1.0 / ratio - 1.0) * knee_factor * 0.5)
                        gain_reduction_db = -knee_gain_db
                        gain_reduction[above_knee_threshold][in_knee] = 10 ** (gain_reduction_db / 20.0)
                    
                    # Above knee: full compression ratio
                    above_knee = amount_over_db > knee_width_db
                    if cp.any(above_knee):
                        amount_above_knee_db = amount_over_db[above_knee] - knee_width_db
                        gain_reduction_db = -(knee_width_db * (1.0 / ratio - 1.0) * 0.5 + 
                                            amount_above_knee_db * (1.0 - 1.0 / ratio))
                        gain_reduction[above_knee_threshold][above_knee] = 10 ** (gain_reduction_db / 20.0)
                
                # Apply gain reduction
                noise_block[ch, :block_size] = noise_block[ch, :block_size] * gain_reduction
        
        # Third stage: Hard limiting for true peaks
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
        """Apply true-peak limiting with 4x oversampling using polyphase implementation"""
        # True peak detection using 4x oversampling
        oversampling_factor = 4
        
        # Quick peak check first to avoid unnecessary processing
        peak_left = cp.max(cp.abs(noise_block[0, :block_size]))
        peak_right = cp.max(cp.abs(noise_block[1, :block_size]))
        peak = cp.maximum(peak_left, peak_right)
            
        peak_threshold = 10 ** (peak_ceiling / 20.0)
        
        # Only do expensive true-peak calculation if we're close to the threshold
        if peak > peak_threshold * 0.8:  # Within ~2dB of threshold
            # Design a high-quality polyphase interpolation filter
            # The filter design is important for accurate true-peak detection
            nyquist_normalized = 1.0 / oversampling_factor
            
            # For better performance, we'll use a predesigned polyphase filter
            # or direct FFT-based resampling which is more efficient for large blocks
            
            # Process each channel
            upsampled_left = cusignal.resample(
                noise_block[0, :block_size], block_size * oversampling_factor
            )
            upsampled_right = cusignal.resample(
                noise_block[1, :block_size], block_size * oversampling_factor
            )
            
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
        if not NOISE_PROFILES.get(self.config.profile, {}).get("pre_emphasis", False):
            return noise_block
        
        # Pre-emphasis filter (boost above 5kHz for YouTube codec)
        nyquist = self.config.sample_rate / 2
        cutoff = 5000 / nyquist
        
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
    
    def generate_block(self, block_size, block_start_idx=0):
        """Generate a block of noise with the specified configuration"""
        # Generate white noise base
        white_noise = self._generate_white_noise_block(block_size)
        
        # Generate pink noise from white noise
        pink_noise = self._apply_pink_filter(white_noise, block_size)
        
        # Generate brown noise from white noise
        brown_noise = self._generate_brown_noise(white_noise, block_size)
        
        # Blend according to color mix
        mixed_noise = self._blend_noise_colors(
            white_noise, pink_noise, brown_noise, self.config.color_mix, block_size
        )
        
        # Apply stereo decorrelation
        mixed_noise = self._apply_stereo_decorrelation(mixed_noise, block_size)
        
        # Apply Haas effect for enhanced stereo imaging if enabled
        if self.config.haas_effect:
            mixed_noise = self._apply_haas_effect(mixed_noise, block_size)
            
        # Apply natural modulation for more organic sound if enabled
        if self.config.natural_modulation:
            mixed_noise = self._apply_natural_modulation(mixed_noise, block_start_idx, block_size)
        
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
    
    def generate_to_file(self, output_path, progress_callback=None):
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
                
                # Large overlap for smooth transitions
                overlap_buffer = None
                
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
                    
                    # Move from GPU to CPU with another CUDA stream
                    with cuda_stream() as transfer_stream:
                        with transfer_stream:
                            # Always stereo, transpose for soundfile's expected format
                            output_data = cp.asnumpy(noise_block[:, :current_block_size].T)
                    
                    # Calculate safe overlap size
                    overlap = min(BLOCK_OVERLAP, current_block_size - 1, samples_remaining)
                    
                    # Apply overlap from previous block if available
                    if overlap_buffer is not None:
                        # Crossfade with previous block's overlap region
                        # Make sure the fade arrays are properly shaped for broadcasting
                        fade_in_shaped = self._fade_in[:overlap, np.newaxis]
                        fade_out_shaped = self._fade_out[:overlap, np.newaxis]
                        
                        output_data[:overlap, :] = (
                            output_data[:overlap, :] * fade_in_shaped + 
                            overlap_buffer[:overlap, :] * fade_out_shaped
                        )
                    
                    # Save overlap buffer for next iteration - ensure explicit array conversion
                    if samples_remaining > overlap:
                        overlap_buffer = np.array(output_data[-overlap:, :].copy())
                    
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
        },
        "white_only": {
            "color_mix": {'white': 1.0, 'pink': 0.0, 'brown': 0.0},
            "rms_target": -63.0,
            "lfo_rate": 0.1
        },
        "pink_only": {
            "color_mix": {'white': 0.0, 'pink': 1.0, 'brown': 0.0},
            "rms_target": -63.0,
            "lfo_rate": 0.1
        },
        "brown_only": {
            "color_mix": {'white': 0.0, 'pink': 0.0, 'brown': 1.0},
            "rms_target": -63.0,
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
    parser = argparse.ArgumentParser(description="Baby-Noise Generator v2.0.2 - Enhanced DSP (Headless)")
    
    # Key parameters - removed channels parameter, now always stereo
    parser.add_argument("--output", type=str, help="Output file path (WAV or FLAC)", default="baby_noise.wav")
    parser.add_argument("--duration", type=int, help="Duration in seconds", default=600)
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
    parser.add_argument("--peak", type=float, help="True-peak ceiling (dBFS)", default=None)
    
    # Enhanced audio quality options - stereo-specific options kept
    parser.add_argument("--natural-mod", action="store_true", help="Enable natural modulation for more organic sound")
    parser.add_argument("--no-natural-mod", action="store_false", dest="natural_mod", help="Disable natural modulation")
    parser.add_argument("--haas", action="store_true", help="Enable Haas effect for enhanced stereo width")
    parser.add_argument("--no-haas", action="store_false", dest="haas", help="Disable Haas effect")
    parser.add_argument("--enhanced-stereo", action="store_true", help="Enable enhanced stereo decorrelation")
    parser.add_argument("--no-enhanced-stereo", action="store_false", dest="enhanced_stereo", help="Use basic stereo decorrelation")
    
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
        profile=args.profile,
        lfo_rate=args.lfo if args.lfo is not None else preset_config.get("lfo_rate"),
        natural_modulation=args.natural_mod,
        haas_effect=args.haas,
        enhanced_stereo=args.enhanced_stereo
    )
    
    # Override with explicit RMS if provided
    if args.rms is not None:
        config.set_rms_target(args.rms)
    elif args.preset and "rms_target" in preset_config:
        config.set_rms_target(preset_config["rms_target"])
    
    # Override with explicit peak ceiling if provided
    if args.peak is not None:
        config.set_peak_ceiling(args.peak)
    
    # Print configuration summary
    logger.info(f"Generating {args.duration}s of stereo noise...")
    logger.info(f"Output: {args.output}")
    logger.info(f"Profile: {args.profile}")
    if color_mix is None:
        logger.info(f"Warmth: {args.warmth}%")
    else:
        logger.info(f"Color mix: white={color_mix['white']:.2f}, pink={color_mix['pink']:.2f}, brown={color_mix['brown']:.2f}")
    logger.info(f"Seed: {seed}")
    
    # Log stereo enhancement options
    logger.info(f"Enhanced stereo: {'ON' if args.enhanced_stereo else 'OFF'}")
    logger.info(f"Haas effect: {'ON' if args.haas else 'OFF'}")
    logger.info(f"Natural modulation: {'ON' if args.natural_mod else 'OFF'}")
    
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
        result = generator.generate_to_file(args.output, print_progress)
        
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