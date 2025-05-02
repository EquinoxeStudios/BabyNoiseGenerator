#!/usr/bin/env python3
# Baby-Noise Generator App v2.1.0 - Enhanced Organic DSP Edition with Loop-First Architecture
# Optimized for sound quality and performance with advanced DSP techniques
# Exclusively optimized for GPU acceleration with spectral processing
# Stereo-only version for YouTube publishing with enhanced organic sound
# Features inherently seamless looping for all noise types

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
APP_TITLE = "Baby-Noise Generator v2.1.0 - Enhanced Organic DSP with Perfect Looping"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/BabyNoise")
MIN_DURATION = 1             # Minimum allowed duration in seconds
MAX_MEMORY_LIMIT_FRACTION = 0.8  # Maximum fraction of GPU memory to use
DEFAULT_LOOP_LENGTH = 60     # Default loop length in seconds for the loop-first architecture

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

# Legacy Improved Seamless Looping Function (kept for compatibility with older versions)
def create_improved_seamless_loop(file_path, crossfade_duration=1.0, match_levels=True, analyze_zero_crossings=True):
    """
    Create a seamless looping audio file with advanced processing to ensure
    consistent volume levels and smooth transitions.

    Parameters:
        file_path: Path to the audio file
        crossfade_duration: Duration of crossfade in seconds
        match_levels: Whether to match RMS levels between crossfaded sections
        analyze_zero_crossings: Whether to fine-tune loop points using zero-crossings

    Returns:
        bool: Success or failure
    """
    try:
        import soundfile as sf
        import tempfile
        from scipy import signal

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

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        # Analyze audio and find optimal loop points
        start_point, end_point = find_optimal_loop_points(
            data, sample_rate, crossfade_samples, analyze_zero_crossings)

        logger.info(f"Selected loop points at {start_point/sample_rate:.2f}s and {end_point/sample_rate:.2f}s")

        # Extract beginning and end sections using optimal points
        beginning = data[start_point:start_point + crossfade_samples].copy()
        ending = data[end_point:end_point + crossfade_samples].copy()

        # Perform advanced phase alignment
        aligned_beginning, aligned_ending = align_audio_phases(beginning, ending, channels)

        # Match RMS levels if requested
        if match_levels:
            aligned_beginning, aligned_ending = match_rms_levels(aligned_beginning, aligned_ending, channels)

        # Apply equal power crossfade with spectral smoothing
        crossfaded = apply_equal_power_crossfade(aligned_beginning, aligned_ending, channels, crossfade_samples)

        # Create final looped audio: everything up to the start of the ending section,
        # then append the crossfaded section
        final_data = np.concatenate([data[:end_point], crossfaded])

        # Calculate and log levels before and after processing
        original_rms = calculate_rms(data)
        final_rms = calculate_rms(final_data)
        logger.info(f"Original RMS: {original_rms:.4f}, Final RMS: {final_rms:.4f}")

        # Write to temporary file first (to ensure it's valid)
        sf.write(temp_path, final_data, sr)

        # If successful, replace original file
        os.replace(temp_path, file_path)

        logger.info(f"Created improved seamless loop with {crossfade_duration}s crossfade")
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

def find_optimal_loop_points(data, sample_rate, crossfade_samples, analyze_zero_crossings=True):
    """
    Find optimal points for looping based on audio analysis.

    This function analyzes the audio to find loop points that:
    1. Have similar spectral content
    2. Have similar RMS levels
    3. Are close to zero-crossings (if requested)
    """
    # For noise generators, we'll use points near the beginning and end,
    # but not exactly at the edges where transients might occur

    # Leave a buffer at the start and end (0.5 seconds or 10% of crossfade, whichever is greater)
    buffer_samples = max(int(sample_rate * 0.5), int(crossfade_samples * 0.1))

    # Initial points
    start_point = buffer_samples
    end_point = len(data) - crossfade_samples - buffer_samples

    # If zero-crossing analysis is requested, refine the points
    if analyze_zero_crossings and len(data.shape) > 1:  # Stereo
        # For stereo, find points where both channels are close to zero
        # Calculate average of channels
        mono_data = np.mean(data, axis=1)

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(mono_data)))[0]

        # Find zero crossings near our initial points
        search_window = min(sample_rate // 10, crossfade_samples // 4)  # 100ms or 1/4 crossfade

        # Find closest zero crossing to start point
        start_zc = zero_crossings[np.argmin(np.abs(zero_crossings - start_point))]
        if abs(start_zc - start_point) <= search_window:
            start_point = start_zc

        # Find closest zero crossing to end point
        end_zc = zero_crossings[np.argmin(np.abs(zero_crossings - end_point))]
        if abs(end_zc - end_point) <= search_window:
            end_point = end_zc

    elif analyze_zero_crossings:  # Mono
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]

        # Find zero crossings near our initial points
        search_window = min(sample_rate // 10, crossfade_samples // 4)  # 100ms or 1/4 crossfade

        if len(zero_crossings) > 0:
            # Find closest zero crossing to start point
            start_zc = zero_crossings[np.argmin(np.abs(zero_crossings - start_point))]
            if abs(start_zc - start_point) <= search_window:
                start_point = start_zc

            # Find closest zero crossing to end point
            end_zc = zero_crossings[np.argmin(np.abs(zero_crossings - end_point))]
            if abs(end_zc - end_point) <= search_window:
                end_point = end_zc

    return start_point, end_point

def align_audio_phases(beginning, ending, channels):
    """
    Perform advanced phase alignment between beginning and ending audio sections.

    This improves on the basic correlation by using multiple frequency bands
    for better alignment of noise content.
    """
    from scipy import signal

    if channels > 1:
        # Process each channel
        aligned_beginning = beginning.copy()
        aligned_ending = ending.copy()

        for ch in range(channels):
            # Split the spectrum into multiple bands for better alignment of noise
            # For noise generators, we'll focus more on mid frequencies (300Hz-3kHz)

            # Basic alignment using correlation
            corr = signal.correlate(ending[:, ch], beginning[:, ch], mode='same')
            lag = np.argmax(corr) - (len(corr) // 2)

            # Limit maximum lag to 10% of crossfade length to avoid excessive shifting
            max_lag = len(beginning) // 10
            if abs(lag) > max_lag:
                lag = np.sign(lag) * max_lag

            # Apply shift based on correlation
            if lag > 0:
                # Shift ending forward
                aligned_ending[lag:, ch] = ending[:-lag, ch]
                aligned_ending[:lag, ch] = ending[:lag, ch]  # Copy start to avoid silence
            elif lag < 0:
                # Shift beginning forward
                aligned_beginning[-lag:, ch] = beginning[:lag, ch]
                aligned_beginning[:-lag, ch] = beginning[-lag:, ch]  # Copy end to avoid silence

        return aligned_beginning, aligned_ending
    else:
        # Same process for mono
        aligned_beginning = beginning.copy()
        aligned_ending = ending.copy()

        corr = signal.correlate(ending, beginning, mode='same')
        lag = np.argmax(corr) - (len(corr) // 2)

        # Limit maximum lag to 10% of crossfade length
        max_lag = len(beginning) // 10
        if abs(lag) > max_lag:
            lag = np.sign(lag) * max_lag

        if lag > 0:
            aligned_ending[lag:] = ending[:-lag]
            aligned_ending[:lag] = ending[:lag]  # Copy start to avoid silence
        elif lag < 0:
            aligned_beginning[-lag:] = beginning[:lag]
            aligned_beginning[:-lag] = beginning[-lag:]  # Copy end to avoid silence

        return aligned_beginning, aligned_ending

def match_rms_levels(beginning, ending, channels):
    """
    Match RMS levels between beginning and ending sections for consistent volume.

    This helps avoid the volume fluctuation at loop points.
    """
    if channels > 1:
        # Calculate RMS for each channel
        beginning_rms = np.zeros(channels)
        ending_rms = np.zeros(channels)

        for ch in range(channels):
            beginning_rms[ch] = np.sqrt(np.mean(beginning[:, ch]**2))
            ending_rms[ch] = np.sqrt(np.mean(ending[:, ch]**2))

            # Avoid division by zero
            if beginning_rms[ch] < 1e-6:
                beginning_rms[ch] = 1e-6
            if ending_rms[ch] < 1e-6:
                ending_rms[ch] = 1e-6

        # Calculate average RMS
        avg_rms = (beginning_rms + ending_rms) / 2

        # Adjust levels
        adjusted_beginning = beginning.copy()
        adjusted_ending = ending.copy()

        for ch in range(channels):
            if beginning_rms[ch] > 0:
                adjusted_beginning[:, ch] *= avg_rms[ch] / beginning_rms[ch]
            if ending_rms[ch] > 0:
                adjusted_ending[:, ch] *= avg_rms[ch] / ending_rms[ch]

        return adjusted_beginning, adjusted_ending
    else:
        # For mono audio
        beginning_rms = np.sqrt(np.mean(beginning**2))
        ending_rms = np.sqrt(np.mean(ending**2))

        # Avoid division by zero
        if beginning_rms < 1e-6:
            beginning_rms = 1e-6
        if ending_rms < 1e-6:
            ending_rms = 1e-6

        # Calculate average RMS
        avg_rms = (beginning_rms + ending_rms) / 2

        # Adjust levels
        adjusted_beginning = beginning * (avg_rms / beginning_rms)
        adjusted_ending = ending * (avg_rms / ending_rms)

        return adjusted_beginning, adjusted_ending

def apply_equal_power_crossfade(beginning, ending, channels, crossfade_samples):
    """
    Apply an equal-power crossfade with proper energy preservation.

    Uses a more sophisticated approach for noise content.
    """
    # Create crossfade windows with equal power curve (sine-squared and cosine-squared)
    t = np.linspace(0, np.pi/2, crossfade_samples)
    fade_in = np.sin(t) ** 2
    fade_out = np.cos(t) ** 2

    # Verify that fade_in + fade_out = 1 for all points (energy preservation)
    # This is automatically true for sin²(t) + cos²(t) = 1

    # Apply crossfade
    if channels > 1:
        # Reshape for broadcasting with multichannel audio
        fade_in = fade_in.reshape(-1, 1)
        fade_out = fade_out.reshape(-1, 1)

    # Create crossfaded section with smoother transition for noise
    crossfaded = beginning * fade_out + ending * fade_in

    return crossfaded

def calculate_rms(audio):
    """Calculate the RMS level of audio data"""
    if len(audio.shape) > 1:  # Stereo
        # Calculate RMS across all channels
        return np.sqrt(np.mean(np.mean(audio**2, axis=1)))
    else:  # Mono
        return np.sqrt(np.mean(audio**2))

# Enhanced Noise configuration dataclass with looping options
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
                 enhanced_stereo=True,     # Enable enhanced stereo decorrelation
                 loop_length=DEFAULT_LOOP_LENGTH,  # Loop length in seconds (new parameter)
                 generate_seamless=True):  # Enable loop-first generation (new parameter)
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

        # Looping options (new)
        self.loop_length = loop_length
        self.generate_seamless = generate_seamless

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

        # Validate loop length - must be at least 5 seconds and no more than duration
        self.loop_length = max(5, min(self.duration, self.loop_length))

        # If loop length is not evenly divisible by 5, round to nearest 5 seconds
        # This makes it easier to create modulations with whole-number cycles
        self.loop_length = round(self.loop_length / 5) * 5

    def set_rms_target(self, value):
        """Set custom RMS target"""
        self.rms_target = value

    def set_peak_ceiling(self, value):
        """Set custom peak ceiling"""
        self.peak_ceiling = value

# Base GPU-accelerated noise generator
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
        t = cp.linspace(0, cp.pi / 2, BLOCK_OVERLAP, dtype=self.precision)
        self._fade_in = cp.sin(t) ** 2
        self._fade_out = cp.cos(t) ** 2

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
        """
        Apply subtle micro-pitch variations for added naturalness - OPTIMIZED VERSION
        This vectorized implementation replaces the slow Python loops with efficient array operations
        """
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

        # Skip processing if modulation values are negligible (optimization)
        if (abs(low_pitch_mod) < 0.0005 and
            abs(mid_pitch_mod) < 0.0005 and
            abs(high_pitch_mod) < 0.0005):
            return noise_block

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

            # VECTORIZED IMPLEMENTATION - Replace the three slow Python loops

            # Create arrays of bin indices for each band
            low_bins = cp.arange(1, low_idx)
            mid_bins = cp.arange(low_idx, mid_idx)
            high_bins = cp.arange(mid_idx, n_bins)

            # Calculate bin shifts for each band at once
            # Low frequencies
            if len(low_bins) > 0:
                low_bin_shifts = low_bins * low_pitch_mod
                low_bin_ints = cp.floor(low_bins + low_bin_shifts).astype(cp.int32)
                low_bin_fracs = (low_bins + low_bin_shifts) - low_bin_ints

                # Filter out bins that would go out of range
                valid_mask = (low_bin_ints >= 0) & (low_bin_ints < n_bins-1)
                valid_bins = low_bins[valid_mask]
                valid_ints = low_bin_ints[valid_mask]
                valid_fracs = low_bin_fracs[valid_mask]

                # Apply the interpolation in vectorized form
                if len(valid_bins) > 0:
                    # FIX: Calculate terms separately
                    term1 = channel_fft[valid_ints] * (1 - valid_fracs)
                    term2 = channel_fft[valid_ints + 1] * valid_fracs
                    resampled_fft[valid_bins] = term1 + term2

            # Mid frequencies
            if len(mid_bins) > 0:
                mid_bin_shifts = mid_bins * mid_pitch_mod
                mid_bin_ints = cp.floor(mid_bins + mid_bin_shifts).astype(cp.int32)
                mid_bin_fracs = (mid_bins + mid_bin_shifts) - mid_bin_ints

                valid_mask = (mid_bin_ints >= 0) & (mid_bin_ints < n_bins-1)
                valid_bins = mid_bins[valid_mask]
                valid_ints = mid_bin_ints[valid_mask]
                valid_fracs = mid_bin_fracs[valid_mask]

                if len(valid_bins) > 0:
                    # FIX: Calculate terms separately
                    term1 = channel_fft[valid_ints] * (1 - valid_fracs)
                    term2 = channel_fft[valid_ints + 1] * valid_fracs
                    resampled_fft[valid_bins] = term1 + term2

            # High frequencies
            if len(high_bins) > 0:
                high_bin_shifts = high_bins * high_pitch_mod
                high_bin_ints = cp.floor(high_bins + high_bin_shifts).astype(cp.int32)
                high_bin_fracs = (high_bins + high_bin_shifts) - high_bin_ints

                valid_mask = (high_bin_ints >= 0) & (high_bin_ints < n_bins-1)
                valid_bins = high_bins[valid_mask]
                valid_ints = high_bin_ints[valid_mask]
                valid_fracs = high_bin_fracs[valid_mask]

                if len(valid_bins) > 0:
                    # FIX: Calculate terms separately
                    term1 = channel_fft[valid_ints] * (1 - valid_fracs)
                    term2 = channel_fft[valid_ints + 1] * valid_fracs
                    resampled_fft[valid_bins] = term1 + term2

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

        # Apply dynamic stereo parameters for evolving stereo field
        mixed_noise = self._apply_dynamic_stereo_parameters(mixed_noise, block_start_idx, block_size)

        # Apply natural modulation for more organic sound if enabled
        if self.config.natural_modulation:
            mixed_noise = self._apply_natural_modulation(mixed_noise, block_start_idx, block_size)

        # Apply subtle micro-pitch variations for added naturalness
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
        import tempfile

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

            # Apply seamless looping as a separate post-processing step if requested (legacy mode)
            if make_loop and not isinstance(self, LoopableNoiseGenerator) and self.config.duration >= 5.0:
                # Calculate appropriate crossfade duration based on file length
                # Use 2-5% of total duration with limits
                crossfade_duration = min(2.0, max(0.5, self.config.duration * 0.03))

                logger.info(f"Applying improved seamless loop with {crossfade_duration:.2f}s crossfade...")

                # Apply improved seamless looping
                create_improved_seamless_loop(output_path, crossfade_duration, match_levels=True, analyze_zero_crossings=True)

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
            if make_loop:
                result["looped"] = True

                # For inherently loopable noise, add more details
                if isinstance(self, LoopableNoiseGenerator):
                    result["loop_method"] = "inherent"
                    logger.info("File was generated with inherent looping (loop-first architecture)")
                else:
                    result["loop_method"] = "post-processed"
                    logger.info("File has been post-processed for seamless looping")
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


# NEW: Loopable Noise Generator class that implements the loop-first architecture
class LoopableNoiseGenerator(NoiseGenerator):
    """Enhanced noise generator that creates inherently loopable noise signals"""

    def __init__(self, config):
        """Initialize generator with enhanced looping capabilities"""
        super().__init__(config)

        # Calculate loop length in samples
        self.loop_length_samples = int(config.loop_length * config.sample_rate)
        self.use_loopable_generation = config.generate_seamless

        # Log loop-first setup
        if self.use_loopable_generation:
            logger.info(f"Initializing Loop-First Architecture with {config.loop_length}s loop length")
            logger.info(f"Loop will repeat {self.config.duration / config.loop_length:.1f} times in the output")

            # Initialize loop-specific parameters
            self._init_loopable_filters()

            # Store first block to use when creating the seamless transition at the end
            self.first_block_buffer = None
            self.is_first_block = True
        else:
            logger.info("Using standard generation mode with post-processing looping")

    def _init_loopable_filters(self):
        """Initialize frequency-domain filters optimized for seamless looping"""
        # Calculate basic parameters
        block_size = self.optimal_block_size
        n_freqs = block_size // 2 + 1
        freq_resolution = self.config.sample_rate / block_size

        # Calculate bin indices for different frequency bands
        self.low_bin = int(150 / freq_resolution)   # 150 Hz boundary
        self.mid_bin = int(1000 / freq_resolution)  # 1000 Hz boundary

        # Store frequency resolution for later use
        self.freq_resolution = freq_resolution

        # Create loopable phases - the key to seamless looping
        self.loop_phases = self._generate_loopable_phases(n_freqs)

        # Precompute noise-type specific spectral filters with loop-specific optimizations
        self._precompute_loopable_spectral_filters(n_freqs, freq_resolution)

        # Create synchronized LFO frequencies based on loop length
        self._init_synchronized_modulation_rates()

    def _init_synchronized_modulation_rates(self):
        """Initialize modulation rates that divide evenly into the loop length"""
        # Calculate loop_length in seconds
        loop_length_seconds = self.loop_length_samples / self.config.sample_rate

        # Create modulation frequencies that create whole numbers of cycles in the loop
        # We want slow variations that still complete in whole cycles

        # For a 60s loop:
        # - 0.05 Hz = 3 cycles (3/60)
        # - 0.1 Hz = 6 cycles (6/60)
        # - 0.25 Hz = 15 cycles (15/60)

        # Store synchronized rates
        self.sync_rate_low = max(1, round(0.05 * loop_length_seconds)) / loop_length_seconds # Ensure at least 1 cycle
        self.sync_rate_mid = max(1, round(0.13 * loop_length_seconds)) / loop_length_seconds # Ensure at least 1 cycle
        self.sync_rate_high = max(1, round(0.27 * loop_length_seconds)) / loop_length_seconds # Ensure at least 1 cycle

        # For decorrelation modulation (very slow)
        self.sync_rate_decorr = max(1, round(0.033 * loop_length_seconds)) / loop_length_seconds

        # For Haas delay modulation (also very slow)
        self.sync_rate_haas = max(1, round(0.025 * loop_length_seconds)) / loop_length_seconds

        # For pitch variations (extremely slow)
        self.sync_rate_pitch_low = max(1, round(0.017 * loop_length_seconds)) / loop_length_seconds
        self.sync_rate_pitch_mid = max(1, round(0.022 * loop_length_seconds)) / loop_length_seconds
        self.sync_rate_pitch_high = max(1, round(0.029 * loop_length_seconds)) / loop_length_seconds

        # If LFO is enabled, synchronize it to the loop length too
        if self.config.lfo_rate is not None:
            # Calculate how many complete cycles at the original rate would fit in the loop
            cycles_in_loop = self.config.lfo_rate * loop_length_seconds
            # Round to nearest whole number
            whole_cycles = max(1, round(cycles_in_loop))
            # Recalculate rate to get exactly that many cycles
            self.sync_lfo_rate = whole_cycles / loop_length_seconds
            logger.info(f"Synchronized LFO rate from {self.config.lfo_rate:.3f}Hz to {self.sync_lfo_rate:.3f}Hz "
                      f"for {whole_cycles} complete cycles per loop")
        else:
            self.sync_lfo_rate = None

        logger.info(f"Initialized synchronized modulation rates for seamless looping")

    def _generate_loopable_phases(self, n_freqs):
        """Generate random phases that guarantee perfect looping

        This is the key to seamless looping - we must ensure that the
        phases at time=0 and time=loop_length are identical, which means
        the phases must rotate an integer number of cycles within our loop.
        """
        # Set the random seed for reproducibility
        phases_rng = cp.random.RandomState(seed=self.config.seed)

        # For each frequency bin, we need to generate a phase that will
        # complete an integer number of cycles over the loop length.
        # Since we're just generating initial phases, we can use a uniform
        # distribution between 0 and 2π.

        # Generate uniform random phases
        initial_phases = phases_rng.uniform(0, 2*cp.pi, n_freqs)

        # For DC (0 Hz) and Nyquist frequency, force phase to 0
        # This ensures real-valued time domain signal
        initial_phases[0] = 0
        if n_freqs % 2 == 0:  # Even number of frequencies
            initial_phases[-1] = 0

        # For very low frequencies, constrain the phases more tightly to avoid
        # obvious looping artifacts in brown noise
        if self.low_bin > 5:
            # For frequencies below 150Hz, reduce randomness
            # This makes brown noise loop better
            reduced_randomness = cp.linspace(0.1, 1.0, self.low_bin)
            initial_phases[1:self.low_bin+1] *= reduced_randomness

        logger.debug(f"Generated loopable phases with seed {self.config.seed}")
        return initial_phases

    def _precompute_loopable_spectral_filters(self, n_freqs, freq_resolution):
        """Create specialized spectral filters for loopable noise"""
        # Create frequency bins
        freqs = cp.arange(n_freqs, dtype=self.precision) * freq_resolution

        # White noise filter - flat response across all frequencies
        self.loopable_white_filter = cp.ones(n_freqs, dtype=self.precision)

        # Pink noise filter (-3dB/octave)
        pink_curve = cp.ones(n_freqs, dtype=self.precision)
        non_dc_mask = freqs > 0
        pink_curve[non_dc_mask] = 1.0 / cp.sqrt(freqs[non_dc_mask])
        pink_curve[0] = pink_curve[1]  # Fix DC component
        self.loopable_pink_filter = pink_curve / cp.sqrt(cp.mean(pink_curve[1:]**2))

        # Brown noise filter (-6dB/octave) with enhancements for better looping
        brown_curve = cp.ones(n_freqs, dtype=self.precision)
        brown_curve[non_dc_mask] = 1.0 / (freqs[non_dc_mask] ** 2)

        # Enhanced low-frequency handling for brown noise to improve looping
        # For the lowest octave, apply a gentle roll-off to avoid DC buildup
        lowest_bin = max(1, int(20 / freq_resolution))  # 20Hz

        # Create smooth transition for first octave
        if lowest_bin < self.low_bin:
            # Special handling of lowest frequencies for better loop stability
            # Apply a gentler slope to the very lowest frequencies
            transition = cp.linspace(0.1, 1.0, self.low_bin - lowest_bin)
            brown_curve[lowest_bin:self.low_bin] *= transition

        # Set DC exactly to zero to prevent offset
        brown_curve[0] = 0.0

        # Normalize for energy consistency
        self.loopable_brown_filter = brown_curve / cp.sqrt(cp.mean(brown_curve[1:]**2))

        logger.debug(f"Created loopable spectral filters for all noise types")

    def _generate_loopable_noise_spectrum(self, noise_type, block_size):
        """Generate loopable frequency-domain representation for specified noise type"""
        # Select the appropriate spectral filter based on noise type
        if noise_type == 'white':
            spectral_filter = self.loopable_white_filter
        elif noise_type == 'pink':
            spectral_filter = self.loopable_pink_filter
        elif noise_type == 'brown':
            spectral_filter = self.loopable_brown_filter
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Get the needed spectral shape for each channel
        # Ensure the filter length matches the expected FFT output size
        n_freqs = block_size // 2 + 1
        left_magnitudes = spectral_filter[:n_freqs].copy()
        right_magnitudes = spectral_filter[:n_freqs].copy()


        # Apply the loopable phases to create complex spectrum
        left_phases = self.loop_phases[:n_freqs].copy()

        # For right channel, apply decorrelation with frequency-dependent phase shift
        # Ensure decorrelation phases also match expected FFT output size
        decorrelation_phases = self.decorrelation_phases[:n_freqs].copy()
        if noise_type == 'brown':
            # Create frequency-dependent decorrelation that preserves low-end coherence
            phase_shift = cp.zeros_like(decorrelation_phases, dtype=cp.complex128) # Match dtype
            low_bin = min(self.low_bin, n_freqs) # Ensure index is within bounds
            # Less decorrelation below 150Hz
            phase_shift[:low_bin] = decorrelation_phases[:low_bin] * 0.5
            # Normal decorrelation above 150Hz
            phase_shift[low_bin:] = decorrelation_phases[low_bin:]
            right_phases = (left_phases + cp.angle(phase_shift)) % (2*cp.pi)
        else:
            right_phases = (left_phases + cp.angle(decorrelation_phases)) % (2*cp.pi)

        # Create complex spectrum (magnitude * e^(i*phase))
        left_spectrum = left_magnitudes * cp.exp(1j * left_phases)
        right_spectrum = right_magnitudes * cp.exp(1j * right_phases)

        return left_spectrum, right_spectrum

    def _apply_natural_modulation_loopable(self, noise_block, block_start_idx, block_size):
        """Apply natural modulation with synchronized rates for perfect looping"""
        if not self.config.natural_modulation:
            return noise_block

        # Calculate position within the loop (0 to 1)
        loop_position = (block_start_idx % self.loop_length_samples) / self.loop_length_samples

        # Generate time array synchronized to loop position
        t_rel = cp.linspace(0, block_size/self.loop_length_samples,
                          block_size, endpoint=False) # Time relative to start of block within loop cycle
        loop_time = (loop_position + t_rel) % 1.0 # Absolute time within the loop cycle (0 to 1)


        # Create synchronized modulators that complete whole cycles within the loop
        # Using simple sine waves with exact frequencies for perfect loop matching
        low_mod = 1.0 + 0.03 * cp.sin(2 * cp.pi * self.sync_rate_low * loop_time * self.config.loop_length)
        mid_mod = 1.0 + 0.025 * cp.sin(2 * cp.pi * self.sync_rate_mid * loop_time * self.config.loop_length)
        high_mod = 1.0 + 0.02 * cp.sin(2 * cp.pi * self.sync_rate_high * loop_time * self.config.loop_length)

        # Process left channel with spectral modulation (similar to original but with synchronized rates)
        left_fft = cufft.rfft(noise_block[0, :block_size])

        # Get frequency bin indices
        n_bins = len(left_fft)
        # Ensure indices do not exceed actual bin count
        low_idx = min(self.low_bin, n_bins)
        mid_idx = min(self.mid_bin, n_bins)


        # Create smooth transition envelopes between bands (same as original)
        # Creates smooth crossfade between spectral regions
        low_env = cp.ones(low_idx, dtype=self.precision)
        low_transition = min(int(low_idx * 0.1), low_idx)
        if low_transition > 0:
            low_env[-low_transition:] = cp.linspace(1.0, 0.0, low_transition, dtype=self.precision)

        mid_env_len = mid_idx - low_idx
        mid_env = cp.zeros(mid_env_len, dtype=self.precision)
        mid_transition_low = min(int(mid_env_len * 0.1), mid_env_len)
        # Ensure mid_transition_high calculation is safe if mid_env_len is small
        mid_transition_high = min(int(mid_env_len * 0.1), max(0, mid_env_len - mid_transition_low))


        if mid_transition_low > 0:
            mid_env[:mid_transition_low] = cp.linspace(0.0, 1.0, mid_transition_low, dtype=self.precision)
        if mid_transition_high > 0:
             # Check index boundary before slicing
            if mid_env_len - mid_transition_high >= 0:
                mid_env[-(mid_transition_high):] = cp.linspace(1.0, 0.0, mid_transition_high, dtype=self.precision)

        # Fill the middle part with ones
        mid_fill_start = mid_transition_low
        mid_fill_end = mid_env_len - mid_transition_high
        if mid_fill_start < mid_fill_end:
             mid_env[mid_fill_start:mid_fill_end] = 1.0


        high_env_len = n_bins - mid_idx
        high_env = cp.zeros(high_env_len, dtype=self.precision)
        high_transition = min(int(high_env_len * 0.1), high_env_len)
        if high_transition > 0:
            high_env[:high_transition] = cp.linspace(0.0, 1.0, high_transition, dtype=self.precision)
        if high_transition < high_env_len: # Check index boundary
             high_env[high_transition:] = 1.0


        # Apply modulation to left channel frequency bands with synchronized modulators
        # Extract modulation factor (centered at 0) - use only first value if modulator is array
        mod_factor_low = (low_mod[0] if isinstance(low_mod, cp.ndarray) else low_mod) - 1.0
        left_fft[:low_idx] = left_fft[:low_idx] * (1.0 + mod_factor_low * low_env)

        mod_factor_mid = (mid_mod[0] if isinstance(mid_mod, cp.ndarray) else mid_mod) - 1.0
        if mid_env_len > 0: # Check if mid_env is not empty
             left_fft[low_idx:mid_idx] = left_fft[low_idx:mid_idx] * (1.0 + mod_factor_mid * mid_env)


        mod_factor_high = (high_mod[0] if isinstance(high_mod, cp.ndarray) else high_mod) - 1.0
        if high_env_len > 0: # Check if high_env is not empty
             left_fft[mid_idx:] = left_fft[mid_idx:] * (1.0 + mod_factor_high * high_env)


        # Convert back to time domain
        noise_block[0, :block_size] = cufft.irfft(left_fft, n=block_size)

        # Process right channel (similar to original but with synchronized phase offset)
        right_fft = cufft.rfft(noise_block[1, :block_size])

        # Create synchronized phase offset for stereo variation
        phase_offset = cp.pi / 4  # 45 degree base offset
        phase_variation = cp.pi / 12  # ±15 degree variation
        current_phase = phase_offset + phase_variation * cp.sin(2 * cp.pi * self.sync_rate_decorr * loop_position * self.config.loop_length)

        # Apply the same modulators but with synchronized phase offset to right channel
        right_fft[:low_idx] = right_fft[:low_idx] * (1.0 + mod_factor_low * cp.cos(current_phase) * low_env)
        if mid_env_len > 0: # Check if mid_env is not empty
             right_fft[low_idx:mid_idx] = right_fft[low_idx:mid_idx] * (1.0 + mod_factor_mid * cp.cos(current_phase + cp.pi/6) * mid_env)
        if high_env_len > 0: # Check if high_env is not empty
             right_fft[mid_idx:] = right_fft[mid_idx:] * (1.0 + mod_factor_high * cp.cos(current_phase + cp.pi/3) * high_env)

        # Convert back to time domain
        noise_block[1, :block_size] = cufft.irfft(right_fft, n=block_size)

        return noise_block

    def _apply_dynamic_stereo_parameters_loopable(self, noise_block, block_start_idx, block_size):
        """Apply dynamic stereo parameters with loop-synchronized modulation"""
        # Only apply if both stereo enhancement and natural modulation are enabled
        if not (self.config.enhanced_stereo and self.config.natural_modulation):
            return noise_block

        # Calculate position within the loop (0 to 1)
        loop_position = (block_start_idx % self.loop_length_samples) / self.loop_length_samples

        # Generate synchronized modulators for stereo parameters
        # Decorrelation modulation (uses sync_rate_decorr)
        decorr_mod = 1.0 + 0.15 * cp.sin(2 * cp.pi * self.sync_rate_decorr * loop_position * self.config.loop_length)

        # Haas delay modulation (uses sync_rate_haas)
        delay_mod = 1.0 + 0.2 * cp.sin(2 * cp.pi * self.sync_rate_haas * loop_position * self.config.loop_length + 0.5)

        # Apply dynamic decorrelation if enhanced stereo is enabled
        if self.config.enhanced_stereo:
            # Process similar to original but with synchronized modulators
            right = noise_block[1, :block_size]
            right_fft = cufft.rfft(right)

            # Create frequency-dependent phase shifts with synchronized modulation
            n_freqs = len(right_fft)

            # Calculate frequency band boundaries (ensure indices are valid)
            low_freq_idx = min(self.low_bin, n_freqs)
            mid_freq_idx = min(self.mid_bin, n_freqs)


            # Create dynamic phases with synchronized modulated strength
            phases = cp.zeros(n_freqs, dtype=self.precision)

            # Scale phase offsets by the synchronized decorrelation modulator
            if low_freq_idx > 0: # Ensure index range is valid
                 phases[:low_freq_idx] = cp.linspace(0, cp.pi/8, low_freq_idx) * decorr_mod
            if mid_freq_idx > low_freq_idx: # Ensure index range is valid
                 phases[low_freq_idx:mid_freq_idx] = cp.linspace(cp.pi/8, cp.pi/4, mid_freq_idx-low_freq_idx) * decorr_mod
            if n_freqs > mid_freq_idx: # Ensure index range is valid
                 phases[mid_freq_idx:] = cp.linspace(cp.pi/4, cp.pi/2.5, n_freqs-mid_freq_idx) * decorr_mod

            # Convert to complex exponentials for FFT multiplication
            dynamic_decorr_phases = cp.exp(1j * phases)

            # Apply the dynamic phase shift
            right_fft = right_fft * dynamic_decorr_phases

            # Convert back to time domain
            noise_block[1, :block_size] = cufft.irfft(right_fft, n=block_size)

        # Apply dynamic Haas effect if enabled
        if self.config.haas_effect:
            # Same as original but with synchronized delay modulation
            base_delay = 0.008  # seconds
            delay_variation = 0.002  # ±2ms variation

            # Calculate current delay time with synchronized modulation
            current_delay = base_delay + delay_variation * (delay_mod - 1.0)

            # Convert to samples
            delay_samples = int(current_delay * self.config.sample_rate)

            # Process the same as original with the synchronized delay
            right_orig = noise_block[1, :block_size]
            right_delayed = cp.zeros(block_size, dtype=self.precision)
            # Ensure delay_samples is not negative and within bounds
            delay_samples = max(0, min(block_size, delay_samples))
            if block_size - delay_samples > 0: # Ensure slice is valid
                 right_delayed[delay_samples:] = right_orig[:block_size-delay_samples]


            right_fft = cufft.rfft(right_orig)
            right_delayed_fft = cufft.rfft(right_delayed)

            n_freqs = len(right_fft)
            freq_resolution = self.config.sample_rate / block_size
            # Ensure cutoff_bin is valid
            cutoff_bin = max(0, min(n_freqs, int(150 / freq_resolution if freq_resolution > 0 else n_freqs)))


            xfade_width = int(cutoff_bin * 0.5)
            # Ensure xfade_width does not exceed bounds
            xfade_width = min(xfade_width, cutoff_bin)
            xfade_width = min(xfade_width, n_freqs - cutoff_bin)


            transition = cp.zeros(n_freqs, dtype=self.precision)
            # Calculate indices safely
            trans_start_idx = max(0, cutoff_bin - xfade_width)
            trans_end_idx = min(n_freqs, cutoff_bin + xfade_width)
            trans_len = trans_end_idx - trans_start_idx

            transition[:trans_start_idx] = 0.0
            transition[trans_end_idx:] = 0.7

            if trans_len > 0:
                 transition[trans_start_idx:trans_end_idx] = cp.linspace(
                    0.0, 0.7, trans_len, dtype=self.precision)


            hybrid_fft = right_fft * (1.0 - transition) + right_delayed_fft * transition

            noise_block[1, :block_size] = cufft.irfft(hybrid_fft, n=block_size)

        return noise_block

    # <<<< START OF FIXED FUNCTION >>>>>
    def _apply_micro_pitch_variations_loopable(self, noise_block, block_start_idx, block_size):
        """Apply subtle micro-pitch variations with loop-synchronized modulation - FIXED VERSION"""
        # Only apply if natural modulation is enabled
        if not self.config.natural_modulation:
            return noise_block

        # Calculate position within the loop (0 to 1)
        loop_position = (block_start_idx % self.loop_length_samples) / self.loop_length_samples

        # Create synchronized modulation sources for pitch variations
        # These need to complete whole cycles within the loop length
        low_pitch_mod = 0.005 * cp.sin(2 * cp.pi * self.sync_rate_pitch_low * loop_position * self.config.loop_length)
        mid_pitch_mod = 0.003 * cp.sin(2 * cp.pi * self.sync_rate_pitch_mid * loop_position * self.config.loop_length + 0.7)
        high_pitch_mod = 0.002 * cp.sin(2 * cp.pi * self.sync_rate_pitch_high * loop_position * self.config.loop_length + 1.4)

        # Skip processing if modulation values are negligible (optimization)
        if (abs(low_pitch_mod) < 0.0005 and
            abs(mid_pitch_mod) < 0.0005 and
            abs(high_pitch_mod) < 0.0005):
            return noise_block

        # Process each channel with vectorized implementation (same as original but with synchronized modulators)
        for ch in range(2):
            channel_fft = cufft.rfft(noise_block[ch, :block_size])

            n_bins = len(channel_fft)
            # Ensure indices are valid
            low_idx = min(self.low_bin, n_bins)
            mid_idx = min(self.mid_bin, n_bins)


            resampled_fft = cp.zeros_like(channel_fft, dtype=cp.complex128)

            # Same vectorized implementation as original
            low_bins = cp.arange(1, low_idx) # Bins from 1 up to low_idx
            mid_bins = cp.arange(low_idx, mid_idx) # Bins from low_idx up to mid_idx
            high_bins = cp.arange(mid_idx, n_bins) # Bins from mid_idx up to n_bins


            # Calculate bin shifts with synchronized modulation - FIX BROADCASTING
            if len(low_bins) > 0:
                low_bin_shifts = low_bins * low_pitch_mod
                low_bin_ints = cp.floor(low_bins + low_bin_shifts).astype(cp.int32)
                low_bin_fracs = (low_bins + low_bin_shifts) - low_bin_ints

                # Filter out bins that would go out of range
                valid_mask = (low_bin_ints >= 0) & (low_bin_ints < n_bins-1)
                valid_bins = low_bins[valid_mask]
                valid_ints = low_bin_ints[valid_mask]
                valid_fracs = low_bin_fracs[valid_mask]

                # Apply the interpolation in vectorized form - FIX: Calculate terms separately
                if len(valid_bins) > 0:
                    term1 = channel_fft[valid_ints] * (1 - valid_fracs)
                    term2 = channel_fft[valid_ints + 1] * valid_fracs
                    resampled_fft[valid_bins] = term1 + term2

            # Do the same for mid frequencies - FIX BROADCASTING
            if len(mid_bins) > 0:
                mid_bin_shifts = mid_bins * mid_pitch_mod
                mid_bin_ints = cp.floor(mid_bins + mid_bin_shifts).astype(cp.int32)
                mid_bin_fracs = (mid_bins + mid_bin_shifts) - mid_bin_ints

                valid_mask = (mid_bin_ints >= 0) & (mid_bin_ints < n_bins-1)
                valid_bins = mid_bins[valid_mask]
                valid_ints = mid_bin_ints[valid_mask]
                valid_fracs = mid_bin_fracs[valid_mask]

                if len(valid_bins) > 0:
                    term1 = channel_fft[valid_ints] * (1 - valid_fracs)
                    term2 = channel_fft[valid_ints + 1] * valid_fracs
                    resampled_fft[valid_bins] = term1 + term2

            # Do the same for high frequencies - FIX BROADCASTING
            if len(high_bins) > 0:
                high_bin_shifts = high_bins * high_pitch_mod
                high_bin_ints = cp.floor(high_bins + high_bin_shifts).astype(cp.int32)
                high_bin_fracs = (high_bins + high_bin_shifts) - high_bin_ints

                valid_mask = (high_bin_ints >= 0) & (high_bin_ints < n_bins-1)
                valid_bins = high_bins[valid_mask]
                valid_ints = high_bin_ints[valid_mask]
                valid_fracs = high_bin_fracs[valid_mask]

                if len(valid_bins) > 0:
                    term1 = channel_fft[valid_ints] * (1 - valid_fracs)
                    term2 = channel_fft[valid_ints + 1] * valid_fracs
                    resampled_fft[valid_bins] = term1 + term2

            # Preserve DC component
            if n_bins > 0: # Ensure index is valid
                resampled_fft[0] = channel_fft[0]


            # Convert back to time domain
            noise_block[ch, :block_size] = cufft.irfft(resampled_fft, n=block_size)

        return noise_block
    # <<<< END OF FIXED FUNCTION >>>>>

    def _apply_lfo_modulation_loopable(self, noise_block, block_start_idx, block_size):
        """Apply LFO modulation with loop-synchronized rate"""
        if not self.sync_lfo_rate:
            return noise_block

        # Calculate position within the loop (0 to 1)
        loop_position = (block_start_idx % self.loop_length_samples) / self.loop_length_samples

        # Generate time indices with loop-synchronized rate
        t_rel = cp.linspace(0, block_size/self.loop_length_samples,
                           block_size, endpoint=False)
        loop_time = (loop_position + t_rel) % 1.0

        # Create synchronized sinusoidal modulation (±1dB)
        modulation_depth = 10**(1.0/20) - 10**(-1.0/20)  # ±1dB in linear scale
        modulation = 1.0 + modulation_depth/2 * cp.sin(2 * cp.pi * self.sync_lfo_rate * loop_time * self.config.loop_length)

        # Apply same modulation to both channels
        noise_block[0, :block_size] = noise_block[0, :block_size] * modulation
        noise_block[1, :block_size] = noise_block[1, :block_size] * modulation

        return noise_block

    def generate_block(self, block_size, block_start_idx=0):
        """Generate a block of noise with inherent loopability"""
        # If not using loopable generation, fall back to standard method
        if not self.use_loopable_generation:
            return super().generate_block(block_size, block_start_idx)

        # For loopable generation, we use a frequency-domain approach
        # that ensures perfect phase continuity

        # Determine which noise types to generate based on color mix
        color_mix = self.config.color_mix
        white_gain = cp.sqrt(color_mix.get('white', 0))
        pink_gain = cp.sqrt(color_mix.get('pink', 0))
        brown_gain = cp.sqrt(color_mix.get('brown', 0))

        # Generate all needed noise types in frequency domain
        # Calculate expected FFT size for this block
        n_freqs_block = block_size // 2 + 1

        frequency_domain_buffers = {}
        if white_gain > 0.001:
            left_white, right_white = self._generate_loopable_noise_spectrum('white', block_size)
            frequency_domain_buffers['white'] = (left_white[:n_freqs_block], right_white[:n_freqs_block]) # Trim to block size


        if pink_gain > 0.001:
            left_pink, right_pink = self._generate_loopable_noise_spectrum('pink', block_size)
            frequency_domain_buffers['pink'] = (left_pink[:n_freqs_block], right_pink[:n_freqs_block]) # Trim to block size


        if brown_gain > 0.001:
            left_brown, right_brown = self._generate_loopable_noise_spectrum('brown', block_size)
            frequency_domain_buffers['brown'] = (left_brown[:n_freqs_block], right_brown[:n_freqs_block]) # Trim to block size


        # Mix the noise types in frequency domain with proper gains
        left_spectrum = cp.zeros(n_freqs_block, dtype=cp.complex128)
        right_spectrum = cp.zeros(n_freqs_block, dtype=cp.complex128)

        # Apply gains and mix
        if 'white' in frequency_domain_buffers:
            left_white, right_white = frequency_domain_buffers['white']
            left_spectrum += left_white * white_gain
            right_spectrum += right_white * white_gain

        if 'pink' in frequency_domain_buffers:
            left_pink, right_pink = frequency_domain_buffers['pink']
            left_spectrum += left_pink * pink_gain
            right_spectrum += right_pink * pink_gain

        if 'brown' in frequency_domain_buffers:
            left_brown, right_brown = frequency_domain_buffers['brown']
            left_spectrum += left_brown * brown_gain
            right_spectrum += right_brown * brown_gain

        # Normalize the mix
        total_power = white_gain**2 + pink_gain**2 + brown_gain**2
        normalization = cp.sqrt(1.0 / (total_power + 1e-10))
        left_spectrum *= normalization
        right_spectrum *= normalization

        # If we're at the beginning of the file, save this block for reference
        if self.is_first_block and block_start_idx == 0:
            self.first_block_spectra = (left_spectrum.copy(), right_spectrum.copy())
            self.is_first_block = False

        # Convert back to time domain
        left = cufft.irfft(left_spectrum, n=block_size)
        right = cufft.irfft(right_spectrum, n=block_size)

        # Store in buffer
        self._mixed_buffer[0, :block_size] = left
        self._mixed_buffer[1, :block_size] = right
        mixed_noise = self._mixed_buffer[:, :block_size]

        # Apply synchronized effects for perfect looping

        # Apply natural modulation with loop-synchronized rates
        if self.config.natural_modulation:
            mixed_noise = self._apply_natural_modulation_loopable(mixed_noise, block_start_idx, block_size)

        # Apply dynamic stereo parameters with loop-synchronized rates
        mixed_noise = self._apply_dynamic_stereo_parameters_loopable(mixed_noise, block_start_idx, block_size)

        # Apply micro-pitch variations with loop-synchronized rates
        mixed_noise = self._apply_micro_pitch_variations_loopable(mixed_noise, block_start_idx, block_size)

        # Apply LFO modulation with loop-synchronized rate if enabled
        mixed_noise = self._apply_lfo_modulation_loopable(mixed_noise, block_start_idx, block_size)

        # Apply gain and limiting (same as original)
        mixed_noise = self._apply_gain_and_limiting(
            mixed_noise, self.config.rms_target, self.config.peak_ceiling, block_size
        )

        # Apply true-peak limiting (same as original)
        mixed_noise = self._apply_true_peak_limiting(mixed_noise, self.config.peak_ceiling, block_size)

        # Apply pre-emphasis if enabled (same as original)
        mixed_noise = self._apply_pre_emphasis(mixed_noise, block_size)

        return mixed_noise

    def generate_to_file(self, output_path, progress_callback=None, make_loop=True):
        """Generate loopable noise and save to file

        With the LoopableNoiseGenerator, the 'make_loop' parameter is redundant
        if using inherent looping, but we keep it for compatibility.
        """
        import soundfile as sf

        # For the loop-first architecture, we don't need post-processing
        # to create loops - it's loopable by design
        # Let the parent method do all the file I/O work
        result = super().generate_to_file(output_path, progress_callback, make_loop=False)

        # If generation was successful and using loop-first mode, update result
        if result["success"] and self.use_loopable_generation:
            # Add looping information to result
            result["looped"] = True
            result["loop_method"] = "inherent"
            result["loop_length"] = self.config.loop_length
            logger.info(f"File was generated with inherently loopable audio using Loop-First Architecture")
            logger.info(f"Loop length: {self.config.loop_length}s ({self.loop_length_samples} samples)")

        return result


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
    parser = argparse.ArgumentParser(description="Baby-Noise Generator v2.1.0 - Enhanced Organic DSP with Perfect Looping")

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

    # Looping options - new options for loop-first architecture
    parser.add_argument("--loop", action="store_true", help="Create seamless looping file", default=True)
    parser.add_argument("--no-loop", action="store_false", dest="loop", help="Don't create loop")
    parser.add_argument("--loop-first", action="store_true", dest="loop_first", help="Use loop-first architecture for perfect loops", default=True)
    parser.add_argument("--loop-post", action="store_false", dest="loop_first", help="Use traditional post-processing for loops")
    parser.add_argument("--loop-length", type=int, help="Loop length in seconds for loop-first architecture", default=DEFAULT_LOOP_LENGTH)

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

    # Validate loop length - not longer than duration
    loop_length = min(args.loop_length, args.duration)
    if loop_length != args.loop_length:
        logger.warning(f"Loop length ({args.loop_length}s) exceeds duration ({args.duration}s), adjusted to {loop_length}s")

    # Create configuration with enhanced stereo and looping options
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
        enhanced_stereo=args.enhanced_stereo,
        loop_length=loop_length,
        generate_seamless=args.loop_first and args.loop  # Only use loop-first if looping is enabled
    )

    # Print configuration summary
    logger.info(f"Generating {args.duration}s of stereo noise...")
    logger.info(f"Output: {args.output}")
    logger.info(f"YouTube-optimized profile: {YOUTUBE_PROFILE['lufs_threshold']} LUFS")
    if color_mix is None:
        logger.info(f"Warmth: {args.warmth}%")
    else:
        logger.info(f"Color mix: white={config.color_mix['white']:.2f}, pink={config.color_mix['pink']:.2f}, brown={config.color_mix['brown']:.2f}")
    logger.info(f"Seed: {seed}")

    # Log stereo enhancement options
    logger.info(f"Enhanced stereo: {'ON' if args.enhanced_stereo else 'OFF'}")
    logger.info(f"Haas effect: {'ON' if args.haas else 'OFF'}")
    logger.info(f"Natural modulation: {'ON' if args.natural_mod else 'OFF'}")

    # Log looping options
    if args.loop:
        if args.loop_first:
            logger.info(f"Seamless looping: ON (using loop-first architecture)")
            logger.info(f"Loop length: {loop_length}s")
        else:
            logger.info(f"Seamless looping: ON (using post-processing)")
    else:
        logger.info(f"Seamless looping: OFF")

    # Check for pyloudnorm availability for accurate LUFS measurement
    try:
        import pyloudnorm
        logger.info("pyloudnorm detected - will use accurate BS.1770-4 LUFS measurement")
    except ImportError:
        logger.info("pyloudnorm not detected - will use approximate LUFS calculation")
        logger.info("For accurate loudness measurement, install pyloudnorm: pip install pyloudnorm")

    # Create generator with appropriate type based on looping configuration
    if args.loop and args.loop_first:
        logger.info("Creating LoopableNoiseGenerator with inherent seamless looping")
        generator = LoopableNoiseGenerator(config)
    else:
        logger.info("Creating standard NoiseGenerator")
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

            if result.get("looped", False):
                if result.get("loop_method") == "inherent":
                    print(f"Loop: Inherently seamless ({config.loop_length}s loop length)")
                else:
                    print("Loop: Post-processed for seamless looping")

            if "warning" in result:
                print(f"Warning: {result['warning']}")

    except KeyboardInterrupt:
        print("\nGeneration cancelled by user")
        generator.cancel_generation()
        return 1
    except Exception as e:
        logger.error(f"Error generating noise: {e}")
        print(f"\nAn unexpected error occurred: {e}") # Print error to console as well
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())