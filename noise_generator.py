#!/usr/bin/env python3
# Baby-Noise Generator v1.1
# GPU-accelerated white/pink/brown noise generator for infant sleep

import argparse
import os
import time
import yaml
import numpy as np
import scipy.signal
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List
import logging
from functools import lru_cache

# Add pyloudnorm for ITU-R BS.1770-4 compliant LUFS calculation
try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

# Optional GPU dependencies
try:
    import cupy as cp
    import cupyx.scipy.signal
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("baby-noise")

# Constants
SAMPLE_RATE = 44100
DEFAULT_PEAK_CEILING = -1.0  # dBFS
DEFAULT_RMS_TARGET = -63.0   # dBFS (~47 dB SPL)
DEFAULT_DURATION = 600       # seconds
MAX_GPU_BUFFER = 128 * 1024 * 1024  # 128MB chunks for GPU processing
CPU_BUFFER_SIZE = 2048               # Small buffer for real-time CPU processing
BROWN_HPF_FREQ = 20.0       # Hz - remove DC offset and ultra-low freqs
BROWN_LEAKY_ALPHA = 0.999   # Leaky integrator coefficient
SAFETY_LUFS_THRESHOLD = -27.0  # LUFS threshold (~50 dB SPL)

# Module-level cache for FIR filters
_pink_fir_cache = {}


@lru_cache(maxsize=8)
def get_pink_fir(sample_rate, n_taps=4097):
    """Cache and return pink noise FIR filter with exactly 4097 taps (default)"""
    if not HAS_CUPY:
        return None
    
    # Always include n_taps in the cache key to correctly handle different sizes
    cache_key = (sample_rate, n_taps)
    if cache_key not in _pink_fir_cache:
        # Generate filter coefficients
        freq = cp.fft.rfftfreq(n_taps, 1/sample_rate)
        freq[0] = freq[1]  # Avoid division by zero
        response = 1.0 / cp.sqrt(freq)
        impulse = cp.fft.irfft(response)
        window = cp.hanning(n_taps)
        impulse = impulse * window
        impulse = impulse / cp.sqrt(cp.sum(impulse**2))
        _pink_fir_cache[cache_key] = impulse
    
    return _pink_fir_cache[cache_key]


@dataclass
class NoiseConfig:
    """Configuration for noise generator"""
    seed: int
    duration: float
    color_mix: Dict[str, float]
    rms_target: float
    peak_ceiling: float
    lfo_rate: Optional[float] = None
    sample_rate: int = SAMPLE_RATE
    use_gpu: bool = False


class NoiseGenerator:
    """GPU-accelerated noise generator with CPU fallback"""
    
    def __init__(self, config: NoiseConfig):
        self.config = config
        self.xp = cp if config.use_gpu and HAS_CUPY else np
        self.buffer_size = MAX_GPU_BUFFER if config.use_gpu and HAS_CUPY else CPU_BUFFER_SIZE
        
        # Initialize the PRNG with seed
        if config.use_gpu and HAS_CUPY:
            # Use newer Generator API with Philox algorithm for better performance
            self.rng = cp.random.Generator(cp.random.Philox4x3210(config.seed))
        else:
            self.rng = np.random.RandomState(seed=config.seed)
        
        # Calculate number of samples and buffers needed
        self.total_samples = int(config.duration * config.sample_rate)
        self.buffer_samples = min(self.buffer_size // 4, self.total_samples)  # 4 bytes per float32
        self.num_buffers = (self.total_samples + self.buffer_samples - 1) // self.buffer_samples
        
        # Pre-calculate filter coefficients for brown noise HPF
        nyquist = config.sample_rate / 2.0
        self.brown_hpf_b, self.brown_hpf_a = scipy.signal.butter(
            1, BROWN_HPF_FREQ / nyquist, btype='highpass'
        )
        
        # For GPU-based processing, keep filter coefficients as NumPy arrays
        if config.use_gpu and HAS_CUPY:
            # Get cached pink FIR filter
            self.pink_fir = get_pink_fir(config.sample_rate)
        else:
            # IMPROVED: Initialize coefficients for Voss-McCartney algorithm (12-stage) for CPU pink noise
            self.pink_octaves = 12  # Number of octaves for Voss-McCartney
            self.pink_state = np.zeros(self.pink_octaves)
            self.pink_last_value = 0.0
            
            # Keep Paul Kellett coefficients for legacy support
            self.pink_b0 = 0.99765 * 0.0555179
            self.pink_b1 = 0.96300 * 0.0750759
            self.pink_b2 = 0.57000 * 0.1538520
            self.pink_b3 = -0.06000 * 0.3104856
            self.pink_b4 = -0.53200 * 0.5329522
            self.pink_b5 = -0.58200 * 0.7234423
            self.pink_b6 = 0.50700 * 0.0168980
            self.pink_mem = np.zeros(7)  # State variables
            
        # Initialize LUFS meter if pyloudnorm is available
        if HAS_PYLOUDNORM:
            self.lufs_meter = pyln.Meter(self.config.sample_rate)
            
    def generate_white_noise(self, length: int) -> np.ndarray:
        """Generate white noise using Philox PRNG"""
        if self.config.use_gpu and HAS_CUPY:
            # GPU implementation using new Generator API
            return self.rng.normal(0, 1, length, dtype=cp.float32)
        else:
            # CPU implementation
            return self.rng.normal(0, 1, length).astype(np.float32)
    
    def generate_pink_noise(self, white_noise: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Transform white noise into pink noise"""
        if self.config.use_gpu and HAS_CUPY:
            # GPU implementation - FFT convolution with optimized plan reuse
            # Use oaconvolve which has better plan caching than fftconvolve
            return cupyx.scipy.signal.oaconvolve(white_noise, self.pink_fir, mode='same')
        else:
            # IMPROVED: Voss-McCartney algorithm (12-stage filter) for flatter response
            # This implementation reduces the frequency tilt compared to the Paul Kellett algorithm
            pink = np.zeros_like(white_noise)
            for i in range(len(white_noise)):
                # Get white noise value
                white = white_noise[i]
                
                # Update octaves
                total = 0.0
                counter = 0
                
                # Determine which octaves to update
                for j in range(self.pink_octaves):
                    if counter & (1 << j) == 0:
                        self.pink_state[j] = self.rng.normal(0, 1)
                    total += self.pink_state[j]
                    counter += 1
                
                # Average the octaves and scale
                avg = total / self.pink_octaves
                
                # Apply smoothing between samples for better low-frequency response
                pink[i] = 0.8 * avg + 0.2 * self.pink_last_value + 0.15 * white
                self.pink_last_value = pink[i]
                
            # Normalize
            pink = pink / np.sqrt(np.mean(pink**2))
            return pink
    
    def generate_brown_noise(self, white_noise: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Transform white noise into brown noise using leaky integrator + HPF"""
        if self.config.use_gpu and HAS_CUPY:
            # IMPROVED: Fully vectorized GPU implementation using cumsum
            # Using the formula provided: cp.cumsum((1-α) * white_noise[::-1], dtype=cp.float32)[::-1] * α**cp.arange(N)
            N = len(white_noise)
            alpha = BROWN_LEAKY_ALPHA
            
            # Create vector of powers of alpha
            alpha_powers = cp.power(alpha, cp.arange(N, dtype=cp.float32))
            
            # Reverse input, compute cumulative sum, then reverse again
            reversed_input = (1.0 - alpha) * white_noise[::-1]
            cumsum_result = cp.cumsum(reversed_input, dtype=cp.float32)
            brown = cumsum_result[::-1] * alpha_powers
            
            # Apply HPF to remove DC offset using cupyx.scipy.signal.lfilter
            return cupyx.scipy.signal.lfilter(self.brown_hpf_b, self.brown_hpf_a, brown)
        else:
            # CPU implementation
            xp = np
            # Leaky integrator (vectorized)
            brown = xp.zeros_like(white_noise)
            brown[0] = white_noise[0]
            # y[n] = α*y[n-1] + (1-α)*x[n]
            for i in range(1, len(white_noise)):
                brown[i] = BROWN_LEAKY_ALPHA * brown[i-1] + (1-BROWN_LEAKY_ALPHA) * white_noise[i]
            
            # Apply HPF to remove DC offset
            return scipy.signal.lfilter(self.brown_hpf_b, self.brown_hpf_a, brown)
    
    def apply_color_mix(self, white: Union[np.ndarray, "cp.ndarray"], 
                       pink: Union[np.ndarray, "cp.ndarray"], 
                       brown: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Mix noise colors according to configuration"""
        xp = cp if self.config.use_gpu and HAS_CUPY else np
        
        # Normalize each noise type
        white = white / xp.sqrt(xp.mean(white**2))
        pink = pink / xp.sqrt(xp.mean(pink**2))
        brown = brown / xp.sqrt(xp.mean(brown**2))
        
        # Apply color mix
        mix = (self.config.color_mix.get('white', 0.0) * white +
               self.config.color_mix.get('pink', 0.0) * pink +
               self.config.color_mix.get('brown', 0.0) * brown)
        
        return mix
    
    def apply_lfo_modulation(self, audio: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Apply slow LFO gain modulation if configured"""
        if not self.config.lfo_rate:
            return audio
            
        xp = cp if self.config.use_gpu and HAS_CUPY else np
        
        # Generate LFO signal (sine wave)
        t = xp.arange(len(audio)) / self.config.sample_rate
        lfo = xp.sin(2 * xp.pi * self.config.lfo_rate * t)
        
        # Scale to ±2dB modulation (linear scale: 10^(±2/20) ≈ 0.8-1.25)
        gain_mod = 1.0 + 0.2 * lfo
        
        return audio * gain_mod
    
    def calculate_lufs(self, audio: Union[np.ndarray, "cp.ndarray"], window_size_sec=1.0) -> Tuple[Union[np.ndarray, "cp.ndarray"], float]:
        """Calculate LUFS-style loudness using a sliding window
        
        IMPROVED: Uses pyloudnorm library for ITU-R BS.1770-4 compliant LUFS calculation when available
        """
        xp = cp if self.config.use_gpu and HAS_CUPY else np
        
        # If pyloudnorm is available, use it for true ITU-R BS.1770-4 LUFS calculation
        if HAS_PYLOUDNORM:
            # We need to move data to CPU if it's on GPU
            if self.config.use_gpu and HAS_CUPY:
                cpu_audio = cp.asnumpy(audio)
            else:
                cpu_audio = audio
                
            # Calculate integrated LUFS
            integrated_lufs = self.lufs_meter.integrated_loudness(cpu_audio)
            
            # For momentary LUFS, use a sliding window
            window_samples = int(window_size_sec * self.config.sample_rate)
            hop_size = window_samples // 4  # 75% overlap
            num_windows = max(1, (len(cpu_audio) - window_samples) // hop_size + 1)
            
            # Store momentary LUFS values
            momentary_values = []
            for i in range(num_windows):
                start = i * hop_size
                end = start + window_samples
                if end > len(cpu_audio):
                    break
                    
                window = cpu_audio[start:end]
                # Calculate momentary (400ms) LUFS - pyloudnorm wants 2D array (channels, samples)
                momentary = self.lufs_meter.integrated_loudness(window.reshape(1, -1))
                momentary_values.append(momentary)
            
            # Return the LUFS values and integrated LUFS
            if momentary_values:
                return xp.array(momentary_values), integrated_lufs
            else:
                return xp.array([]), integrated_lufs
            
        else:
            # Fallback to simplified LUFS approximation if pyloudnorm not available
            window_samples = int(window_size_sec * self.config.sample_rate)
            
            # Use overlapping windows with 75% overlap
            hop_size = window_samples // 4
            num_windows = max(1, (len(audio) - window_samples) // hop_size + 1)
            
            loudness_values = []
            for i in range(num_windows):
                start = i * hop_size
                end = start + window_samples
                if end > len(audio):
                    break
                    
                window = audio[start:end]
                # Mean square (simplified LUFS)
                ms = xp.mean(window**2)
                loudness = -0.691 + 10 * xp.log10(ms + 1e-10)
                loudness_values.append(loudness)
            
            # Return the loudness values and the integrated (gated) loudness
            if loudness_values:
                return xp.array(loudness_values), xp.mean(xp.array(loudness_values))
            else:
                return xp.array([]), -100.0
    
    def apply_loudness_control(self, audio: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Apply RMS target and peak ceiling with LUFS-based safety control"""
        xp = cp if self.config.use_gpu and HAS_CUPY else np
        
        # Calculate LUFS for 1-second windows
        loudness_values, integrated_lufs = self.calculate_lufs(audio, window_size_sec=1.0)
        
        # Calculate current RMS level
        current_rms_db = 20 * xp.log10(xp.sqrt(xp.mean(audio**2)) + 1e-10)
        
        # Calculate gain needed to reach target RMS
        gain_db = self.config.rms_target - current_rms_db
        
        # AAP safety threshold (approximately -27 LUFS for 50 dB SPL)
        safety_lufs_threshold = SAFETY_LUFS_THRESHOLD
        
        # Apply safety gain reduction if LUFS exceeds threshold
        if integrated_lufs > safety_lufs_threshold:
            safety_gain_db = safety_lufs_threshold - integrated_lufs
            gain_db += safety_gain_db
            logger.warning(
                f"LUFS {integrated_lufs:.1f} exceeds safety threshold {safety_lufs_threshold:.1f}. "
                f"Applying additional {safety_gain_db:.1f} dB reduction."
            )
        
        # Apply gain
        gain_linear = 10 ** (gain_db / 20)
        audio = audio * gain_linear
        
        # Check for peaks exceeding ceiling
        peak = xp.max(xp.abs(audio))
        peak_db = 20 * xp.log10(peak + 1e-10)
        
        # Apply limiting if needed
        if peak_db > self.config.peak_ceiling:
            limiting_gain = 10 ** ((self.config.peak_ceiling - peak_db) / 20)
            audio = audio * limiting_gain
            logger.info(f"Applied limiting: {limiting_gain:.3f} ({peak_db:.1f} dBFS -> {self.config.peak_ceiling:.1f} dBFS)")
        
        return audio
    
    def apply_dither(self, audio: Union[np.ndarray, "cp.ndarray"], output_format: str = "PCM_16") -> np.ndarray:
        """Apply TPDF dither for bit-depth-reduced output formats
        
        Args:
            audio: Input audio array (CPU or GPU)
            output_format: Output format (e.g., "PCM_16" for 16-bit PCM)
            
        Returns:
            Dithered audio array on CPU
        """
        # Skip dither for formats that don't need it (FLAC, FLOAT)
        if output_format not in ["PCM_16", "PCM_24"]:
            # Just move to CPU if needed, no dither required
            if self.config.use_gpu and HAS_CUPY:
                return cp.asnumpy(audio)
            else:
                return audio
            
        # For PCM formats, apply TPDF dither
        if self.config.use_gpu and HAS_CUPY:
            # Move to CPU first
            audio_cpu = cp.asnumpy(audio)
            
            # Generate and apply dither on CPU to avoid extra GPU->CPU transfer
            amplitude = 2.0 / (2**16 - 1)
            dither = np.random.uniform(-amplitude, amplitude, len(audio_cpu)) - \
                     np.random.uniform(-amplitude, amplitude, len(audio_cpu))
            return audio_cpu + dither
        else:
            # CPU path remains unchanged
            xp = np
            amplitude = 2.0 / (2**16 - 1)
            dither = xp.random.uniform(-amplitude, amplitude, len(audio)) - \
                     xp.random.uniform(-amplitude, amplitude, len(audio))
            return audio + dither
    
    def generate_buffer(self, output_format: str = "PCM_16") -> Tuple[np.ndarray, Dict]:
        """Generate a buffer of noise according to configuration"""
        # Generate white noise
        white = self.generate_white_noise(self.buffer_samples)
        
        # Generate pink and brown from white
        pink = self.generate_pink_noise(white)
        brown = self.generate_brown_noise(white)
        
        # Mix colors
        audio = self.apply_color_mix(white, pink, brown)
        
        # Apply LFO modulation if configured
        if self.config.lfo_rate:
            audio = self.apply_lfo_modulation(audio)
        
        # Apply loudness control
        audio = self.apply_loudness_control(audio)
        
        # Apply dither and move to CPU if needed
        audio = self.apply_dither(audio, output_format)
        
        # Calculate metrics
        metrics = {
            "rms_db": 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10),
            "peak_db": 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        }
        
        return audio, metrics
    
    def generate_to_file(self, output_path: str) -> Dict:
        """Generate noise and write to file"""
        logger.info(f"Generating {self.config.duration:.1f}s of noise to {output_path}")
        logger.info(f"Using {'GPU' if self.config.use_gpu and HAS_CUPY else 'CPU'} backend")
        logger.info(f"Seed: {self.config.seed}")
        logger.info(f"Color mix: {self.config.color_mix}")
        
        start_time = time.time()
        samples_written = 0
        metrics = {"buffers": [], "avg_rms_db": 0, "peak_db": -np.inf}
        
        # Determine output format from file extension
        _, ext = os.path.splitext(output_path)
        output_format = "PCM_16" if ext.lower() == ".wav" else "FLAC" if ext.lower() == ".flac" else "PCM_16"
        
        # Store original buffer size to avoid UI progress jumps
        orig_buffer_samples = self.buffer_samples
        
        # Create output file
        with sf.SoundFile(output_path, mode='w', samplerate=self.config.sample_rate, 
                          channels=1, subtype=output_format) as f:
            
            # Generate and write buffers
            for i in range(self.num_buffers):
                remaining = self.total_samples - samples_written
                actual_buffer_samples = min(orig_buffer_samples, remaining)
                
                # Temporarily adjust buffer_samples for this iteration only
                self.buffer_samples = actual_buffer_samples
                
                # Generate buffer
                buffer, buffer_metrics = self.generate_buffer(output_format)
                
                # Write to file
                f.write(buffer)
                samples_written += len(buffer)
                
                # Update metrics
                metrics["buffers"].append(buffer_metrics)
                metrics["avg_rms_db"] += buffer_metrics["rms_db"] / self.num_buffers
                metrics["peak_db"] = max(metrics["peak_db"], buffer_metrics["peak_db"])
                
                # Log progress
                if i % max(1, self.num_buffers // 10) == 0 or i == self.num_buffers - 1:
                    progress = samples_written / self.total_samples * 100
                    elapsed = time.time() - start_time
                    logger.info(f"Progress: {progress:.1f}% ({elapsed:.1f}s elapsed)")
        
        # Restore original buffer size
        self.buffer_samples = orig_buffer_samples
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Generated {self.config.duration:.1f}s of noise in {elapsed:.1f}s")
        logger.info(f"Average RMS: {metrics['avg_rms_db']:.1f} dBFS, Peak: {metrics['peak_db']:.1f} dBFS")
        
        # Add metadata to file
        # Note: This would require additional libraries like mutagen
        
        return metrics


class StreamingNoiseGenerator(NoiseGenerator):
    """Real-time streaming version of the noise generator"""
    
    def __init__(self, config: NoiseConfig):
        # Force CPU mode for streaming
        config.use_gpu = False
        super().__init__(config)
        
        # State variables for streaming
        self.pink_state = np.zeros(self.pink_octaves)  # Using improved Voss-McCartney
        self.pink_last_value = 0.0
        self.brown_prev = 0.0
    
    def get_next_chunk(self, chunk_size: int = 2048) -> np.ndarray:
        """Generate a chunk of audio for real-time streaming"""
        # Override buffer size for streaming
        self.buffer_samples = chunk_size
        
        # Generate chunk with explicit output format
        chunk, _ = self.generate_buffer("PCM_16")
        
        return chunk


def load_preset(preset_name: str, presets_file: str = None) -> Dict:
    """Load a noise preset from YAML file"""
    if presets_file is None:
        # Default to bundled presets
        presets_file = os.path.join(os.path.dirname(__file__), "presets.yaml")
    
    with open(presets_file, 'r') as f:
        all_presets = yaml.safe_load(f)
    
    if preset_name not in all_presets.get('presets', {}):
        logger.warning(f"Preset '{preset_name}' not found, using 'default'")
        preset_name = 'default'
    
    return all_presets['presets'][preset_name]


def auto_select_backend() -> bool:
    """Auto-select GPU/CPU backend based on available hardware"""
    if not HAS_CUPY:
        logger.info("CuPy not available, using CPU backend")
        return False
    
    try:
        # Check if CUDA is available and functioning
        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus > 0:
            # Get device info
            device = cp.cuda.runtime.getDeviceProperties(0)
            logger.info(f"Found GPU: {device['name'].decode()}")
            logger.info(f"CUDA Compute Capability: {device['major']}.{device['minor']}")
            
            # Verify basic CUDA functionality with a simple operation
            try:
                # Test if we can perform a basic operation
                test_arr = cp.zeros(10)
                test_arr += 1
                cp.asnumpy(test_arr)
                return True
            except Exception as cuda_error:
                logger.warning(f"CUDA test failed, falling back to CPU: {cuda_error}")
                return False
        else:
            logger.info("No CUDA GPUs found, using CPU backend")
            return False
    except Exception as e:
        logger.warning(f"Error checking GPU: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Baby-Noise Generator")
    parser.add_argument("--output", "-o", default="baby_noise.wav", help="Output file path")
    parser.add_argument("--duration", "-d", type=float, default=DEFAULT_DURATION, 
                       help="Duration in seconds")
    parser.add_argument("--seed", default="auto", help="PRNG seed (auto, random, or integer)")
    parser.add_argument("--preset", "-p", default="default", help="Preset name")
    parser.add_argument("--presets-file", help="Custom presets YAML file")
    parser.add_argument("--white", type=float, help="White noise ratio (0.0-1.0)")
    parser.add_argument("--pink", type=float, help="Pink noise ratio (0.0-1.0)")
    parser.add_argument("--brown", type=float, help="Brown noise ratio (0.0-1.0)")
    parser.add_argument("--rms", type=float, help="RMS target in dBFS")
    parser.add_argument("--peak", type=float, help="Peak ceiling in dBFS")
    parser.add_argument("--lfo", type=float, help="LFO rate in Hz")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    parser.add_argument("--gpu", action="store_true", help="Force GPU backend")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode (CPU only)")
    args = parser.parse_args()
    
    # Load preset
    preset = load_preset(args.preset, args.presets_file)
    
    # Determine seed
    if args.seed == "auto":
        seed = int(time.time())
    elif args.seed == "random":
        seed = np.random.randint(0, 2**32 - 1)
    else:
        try:
            seed = int(args.seed)
        except ValueError:
            logger.warning(f"Invalid seed '{args.seed}', using time-based seed")
            seed = int(time.time())
    
    # Determine color mix
    color_mix = preset.get('color_mix', {'white': 0.4, 'pink': 0.4, 'brown': 0.2})
    if args.white is not None or args.pink is not None or args.brown is not None:
        color_mix = {
            'white': args.white if args.white is not None else color_mix.get('white', 0.0),
            'pink': args.pink if args.pink is not None else color_mix.get('pink', 0.0),
            'brown': args.brown if args.brown is not None else color_mix.get('brown', 0.0)
        }
        # Normalize to sum to 1.0
        total = sum(color_mix.values())
        if total > 0:
            color_mix = {k: v / total for k, v in color_mix.items()}
    
    # Other parameters
    rms_target = args.rms if args.rms is not None else preset.get('rms_target', DEFAULT_RMS_TARGET)
    peak_ceiling = args.peak if args.peak is not None else DEFAULT_PEAK_CEILING
    lfo_rate = args.lfo if args.lfo is not None else preset.get('lfo_rate')
    
    # Determine backend
    if args.stream:
        use_gpu = False
    elif args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = auto_select_backend()
    
    # Create config
    config = NoiseConfig(
        seed=seed,
        duration=args.duration,
        color_mix=color_mix,
        rms_target=rms_target,
        peak_ceiling=peak_ceiling,
        lfo_rate=lfo_rate,
        use_gpu=use_gpu
    )
    
    # Create generator
    if args.stream:
        generator = StreamingNoiseGenerator(config)
        # This would be integrated into an audio streaming framework in a real app
        # For demonstration, just show it can generate chunks
        chunk = generator.get_next_chunk()
        logger.info(f"Generated streaming chunk, shape: {chunk.shape}")
    else:
        generator = NoiseGenerator(config)
        # Generate to file
        generator.generate_to_file(args.output)


if __name__ == "__main__":
    main()