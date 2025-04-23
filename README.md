# Baby-Noise Generator v2.0 (GPU Accelerated)

A CUDA-accelerated white/pink/brown noise generator for infant sleep, capable of ultra-fast rendering for YouTube videos.

## Features

- **Three noise colors** with intuitive "warmth" parameter:
  - White noise (flat spectrum)
  - Pink noise (-3 dB/octave)
  - Brown noise (-6 dB/octave)
  
- **Stereo output support**:
  - Advanced decorrelated stereo channels for rich spatial sound
  - Frequency-dependent phase decorrelation for natural sound field
  - Enhanced experience with headphones or multi-speaker setups
  
- **Two output profiles**:
  - **Baby-safe**: AAP-compliant levels (~47 dB SPL)
  - **YouTube-pub**: Optimized for YouTube publishing (-16 LUFS)
  
- **CUDA GPU acceleration** for ultra-fast rendering:
  - 10-hour files render in under 8 minutes on high-end GPUs
  - Dynamic memory management for optimal GPU utilization
  - Automatic buffer optimization based on GPU capabilities
  
- **Medical-safe output levels**:
  - Default RMS level ~47 dB SPL (AAP guideline compliant)
  - True-peak limiter with 4x oversampling
  - LUFS-style loudness monitoring
  - Automatic gain reduction for levels exceeding safety thresholds
  
- **Presets for different ages and sleep stages**:
  - Newborn (deep & light sleep)
  - 3-month infant (deep & light sleep)
  - 6-month infant (deep & light sleep)
  - Toddler (deep & light sleep)
  
- **Advanced features**:
  - Deterministic seeds for reproducible renders using Philox PRNG
  - Optional gentle gain modulation to reduce habituation
  - High-quality 24-bit WAV/FLAC output with proper dithering
  - High-frequency pre-emphasis for YouTube codec resilience
  - Batch processing capabilities for multiple files

## System Requirements

- Python 3.8 or newer
- **REQUIRED**: NVIDIA GPU with CUDA support
- 4GB+ GPU memory recommended (8GB+ for optimal performance)
- CUDA Toolkit 11.x or 12.x

## Installation

```bash
pip install -r requirements.txt
pip install cupy-cuda12x  # Required for GPU acceleration
```

See the [Installation Guide](INSTALL.md) for detailed instructions.

## Usage

### Python API

```python
from noise_generator import NoiseGenerator, NoiseConfig

# Create a configuration
config = NoiseConfig(
    seed=12345,
    duration=600,  # 10 minutes
    color_mix={'white': 0.3, 'pink': 0.4, 'brown': 0.3},
    rms_target=-63.0,
    peak_ceiling=-3.0,
    lfo_rate=0.1,  # gentle modulation
    sample_rate=44100,
    channels=2,    # stereo output
    profile="baby-safe"
)

# Create generator and render file
generator = NoiseGenerator(config)
result = generator.generate_to_file("output_noise.wav")

# Check results
print(f"Generated file with {result['integrated_lufs']:.1f} LUFS, {result['peak_db']:.1f} dB peak")
print(f"Processing time: {result['processing_time']:.1f} seconds")
print(f"Real-time factor: {result['real_time_factor']:.1f}x")
```

### Converting Warmth to Color Mix

```python
def warmth_to_color_mix(warmth):
    """Convert warmth (0-100) to color mix"""
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
    
    # Normalize to sum to 1.0
    total = white + pink + brown
    return {
        'white': white / total,
        'pink': pink / total,
        'brown': brown / total
    }

# Usage
color_mix = warmth_to_color_mix(75)  # warmer noise
```

### Command Line Interface

```bash
# Basic usage with preset
python noise_generator.py --output baby_sleep.wav --duration 3600 --preset infant_3m_deep

# Stereo output with warmth control
python noise_generator.py --output baby_sleep.flac --channels 2 --warmth 75 --profile youtube-pub

# High-quality 24-bit WAV with specific GPU device
python noise_generator.py --output noise_10h_24bit.wav --duration 36000 --output-format "WAV (24-bit)" --seed 12345

# Batch processing with configuration
python batch_generate.py --config batch_config.yaml --output-dir ./outputs
```

For help with command line options:

```bash
python noise_generator.py --help
```

## Performance Benchmarks

The GPU-accelerated algorithm provides extraordinary rendering speeds:

| GPU Model | Memory | 1-hour mono | 10-hour stereo | Real-time Factor |
|-----------|--------|-------------|----------------|------------------|
| RTX 4090  | 24GB   | 32 seconds  | 6 minutes      | ~100x            |
| RTX 3080  | 10GB   | 45 seconds  | 8 minutes      | ~75x             |
| RTX 4060  | 8GB    | 1.5 minutes | 15 minutes     | ~40x             |
| RTX 2060  | 6GB    | 2 minutes   | 22 minutes     | ~27x             |
| GTX 1660  | 6GB    | 3 minutes   | 32 minutes     | ~19x             |

*Note: Performance may vary based on system configuration and other factors*

## Output Profiles

The Baby-Noise Generator supports two main output profiles:

### Baby-safe Profile

- Follows American Academy of Pediatrics (AAP) guidelines for infant noise exposure
- Default RMS level: -63 dBFS (~47 dB SPL)
- LUFS threshold: -27 LUFS
- True-peak ceiling: -3 dBTP
- Includes automatic safety monitoring and gain reduction
- Recommended for all infant sleep applications

### YouTube-pub Profile

- Optimized for YouTube and other streaming platforms
- RMS level: -20 dBFS (~-16 LUFS)
- LUFS threshold: -16 LUFS
- True-peak ceiling: -2 dBTP
- Includes high-frequency pre-emphasis for codec resilience
- **NOT recommended for infant sleep** - use only for publishing

## Technical Details

- **White Noise**: Generated using Philox counter-based PRNG (2²⁵⁶ period)
- **Pink Noise**: FFT-based convolution with optimized filter implementation
- **Brown Noise**: Optimized leaky integration with high-pass filtering
- **GPU Pipeline**: CuPy implementation with optimized memory management
- **LUFS Monitoring**: ITU-R BS.1770-4 compliant loudness measurement
- **True-peak Detection**: 4x oversampling to catch intersample peaks
- **Stereo Generation**: Frequency-dependent phase decorrelation

## Medical Safety

This application follows American Academy of Pediatrics guidelines for infant noise exposure:
- Default levels are set to ~47 dB SPL (well below the 50 dB SPL recommendation)
- LUFS monitoring ensures consistent loudness across devices
- Automatic safety gain reduction when threshold is exceeded
- Use in conjunction with proper sleep practices and monitoring

## License

MIT License - See [LICENSE](LICENSE) file for details

## What's New in v2.0

- **GPU-only implementation**: Fully optimized for CUDA GPU processing
- **Performance boost**: 2-3x faster rendering than previous version
- **Dynamic memory management**: Auto-optimizes for different GPU capabilities
- **Enhanced stereo decorrelation**: Frequency-dependent phase manipulation
- **Advanced true-peak limiting**: 4x oversampling for better peak detection
- **Improved buffer handling**: Larger, optimized blocks for faster processing
- **Real-time factor reporting**: Built-in benchmarking metrics
- **Memory optimization**: Efficient GPU memory usage across all operations