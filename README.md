# Baby-Noise Generator v1.2

A GPU-accelerated white/pink/brown noise generator for infant sleep, capable of high-quality rendering for YouTube videos.

## Features

- **Three noise colors** with intuitive "warmth" parameter:
  - White noise (flat spectrum)
  - Pink noise (-3 dB/octave)
  - Brown noise (-6 dB/octave)
  
- **Stereo output support**:
  - Decorrelated stereo channels for rich spatial sound
  - Better experience with headphones or multi-speaker setups
  
- **Two output profiles**:
  - **Baby-safe**: AAP-compliant levels (~47 dB SPL)
  - **YouTube-pub**: Optimized for YouTube publishing (-16 LUFS)
  
- **GPU acceleration** for rendering long files:
  - 10-hour files render in under 15 minutes on modern GPUs
  - Optimized memory usage with vectorized algorithms
  - Automatic fallback to CPU when GPU unavailable
  
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
  - High-quality 24-bit WAV/FLAC output with TPDF dither
  - High-frequency pre-emphasis for YouTube codec resilience
  - Batch processing capabilities for multiple files

## System Requirements

- Python 3.8 or newer
- NVIDIA GPU with CUDA support (optional, for accelerated rendering)
- 4GB+ GPU memory recommended for long renders
- Any modern CPU for real-time streaming

## Installation

```bash
pip install -r requirements.txt
pip install cupy-cuda12x  # Optional: for GPU acceleration
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
    use_gpu=True,  # auto-selects based on availability
    channels=2,    # stereo output
    profile="baby-safe"
)

# Create generator and render file
generator = NoiseGenerator(config)
result = generator.generate_to_file("output_noise.wav")

# Check results
print(f"Generated file with {result['integrated_lufs']:.1f} LUFS, {result['peak_db']:.1f} dB peak")
print(f"Processing time: {result['processing_time']:.1f} seconds")
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

# Batch rendering with 24-bit WAV
python noise_generator.py --output noise_10h_24bit.wav --duration 36000 --bits 24 --seed 12345

# Batch processing with configuration
python batch_generate.py --config batch_config.yaml --output-dir ./outputs
```

For help with command line options:

```bash
python noise_generator.py --help
```

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
- **Pink Noise**: FFT-based convolution with cached 4097-tap FIR filter on GPU
- **Brown Noise**: Optimized sequential implementation with high-pass filter
- **GPU Pipeline**: CuPy implementation with optimized device memory usage
- **CPU Fallback**: Block-based algorithm for efficient pink noise on CPU
- **LUFS Monitoring**: ITU-R BS.1770-4 compliant loudness measurement
- **True-peak Detection**: 4x oversampling to catch intersample peaks
- **Stereo Generation**: Decorrelated channels with precise phase control

## Medical Safety

This application follows American Academy of Pediatrics guidelines for infant noise exposure:
- Default levels are set to ~47 dB SPL (well below the 50 dB SPL recommendation)
- LUFS monitoring ensures consistent loudness across devices
- Automatic safety gain reduction when threshold is exceeded
- Use in conjunction with proper sleep practices and monitoring

## License

MIT License - See [LICENSE](LICENSE) file for details

## What's New in v1.2

- **Stereo support**: Added decorrelated stereo output for richer sound
- **Output profiles**: Baby-safe (AAP-compliant) and YouTube publishing presets
- **Improved CPU generation**: Faster block-based algorithm for CPU pink noise
- **True-peak limiting**: 4x oversampling to catch intersample peaks
- **Format upgrades**: Default to 24-bit WAV and FLAC for higher quality
- **Warmth parameter**: Simplified noise color control from bright to warm
- **YouTube optimization**: High-frequency pre-emphasis for better codec resilience
- **Enhanced testing**: Automated level verification ensures consistent output
- **Code simplification**: Focused on core functionality and command-line usage