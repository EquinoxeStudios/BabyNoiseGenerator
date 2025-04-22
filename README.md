# Baby-Noise Generator v1.1

A GPU-accelerated white/pink/brown noise generator for infant sleep, capable of both real-time streaming and high-quality rendering for YouTube videos.

![Baby-Noise Generator Screenshot](https://via.placeholder.com/800x600)

## Features

- **Three noise colors** with intuitive "warmth" slider:
  - White noise (flat spectrum)
  - Pink noise (-3 dB/octave)
  - Brown noise (-6 dB/octave)
  
- **GPU acceleration** for rendering long files:
  - 10-hour files render in under 15 minutes on modern GPUs (up to 50% faster than v1.0)
  - Optimized memory usage with vectorized algorithms
  - Automatic fallback to CPU when GPU unavailable
  
- **Medical-safe output levels**:
  - Default RMS level ~47 dB SPL (AAP guideline compliant)
  - Brick-wall limiter at -1 dBFS
  - LUFS-style loudness monitoring with visual alerts for unsafe levels
  - Automatic gain reduction for levels exceeding safety thresholds
  
- **Presets for different ages and sleep stages**:
  - Newborn (deep & light sleep)
  - 3-month infant (deep & light sleep)
  - 6-month infant (deep & light sleep)
  - Toddler (deep & light sleep)
  
- **Advanced features**:
  - Deterministic seeds for reproducible renders using Philox PRNG
  - Optional gentle gain modulation to reduce habituation
  - Real-time spectral visualization
  - High-quality 16-bit WAV/FLAC output with TPDF dither
  - Progress bar for long renders

## System Requirements

- Python 3.8 or newer
- NVIDIA GPU with CUDA support (optional, for accelerated rendering)
- 4GB+ GPU memory recommended for long renders
- Any modern CPU for real-time streaming

## Installation

### Quick Installation

```bash
pip install -r requirements.txt
pip install cupy-cuda12x  # Optional: for GPU acceleration
```

See the [Installation Guide](INSTALL.md) for detailed instructions.

## Usage

### GUI Application

Start the graphical interface:

```bash
python baby_noise_app.py
```

The GUI allows you to:
- Adjust noise color mix with the warmth slider
- Select from age-appropriate presets
- Play noise in real-time
- Render long files for YouTube or mobile devices
- Visualize the noise spectrum and level
- Monitor output levels with AAP safety indicators

### Command Line Interface

For headless or batch rendering:

```bash
python noise_generator.py --output baby_sleep.wav --duration 36000 --preset infant_3m_deep
```

For help with command line options:

```bash
python noise_generator.py --help
```

## Technical Details

- **White Noise**: Generated using Philox counter-based PRNG (2²⁵⁶ period)
- **Pink Noise**: FFT-based convolution with cached 4097-tap FIR filter on GPU
- **Brown Noise**: Optimized sequential implementation with high-pass filter
- **GPU Pipeline**: CuPy implementation with optimized device memory usage
- **CPU Fallback**: Paul Kellett algorithm for efficient pink noise on CPU
- **LUFS Monitoring**: Sliding-window loudness measurement with AAP guideline alerts

## Medical Safety

This application follows American Academy of Pediatrics guidelines for infant noise exposure:
- Default levels are set to ~47 dB SPL (well below the 50 dB SPL recommendation)
- LUFS monitoring ensures consistent loudness across devices
- Visual alerts when settings exceed recommended levels
- Automatic safety gain reduction when threshold is exceeded
- Use in conjunction with proper sleep practices and monitoring

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

- Based on Audiolab research on infant sleep noise characteristics
- GPU acceleration inspired by work from the CuPy community
- Color mixing algorithms adapted from established DSP research

## Future Development (Q3 2025)

- Heartbeat underlayment option
- Mobile app versions for iOS and Android
- Cloud-based rendering API
- Enhanced presets based on sleep research

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## What's New in v1.1

- **Performance optimizations**: Up to 50% faster GPU rendering
- **Memory efficiency**: Improved memory usage for longer renders
- **Enhanced safety**: LUFS monitoring with AAP compliance indicators and auto-gain reduction
- **UI improvements**: Added progress bar and visual safety indicators
- **Format support**: Added FLAC output option with optimized dithering
- **Algorithm improvements**: 
  - Used modern Philox PRNG via CuPy's Generator API
  - Implemented cached FIR filters with improved FFT plan reuse
  - Enhanced brown noise generation for deterministic output
  - Optimized memory transfers for dithering