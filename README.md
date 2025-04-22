# Baby-Noise Generator v1.0

A GPU-accelerated white/pink/brown noise generator for infant sleep, capable of both real-time streaming and high-quality rendering for YouTube videos.

![Baby-Noise Generator Screenshot](https://via.placeholder.com/800x600)

## Features

- **Three noise colors** with intuitive "warmth" slider:
  - White noise (flat spectrum)
  - Pink noise (-3 dB/octave)
  - Brown noise (-6 dB/octave)
  
- **GPU acceleration** for rendering long files:
  - 10-hour files render in under 30 minutes on modern GPUs
  - Automatic fallback to CPU when GPU unavailable
  
- **Medical-safe output levels**:
  - Default RMS level ~47 dB SPL (AAP guideline compliant)
  - Brick-wall limiter at -1 dBFS
  
- **Presets for different ages and sleep stages**:
  - Newborn (deep & light sleep)
  - 3-month infant (deep & light sleep)
  - 6-month infant (deep & light sleep)
  - Toddler (deep & light sleep)
  
- **Advanced features**:
  - Deterministic seeds for reproducible renders
  - Optional gentle gain modulation to reduce habituation
  - Real-time spectral visualization
  - High-quality 16-bit WAV/FLAC output with TPDF dither

## System Requirements

- Python 3.8 or newer
- NVIDIA GPU with CUDA support (optional, for accelerated rendering)
- 4GB+ GPU memory recommended for long renders
- Any modern CPU for real-time streaming

## Installation

### Quick Installation

```bash
pip install -r requirements.txt
pip install cupy-cuda11x  # Optional: for GPU acceleration
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
- **Pink Noise**: FFT-based convolution with 4097-tap FIR filter on GPU
- **Brown Noise**: Leaky integrator with high-pass filter, fully vectorized
- **GPU Pipeline**: CuPy implementation with minimal host-device transfers
- **CPU Fallback**: Paul Kellett algorithm for efficient pink noise on CPU

## Medical Safety

This application follows American Academy of Pediatrics guidelines for infant noise exposure:
- Default levels are set to ~47 dB SPL (well below the 50 dB SPL recommendation)
- LUFS monitoring ensures consistent loudness across devices
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