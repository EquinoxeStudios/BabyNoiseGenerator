# Changelog for Baby-Noise Generator

## v1.2.0 (April 2025)

### New Features
- **Stereo Support**: Added decorrelated stereo/multi-channel output for richer sound
- **Output Profiles**: Added two profiles with different settings:
  - "baby-safe": AAP-compliant levels for infant sleep (-63 RMS / -27 LUFS / -3 dBTP)
  - "youtube-pub": Optimized for YouTube publishing (-20 RMS / -16 LUFS / -2 dBTP)
- **Warmth Control**: Replaced individual color sliders with a single 0-100 warmth parameter
- **True-Peak Safety**: Implemented 4x oversampling to catch and prevent intersample peaks
- **Format Upgrades**: Changed default output to 24-bit WAV and added FLAC support
- **YouTube Optimization**: Added high-frequency pre-emphasis for better codec resilience
- **Batch Processing**: Added batch generation capabilities for creating multiple files

### Performance Improvements
- **Improved CPU Generation**: Replaced slow per-sample algorithm with efficient block-based method
- **Faster Pink Noise**: Implemented FFT-based filtering for CPU pink noise on long renders
- **Memory Optimization**: Better buffer handling for large files

### Technical Enhancements
- **Enhanced LUFS Metering**: Fully implemented ITU-R BS.1770-4 compliant loudness measurement
- **Improved Testing**: Added comprehensive test suite with level verification
- **Decorrelation Logic**: Sophisticated stereo decorrelation while maintaining timbral balance
- **Enhanced File Processing**: New file handling with chunked processing and appropriate dithering

### Bug Fixes
- Fixed peak ceiling handling in batch processing
- Fixed progress reporting for long renders
- Fixed cache issues with filter coefficients
- Addressed memory leaks in long renders

### Documentation
- Updated README with stereo and profile information
- Added detailed INSTALL.md with platform-specific instructions
- Better comments and docstrings throughout the codebase

## v1.1.0 (February 2025)

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

## v1.0.0 (December 2024)

- Initial release
- White, pink, and brown noise generation
- GPU acceleration via CuPy
- Basic presets for different ages and sleep stages
- Level control with safety limits
- Simple GUI and command-line interface