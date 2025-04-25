# Changelog for Baby-Noise Generator

## v2.0.2 (April 2025) - Enhanced DSP Edition

### Sound Quality Enhancements
- **Enhanced pink noise algorithm**: Upgraded to 8th-order IIR coefficients for more accurate low-frequency response
- **Improved brown noise algorithm**: Added low-shelf filter (+3dB @ 75Hz) for enhanced low-end presence
- **Natural sound modulation**: Added subtle multi-band modulation for more organic sound
- **Enhanced stereo field**: Frequency-dependent phase decorrelation that preserves mono compatibility
- **Haas effect stereo enhancement**: Added delay-based enhancement with frequency protection
- **Psychoacoustic warmth curve**: More perceptually natural warmth parameter response

### Technical Improvements
- **Numerical optimization**: Corrected filter symmetry for proper phase response
- **Frequency calculation precision**: Enhanced bin index calculation for better accuracy
- **Logarithmic-domain compression**: Improved soft-knee compression algorithm
- **Polyphase true-peak detection**: Better intersample peak detection with 4x oversampling
- **Dynamic precision control**: Adaptive precision based on render duration
- **Enhanced filter initialization**: Proper filter state dimensions for better stability
- **Improved memory management**: Better handling of long renders and large buffer sizes

### New Features
- **Enhanced presets**: Added new presets focused on specific sound qualities
  - `enhanced_organic`: Optimized for natural sound with subtle modulation
  - `enhanced_spatial`: Optimized for immersive stereo field
  - `enhanced_warm`: Optimized for rich, warm sound character
- **Command-line options**: New options for enhanced DSP features
  - `--natural-mod / --no-natural-mod`: Enable/disable organic sound modulation
  - `--haas / --no-haas`: Enable/disable Haas effect for stereo width
  - `--enhanced-stereo / --no-enhanced-stereo`: Enable/disable advanced stereo decorrelation

### Documentation
- Updated README with enhanced feature descriptions
- Added detailed technical information about DSP algorithms
- Updated performance benchmarks with enhanced algorithm metrics

## v2.0.1 (April 2025)

### Bug Fixes
- **Device Name Handling**: Fixed CUDA device name decoding to properly handle both string and byte formats
- **Code Cleanup**: Removed unused parameter from render thread implementation

## v2.0.0 (April 2025)

### New Features
- **GPU-only implementation**: Fully optimized for CUDA GPU processing
- **Dynamic memory management**: Auto-optimizes for different GPU capabilities
- **Enhanced stereo decorrelation**: Frequency-dependent phase manipulation
- **Advanced true-peak limiting**: 4x oversampling for better peak detection
- **Improved buffer handling**: Larger, optimized blocks for faster processing
- **Real-time factor reporting**: Built-in benchmarking metrics

### Performance Improvements
- **Fully vectorized algorithms**: All DSP operations optimized for GPU execution
- **Performance boost**: 15-20% faster rendering than previous version
- **Second-order sections filtering**: Improved numerical stability for brown noise
- **Adaptive progress reporting**: Smart throttling based on render duration
- **Optimized memory layout**: Efficient GPU memory usage across all operations
- **CuPy scalar conversions**: Proper handling of GPU arrays for numeric stability
- **Optimal block sizes**: Dynamic adjustment based on available GPU memory

### Technical Enhancements
- **Thread safety enhancements**: Comprehensive exception handling in background threads
- **Fallback mechanisms**: Silent output generation when errors occur
- **Resource management**: Proper cleanup of threads and GPU resources
- **Error propagation**: Improved feedback for file measurements and rendering issues
- **Pink filter refinements**: Fixed off-by-one issues in filter tap generation
- **Standardized output formats**: 24-bit WAV and FLAC for all exports
- **Optimized filter caching**: Precomputed coefficients for improved performance

### Bug Fixes
- Fixed infinite recursion in stereo brown noise generation
- Fixed streaming generator thread resource leaks
- Fixed numerical errors in filter coefficient generation
- Fixed CuPy array scalar formatting in log messages
- Corrected inconsistent color mix normalization
- Fixed memory leaks in long renders
- Fixed potential crashes in stereo decorrelation

### Documentation
- Updated README with improved performance benchmarks
- Added detailed technical outline of the GPU acceleration strategy
- Improved inline code documentation for key algorithms
- Updated system requirements

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