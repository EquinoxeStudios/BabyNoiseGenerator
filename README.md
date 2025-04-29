# Baby-Noise Generator v2.0.5 (Simplified DSP Edition)

A CUDA-accelerated stereo noise generator with integrated high-quality DSP optimized for cloud GPU services, capable of ultra-fast rendering with exceptional sound quality. Designed specifically for creating high-quality YouTube sleep noise content.

## Features

- **Three noise colors** with intuitive "warmth" parameter:
  - White noise (flat spectrum with enhanced uniformity)
  - Pink noise (optimized -3 dB/octave with 8th-order precision)
  - Brown noise (enhanced -6 dB/octave with low-end body)
  
- **Integrated high-quality stereo processing**:
  - Frequency-dependent phase decorrelation for natural stereo field
  - Haas-effect enhancement with bass protection
  - Enhanced spatial imaging optimized for YouTube playback
  
- **Built-in natural sound modulation**:
  - Subtle multi-band modulation for more organic sound
  - Frequency-dependent processing with phase offsets between channels
  - Removes the "static" quality found in synthetic noise
  
- **YouTube-optimized output**:
  - 48 kHz sample rate (matches YouTube's processing pipeline)
  - Optimized for YouTube and other streaming platforms (-16 LUFS)
  - High-frequency pre-emphasis for codec resilience
  - True-peak ceiling: -2 dBTP
  
- **Optimized CUDA GPU acceleration** for ultra-fast rendering:
  - 10-hour files render in under 5 minutes on high-end GPUs
  - Dynamic memory management with adaptive precision
  - Automatic buffer optimization based on GPU capabilities
  - Smart skipping of unused noise types based on color mix
  
- **Enhanced dynamics processing**:
  - Psychoacoustically-optimized soft-knee compression
  - True-peak limiting with 4x oversampling
  - Logarithmic-domain processing for better numerical stability
  - BS.1770-4 compliant LUFS loudness metering
  
- **Presets for different sound characteristics**:
  - Enhanced presets with organic, spatial, and warm variations
  - Pure color presets (white only, pink only, brown only)
  - YouTube-optimized stereo enhancement presets
  
- **Advanced features**:
  - Deterministic seeds for reproducible renders using Philox PRNG
  - High-quality 24-bit WAV/FLAC output with proper dithering
  - High-frequency pre-emphasis for YouTube codec resilience
  - Comprehensive error handling and failsafe mechanisms

## What's New in v2.0.5

- **Simplified architecture**: Enhanced stereo, Haas effect, and natural modulation are now always enabled for best quality
- **Reduced config options**: Removed unnecessary toggle flags for core DSP features
- **Streamlined presets**: Simplified preset structure with focused sound characteristics 
- **Enhanced reliability**: Fixed-coefficient filters ensure consistent performance across all GPU types
- **Intelligent processing**: Only generates noise types that contribute to the final output
- **Better error handling**: Added detailed logging for effective troubleshooting
- **Pure noise optimization**: Much faster when using single noise types