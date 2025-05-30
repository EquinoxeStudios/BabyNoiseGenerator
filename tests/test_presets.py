#!/usr/bin/env python3
"""
Tests for Baby-Noise Generator presets
"""

import os
import pytest
import yaml
from pathlib import Path

# More robust path finding using pathlib
PRESETS_FILE = Path(__file__).resolve().parent.parent / "presets.yaml"


def test_presets_exist():
    """Test that presets file exists"""
    assert PRESETS_FILE.exists(), f"Presets file not found: {PRESETS_FILE}"


def test_presets_yaml_valid():
    """Test that presets file is valid YAML"""
    with open(PRESETS_FILE, 'r') as f:
        presets_data = yaml.safe_load(f)
    
    assert presets_data is not None, "Failed to parse presets YAML"
    assert "presets" in presets_data, "No 'presets' section found in YAML"
    assert isinstance(presets_data["presets"], dict), "'presets' should be a dictionary"


def test_preset_schema():
    """Test that all presets follow the expected schema"""
    with open(PRESETS_FILE, 'r') as f:
        presets_data = yaml.safe_load(f)
    
    presets = presets_data["presets"]
    
    # Check each preset
    for name, preset in presets.items():
        # Required fields
        assert "color_mix" in preset, f"Preset '{name}' missing 'color_mix'"
        assert "rms_target" in preset, f"Preset '{name}' missing 'rms_target'"
        assert "lfo_rate" in preset or preset.get("lfo_rate") is None, f"Preset '{name}' has invalid 'lfo_rate'"
        
        # Stereo-specific settings
        assert "haas_effect" in preset, f"Preset '{name}' missing 'haas_effect'"
        assert "enhanced_stereo" in preset, f"Preset '{name}' missing 'enhanced_stereo'"
        assert "natural_modulation" in preset, f"Preset '{name}' missing 'natural_modulation'"
        
        # Check color_mix
        color_mix = preset["color_mix"]
        assert isinstance(color_mix, dict), f"Preset '{name}' 'color_mix' should be a dictionary"
        assert "white" in color_mix, f"Preset '{name}' 'color_mix' missing 'white'"
        assert "pink" in color_mix, f"Preset '{name}' 'color_mix' missing 'pink'"
        assert "brown" in color_mix, f"Preset '{name}' 'color_mix' missing 'brown'"
        
        # Check types
        assert isinstance(preset["rms_target"], (int, float)), f"Preset '{name}' 'rms_target' should be a number"
        if "lfo_rate" in preset and preset["lfo_rate"] is not None:
            assert isinstance(preset["lfo_rate"], (int, float)), f"Preset '{name}' 'lfo_rate' should be a number"


def test_color_mix_sums_to_one():
    """Test that color_mix values sum approximately to 1.0"""
    with open(PRESETS_FILE, 'r') as f:
        presets_data = yaml.safe_load(f)
    
    presets = presets_data["presets"]
    
    for name, preset in presets.items():
        color_mix = preset["color_mix"]
        total = sum(color_mix.values())
        # Allow small floating point errors
        assert 0.99 <= total <= 1.01, f"Preset '{name}' color_mix values sum to {total}, should be ~1.0"


def test_safety_limits():
    """Test that presets adhere to safety limits"""
    with open(PRESETS_FILE, 'r') as f:
        presets_data = yaml.safe_load(f)
    
    presets = presets_data["presets"]
    
    # AAP safety threshold is approximately -60 dBFS
    safety_threshold = -60.0
    
    for name, preset in presets.items():
        rms_target = preset["rms_target"]
        # Special case for YouTube presets which can be louder
        if not name.startswith("youtube"):
            assert rms_target <= safety_threshold, (
                f"Preset '{name}' rms_target ({rms_target} dBFS) exceeds safety threshold "
                f"({safety_threshold} dBFS / 50 dB SPL)"
            )


def test_stereo_settings():
    """Test that all presets have stereo settings"""
    with open(PRESETS_FILE, 'r') as f:
        presets_data = yaml.safe_load(f)
    
    presets = presets_data["presets"]
    
    for name, preset in presets.items():
        # Ensure stereo settings are present
        assert "haas_effect" in preset, f"Preset '{name}' missing 'haas_effect'"
        assert "enhanced_stereo" in preset, f"Preset '{name}' missing 'enhanced_stereo'"
        
        # Check types of stereo settings
        assert isinstance(preset["haas_effect"], bool), f"Preset '{name}' 'haas_effect' should be a boolean"
        assert isinstance(preset["enhanced_stereo"], bool), f"Preset '{name}' 'enhanced_stereo' should be a boolean"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main(["-xvs", __file__])