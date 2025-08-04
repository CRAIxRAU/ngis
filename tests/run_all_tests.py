#!/usr/bin/env python3
"""
Run all EEG loader and channel selection tests.

This script runs all validation tests for the EEG processing pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_test(script_name, description):
    """Run a test script and report results."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              cwd=Path(__file__).parent,
                              check=False)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running All EEG Processing Tests")
    print("="*60)
    
    tests = [
        ("test_eeg_loader.py", "EEG Loader and Channel Selection"),
        ("test_university_standard.py", "University Standard Channel Selection"),
        ("validate_channel_selection.py", "Channel Selection Validation"),
        ("analyze_electrode_layout.py", "Electrode Layout Analysis")
    ]
    
    results = []
    
    for script, description in tests:
        script_path = Path(__file__).parent / script
        if script_path.exists():
            success = run_test(script, description)
            results.append((description, success))
        else:
            print(f"⚠️  Test script not found: {script}")
            results.append((description, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {description:<40} {status}")
    
    print(f"\n🏁 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EEG processing pipeline is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())