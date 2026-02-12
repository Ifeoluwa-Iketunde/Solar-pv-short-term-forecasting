#!/usr/bin/env python3
"""
Complete Pipeline Runner for Solar PV Forecasting Project
This script runs all data processing and modeling steps in sequence.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and display progress"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! ({execution_time:.2f}s)")
            # Print last few lines of output for quick review
            output_lines = result.stdout.strip().split('\n')[-10:]  # Last 10 lines
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"❌ Error!")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
        
    return True

def main():
    print(" Solar PV Short-Term Forecasting - Complete Pipeline ")
    print("=" * 60)
    
    # Define pipeline steps (skip data processing if already done)
    steps = [
        # Check if data processing is needed
        ("python scripts/create_final_full_dataset.py", "Data Processing and Dataset Creation"),
        ("python scripts/temporal_split_full.py", "Temporal Train/Test Split"),
        ("python scripts/persistence_model.py", "Persistence Baseline Model"),
        ("python scripts/random_forest_model.py", "Basic Random Forest Model"),
        ("python scripts/enhanced_random_forest.py", "Enhanced Random Forest Model"),
        ("python scripts/lstm_model.py", "LSTM Neural Network Model")
    ]
    
    # Execute pipeline
    start_time = time.time()
    success_count = 0
    
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] ", end="")
        if run_command(command, description):
            success_count += 1
        else:
            print(f"\n⚠️  Pipeline stopped due to error in step {i}")
            break
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Steps completed: {success_count}/{len(steps)}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per step: {total_time/len(steps):.2f} seconds")
    
    if success_count == len(steps):
        print("\n🎉 All steps completed successfully!")
        print("\n📊 Next steps:")
        print("1. Review the results in each model's output")
        print("2. Check the README.md for detailed explanations")
        print("3. Compare model performances")
        print("4. Analyze which approach works best for your use case")
    else:
        print(f"\n❌ {len(steps) - success_count} steps failed")
        print("Please check the error messages above and fix issues before re-running")

if __name__ == "__main__":
    main()