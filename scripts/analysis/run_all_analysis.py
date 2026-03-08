#!/usr/bin/env python3
"""
Master script to run all analysis scripts for DJC Jewellers project.

Usage:
    python scripts/analysis/run_all_analysis.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

SCRIPTS = [
    ("07_inventory_analysis.py", "Inventory Management Analysis"),
    ("08_demand_forecasting.py", "Seasonal Demand Forecasting"),
    ("09_pricing_analysis.py", "Pricing & Discount Analysis"),
]


def run_script(script_name: str, description: str) -> bool:
    """Run a single script and return success status."""
    script_path = Path(__file__).parent / script_name

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent.parent.parent),
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {script_name} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {script_name} failed with exit code {e.returncode} after {elapsed:.1f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {script_name} failed with error: {e} after {elapsed:.1f}s")
        return False


def main():
    print("\n" + "="*60)
    print("DJC JEWELLERS - ANALYSIS SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()
    results = {}

    for script_name, description in SCRIPTS:
        success = run_script(script_name, description)
        results[script_name] = success

    # Summary
    total_elapsed = time.time() - total_start
    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Scripts run: {successful}/{total} successful")

    print("\nResults:")
    for script_name, description in SCRIPTS:
        status = "✓" if results.get(script_name, False) else "✗"
        print(f"  {status} {script_name} - {description}")

    # List outputs
    viz_dir = Path(__file__).parent.parent.parent / "outputs" / "visualizations"
    reports_dir = Path(__file__).parent.parent.parent / "outputs" / "reports"

    if viz_dir.exists():
        print(f"\nVisualizations in {viz_dir}:")
        for f in sorted(viz_dir.glob("*.png")):
            print(f"  - {f.name}")

    if reports_dir.exists():
        print(f"\nReports in {reports_dir}:")
        for f in sorted(reports_dir.glob("*.*")):
            print(f"  - {f.name}")

    print("\n" + "="*60)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
