#!/usr/bin/env python3
"""
Cross-platform CLI runner for workflow evaluation pipeline
Works on both Windows and Linux/macOS
"""
import os
import sys
import argparse
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


class Colors:
    """ANSI color codes for cross-platform colored output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def is_supported(cls):
        """Check if colored output is supported"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Colorize text if colors are supported"""
        if cls.is_supported():
            return f"{color}{text}{cls.NC}"
        return text


def log_info(message: str):
    """Print info message"""
    print(Colors.colorize(f"[INFO] {message}", Colors.BLUE))


def log_success(message: str):
    """Print success message"""
    print(Colors.colorize(f"[SUCCESS] {message}", Colors.GREEN))


def log_warning(message: str):
    """Print warning message"""
    print(Colors.colorize(f"[WARNING] {message}", Colors.YELLOW))


def log_error(message: str):
    """Print error message"""
    print(Colors.colorize(f"[ERROR] {message}", Colors.RED))


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        log_error("Python 3.7 or higher is required")
        return False
    return True


def check_dependencies(data_dir: str) -> bool:
    """Check if all dependencies are satisfied"""
    log_info("Checking dependencies...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        log_error(f"Data directory not found: {data_dir}")
        return False
    
    # Check if there are any test data files
    json_files = list(Path(data_dir).glob("*.json"))
    if not json_files:
        log_error(f"No JSON test data files found in: {data_dir}")
        return False
    
    # Check if the main evaluation script exists
    eval_script = current_dir / "run_evaluation.py"
    if not eval_script.exists():
        log_error(f"Evaluation script not found: {eval_script}")
        return False
    
    log_success("All dependencies satisfied")
    return True


def clean_previous_results():
    """Clean up previous results and checkpoints"""
    log_info("Cleaning previous results and checkpoints...")
    
    # Remove results directory
    results_dir = current_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
        log_success("Removed results directory")
    
    # Remove checkpoints directory
    checkpoints_dir = current_dir / "checkpoints"
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        log_success("Removed checkpoints directory")
    
    log_success("Cleanup completed")


def build_command(args) -> List[str]:
    """Build the evaluation command"""
    cmd = [sys.executable, str(current_dir / "run_evaluation.py")]
    
    if args.layers:
        cmd.extend(["--layers", args.layers])
    
    if args.max_processes:
        cmd.extend(["--max-processes", str(args.max_processes)])
    
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    
    if args.config:
        cmd.extend(["--config", args.config])
    
    return cmd


def run_evaluation(args) -> bool:
    """Run the evaluation pipeline"""
    log_info("Starting workflow evaluation pipeline...")
    log_info(f"Layers: {args.layers}")
    log_info(f"Max processes: {args.max_processes}")
    log_info(f"Batch size: {args.batch_size}")
    log_info(f"Data directory: {args.data_dir}")
    
    # Build command
    cmd = build_command(args)
    
    if args.dry_run:
        log_info("Dry run mode - would execute:")
        print(" ".join(cmd))
        return True
    
    # Execute the evaluation
    log_info(f"Executing: {' '.join(cmd)}")
    print("")
    
    try:
        # Run the command and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Wait for process to complete
        return_code = process.poll()
        
        if return_code == 0:
            log_success("Evaluation pipeline completed successfully!")
            return True
        else:
            log_error("Evaluation pipeline failed!")
            return False
            
    except KeyboardInterrupt:
        log_warning("Evaluation interrupted by user")
        return False
    except Exception as e:
        log_error(f"Failed to run evaluation: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "max_processes": 4,
        "batch_size": 10,
        "max_retries": 3,
        "retry_delay": 2.0,
        "layers_to_run": [1, 2, 3],
        "evaluation_llm_model": "gpt-4",
        "evaluation_llm_temperature": 0.7,
        "workflow_execution_timeout": 300.0
    }
    
    config_file = current_dir / "sample_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2)
    
    log_success(f"Sample configuration created: {config_file}")
    print("You can modify this file and use it with --config option")


def show_status():
    """Show current status of evaluation results"""
    results_dir = current_dir / "results"
    checkpoints_dir = current_dir / "checkpoints"
    
    print("=" * 50)
    print("EVALUATION PIPELINE STATUS")
    print("=" * 50)
    
    # Check results
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        if result_files:
            print(f"Results directory: {results_dir}")
            print("Available result files:")
            for file in sorted(result_files):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
        else:
            print("Results directory exists but is empty")
    else:
        print("No results directory found")
    
    print("")
    
    # Check checkpoints
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.json"))
        if checkpoint_files:
            print(f"Checkpoints directory: {checkpoints_dir}")
            print("Available checkpoint files:")
            for file in sorted(checkpoint_files):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
        else:
            print("Checkpoints directory exists but is empty")
    else:
        print("No checkpoints directory found")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Cross-platform Workflow Evaluation Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all layers with default settings
  %(prog)s -l 1,2                           # Run only layers 1 and 2
  %(prog)s -p 8 -b 20                       # Use 8 processes with batch size 20
  %(prog)s --clean                          # Clean previous results
  %(prog)s -l 3 --config custom_config.json # Run layer 3 with custom config
  %(prog)s --status                         # Show current status
  %(prog)s --create-config                  # Create sample configuration file
        """
    )
    
    # Main options
    parser.add_argument(
        "-l", "--layers",
        default="1,2,3",
        help="Comma-separated layers to run (default: 1,2,3)"
    )
    parser.add_argument(
        "-p", "--max-processes",
        type=int,
        default=4,
        help="Maximum number of processes (default: 4)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)"
    )
    parser.add_argument(
        "-d", "--data-dir",
        default="evaluation_pipeline/workflow_generation_eval_data",
        help="Directory containing test data (default: evaluation_pipeline/workflow_generation_eval_data)"
    )
    parser.add_argument(
        "-c", "--config",
        help="Configuration file path"
    )
    
    # Action options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up previous results and checkpoints"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status of evaluation results"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a sample configuration file"
    )
    
    args = parser.parse_args()
    
    # Show header
    print("=" * 60)
    print("  Workflow Evaluation Pipeline Runner (Cross-platform)")
    print("=" * 60)
    print("")
    
    try:
        # Handle special actions first
        if args.create_config:
            create_sample_config()
            return 0
        
        if args.status:
            show_status()
            return 0
        
        if args.clean:
            clean_previous_results()
            print("")
        
        # Check dependencies
        if not check_dependencies(args.data_dir):
            return 1
        
        print("")
        
        # Run evaluation
        success = run_evaluation(args)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        log_warning("Operation interrupted by user")
        return 1
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
