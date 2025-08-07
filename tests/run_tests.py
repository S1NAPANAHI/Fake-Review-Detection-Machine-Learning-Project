#!/usr/bin/env python3
"""
Test runner script for the fake review detection system.

This script provides an easy way to run tests with various configurations
and generate reports for CI/CD pipelines.

Usage:
    python tests/run_tests.py --help
    python tests/run_tests.py --unit
    python tests/run_tests.py --integration
    python tests/run_tests.py --all --coverage
    python tests/run_tests.py --api --verbose
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(command, description=None):
    """Run a shell command and handle errors."""
    if description:
        print(f"\n=== {description} ===")
    
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for fake review detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_tests.py --unit
    python tests/run_tests.py --integration --verbose
    python tests/run_tests.py --all --coverage --html-report
    python tests/run_tests.py --api --parallel
    python tests/run_tests.py --smoke --fast
        """
    )
    
    # Test selection arguments
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--unit', action='store_true',
                          help='Run only unit tests')
    test_group.add_argument('--integration', action='store_true',
                          help='Run only integration tests')
    test_group.add_argument('--api', action='store_true',
                          help='Run only API tests')
    test_group.add_argument('--all', action='store_true', default=True,
                          help='Run all tests (default)')
    test_group.add_argument('--smoke', action='store_true',
                          help='Run smoke tests only')
    
    # Test configuration arguments
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML test report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    
    # Environment arguments
    parser.add_argument('--install-deps', action='store_true',
                       help='Install test dependencies first')
    parser.add_argument('--python', default='python',
                       help='Python executable to use (default: python)')
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(PROJECT_ROOT)
    
    # Install dependencies if requested
    if args.install_deps:
        if not run_command([
            args.python, '-m', 'pip', 'install', '-r', 'tests/requirements-test.txt'
        ], "Installing test dependencies"):
            return 1
    
    # Build pytest command
    cmd = [args.python, '-m', 'pytest']
    
    # Add test selection
    if args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    elif args.api:
        cmd.extend(['-m', 'api'])
    elif args.smoke:
        cmd.extend(['-m', 'not slow'])
    # --all is default, no extra args needed
    
    # Add configuration options
    if args.verbose:
        cmd.append('-v')
    
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    
    if args.coverage:
        cmd.extend([
            '--cov=src',
            '--cov-report=term-missing',
            '--cov-report=xml'
        ])
    
    if args.html_report:
        cmd.extend(['--html=test-report.html', '--self-contained-html'])
    
    if args.benchmark:
        cmd.append('--benchmark-only')
    
    # Add test directory
    cmd.append('tests/')
    
    # Run tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print("\n=== Test run completed successfully! ===")
        
        # Show coverage report location if generated
        if args.coverage:
            print("Coverage report generated:")
            print("  - Terminal: (shown above)")
            print("  - XML: coverage.xml")
            if args.html_report:
                print("  - HTML: htmlcov/index.html")
        
        # Show HTML report location if generated
        if args.html_report:
            print("HTML test report: test-report.html")
        
        return 0
    else:
        print("\n=== Test run failed! ===")
        return 1


if __name__ == '__main__':
    sys.exit(main())
