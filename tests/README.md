# Tests for Fake Review Detection System

This directory contains comprehensive unit and integration tests for the fake review detection system, validating functionality across all modules with small synthetic fixtures and metric threshold validation.

## Test Structure

### Test Files
- `test_utils.py` - Tests for utility functions (path resolution, config loading, data I/O, validation, metrics)
- `test_data.py` - Tests for data collection module (Yelp/Amazon datasets, API integration, validation)
- `test_preprocessing.py` - Tests for text preprocessing (cleaning, tokenization, feature engineering, SMOTE)
- `test_features.py` - Tests for feature engineering (TF-IDF, behavioral features, graph features, sentiment)
- `test_model.py` - Tests for model training and selection (hyperparameter optimization, cross-validation, persistence)
- `test_api.py` - Tests for FastAPI endpoints (prediction, health checks, batch processing, error handling)

### Configuration Files
- `pytest.ini` - Pytest configuration with coverage and reporting settings
- `requirements-test.txt` - Testing dependencies
- `run_tests.py` - Test runner script with various options
- `README.md` - This documentation

## Quick Start

### Install Test Dependencies
```bash
pip install -r tests/requirements-test.txt
```

### Run All Tests
```bash
# Simple run
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=src --cov-report=html

# Using test runner script
python tests/run_tests.py --all --coverage --html-report
```

## Test Categories

### Unit Tests
Test individual components in isolation:
```bash
python tests/run_tests.py --unit
python -m pytest tests/ -m unit
```

### Integration Tests
Test component interactions:
```bash
python tests/run_tests.py --integration
python -m pytest tests/ -m integration
```

### API Tests
Test FastAPI endpoints:
```bash
python tests/run_tests.py --api
python -m pytest tests/ -m api
```

## Test Runner Options

The `run_tests.py` script provides various testing options:

### Basic Usage
```bash
# Run all tests with coverage
python tests/run_tests.py --all --coverage

# Run unit tests with verbose output
python tests/run_tests.py --unit --verbose

# Run API tests in parallel
python tests/run_tests.py --api --parallel

# Quick smoke tests (skip slow tests)
python tests/run_tests.py --smoke --fast
```

### Advanced Options
```bash
# Generate HTML report
python tests/run_tests.py --all --html-report

# Run with benchmarking
python tests/run_tests.py --benchmark

# Install dependencies and run tests
python tests/run_tests.py --install-deps --all --coverage
```

## Test Data and Fixtures

All tests use small synthetic fixtures to ensure:
- Fast execution
- No external dependencies
- Reproducible results
- Clear test scenarios

### Example Synthetic Fixtures
- **Text data**: Small review samples with known characteristics
- **Numerical data**: Generated arrays with controlled distributions
- **API responses**: Mock responses with predictable structures
- **Model outputs**: Controlled predictions for validation

## Validation Approach

Tests validate:
- **Shapes**: Output dimensions match expected sizes
- **Types**: Correct data types returned
- **Ranges**: Values within expected bounds
- **Thresholds**: Performance metrics above minimum thresholds
- **API responses**: HTTP status codes and response schemas
- **Error handling**: Proper exception raising and handling

## Coverage Reports

Generate coverage reports to ensure comprehensive testing:

```bash
# Terminal report
python -m pytest tests/ --cov=src --cov-report=term-missing

# HTML report (view at htmlcov/index.html)
python -m pytest tests/ --cov=src --cov-report=html

# XML report (for CI/CD)
python -m pytest tests/ --cov=src --cov-report=xml
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Install dependencies
pip install -r tests/requirements-test.txt

# Run tests with reports
python -m pytest tests/ \
    --cov=src \
    --cov-report=xml \
    --cov-report=term \
    --junitxml=test-results.xml \
    --html=test-report.html \
    --self-contained-html
```

## Test Configuration

### Pytest Markers
Tests are marked for easy categorization:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.api` - API tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_data` - Tests needing external data
- `@pytest.mark.requires_model` - Tests needing trained models
- `@pytest.mark.requires_network` - Tests needing network access

### Environment Variables
Set these for specific test scenarios:
- `SKIP_SLOW_TESTS=1` - Skip slow tests
- `TEST_DATA_PATH=/path/to/test/data` - Custom test data location
- `MOCK_EXTERNAL_APIS=1` - Mock all external API calls

## Debugging Tests

### Run Specific Tests
```bash
# Single test file
python -m pytest tests/test_utils.py -v

# Specific test class
python -m pytest tests/test_model.py::TestModelTrainer -v

# Specific test method
python -m pytest tests/test_api.py::TestFastAPIEndpoints::test_health_endpoint -v
```

### Debug Mode
```bash
# Drop into debugger on failure
python -m pytest tests/ --pdb

# Stop on first failure
python -m pytest tests/ -x

# Show local variables in tracebacks
python -m pytest tests/ -l
```

## Performance Testing

Run benchmarks to validate performance:
```bash
# Run benchmark tests only
python tests/run_tests.py --benchmark

# Profile test execution
python -m pytest tests/ --profile
```

## Adding New Tests

### Guidelines
1. **Small fixtures**: Use minimal synthetic data
2. **Clear assertions**: Test one thing at a time
3. **Proper mocking**: Mock external dependencies
4. **Error cases**: Test both success and failure paths
5. **Documentation**: Add docstrings explaining test purpose

### Template
```python
def test_function_name(self):
    """Test specific functionality with clear description."""
    # Arrange: Set up test data
    test_input = create_synthetic_fixture()
    
    # Act: Execute the function
    result = function_under_test(test_input)
    
    # Assert: Validate results
    self.assertEqual(result.shape, expected_shape)
    self.assertGreater(result.metric, threshold)
    self.assertIn('expected_key', result)
```

## Common Issues

### Import Errors
Ensure the project root is in Python path:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Mock Failures
Use proper patching for external dependencies:
```python
@patch('module.external_function')
def test_with_mock(self, mock_function):
    mock_function.return_value = expected_value
```

### Async Tests
For FastAPI testing:
```python
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.get("/endpoint")
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
