# Contributing to Fake Review Detection System

Thank you for your interest in contributing to the Fake Review Detection System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected to see**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see**
- **Explain why this enhancement would be useful**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install dependencies** in development mode
3. **Make your changes** following our coding standards
4. **Add tests** for your changes
5. **Run the test suite** to ensure everything passes
6. **Update documentation** if necessary
7. **Commit your changes** with clear commit messages
8. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Git
- Docker (optional, for containerized development)

### Local Development Setup

1. **Clone your fork:**
```bash
git clone https://github.com/your-username/fake-review-detection.git
cd fake-review-detection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
```

4. **Install development tools:**
```bash
pip install black isort flake8 mypy pre-commit
```

5. **Download required models:**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')"
python -m spacy download en_core_web_sm
```

6. **Set up pre-commit hooks:**
```bash
pre-commit install
```

### Development Workflow

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the coding standards below

3. **Run tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api --cov-report=html

# Run specific tests
pytest tests/test_specific_module.py
```

4. **Check code quality:**
```bash
# Format code
black src/ tests/ api/

# Sort imports
isort src/ tests/ api/

# Lint code
flake8 src/ tests/ api/

# Type check
mypy src/ --ignore-missing-imports
```

5. **Commit your changes:**
```bash
git add .
git commit -m "feat: add your feature description"
```

6. **Push and create PR:**
```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length:** Maximum 127 characters
- **Imports:** Use `isort` for import sorting
- **Formatting:** Use `black` for code formatting
- **Type hints:** Use type hints for all public functions
- **Docstrings:** Use Google-style docstrings

### Code Organization

```python
"""Module docstring describing the module's purpose.

This should include a brief description of what the module does,
any important classes or functions it contains, and usage examples.
"""

import standard_library_imports
import third_party_imports
import local_imports


class ExampleClass:
    """Brief class description.
    
    Longer class description with more details about the class purpose,
    its main responsibilities, and how to use it.
    
    Attributes:
        attribute_name: Description of the attribute.
        
    Example:
        Basic usage of the class:
        
        >>> example = ExampleClass("param")
        >>> example.method()
        "result"
    """
    
    def __init__(self, param: str) -> None:
        """Initialize the class.
        
        Args:
            param: Description of the parameter.
        """
        self.attribute = param
    
    def method(self) -> str:
        """Brief method description.
        
        Longer description of what the method does, including any
        side effects or important behavior.
        
        Returns:
            Description of the return value.
            
        Raises:
            ValueError: Description of when this exception is raised.
        """
        return self.attribute
```

### Testing Guidelines

- **Write tests for all new functionality**
- **Maintain test coverage above 80%**
- **Use descriptive test names**
- **Include both positive and negative test cases**
- **Mock external dependencies**

```python
def test_should_return_expected_result_when_given_valid_input():
    """Test that the function returns expected result for valid input."""
    # Arrange
    input_data = "test input"
    expected = "expected result"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected
```

### Documentation Guidelines

- **Update README.md** for user-facing changes
- **Update API documentation** for API changes
- **Add docstrings** to all public functions and classes
- **Include usage examples** in docstrings
- **Update changelog** for significant changes

## Project Structure

Understanding the project structure will help you contribute effectively:

```
fake-review-detection/
├── api/                    # FastAPI application
├── src/                    # Core source code
│   ├── data_collection.py  # Data collection utilities
│   ├── preprocessing.py    # Data preprocessing
│   ├── feature_engineering.py  # Feature extraction
│   ├── modeling.py         # ML model implementations
│   ├── evaluation.py       # Model evaluation
│   └── utils.py           # Utility functions
├── config/                 # Configuration files
├── tests/                  # Test suites
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
├── artifacts/              # Model artifacts
├── data/                   # Data storage
└── deployment/             # Deployment configurations
```

## Types of Contributions

### 🐛 Bug Fixes
- Fix existing bugs
- Improve error handling
- Add missing validation

### ✨ New Features
- Add new ML models
- Implement new feature extraction methods
- Enhance API functionality
- Add monitoring capabilities

### 📚 Documentation
- Improve README
- Add code examples
- Write tutorials
- Update API documentation

### 🧪 Testing
- Add missing tests
- Improve test coverage
- Add integration tests
- Performance testing

### 🎨 Code Quality
- Refactor existing code
- Improve performance
- Add type hints
- Enhance error messages

## Review Process

All pull requests go through a review process:

1. **Automated checks** (CI/CD pipeline)
   - Code quality checks
   - Test suite execution
   - Security scanning

2. **Code review** by maintainers
   - Code quality assessment
   - Architecture review
   - Documentation review

3. **Testing** in different environments
4. **Merge** after approval

## Getting Help

- **GitHub Discussions:** For questions and general discussion
- **GitHub Issues:** For bug reports and feature requests
- **Documentation:** Check existing docs first
- **Code Examples:** Look at existing implementations

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **GitHub contributors** section
- **Release notes** for significant contributions

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Fake Review Detection System! 🚀
