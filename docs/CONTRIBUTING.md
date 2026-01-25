# Contributing to Text Segmentation & Reflow

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/segmentation.git
   cd segmentation
   ```

2. **Set up the development environment**
   ```bash
   pixi install
   pixi shell
   ```

3. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

This project uses **Black** for code formatting.

### Format your code before committing:
```bash
pixi run black src/
pixi run black tests/
```

### Recommended settings:
- Line length: 100 characters (Black's default: 88)
- Use type hints where applicable
- Add docstrings for public functions

## Testing

### Running Tests

```bash
# Run all tests
python test_*.py

# Run specific test
python test_1950.py
```

### Adding Tests

Create new test files following the pattern:
```python
#!/usr/bin/env python3
"""
Description of what this test validates
"""
import sys
sys.path.insert(0, '/path/to/src')

from reflow import create_page_with_word_wrapping, Letter
import numpy as np
import cv2

def create_test_case():
    # Create test data
    pass

def main():
    print("Testing: <what you're testing>")
    # Run test
    # Save results
    pass

if __name__ == "__main__":
    main()
```

## Pull Request Process

1. **Update documentation** if you've added/changed functionality
2. **Add tests** for new features
3. **Run Black** to format code
4. **Test your changes** thoroughly
5. **Write a clear PR description** explaining:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?

### PR Checklist

- [ ] Code is formatted with Black
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No unnecessary files included (check .gitignore)

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: Add word split prevention for single-letter cases
fix: Correct line spacing calculation with outliers
docs: Update README with Pixi installation instructions
test: Add test case for number splitting
refactor: Extract baseline calculation into separate function
```

Prefixes:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Bug Reports

When reporting a bug, please include:

1. **Description**: What happened?
2. **Expected behavior**: What should happen?
3. **Steps to reproduce**:
   - Input image (if possible)
   - Commands run
   - Environment details
4. **Error messages**: Full traceback if applicable
5. **System info**:
   - OS version
   - Python version
   - CUDA version (if relevant)

## Feature Requests

When requesting a feature:

1. **Use case**: Why is this needed?
2. **Proposed solution**: How would it work?
3. **Alternatives**: What other approaches did you consider?
4. **Examples**: Provide examples if applicable

## Code Review

All submissions require review. We'll look at:

- **Correctness**: Does it work as intended?
- **Performance**: Is it efficient?
- **Readability**: Is the code clear?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it well-documented?

## Areas for Contribution

### Easy/Good First Issues

- Add more test cases
- Improve documentation
- Add type hints
- Fix typos or formatting

### Medium Difficulty

- Performance optimizations
- Better error handling
- Additional configuration options
- Improved paragraph detection

### Advanced

- Support for different languages/scripts
- Machine learning integration
- Multi-column layout support
- Table detection and preservation

## Questions?

- Open an issue with the `question` label
- Contact: Sergey Mikhno <sergey.mikhno@gmail.com>

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing! 🎉
