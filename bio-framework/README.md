# BIOpt — Bilevel Isoperformance Optimization

**A field-agnostic Python framework for discovering geometrically distinct designs that achieve equivalent performance across parameterization families.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Installation

```bash
# From source (editable / development mode)
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# With CMA-ES optimizer support
pip install -e ".[cmaes]"
```

## Project Structure

```
bio-framework/
├── src/
│   └── biopt/                     # Installable core package
│       ├── __init__.py            # Public API, version
│       ├── families.py            # ParameterizationFamily ABC
│       ├── residual.py            # IsoperformanceResidual ℛ(x₁, x₂)
│       ├── distance.py            # Shape-space distance dS and exclusion zones
│       ├── optimizers.py          # Pluggable optimizers (DE, CMA-ES)
│       ├── pair.py                # Level A: pair problem (fixed reference)
│       ├── bilevel.py             # Level C: bilevel optimization
│       └── sampling.py            # Sequential manifold sampling
├── applications/                  # Domain-specific families
│   └── structural/
│       └── families.py            # I-beam, RHS
├── analysis/                      # Post-processing tools
│   └── sweeps.py                  # Stringency and δ-sensitivity analyses
├── examples/                      # Usage examples
└── tests/                         # Test suite
```

## Quick Start

```python
from biopt.residual import IsoperformanceResidual
from biopt.pair import PairProblem
from biopt.optimizers import DEOptimizer
from applications.structural.families import IBeamFamily, RHSFamily
import numpy as np

# Define families
ibeam = IBeamFamily()
rhs   = RHSFamily()

# Reference I-beam
x1 = np.array([0.10, 0.008, 0.184, 0.006])

# Configure residual (condition-independent → single dummy eval param)
residual = IsoperformanceResidual(ibeam, rhs, eval_params=[np.array([0.0])])

# Solve pair problem: find an RHS equivalent to the I-beam
pair = PairProblem(ibeam, rhs, residual, delta=0.01, epsilon_tol=1e-4)
result = pair.solve(x1)

print(f"Equivalent RHS: {result.x2}")
print(f"Residual ε* = {result.epsilon_star:.2e}")
print(f"Feasible: {result.feasible}")
```

## Adding a New Application Case

Subclass `ParameterizationFamily` and implement three methods:

```python
from biopt.families import ParameterizationFamily

class MyFamily(ParameterizationFamily):
    @property
    def n_params(self) -> int:
        return 4

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[0.1, 1.0], ...])  # (n_params, 2)

    def realize(self, x):
        """φ(x) → shape-space vector."""
        ...

    def performance(self, x, alpha):
        """F(x, α) → performance vector ∈ ℝᵐ."""
        ...
```

## Development

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/biopt/ applications/ analysis/ tests/
isort src/biopt/ applications/ analysis/ tests/
```

## License

MIT — see [LICENSE](LICENSE) for details.
