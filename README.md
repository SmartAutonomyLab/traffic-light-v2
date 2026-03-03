Here is a clean, professional `README.md` suitable for your project.

You can copy this directly into `README.md` at the project root.

---

# Traffic Light V2 – Supervisory Control Under Strategic Attacks

## Overview

This project implements data-driven supervisory control methods for signalized traffic networks subject to strategic cyber-attacks.

The framework integrates:

* Finite-State Transducer (FST) modeling using **Pynini / OpenFst**
* Spectral learning for attacker model inference
* Supervisory control logic
* Attack sequence data collection and persistence

The goal is to learn attacker behavior from observed sequences and synthesize resilient supervisors that enforce temporal specifications.

---

## Repository Structure

```
traffic_light_v2/
│
├── src/                        # Core project source code
│   └── TL_pckgs/               # Main project package
│
├── external/                   # Modified third-party dependencies
│   └── scikit-splearn-1.2.1/   # Edited spectral learning library
│
├── test scripts/               # Experimental and validation scripts
│
├── pyproject.toml              # Project packaging configuration
├── pyrightconfig.json          # Static type analysis configuration
└── README.md
```

---

## Requirements

* Python 3.11
* Conda (recommended)
* Pynini / OpenFst
* NumPy
* SciPy
* scikit-learn

---

## Setup

Create and activate a dedicated conda environment:

```bash
conda create -n traffic_control_fst python=3.11
conda activate traffic_control_fst
```

Install the main project in editable mode:

```bash
pip install -e .
```

Install the modified spectral learning library:

```bash
pip install -e external/scikit-splearn-1.2.1
```

Verify installation:

```bash
python -c "import TL_pckgs; print('Main package OK')"
python -c "import splearn; print('splearn OK')"
```

---

## Running Experiments

Example:

```bash
python test scripts/fst_tests.py
```

Ensure the correct conda environment is active before execution.

---

## Saving Learned Models

Learned attacker models are saved as timestamped pickle files:

```
learned_model_info_YYYYMMDD_HHMMSS.pkl
```

These files store:

* Spectral learning runtime
* Number of attack sequences
* Learned FST model
* Raw attack sequences

Generated artifacts are not intended to be version-controlled.

---

## Development Notes

* The project uses a `src/` layout.
* Editable installs (`pip install -e .`) are required for proper module resolution.
* Third-party modifications are isolated under `external/`.
* Static type checking is configured via `pyrightconfig.json`.

---

## Recommended Workflow

1. Activate conda environment.
2. Run experiments from project root.
3. Use VS Code with Pylance enabled.
4. Avoid committing generated model files or environment folders.


