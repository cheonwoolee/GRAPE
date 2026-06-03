# Streaming Virtual Metrology via Lightweight Adaptation under Temporal Distribution Shifts

This repository contains the official implementation of GRAPE (Global Regression with Adaptive Prediction Enhancement), an online learning framework for VM in evolving wafer-processing environments.

## Installation

Requires **Python 3.10+**. You can set up the environment with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
python run.py --dataset data-1
python run.py --dataset data-2
```

---

## Configuration

The grid-search configuration is defined as follows:

```yaml
active_keys: [alpha, nf, adapter, na, ema]

pools:
  alpha: [1, 10, 100, 1000, 10000]
  nf: [50, 100, 500]
  adapter: [nlms, rls]
  na: [5, 10, 50]
  ema: [0.01, 0.05, 0.1, 0.5]
```

---

## License

This project is licensed under the **Apache License 2.0**.
See the `LICENSE` file for details.
