# Security Policy

## Supported Versions

- Only the latest commit on `master` is supported. Run `bash run_unittests.sh` with current dependencies before opening a report.
- Older notebooks or volumes (`vol1/`, `vol2/`, `vol3/`) are not patched once superseded.

## Ecosystem & Compatibility

| Component              | Version(s) / Tooling             | Notes |
| ---------------------- | -------------------------------- | ----- |
| OS baseline            | WSL (Ubuntu 24.4.3 LTS)          | Matches the environment documented in the README. |
| Python runtime         | CPython 3.14.2 (`.python-version`) | Managed via pyenv. |
| Core libraries         | `numpy`, `matplotlib` (see `requirements.txt`) | Install with `pip install -r requirements.txt`; additional lab-specific deps live in each volume. |
| Datasets & tooling     | MNIST loaders and local datasets | Ensure datasets are available locally before running proofs-of-concept. |

## Backward Compatibility

- Model notebooks are versioned with the repository; backward compatibility is only guaranteed within the same Python 3.14.x runtime and dependency lock.
- Historical experiment folders are preserved for reference but are not updated once superseded. Re-run them on the latest stack if you need a security fix.

## Reporting a Vulnerability

Please report issues privately via:

1. **GitHub Security Advisory** (preferred) — open through the repository’s  **Security → Report a vulnerability** workflow.
2. **Email** — contact `security@project.org` with reproduction steps, sample  notebooks/scripts, and environment details.

Acknowledgement occurs within **3 business days**; status updates follow at least every **7 business days** until resolution.  
After remediation we publish guidance alongside required dependency updates.
