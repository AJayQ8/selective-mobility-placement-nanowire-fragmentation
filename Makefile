SYSTEM_PYTHON ?= python3.12
VENV_PYTHON := .venv/bin/python
PYTHON ?= $(VENV_PYTHON)
export PYTHONDONTWRITEBYTECODE := 1
PAPER_MODULE := science_lab.papers.selective_mobility_scripta

.PHONY: setup check-python test test-all test-core figures provenance manifest clean

setup:
	$(SYSTEM_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --disable-pip-version-check -r requirements-lock.txt

check-python:
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { \
		echo "Python environment not found. Run 'make setup' first." >&2; \
		exit 2; \
	}

test: check-python
	$(PYTHON) scripts/audit_release.py
	$(PYTHON) -m unittest \
		$(PAPER_MODULE).figures.test_figures \
		$(PAPER_MODULE).precursor_synthesis.test_analyze \
		$(PAPER_MODULE).test_precursor_context_audit \
		$(PAPER_MODULE).source_data.test_cms_analysis \
		$(PAPER_MODULE).provenance.transport_confirmation_bc.test_analysis -v

test-all: check-python
	$(PYTHON) -m unittest discover \
		-s science_lab/papers -p 'test_*.py' -v

test-core: check-python
	$(PYTHON) -m unittest \
		science_lab.papers.nanowire_gb_junction.roy_2021_reproduction.test_model \
		science_lab.papers.nanowire_gb_junction.roy_fixed_gb_bridge.test_geometry_metrics \
		science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.test_diagnostics \
		science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.test_geometry -v

figures: check-python
	$(PYTHON) scripts/regenerate_figures.py

provenance: check-python
	$(PYTHON) scripts/build_public_provenance.py

manifest: check-python
	$(PYTHON) scripts/build_manifest.py

clean:
	rm -rf reproduced_artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d \( -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
