PYTHON ?= python3
VENV ?= ve_dooms
VENV_PYTHON := $(VENV)/bin/python
SMALL_INPUT ?= ../../2025-238_Proteome_Remeasure_report_short_n_2_dt_long.csv
CHECK_OUTPUT ?= /tmp/dooms_cli_check

.PHONY: help venv install check cli-check clean-venv

help:
	@echo "dooms: ANOVA and Tukey post-hoc analysis"
	@echo "Targets:"
	@echo "  make venv       Create local virtualenv at $(VENV)"
	@echo "  make install    Install package into $(VENV)"
	@echo "  make check      Syntax-check package modules"
	@echo "  make cli-check  Run CLI on SMALL_INPUT=$(SMALL_INPUT)"
	@echo "  make clean-venv Remove $(VENV)"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install: venv
	$(VENV_PYTHON) -m pip install -e .

check: install
	$(VENV_PYTHON) -m py_compile \
		src/dooms/analysis.py src/dooms/fdr.py src/dooms/masks.py \
		src/dooms/io.py src/dooms/cli/main.py src/dooms/__init__.py src/dooms/__main__.py
	$(VENV_PYTHON) -m dooms --help >/dev/null

cli-check: install
	rm -rf $(CHECK_OUTPUT)
	$(VENV_PYTHON) -m dooms $(SMALL_INPUT) \
		--output-dir $(CHECK_OUTPUT) --tukey-backend numba_grouped_interp -q
	find $(CHECK_OUTPUT) -maxdepth 1 -type f -printf '%f\n' | sort

clean-venv:
	rm -rf $(VENV)
