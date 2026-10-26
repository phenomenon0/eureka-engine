# CI / hygiene gates. `make gpl-tts-check` is the entrypoint CI should run.
PYTHON ?= python3

.PHONY: gpl-tts-check
gpl-tts-check:
	$(PYTHON) tools/check_no_gpl_tts.py
