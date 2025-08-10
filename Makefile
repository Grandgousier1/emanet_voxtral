# Makefile — Emanet Runpod final
VENV := $(HOME)/emanet-env
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQ := requirements.txt

.PHONY: install run batch debug clean

install:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r $(REQ)
	chmod +x setup_runpod.sh
	./setup_runpod.sh

# run single episode
run:
	@if [ -z "$(URL)" ]; then echo "Please provide URL or FILE variable: make run URL=\"https://..\" or make run FILE=\"/path/ep.mp4\""; exit 1; fi
	$(PY) main.py --url "$(URL)" --out "$(OUT)"

# batch: provide a text file with one URL or local path per line
batch:
	@if [ -z "$(LIST)" ]; then echo "Please provide LIST=path/to/list.txt"; exit 1; fi
	$(PY) main.py --batch-list "$(LIST)" --out-dir "$(OUTDIR)"

# quick debug smoke tests
debug:
	$(PY) main.py --dry-run

clean:
	rm -rf work_* *.srt ./*.db
	rm -rf $(VENV)
