# Molt — elastic on-device inference
.PHONY: help install test test-fast projectors bench figures serve demo lint clean

LADDER ?= qwen
DEVICE ?= cpu
DTYPE  ?= float32
PORT   ?= 8000
OUT    ?= benchmarks/results

help:
	@echo "make install     install dependencies"
	@echo "make test        full hermetic test-suite (no downloads)"
	@echo "make projectors  fit the KV projections for LADDER=$(LADDER)"
	@echo "make bench       run all conditions + cost sweep + QoS"
	@echo "make figures     render figures, splice the README table, build the report"
	@echo "make serve       start the streaming service on port $(PORT)"
	@echo "make demo        serve + open the browser demo"

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest -q

test-fast:
	python3 -m pytest -q -x -m "not slow"

projectors:
	python3 scripts/train_projectors.py --ladder $(LADDER) --device $(DEVICE) \
		--dtype $(DTYPE) --recompute-top-k 6 --max-total-tokens 6144

bench:
	python3 benchmarks/run.py --ladder $(LADDER) --device $(DEVICE) --dtype $(DTYPE) \
		--trace spike_mid_answer --conditions all --n-prompts 5 --max-new-tokens 48 \
		--virtual-step 1.0 --recompute-top-k 6 --up-patience 4 --cooldown 4 \
		--cost-sweep --qos --out $(OUT)

figures:
	python3 figures/make_figures.py --results $(OUT) --out figures
	python3 scripts/update_readme.py --table figures/results_table.md
	python3 figures/make_report.py --results $(OUT) --out artifacts/report.html

serve:
	python3 -m molt.service --ladder $(LADDER) --device $(DEVICE) --dtype $(DTYPE) \
		--port $(PORT)

demo: serve

lint:
	python3 -m compileall -q molt benchmarks figures scripts examples tests

clean:
	rm -rf .pytest_cache **/__pycache__ benchmarks/results_* figures/*.png
