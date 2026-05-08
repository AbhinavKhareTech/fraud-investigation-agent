.PHONY: install install-full run serve eval test clean

install:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[dev,trident]"

run:
	python -m agent.cli

serve:
	uvicorn api.server:app --reload --port 8000

eval:
	python -m eval.run_eval

eval-tp:
	python -m eval.run_eval --category true_positive

eval-tn:
	python -m eval.run_eval --category true_negative

eval-degraded:
	python -m eval.run_eval --category degraded

test:
	pytest tests/ -v

clean:
	rm -rf eval/results/*.json __pycache__ .pytest_cache *.egg-info
