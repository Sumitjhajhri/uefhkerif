.PHONY: install test eval eval-v1 eval-v2 clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -q

eval-v1:
	python -m eval.run_eval --version v1

eval-v2:
	python -m eval.run_eval --version v2

eval: eval-v1 eval-v2

clean:
	rm -rf results/manifests __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} +
