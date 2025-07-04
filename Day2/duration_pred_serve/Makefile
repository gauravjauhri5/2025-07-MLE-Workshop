.PHONY: run
run:
	export MODEL_PATH="./models/2022-01.pkl" ; \
	export VERSION="0.0.1" ; \
	uv run python src/duration_pred_serve/serve.py
	
.PHONY: predict-test
predict-test:
	uv run python integration-tests/predict-test.py