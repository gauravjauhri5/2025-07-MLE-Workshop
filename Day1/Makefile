.PHONY: train
train:
	uv run python -m duration_prediction.main --train-date 2022-01 --val-date 2022-02 --model-save-path models/2022-01.pkl
	
.PHONY: test
test:
	uv run pytest