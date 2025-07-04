
import argparse
from datetime import date

from duration_prediction.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on specified dates and save it to a given path")
    parser.add_argument("--train-date", required=True, help="train month in the YYYY-MM format")
    parser.add_argument("--val-date", required=True, help="validation month in the YYYY-MM format")
    parser.add_argument("--model-save-path", required=True, help="Path where the trained model should be saved")
    
    args = parser.parse_args()
    train_year, train_month = args.train_date.split("-")
    val_year, val_month = args.val_date.split("-")
    
    train_date = date(int(train_year), int(train_month), 1)
    val_date = date(int(val_year), int(val_month), 1)
    out_path = args.model_save_path

    train(train_date, val_date, out_path)