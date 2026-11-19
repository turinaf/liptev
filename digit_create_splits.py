import argparse
from create_splits import create_splits


def create_digit_splits(config_path="config.json", processed_root=None):
    return create_splits(config_path, dataset="digit", processed_root=processed_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits for processed digit data.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON config file")
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Optional override for digit_processed_data_dir (defaults to config value)",
    )
    args = parser.parse_args()

    create_digit_splits(config_path=args.config, processed_root=args.processed_root)
