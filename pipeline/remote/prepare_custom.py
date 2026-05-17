from utils import prepare_custom_dataset
prepare_custom_dataset(
    csv_path="datasets/custom/stream_A.csv",
    label_path="datasets/custom/stream_A_gt.csv",
    train_ratio=0.4,   # adjust so anomalies fall in test
    normalize=True,
    save_dir="datasets/custom"
)