import pickle
import numpy as np

with open(r"C:\Users\user\OneDrive\Desktop\MTS\Pipeline\local\handoff\CUSTOM_test_label.pkl", "rb") as f:
    labels = pickle.load(f)

print(f"Label length: {len(labels)}")
print(f"Anomalous timesteps: {int(labels.sum())}")
print(f"First anomalous index: {np.where(labels)[0][0] if labels.sum() > 0 else 'none'}")
print(f"Last anomalous index: {np.where(labels)[0][-1] if labels.sum() > 0 else 'none'}")