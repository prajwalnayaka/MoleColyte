import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from Utils import tox21_test_loader
from Training_Scripts.train_Tox21 import MoleColyteModel


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kiln: {device}")

    model = MoleColyteModel(in_node_features=8, out_features=12).to(device)

    model.load_state_dict(
        torch.load(r"Trained_Models/molecolyte_tox21_random_spilt_finetuned_best.pt", weights_only=True))
    model.eval()

    all_preds = []
    all_targets = []

    print("Running inference on the unseen Test Set...")
    with torch.no_grad():
        for batch in tox21_test_loader:
            batch = batch.to(device)

            # Forward pass
            logits = model(batch.x.to(torch.float), batch.pos, batch.edge_index, batch.edge_attr, batch.batch)

            probs = torch.sigmoid(logits)

            # Move data back to CPU and store as numpy arrays for scikit-learn
            all_preds.append(probs.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    roc_aucs = []

    print("\n=== Tox21 Evaluation Results ===")
    for i in range(12):  # Iterate over the 12 toxicity pathways

        # The NumPy version of our NaN Masking trick
        valid_mask = ~np.isnan(all_targets[:, i])

        task_targets = all_targets[valid_mask, i]
        task_preds = all_preds[valid_mask, i]

        # Scikit-learn crashes if a test set is so small that it only contains 0s and no 1s.
        # This safety check ensures we only score pathways that actually have both safe and toxic examples.
        if len(np.unique(task_targets)) > 1:
            score = roc_auc_score(task_targets, task_preds)
            roc_aucs.append(score)
            print(f"Assay {i + 1:02d} AUC-ROC: {score:.4f}")
        else:
            print(f"Assay {i + 1:02d} Skipped (Only one class present in test set)")

    print("--------------------------------")
    print(f"🏆 FINAL MEAN AUC-ROC: {np.mean(roc_aucs):.4f}")


if __name__ == "__main__":
    evaluate()