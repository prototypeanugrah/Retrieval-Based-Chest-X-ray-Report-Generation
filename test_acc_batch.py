import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import f1_score
import scipy.stats as st

useful_labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def bootstrap_f1(args):
    """
    Calculate the F1 score for the given labels using bootstrap samples

    Args:
        args (argparse): Arguments
    """
    true_labels = pd.read_csv(args.bootstrap_dir + "labels.csv").fillna(0)[
        useful_labels
    ]

    pred_labels = pd.read_csv(args.dir + "labeled_reports.csv").fillna(0)[useful_labels]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_pred_labels[np_pred_labels == -1] = 0  # convert -1 to 0
    np_true_labels[np_true_labels == -1] = 0  # convert -1 to 0
    opts = np.array([0, 1])
    assert np.all(np.isin(np_pred_labels, opts))  # make sure all 0s and 1s

    scores = []
    for i in range(10):  # 10 bootstrap samples
        indices = np.loadtxt(args.bootstrap_dir + str(i) + "/indices.txt", dtype=int)
        batch_score = f1_batch(
            indices, np_pred_labels, np_true_labels
        )  # calculate f1 score for each batch
        scores.append(batch_score)
    interval = st.t.interval(
        0.95, df=len(scores) - 1, loc=np.mean(scores), scale=st.sem(scores)
    )
    mean = sum(scores) / len(scores)
    print(f"F1 score mean: {round(mean, 3)}, +/- {round(mean - interval[0], 3)}")


def f1_batch(indices, pred_labels, true_labels):
    """
    Calculate the F1 score for the given labels for a batch of indices

    Args:
        indices (int): Indices of the batch
        pred_labels (int): Predicted labels (0 or 1)
        true_labels (int): True labels (0 or 1)

    Returns:
        float: F1 score
    """
    f1_macro = f1_score(
        true_labels[indices, :], pred_labels[indices, :], average="macro"
    )  # calculate f1 macro
    return f1_macro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test accuracy of the model on the test set."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory where the labeled reports are stored",
    )
    parser.add_argument(
        "--bootstrap_dir",
        type=str,
        required=True,
        help="directory where the bootstrap indices are stored",
    )

    args = parser.parse_args()

    bootstrap_f1(args)

    # Code to run this script
    # python test_acc_batch.py --dir mimic_results/ --bootstrap_dir bootstrap_test/
