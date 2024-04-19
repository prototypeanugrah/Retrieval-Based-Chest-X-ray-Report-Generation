import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate
from sklearn.metrics import f1_score, precision_recall_fscore_support

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


def main(args):
    gt_labels = pd.read_csv(args.gt_labels_path + "labels.csv").fillna(0)
    test_reports = pd.read_csv(args.gt_labels_path + "reports.csv").fillna(0)
    gt_test_labels = pd.merge(
        test_reports, gt_labels, on=["subject_id", "study_id"]
    ).drop(
        columns=["dicom_id", "report"]
    )  # merge the reports and labels
    print("F1")
    calculate_f1(args.dir, gt_test_labels)
    print("Precision Recall:")
    calculate_prec_recall(args.dir, gt_test_labels)


def calculate_f1(dir, gt_labels_path):
    """
    Calculate the F1 score for the given labels

    Args:
        dir (dir): Directory where the predicted labels are stored
        gt_labels_path (pd.DataFrame): Ground truth labels
    """
    true_labels = gt_labels_path[useful_labels]

    pred_labels = pd.read_csv(dir + "labeled_reports.csv", index_col=False).fillna(0)[
        useful_labels
    ]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_pred_labels[np_pred_labels == -1] = 0  # convert -1 to 0
    np_true_labels[np_true_labels == -1] = 0  # convert -1 to 0
    opts = np.array([0, 1])  # 0 and 1 are the only valid labels
    assert np.all(np.isin(np_pred_labels, opts))  # make sure all 0s and 1s

    f1_macro = f1_score(
        np_true_labels, np_pred_labels, average="macro"
    )  # calculate f1 macro
    f1_micro = f1_score(
        np_true_labels, np_pred_labels, average="micro"
    )  # calculate f1 micro
    print("F1 Macro score:", f1_macro)
    print("F1 Micro score:", f1_micro)


def calculate_prec_recall(dir, gt_labels_path):
    """
    Calculate the precision and recall for the given labels

    Args:
        dir (dir): Directory where the predicted labels are stored
        gt_labels_path (pd.DataFrame): Ground truth labels
    """
    true_labels = gt_labels_path[useful_labels]

    pred_labels = pd.read_csv(dir + "labeled_reports.csv", index_col=False).fillna(0)[
        useful_labels
    ]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_true_labels[np_true_labels == -1] = 0  # convert -1 to 0
    np_pred_labels[np_pred_labels == -1] = 0  # convert -1 to 0

    precs = []
    recalls = []
    for i in range(len(useful_labels)):  # calculate precision and recall for each label
        y_true = np_true_labels[:, i]  # get the true labels
        y_pred = np_pred_labels[:, i]  # get the predicted labels
        opts = np.array([0, 1])  # 0 and 1 are the only valid labels
        if not np.all(np.isin(y_true, opts)):  # make sure all 0s and 1s
            print(np.unique(y_true))  # print the unique values

        prec, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=1, average="binary"
        )  # calculate precision and recall
        precs.append(prec)
        recalls.append(recall)
    precs.append(np.array(precs).mean())
    recalls.append(np.array(recalls).mean())

    _df = pd.DataFrame([precs], columns=[*useful_labels, "Average"])
    print("Precision:")
    print(
        tabulate(_df, headers="keys", tablefmt="psql", showindex=False)
    )  # print the precision table

    # Save the precision table to a csv
    _df.to_csv(dir + "precision.csv", index=False)

    _df = pd.DataFrame([recalls], columns=[*useful_labels, "Average"])
    print("Recall:")
    print(
        tabulate(_df, headers="keys", tablefmt="psql", showindex=False)
    )  # print the recall table

    # Save the precision table to a csv
    _df.to_csv(dir + "recall.csv", index=False)


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
        "--gt_labels_path",
        type=str,
        required=True,
        help="directory where the ground truth labels are stored",
    )
    args = parser.parse_args()

    main(args)
