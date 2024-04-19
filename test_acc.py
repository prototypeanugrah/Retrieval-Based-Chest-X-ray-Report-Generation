import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tabulate import tabulate
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

cxr_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion','Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
useful_labels = cxr_labels

def main(args):
    gt_labels = pd.read_csv(args.gt_labels_path + 'labels.csv').fillna(0)
    test_reports = pd.read_csv(args.gt_labels_path + 'reports.csv').fillna(0)
    gt_test_labels = pd.merge(test_reports, gt_labels, on=['subject_id', 'study_id']).drop(columns=['dicom_id', 'report'])
    # gt_test_labels = gt_test_labels[['subject_id', 'study_id'] + cxr_labels]
    print("F1")
    calculate_f1(args.dir, gt_test_labels)
    print("Precision Recall:")
    calculate_prec_recall(args.dir, gt_test_labels)
    
def calculate_f1(dir, gt_labels_path):
    # true_labels = pd.read_csv(gt_labels_path).fillna(0)[useful_labels]
    true_labels = gt_labels_path[useful_labels]

    pred_labels = pd.read_csv(dir + 'labeled_reports.csv', index_col=False).fillna(0)[useful_labels]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_pred_labels[np_pred_labels == -1] = 0
    np_true_labels[np_true_labels == -1] = 0
    opts = np.array([0,1])
    assert np.all(np.isin(np_pred_labels, opts))

    f1_macro = f1_score(np_true_labels, np_pred_labels, average='macro')
    f1_micro = f1_score(np_true_labels, np_pred_labels, average='micro')
    print('F1 Macro score:', f1_macro)
    print('F1 Micro score:', f1_micro)

def calculate_prec_recall(dir, gt_labels_path):
    # true_labels = pd.read_csv(gt_labels_path).fillna(0)[useful_labels]
    true_labels = gt_labels_path[useful_labels]

    pred_labels = pd.read_csv(dir + 'labeled_reports.csv', index_col=False).fillna(0)[useful_labels]

    np_true_labels = true_labels.to_numpy()
    np_pred_labels = pred_labels.to_numpy()
    np_true_labels[np_true_labels == -1] = 0
    np_pred_labels[np_pred_labels == -1] = 0

    precs = []
    recalls = []
    for i in range(len(useful_labels)):
        y_true = np_true_labels[:, i]
        y_pred = np_pred_labels[:, i]
        opts = np.array([0,1])
        if not np.all(np.isin(y_true, opts)):
            print(np.unique(y_true))

        assert np.all(np.isin(y_true, opts))
        assert np.all(np.isin(y_pred, opts))
        prec, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=1, average='binary')
        precs.append(prec)
        recalls.append(recall)
    precs.append(np.array(precs).mean())
    recalls.append(np.array(recalls).mean())

    _df = pd.DataFrame([precs], columns=[*useful_labels, "Average"])
    print("Precision:")
    print(tabulate(_df, headers='keys', tablefmt='psql', showindex=False))
    
    # Save the precision table to a csv
    _df.to_csv(dir + 'precision.csv', index=False)
    
    _df = pd.DataFrame([recalls], columns=[*useful_labels, "Average"])
    print("Recall:")
    print(tabulate(_df, headers='keys', tablefmt='psql', showindex=False))

    # Save the precision table to a csv
    _df.to_csv(dir + 'recall.csv', index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing framework for CXR-RePaiR and baseline methods')
    parser.add_argument('--dir', type=str, required=True, help='directory where predicted labels and embeddings are')
    parser.add_argument('--gt_labels_path', type=str, required=True, help='path to where gt labels are stored')
    args = parser.parse_args()

    main(args)


