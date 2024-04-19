import pandas as pd
import os
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()


def main(args):
    generated_reports = pd.read_csv(args.dir + "generated_reports.csv")
    test_reports = pd.read_csv(args.bootstrap_dir + "reports.csv")

    # Add the generated reports to the test reports dataframe
    test_reports["generated_report"] = generated_reports["Report Impression"]

    # Calculate BLEU scores
    test_reports["bleu_score"] = test_reports.progress_apply(
        lambda row: calculate_bleu_score(
            row["report"], row["generated_report"]
        ),
        axis=1,
    )
    
    test_reports = test_reports[['report', 'generated_report', 'bleu_score']]

    # Save the dataframe with BLEU scores
    test_reports.to_csv(args.dir + "bleu_scores_test.csv", index=False)


def calculate_bleu_score(references, hypothesis):
    # Calculate BLEU score
    bleu_score = sentence_bleu(
        references, hypothesis, smoothing_function=SmoothingFunction().method7
    )
    return bleu_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU scores")
    parser.add_argument(
        "--bootstrap_dir", type=str, help="Path to directory containing test reports"
    )
    parser.add_argument(
        "--dir", type=str, help="Path to directory containing the generated reports"
    )
    args = parser.parse_args()
    main(args)
