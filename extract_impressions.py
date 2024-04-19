import nltk.data
import pandas as pd
import argparse
import os

# """Finds line index that is the start of the section."""


def sec_start(lines, section=" IMPRESSION"):
    """
    Given a list of lines, find the index of the line that starts the section.

    Args:
        lines (list): List of lines in the report.
        section (str): Section to find the start of.

    Returns:
        int: Index of the line that starts the section.
    """
    for idx, line in enumerate(lines):
        if line.startswith(section):  # Check if the line starts with the section
            return idx
    return -1

    # """Generates a csv containing report impressions."""


def gen_full_report_csv(df, split, dir):
    """
    Given a dataframe, generates a csv containing the impression section of the report.

    Args:
        df (pd.DataFrame): Dataframe containing the reports.
        split (str): Split of the data.
        dir (str): Directory to save the csv.
    """
    df_imp = df.copy()
    for index, row in df_imp.iterrows():
        report = row["report"].splitlines()

        imp_idx = sec_start(
            report
        )  # Find the index of the start of the section starting with 'IMPRESSION:'
        imp_find_idx = sec_start(
            report, section=" FINDINGS AND IMPRESSION:"
        )  # Find the index of the start of the section starting with 'FINDINGS AND IMPRESSION:'
        sep = ""
        if imp_idx != -1:  # If the impression index is found
            impression = (
                sep.join(report[imp_idx:])
                .replace("IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        elif imp_find_idx != -1:  # If the impression and findings index is found
            impression = (
                sep.join(report[imp_find_idx:])
                .replace("FINDINGS AND IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        else:  # If neither index is found
            impression = ""

        df_imp.at[index, "report"] = (
            impression  # Update the report column with the impression
        )

    out_name = f"mimic_{split}_impressions.csv"
    out_path = os.path.join(dir, out_name)
    df_imp.to_csv(out_path, index=False)

    # """Generates a csv containing all impression sentences."""


def gen_sentence_imp_csv(df, split, dir, tokenizer):
    """
    Given a dataframe, generates a csv containing all impression sentences.

    Args:
        df (pd.DataFrame): Dataframe containing the reports.
        split (str): Split of the data.
        dir (str): Directory to save the csv.
        tokenizer (nltk tokenizer): Tokenizer to split sentences.
    """
    df_imp = []
    for index, row in df.iterrows():
        report = row["report"].splitlines()

        imp_idx = sec_start(
            report
        )  # Find the index of the start of the section starting with 'IMPRESSION:'
        imp_find_idx = sec_start(
            report, section=" FINDINGS AND IMPRESSION:"
        )  # Find the index of the start of the section starting with 'FINDINGS AND IMPRESSION:'
        sep = ""
        if imp_idx != -1:  # If the impression index is found
            impression = (
                sep.join(report[imp_idx:])
                .replace("IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        elif imp_find_idx != -1:  # If the impression and findings index is found
            impression = (
                sep.join(report[imp_find_idx:])
                .replace("FINDINGS AND IMPRESSION:", "")
                .replace("\n", "")
                .strip()
            )
        else:  # If neither index is found
            impression = ""

        for sent_index, sent in enumerate(
            split_sentences(impression, tokenizer)
        ):  # Split the impression into sentences
            df_imp.append(
                [row["dicom_id"], row["study_id"], row["subject_id"], sent_index, sent]
            )

    df_imp = pd.DataFrame(
        df_imp, columns=["dicom_id", "study_id", "subject_id", "sentence_id", "report"]
    )

    out_name = f"mimic_{split}_sentence_impressions.csv"
    out_path = os.path.join(dir, out_name)
    df_imp.to_csv(out_path, index=False)

    # """Splits sentences by periods and removes numbering and nans."""


def split_sentences(report, tokenizer):
    """
    Splits sentences by periods and removes numbering and nans.

    Args:
        report (str): Report to split into sentences.
        tokenizer (nltk tokenizer): Tokenizer to split sentences.

    Returns:
        list: List of sentences.
    """
    sentences = []
    if not (
        isinstance(report, float) and math.isnan(report)
    ):  # Check if the report is not a float and not nan
        for sentence in tokenizer.tokenize(
            report
        ):  # Tokenize the report into sentences
            try:
                float(sentence)  # Remove numbering
            except ValueError:
                sentences.append(sentence)
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract impression sections from reports at full report and sentence levels."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory where train and test report reports are stored and where impression sections will be stored",
    )
    args = parser.parse_args()

    train_path = os.path.join(args.dir, "mimic_train_full.csv")
    test_path = os.path.join(args.dir, "mimic_test_full.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # whole reports
    gen_full_report_csv(train_df, "train", args.dir)
    gen_full_report_csv(test_df, "test", args.dir)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # sentences
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    gen_sentence_imp_csv(train_df, "train", args.dir, tokenizer)
