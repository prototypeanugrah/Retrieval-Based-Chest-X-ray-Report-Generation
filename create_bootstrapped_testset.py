import pandas as pd
import argparse
import os
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np


def filter_csv(input_csv, output_csv, train_test):
    if train_test == "test":
        df = pd.read_csv(input_csv, index_col=[0])
    else:
        df = pd.read_csv(input_csv)
    filtered = df[df["report"].notnull()]
    filtered.to_csv(output_csv, index=False)


def create_cxr_h5(reports_path, cxr_files_dir, cxr_outpath):
    cxr_paths = get_cxr_paths(reports_path, cxr_files_dir)
    img_to_hdf5(cxr_paths, cxr_outpath)


def get_cxr_paths(reports_path, cxr_files_dir):
    df = pd.read_csv(reports_path)
    df["Path"] = (
        cxr_files_dir
        + "p"
        + df["subject_id"].astype(str).str[:2]
        + "/p"
        + df["subject_id"].astype(str)
        + "/s"
        + df["study_id"].astype(str)
        + "/"
        + df["dicom_id"]
        + ".jpg"
    )
    cxr_paths = df["Path"]
    return cxr_paths


def img_to_hdf5(cxr_paths, out_filepath, resolution=320):
    dset_size = len(cxr_paths)
    with h5py.File(out_filepath, "w") as h5f:
        img_dset = h5f.create_dataset("cxr", shape=(dset_size, resolution, resolution))

        for idx, path in enumerate(tqdm(cxr_paths)):
            img = Image.open(path)
            img = preprocess(img, resolution)
            img_dset[idx] = img


def preprocess(img, desired_size):
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.LANCZOS)

    new_img = Image.new("L", (desired_size, desired_size))
    new_img.paste(
        img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )
    return new_img


def create_bootstrap_indices(bootstrap_dir, n=10):
    test_h5_path = os.path.join(bootstrap_dir, "mimic_cxr_test.h5")
    h5 = h5py.File(test_h5_path, "r")["cxr"]
    dset_size = len(h5)

    for i in range(n):
        subdir = os.path.join(bootstrap_dir, f"{str(i)}/")
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        indices = np.random.choice(dset_size, size=dset_size, replace=True)
        np.savetxt(subdir + "indices.txt", indices, fmt="%d")
        print(f"Bootstrap {i} indices saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the bootstrapped test-set, including the report impressions, images, and bootstrap indices."
    )
    parser.add_argument(
        "--imp_dir",
        type=str,
        required=True,
        help="directory where report impressions are stored",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default="bootstrap_test",
        help="directory where train/test set will be stored",
    )
    parser.add_argument(
        "--cxr_files_dir",
        type=str,
        required=True,
        help="mimic-cxr-jpg files directory containing chest X-rays",
    )
    parser.add_argument(
        "--train_val_test",
        type=str,
        required=True,
        help="Whether to create the train, val or test set",
    )
    args = parser.parse_args()

    unfiltered_test_impressions_path = (
        os.path.join(args.imp_dir, "mimic_test_impressions.csv")
        if args.train_val_test == "test"
        else (
            os.path.join(args.imp_dir, "mimic_train_sentence_impressions.csv")
            if args.train_val_test == "train"
            else os.path.join(args.imp_dir, "mimic_val_sentence_impressions.csv")
        )
    )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # filter out impressions that are nan
    filter_csv(
        unfiltered_test_impressions_path,
        os.path.join(args.out_dir, f"{args.train_val_test}_reports.csv"),
        args.train_val_test,
    )

    # crop and save cxrs as an h5 file
    cxr_outpath = (
        os.path.join(args.out_dir, "mimic_cxr_test.h5")
        if args.train_val_test == "test"
        else (
            os.path.join(args.out_dir, "mimic_cxr_train.h5")
            if args.train_val_test == "train"
            else os.path.join(args.out_dir, "mimic_cxr_val.h5")
        )
    )
    create_cxr_h5(
        os.path.join(args.out_dir, f"{args.train_val_test}_reports.csv"),
        args.cxr_files_dir,
        cxr_outpath,
    )

    if args.train_val_test != "test":
        print(f"{(args.train_val_test).upper()} set created.")
        exit()
    else:
        # create files of indices to sample for testing
        create_bootstrap_indices(args.out_dir)
        print("Bootstrapped test set created.")

    # Code to run the script
    # For val split - python create_bootstrapped_testset.py --imp_dir splits --out_dir reports --cxr_files_dir /work/pi_rachel_melamed_uml_edu/anugrah_vaishnav_student_uml_edu/mimic_cxr_jpg/val_images/mimic-cxr-jpg/2.1.0/files/ --train_val_test val
    # For train split - python create_bootstrapped_testset.py --imp_dir splits --out_dir reports --cxr_files_dir /work/pi_rachel_melamed_uml_edu/anugrah_vaishnav_student_uml_edu/mimic_cxr_jpg/train_images/mimic-cxr-jpg/2.1.0/files/ --train_val_test train
    # For test split - python create_bootstrapped_testset.py --imp_dir splits --cxr_files_dir /work/pi_rachel_melamed_uml_edu/anugrah_vaishnav_student_uml_edu/mimic_cxr_jpg/test_images/mimic-cxr-jpg/2.1.0/files/ --train_val_test test