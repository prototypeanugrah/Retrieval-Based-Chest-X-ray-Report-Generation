# Chest X-Ray Report Generation using CLIP model
 
## Project members

```
Anugrah Vaishnav
Neha Miryala
Divij Shivaji Pawar
Aaron Mailhot
```

## Dependencies

```
Python 3.11.8
PyTorch 2.2.1
NumPy 1.26.4
Pandas 2.2.1
torchvision 0.17.1
CLIP 1.0
PILLOW 9.3.0
```

## Data Preprocessing

### Data access
In order to run the scripts in this repository, you must have access to the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/files/#files-panel) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) databases.
Once you have access to these databases, you will have access to the train/test splits, the images and the radiology reports.

### Create data split

```
python split_mimic.py \
    --report_files_dir=<path containing all the radiology reports in the original format provided by Physionet> \
    --split_path=<path to the file containing the splits of data in mimic-cxr-jpg (file ending in .csv.gz)> \
    --out_dir=<path where you want to store the train and test report csv>
```

### Extract Impressions section

The original radiology reports contain <i>examination,  indication, technique, findings, and impression</i> sections. We are only interested in the <b>impressions</b> section.

```
python extract_impressions.py \
    --dir=<path to directory containing the train and test report splits>
```

### Create test set of reports and CXR pairs

```
python create_bootstrapped_testset \
    --imp_dir=<directory where report impressions are stored> \
    --out_dir=<directory where train/test set will be stored (containing .h5 file)> \
    --cxr_files_dir=<mimic-cxr-jpg files directory containing chest X-rays> \
    --train_val_test=<whether to create the train, val or test set>
```

## Ground truth labels for test reports

Labeled reports for the test set are provided in the mimic-cxr-2.0.0-chexpert.csv.gz file. Extract and rename the file to labels.csv and place it in the boostrap_test directory.

## Pre-trained CLIP model

The CLIP model checkpoint pre-trained on MIMIC-CXR train set is available here [CLIP weights](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml)

## Generate embeddings for the train corpus

```
python gen_corpus_embeddings.py \
    --clip_model_path=<name of clip model state dictionary for generating embeddings> \
    --data_path=<path of csv file containing training corpus (either sentence level or report level)> \
    --output_save_model=<name for saved corpus embeddings (include .pt extension)>
```

## Creating reports

```
python run_test.py \
    --corpus_embedding_path=<name of corpus embeddings file generated by CLIP (ending in .pt)> \
    --clip_model_path=<name of clip model state dictionary (ending in .pt)> \
    --out_dir=<directory to save outputted generated reports> \
    --cxr_path=<path of X-rays, (.h5 file) for MIMIC-CXR dataset> \
    --topk=<number of top sentences to retrieve>
```

## Generating labels for generated reports

Use [CheXbert](https://github.com/stanfordmlgroup/CheXbert/tree/master) to generate prediction labels for the generated reports.

```
python label.py -d={path to generated reports} -o={path to output dir} -c={path to checkpoint}
```

## Testing the performance of the model and the generated reports

We evaluate the following metrics

### BLEU score

```
python test_bleu.py --dir mimic_results/ --bootstrap_dir bootstrap_test/
```

### F1 score, precision, recall

```
python test_acc.py --dir mimic_results/ --gt_labels_path bootstrap_test/
```

### F1 score on bootstrapped test data (10 bootstraps)
```
python test_acc_batch.py --dir mimic_results/ --bootstrap_dir bootstrap_test/
```

Mean F1 score achieved on bootstrapped test data is 0.123 +/- 0.002


### Original paper citation

```
@InProceedings{pmlr-v158-endo21a,
  title = 	 {Retrieval-Based Chest X-Ray Report Generation Using a Pre-trained Contrastive Language-Image Model},
  author =       {Endo, Mark and Krishnan, Rayan and Krishna, Viswesh and Ng, Andrew Y. and Rajpurkar, Pranav},
  booktitle = 	 {Proceedings of Machine Learning for Health},
  pages = 	 {209--219},
  year = 	 {2021},
  volume = 	 {158},
  series = 	 {Proceedings of Machine Learning Research}
}
```
