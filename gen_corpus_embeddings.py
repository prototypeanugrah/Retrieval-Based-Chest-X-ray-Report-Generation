import os

import pandas as pd
import argparse
import torch
from tqdm import tqdm

import clip

from utils import nonpretrained_params


def encode_texts(impressions, model, device):
    """
    Create embeddings for the impressions using the CLIP model.

    Args:
        impressions (batch): Batch of impressions to encode (list of strings)
        model (CLIP model): CLIP model to use for encoding
        device (torch.device): Device to run the model on

    Returns:
        embeddings (tensor): Embeddings for the impressions
    """
    trimmed_impressions = impressions
    with torch.no_grad():
        imp_toks = clip.tokenize(
            trimmed_impressions, context_length=model.context_length
        ).to(device)
        embeddings = model.encode_text(imp_toks)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, _ = clip.load("ViT-B/32", device=device, jit=False)  # Load the model
    print("Loaded in pretrained model.")

    print(f"Model context length: {model.context_length}")  # 77
    model.load_state_dict(
        torch.load(args.clip_model_path, map_location=device)
    )  # Load the model state dictionary
    model = model.to(device)

    impressions = pd.read_csv(args.data_path)
    impressions = impressions[
        ~impressions.report.isin(
            [
                "AP chest compared to ___, 5:49 a.m., read in conjunction with abdomen CT on ___, 10:14 p.m.  Large left pleural effusion is slightly larger today than 24 hours ago, further reducing the volume of the nearly collapsed left lung, with moderate stable rightward mediastinal shift and probably responsible for moderately severe atelectasis in the right lower lobe.",
                "Although the right lung with cephalization and background interstitial abnormality is likely pulmonary edema, either on the basis of volume overload, cardiac decompensation or blood transfusion-related acute lung injury, left lung has a distinctly and progressively nodular abnormality, although occasionally pulmonary edema present as acinar filling with round opacities likely these, one cannot make that assumption and I discussed with Dr. ___ ___ possibility of disseminated infection, including septicemia.",
            ]
        )  # Remove these two reports from the dataset as their context length is too long
    ]["report"].reset_index(drop=True)
    impressions_size = impressions.shape[0]

    batch_size = args.batch_size
    num_batches = impressions_size // batch_size
    weight_tensors = []
    for i in tqdm(range(num_batches)):
        batch = impressions[
            batch_size * i : batch_size * i + batch_size
        ]  # Get the batch
        weights = encode_texts(batch, model, device)  # Encode the batch
        weight_tensors.append(weights)  # Append the weights to the list
    weights = encode_texts(
        impressions[batch_size * num_batches :], model, device
    )  # Encode the remaining impressions
    weight_tensors.append(weights)  # Append the weights to the list

    clip_embeddings = torch.cat(weight_tensors)  # Concatenate the weights
    out_data = (impressions, clip_embeddings)  # Save the impressions and the embeddings

    if not os.path.exists("corpus_embeddings"):
        os.makedirs("corpus_embeddings")
    out_path = "corpus_embeddings/" + args.output_saved_model  # Save the embeddings
    torch.save(out_data, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate clip embeddings for a training corpus (either sentence level or report level"
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        required=True,
        help="name of clip model state dictionary for generating embeddings",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path of csv file containing training corpus (either sentence level or report level)",
    )
    parser.add_argument(
        "--output_saved_model",
        type=str,
        required=True,
        help="name for saved corpus embeddings (include .pt extension)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6000,
        help="Batch size for generating clip embeddings",
    )
    args = parser.parse_args()

    main(args)

    # Code to run this script:
    # python gen_corpus_embeddings.py --clip_model_path clip_pretrained.pt --data_path splits/mimic_train_sentence_impressions.csv --output_saved_model mimic_train_embeddings.pt
