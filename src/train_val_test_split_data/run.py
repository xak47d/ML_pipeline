#!/usr/bin/env python
"""
Performs basic cleaning on the data and store the results in Weights & Biases
"""
import os
import argparse
import logging
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Main function, implementation of basic cleaning operations
    """

    run = wandb.init(project="my_ml_project", job_type="segregate")
    run.config.update(args)

    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    logger.info(f"Loaded data: {df.shape}")

    stratify = None
    if args.stratify_column != "none" and args.stratify_column in df.columns:
        stratify = df[args.stratify_column]
        logger.info(f"Stratifying by column: {args.stratify_column}")

    train_val, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify
    )

    val_size_adjusted = args.val_size / (1 - args.test_size)
    
    stratify_train_val = None
    if stratify is not None:
        stratify_train_val = train_val[args.stratify_column]
    
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=args.random_state,
        stratify=stratify_train_val
    )

    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    logger.info(f"  Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    logger.info(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

    for dataframe, k in zip([train, val, test], ["train", "val", "test"]):
        logger.info(f"Uploading {k}_data.csv set to W&B")
        with tempfile.NamedTemporaryFile("w") as fp:

            dataframe.to_csv(fp.name, index=False)

            artifact = wandb.Artifact(
                name=f"{k}_data.csv",
                type=f"{k}_data",
                description=f"{k} data split",
            )

            artifact.add_file(fp.name)
            run.log_artifact(artifact)

            artifact.wait()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Segregate data into train/val/test")
    
    parser.add_argument("input_artifact",
                        type=str,
                        help="Input W&B artifact")

    parser.add_argument("test_size",
                        type=float,
                        help="Test set proportion")

    parser.add_argument("val_size",
                        type=float,
                        help="Validation set proportion")

    parser.add_argument("random_state",
                        type=int,
                        help="Random seed")

    parser.add_argument("stratify_column",
                        type=str,
                        help="Column for stratification")
    
    args = parser.parse_args()
    
    go(args)