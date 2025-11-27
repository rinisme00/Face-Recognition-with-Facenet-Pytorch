import os
from typing import List

import pandas as pd


def collect_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    Assumes dataset_dir structure:
        dataset_dir/
            user1/*.jpg
            user2/*.jpg
            ...
    Returns DataFrame with columns: image_path, label
    """
    dataset_dir = os.path.abspath(dataset_dir)
    image_paths: List[str] = []
    labels: List[str] = []

    for user_name in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_dir):
            continue

        for filename in os.listdir(user_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(user_dir, filename)
                image_paths.append(full_path)
                labels.append(user_name)

    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    return df