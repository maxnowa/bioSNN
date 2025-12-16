# utils.py
import numpy as np
import json
import os
import uuid


def get_path(file_name):
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, f"../{file_name}")
    return file_path


def load_json_config(filename):
    config_path = get_path(filename)
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def create_batches(dataset, batch_size):
    num_samples = dataset.shape[0]
    num_batches = num_samples // batch_size
    batches = []

    for i in range(num_batches):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batches.append(batch)

    # Handle the remaining data
    if num_samples % batch_size != 0:
        remaining_batch = dataset[num_batches * batch_size :]
        batches.append(remaining_batch)
    return batches


def create_experiment_folder(dataset, inference, experiment_name="", **kwargs):
    config_path = "config/meta.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    model = config["model"]
    version = config["version"]
    base_dir = config["base_dir"]
    # Base folder name with model, dataset, date, and experiment ID
    exp_id = str(uuid.uuid4())[:8]
    if inference:
        folder_name = f"{model}-v{version}_{dataset}_{exp_id}_{experiment_name}"
    else: 
        folder_name = f"{model}-v{version}_TEST_{dataset}_{exp_id}"
    # Add varied parameters to the folder name
    if kwargs:
        param_str = "_".join(f"{key}-{value}" for key, value in kwargs.items())
        folder_name += f"_{param_str}"
    # TODO make work
    # if kwargs:
    #     param_str = "_".join(f"output-{kwargs["architecture"]}_epochs-{kwargs["epochs"]}_WTA-{kwargs["WTA"]}")
    #     folder_name += f"_{param_str}"
    
    # Create the full folder path
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    subfolders = ["weights", "parameters", "plots", "results"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(folder_path, subfolder), exist_ok=True)
    return folder_path

