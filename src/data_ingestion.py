from datasets import load_dataset
from transformers import AutoTokenizer
from config  import *

def load_data(dataset_name: str):
    dataset = load_data(dataset_name)
    return dataset

def split_data(dataset):
    train_data = dataset["train"].select(range(2000))
    val_data = dataset["validation"].select(range(500))
    test_data = dataset["test"].select(range(500))



