import random
import pandas as pd
from datasets import load_dataset, Dataset


def count_mountain_names(dataset, mountain_label=24):
    mountain_count = 0
    mountain_names_set = set()

    for example in dataset:
        fine_ner_tags = example["fine_ner_tags"]
        tokens = example["tokens"]

        if mountain_label in fine_ner_tags:
            mountain_count += 1
            mountain_words = [tokens[i] for i, label in enumerate(fine_ner_tags) if label == mountain_label]
            mountain_names_set.update(mountain_words)

    mountain_names = list(mountain_names_set)
    return mountain_count, mountain_names


def balance_data(dataset, mountain_label=24, mountain_rows_count=1502, additional_rows_count=498):
    mountain_rows = [example for example in dataset if mountain_label in example["fine_ner_tags"]]
    non_mountain_rows = [example for example in dataset if mountain_label not in example["fine_ner_tags"]]

    random.seed(42)
    additional_rows = random.sample(non_mountain_rows, additional_rows_count)

    balanced_data = mountain_rows + additional_rows
    random.shuffle(balanced_data)

    for example in balanced_data:
        example["fine_ner_tags"] = [1 if label == mountain_label else 0 for label in example["fine_ner_tags"]]

    for example in balanced_data:
        del example['ner_tags']
        example['ner_tags'] = example.pop('fine_ner_tags')

    return Dataset.from_dict({key: [example[key] for example in balanced_data] for key in balanced_data[0]})


def save_to_csv(dataset, file_path):
    df = pd.DataFrame(dataset)
    df.to_csv(file_path, index=False)


# Load dataset
all_data = load_dataset("DFKI-SLT/few-nerd", "supervised")
df_train, df_val, df_test = all_data["train"], all_data["validation"], all_data["test"]

# Balance datasets
balanced_train_dataset = balance_data(df_train, mountain_rows_count=1502, additional_rows_count=2000 - 1502)
balanced_val_dataset = balance_data(df_val, mountain_rows_count=218, additional_rows_count=300 - 218)
balanced_test_dataset = balance_data(df_test, mountain_rows_count=448, additional_rows_count=550 - 448)

# Save datasets to CSV
save_to_csv(balanced_train_dataset, "balanced_train_dataset.csv")
save_to_csv(balanced_val_dataset, "balanced_val_dataset.csv")
save_to_csv(balanced_test_dataset, "balanced_test_dataset.csv")
