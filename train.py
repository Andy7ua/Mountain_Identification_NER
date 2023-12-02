import argparse
import numpy as np
import torch
import pandas as pd

from datasets import load_from_disk, load_metric

from transformers import (DataCollatorForTokenClassification, AutoTokenizer,
                          AutoModelForTokenClassification, TrainingArguments, Trainer)

from sklearn.utils.class_weight import compute_class_weight

from torch import nn


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model for mountain name recognition.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for saving model checkpoints and logs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=float, default=5,
                        help="Number of training epochs.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load balanced datasets from CSV files
    balanced_train_dataset = pd.read_csv('balanced_train_dataset.csv')
    balanced_val_dataset = pd.read_csv('balanced_val_dataset.csv')
    balanced_test_dataset = pd.read_csv('balanced_test_dataset.csv')

    # Initialize the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER",
                                                            num_labels=2, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    tokenizer.save_pretrained("./tokenizers/tokenizer")

    # Tokenize and adjust labels for the datasets
    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True)
        total_adjusted_labels = []

        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if wid is None:
                    adjusted_label_ids.append(-100)
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])

            total_adjusted_labels.append(adjusted_label_ids)

        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples

    tokenized_dataset_train = balanced_train_dataset.map(tokenize_adjust_labels, batched=True)
    tokenized_dataset_val = balanced_val_dataset.map(tokenize_adjust_labels, batched=True)
    tokenized_dataset_test = balanced_test_dataset.map(tokenize_adjust_labels, batched=True)

    # Calculate class weights for imbalanced classes
    labels = [0] * (68395 - 8166) + [1] * 8166
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=labels)
    class_weights_dict = {i: weight for i, weight in zip([0, 1], class_weights)}
    print("Class Weights:", class_weights_dict)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Load seqeval metric for evaluation
    metric = load_metric("seqeval")

    # Define custom metrics for evaluation
    def compute_metrics(predictions):
        predictions, labels = predictions
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        class_1_tp = 0
        class_1_fp = 0
        class_1_fn = 0

        for i in range(len(true_predictions)):
            class_1_tp += np.sum([(true_predictions[i][j] == 1 and true_labels[i][j] == 1) for j in range(len(true_predictions[i]))])
            class_1_fp += np.sum([(true_predictions[i][j] == 1 and true_labels[i][j] != 1) for j in range(len(true_predictions[i]))])
            class_1_fn += np.sum([(true_predictions[i][j] != 1 and true_labels[i][j] == 1) for j in range(len(true_predictions[i]))])

        class_1_precision = class_1_tp / max(float(class_1_tp + class_1_fp), 1e-9)
        class_1_recall = class_1_tp / max(float(class_1_tp + class_1_fn), 1e-9)
        class_1_f1 = 2 * (class_1_precision * class_1_recall) / max(class_1_precision + class_1_recall, 1e-9)

        return {
            "class_1_f1": class_1_f1,
            "class_1_precision": class_1_precision,
            "class_1_recall": class_1_recall,
            "overall_accuracy": results["overall_accuracy"]
        }

    # Define a weighted trainer class for handling class imbalance
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.57, 4.19], device=logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        run_name="ep_10_tokenized_11",
        save_strategy='no',
        report_to=None
    )

    # Initialize the weighted trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save the trained model
    model.save_pretrained("./models/large_model")

    # Evaluate the model on the test dataset
    evaluation_results = trainer.evaluate(eval_dataset=tokenized_dataset_test)

    print(evaluation_results)

    print(f"The F1 Score on test data is: {evaluation_results['eval_class_1_f1']}")


if __name__ == "__main__":
    main()
