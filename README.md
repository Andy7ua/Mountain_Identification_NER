# Named Entity Recognition for Mountain Identification
This project revolves around training a Named Entity Recognition (NER) model specifically tailored to detect mountain names within textual content. The project comprises the following elements:

# 1. Dataset Generation
The dataset utilized in this project is derived from the [DFKI-SLT/few-nerd](https://huggingface.co/datasets/DFKI-SLT/few-nerd) dataset. Pertinent rows were selected, and the data underwent relabeling. Detailed information on dataset statistics and preprocessing can be found in the `visual_data.ipynb` Jupyter notebook and dataSet.py.

The balanced dataset is stored in the local directory, segmented into training, validation, and test sets.

# 2. Model Training
For NER, the [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER) model and tokenizer were employed. To initiate training, use the following command:

`python train.py --output_dir [output_dir] --learning_rate [learning_rate] --num_train_epochs [num_train_epochs]`

(Default arguments are *2e-5* for `learning_rate` and *5* for `num_train_epochs`).

# 3. Model Inference
Performing inference on new text samples is achievable with the following command:

`python inference.py --text [text]`

# Project Structure
- `visual_data.ipynb`: Jupyter notebook encompassing dataset creation and statistics.
- `train.py`: Python script for model training.
- `inference.py`: Python script for model inference.
- `dataSet.py`: The script balances mountain-related NER dataset and saves as CSVs.
- `requirements.txt`: Requirements file, install using `pip install -r requirements.txt`

Explore the notebooks and scripts to gain a deeper understanding of the dataset, training, and inference processes. If you encounter any obstacles or have inquiries, don't hesitate to get in touch!