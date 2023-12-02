import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def extract_mountain_names(text):
    tags = []
    words = []
    # Mapping from label to category
    label_to_category = {'LABEL_0': 'other', 'LABEL_1': 'mountain'}
    # Perform NER using the classifier pipeline
    result = ner_classifier(text)
    current_word = ''
    # Process NER results
    for element in result:
        # Check for tokenization
        if element['word'][0] != '#':
            tags.append(label_to_category[element['entity']])
            # Check if the current word is not empty
            if current_word != '':
                words.append(current_word)
                current_word = ''
            current_word += element['word']
        else:
            current_word += element['word'][2:]
    # Append the last word
    words.append(current_word)
    # Create a dictionary mapping words to tags
    return {word: tag for word, tag in zip(words, tags)}


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Identify Mountain Names in Text.')
    parser.add_argument('--text', type=str, required=True, help='Input text for NER.')

    # Parse command-line arguments
    args = parser.parse_args()
    input_text = args.text

    # Extract mountain names from the input text
    result_tags = extract_mountain_names(input_text)
    print(result_tags)


if __name__ == "__main__":
    # Load pre-trained NER model and tokenizer from Hugging Face Model Hub
    mountain_model = AutoModelForTokenClassification.from_pretrained("dieumerci/mountain-recognition-ner")
    mountain_tokenizer = AutoTokenizer.from_pretrained("dieumerci/mountain-recognition-ner")
    ner_classifier = pipeline("ner", model=mountain_model, tokenizer=mountain_tokenizer)

    main()
