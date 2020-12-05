# import argparse
# import numpy as np
# import re
# import nltk
# from typing import List
# from sklearn.datasets import load_files
# import pickle
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# #nltk.download('stopwords')
# #nltk.download('wordnet')


# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description='Making the dataset.'
#     )
#     parser.add_argument(
#         '--input_path',
#         type=str,
#         required=True,
#         help='File to input',
#     )
#     parser.add_argument(
#         '--output_file',
#         type=str,
#         required=True,
#         help='File to output',
#     )
#     return parser.parse_args()


# def text_preprocessing(sentences: List[str]) -> List[str]:
#     documents = []

#     stemmer = WordNetLemmatizer()

#     for sen in range(0, len(sentences)):
#         # Remove all the special characters
#         document = re.sub(r'\W', ' ', str(sentences[sen]))

#         # remove all single characters
#         document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

#         # Remove single characters from the start
#         document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

#         # Substituting multiple spaces with single space
#         document = re.sub(r'\s+', ' ', document, flags=re.I)

#         # Removing prefixed 'b'
#         document = re.sub(r'^b\s+', '', document)

#         # Converting to Lowercase
#         document = document.lower()

#         # Lemmatization
#         document = document.split()

#         document = [stemmer.lemmatize(word) for word in document]
#         document = ' '.join(document)

#         documents.append(document)

#     return documents


# def main():
#     args = parse_arguments()
#     movie_data = load_files(args.input_path)
#     sentences, labels = movie_data.data, movie_data.target
#     clean_sentences = text_preprocessing(sentences)
#     with open(args.output_file, 'w') as output_handler:
#         for X, y in zip(clean_sentences, labels):
#             output_handler.write('{}\t{}\n'.format(X, y))


# if __name__ == "__main__":
#     main()

import os

DATASET_URL = {
    "TRAIN_SET": "https://ndownloader.figshare.com/files/15738824",
    "TEST_SET": "https://ndownloader.figshare.com/files/15738818",
    "FULL_DATASET": "https://ndownloader.figshare.com/files/14878388"
}