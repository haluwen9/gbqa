import os
import pathlib
import zipfile


import click
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gbqa import GBQA
from gbqa.configs import MODEL_PATH, PRETRAIN_MODEL_DOWNLOAD_URL, DATASET_CONFIG
from gbqa.file_utils import download_file, load_preprocessor, get_model_paths
from gbqa.models.preprocess import EnglishSequenceProcessor, SparqlSequenceProcessor
from gbqa.models import NMTModel, TemplateBasedModel

@click.group()
def main():
    pass

@main.command()
@click.option("-l", "--list",
              is_flag=True,
              help="Show all available datasets for download.")
@click.option("-d", "--download",
              type=str,
              help="Download dataset.")
def dataset(list, download):
    print("This feature has not been implemented yet")

@main.command()
@click.option("-l", "--lists",
              is_flag=True,
              help="Show all installed models.")
@click.option("-t", "--train",
              type=str,
              help="Train and Save new model.\nUsage: `gbqa model --train MODELNAME --MODEL_TYPE`")
@click.option("-e", "--evaluate",
              type=str,
              help="Evaluate model.\nUsage: `gbqa model --evaluate MODELNAME --MODEL_TYPE`")
@click.option("--template/--nmt", default="True",
              help="Model type to use. There are two type available, TEMPLATE_BASED and NMT. \nDefault: TEMPLATE_BASED")
@click.option("-p", "--pretrain",
              is_flag=True,
              help="Download pretrain model.")
@click.option("-i", "--install",
              type=str,
              help="Install your model. \nUsage: `gbqa model -i path_to_your_model/MODELNAME_MODELTYPE.hdf5`")
def model(lists, train, evaluate, template, pretrain, install):
    if lists or train or evaluate:
        print("This feature has not been implemented yet")

    if pretrain:
        downloaded_file = os.path.join(MODEL_PATH, "pretrain.zip")
        download_file(PRETRAIN_MODEL_DOWNLOAD_URL, downloaded_file)
        zip = zipfile.ZipFile(downloaded_file)
        zip.extractall(MODEL_PATH)
        os.remove(downloaded_file)

        print("Download Pretrain models completed.")

@main.command()
@click.option("--stdin/--file", default="True",
              help="Predict from stdin or file.")
@click.option("-m", "model",
              default="PRETRAIN",
              help="Model name to use. Use \"gbqa model --list\" to see all available model to use.\nDefault: PRETRAIN")
@click.option("--template/--nmt", default="True",
              help="Model type to use. There are two type available, TEMPLATE_BASED and NMT. \nDefault: TEMPLATE_BASED")
def predict(stdin, model, template):
    if stdin == False:
        print("File prediction is not supported at the moment. Stdin will be used")

    if np.sum([os.path.exists(get_model_paths("pretrain", "ENGLISH_PROCESSOR")),
              os.path.exists(get_model_paths("pretrain", "SPARQL_PROCESSOR")),
              os.path.exists(get_model_paths("pretrain", "ENTITY_PROCESSOR"))]) < 3:
        print("Pre-train model can not be found on your disk. Please download it using `gbqa model --pretrain` command.")

    gbqa_instance = GBQA(model, "TEMPLATE_BASED" if template else "NMT")

    while True:
        question = input("Ask me: \n")
        
        predicted = gbqa_instance.predict_sparql(question)

        if predicted["status"] == 0:
            print(">>>", predicted["sparql"])
        else:
            print("ERROR:", predicted["msg"])

        print()
    

if __name__ == "__main__":
    main()

# What is the name of the capital of America?