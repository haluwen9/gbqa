import os
import pathlib

HOME_PATH = pathlib.Path.home()
RESOURCE_PATH = os.path.join(HOME_PATH, ".gbqa")
MODEL_PATH = os.path.join(RESOURCE_PATH, "models")

PRETRAIN_MODEL_DOWNLOAD_URL = "https://www.dropbox.com/s/8mv7z0odw1cc4cp/models.zip?dl=1"

DEFAULT_NMT_MODEL_CONFIG = {
    "EMBEDDING_DIM" : 1024,
    "LATENT_DIM" : 512,
}

DATASET_CONFIG = {
    "MAX_INPUT_LENGTH" : 40,
    "MAX_DECODER_LENGTH" : 100,
    "MAX_ENTITY_NUMBER" : 6,
    "INPUT_VOCABULARY_SIZE" : 18831,
    "SPARQL_VOCABULARY_SIZE" : 17312,
    "ENTITY_VOCABULARY_SIZE" : 17320,
    "TEMPLATE_COUNT" : 18
}
