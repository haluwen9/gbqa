import os
import pickle
from urllib.request import urlretrieve

from tqdm import tqdm

from gbqa import configs

class DownloadTqdm(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize

def download_file(url, destination):
    """
    url  : string
        File's URL
    desc  : string
        Save location.
    """
    with DownloadTqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=f'Downloading {url.split("/")[-1]}') as bar:
        urlretrieve(url, filename=destination,
                           reporthook=bar.update_to, data=None)
        bar.total = bar.n

class PreprocessUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "gbqa.models.preprocess"
        return super().find_class(module, name)

def load_preprocessor(path):
    with open(path, "rb") as f:
        unpickler = PreprocessUnpickler(f)
        obj = unpickler.load()
    return obj

def get_model_paths(model_name, model_type):
    MODEL_TYPE = {
        "NMT": "_NMT.hdf5",
        "TEMPLATE_BASED": "_TMP_",
        "ENGLISH_PROCESSOR": "_EPROC.pkl",
        "SPARQL_PROCESSOR": "_SPROC.pkl",
        "ENTITY_PROCESSOR": "_ENTPROC.pkl"
    }

    model_path = os.path.join(configs.MODEL_PATH, model_name + MODEL_TYPE[model_type])

    if model_type == "TEMPLATE_BASED":
        model_path = [model_path + postfix + ".hdf5" for postfix in ["CLS", "REC"]]

    return model_path