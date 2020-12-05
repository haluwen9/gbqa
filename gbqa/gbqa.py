import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gbqa import configs
from gbqa.models import NMTModel, TemplateBasedModel
from gbqa.models.preprocess import EnglishSequenceProcessor, SparqlSequenceProcessor, EntityProcessor
from gbqa.models.postprocess import postprocess_template
from gbqa.wikidata import wikidata_query, wikidata_get_sitelink
from gbqa.file_utils import load_preprocessor, get_model_paths


class GBQA:
    def __init__(self, model_name="PRETRAIN", model_type="TEMPLATE_BASED"):
        if model_type not in ["NMT", "TEMPLATE_BASED"]:
            raise Exception("INVALID_MODEL_TYPE", "Invalid Model Type!")
        
        self.model_name = model_name
        self.model_type = model_type

        self.__load_model()

    def predict_sparql(self, question):
        x = self.__pre_processing(question)

        if self.model_type == "NMT":
            y_pred = self.model.predict([x])
            return {
                "status" : 0,
                "msg": "",
                "sparql" : " ".join(self.__post_processing_sparql(y_pred)[0]).replace("<end>", "").strip()
            }
        else:
            template_id, entities = self.model.predict([x])
            template_id = np.argmax(template_id[0])
            entities = self.__post_processing_entity(entities)[0]
            print(template_id, entities)
            return postprocess_template(template_id, entities)


    def ask(self, question):
        predicted = self.predict_sparql(question)
        if predicted["status"] != 0:
            return predicted
        
        sparql = predicted["sparql"]
        result = wikidata_query(sparql)

        if result["status"] != 0:
            print(result["msg"])
            return {
                **result,
                "sparql": sparql
            }
        
        if result["query_type"] == "ASK":
            return {
                **result,
                "sparql": sparql,
                "msg": "Success!"
            }
        elif len(result["answers"]) != 0:
            key = ""
            for k in result["answers"][0].keys():
                if k in ["value", "obj", "sbj", "ent", "answer"]:
                    key = k
                    break
            if key == "":
                del result["answers"]
                return {
                    **result,
                    "sparql": sparql,
                    "status": 404,
                    "msg": "No answer."
                }
            
            values = []
            answers = []
            for answer in result["answers"]:
                if answer[key]["type"] != "uri":
                    values.append(answer[key]["value"])
                else:
                    id = answer[key]["value"].split("/")[-1]
                    sitelink = wikidata_get_sitelink(id) 

                    if "error" in sitelink:
                        print(sitelink)
                        values.append(answer[key]["value"])
                    else:
                        answers.append(sitelink)        

            if len(values) > 0:
                result["values"] = values

            if len(answers) > 0:
                result["answers"] = answers
            else:
                del result["answers"]   

            return {
                **result,
                "status": 404 if len(values) + len(answers) == 0 else 0,
                "sparql": sparql,
                "msg": "No answer." if len(values) + len(answers) == 0 else "Success"
            }
        else:
            del result["answers"]
            return {
                **result,
                "sparql": sparql,
                "status": 404,
                "msg": "No answer."
            }

    def __load_model(self):
        model_path = get_model_paths(self.model_name, self.model_type)

        # print(model_path)

        if self.model_type == "NMT":
            if not os.path.exists(model_path):
                raise Exception("MODEL_N_EXIST", "Model is not exist.")

            self.model = NMTModel(model_path)
        else:
            if not all([os.path.exists(path) for path in model_path]):
                raise Exception("MODEL_N_EXIST", "Model is not exist or missing part.")

            self.model = TemplateBasedModel(model_path[0], model_path[1])

        self.eng_proc = load_preprocessor(get_model_paths("pretrain", "ENGLISH_PROCESSOR"))
        self.sparql_proc = load_preprocessor(get_model_paths("pretrain", "SPARQL_PROCESSOR"))
        self.ent_proc = load_preprocessor(get_model_paths("pretrain", "ENTITY_PROCESSOR"))

    def __pre_processing(self, x):
        return pad_sequences(self.eng_proc.tokenize([x]), maxlen=configs.DATASET_CONFIG["MAX_INPUT_LENGTH"], padding='post')

    def __post_processing_sparql(self, y_pred):
        return self.sparql_proc.indices_to_words(np.argmax(y_pred, 2))

    def __post_processing_entity(self, y_pred):
        return self.ent_proc.indices_to_words(np.argmax(y_pred, 2))
