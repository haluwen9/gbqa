import re

import numpy as np
import tensorflow as tf

TEMPLATE_REGEX = [
    {"regex" : r"SELECT DISTINCT \?sbj \?sbj_label WHERE { \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj rdfs:label \?sbj_label \. FILTER\(CONTAINS\(lcase\(\?sbj_label\), '([^']+)'\)\) \. FILTER[\s]*\(lang\(\?sbj_label\) = 'en'\) } LIMIT [\d]+", "template" : "<?S P O ; ?S instanceOf Type ; contains word TEMP1>"},
    {"regex" : r"SELECT DISTINCT \?sbj \?sbj_label WHERE { \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj rdfs:label \?sbj_label \. FILTER\(CONTAINS\(lcase\(\?sbj_label\), '([^']+)'\)\) \. FILTER[\s]*\(lang\(\?sbj_label\) = 'en'\) } LIMIT [\d]+", "template" : "<?S P O ; ?S instanceOf Type ; contains word TEMP2>" },
    {"regex" : r"SELECT DISTINCT \?sbj \?sbj_label WHERE { \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj rdfs:label \?sbj_label \. FILTER\(STRSTARTS\(lcase\(\?sbj_label\), '([^']+)'\)\) \. FILTER \(lang\(\?sbj_label\) = 'en'\) } LIMIT [\d]+", "template" : "<?S P O ; ?S instanceOf Type ; starts with character >" },
    {"regex" : r"SELECT DISTINCT \?sbj \?sbj_label WHERE { \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj rdfs:label \?sbj_label \. FILTER\(STRSTARTS\(lcase\(\?sbj_label\), '([^']+)'\)\) \. FILTER \(lang\(\?sbj_label\) = 'en'\) } LIMIT [\d]+", "template" : "<?S P O ; ?S instanceOf Type ; starts with character TYPE2>" },
    {"regex" : r"SELECT DISTINCT \?obj WHERE { wd:([^\s]+) wdt:([^\s]+) \?obj \. \?obj wdt:([^\s]+) wd:([^\s]+) }", "template" : "<S P ?O ; ?O instanceOf Type>" },
    {"regex" : r"SELECT \?value WHERE { wd:([^\s]+) p:([^\s]+) \?s \. \?s ps:([^\s]+) wd:([^\s]+) \. \?s pq:([^\s]+) \?value}", "template" : "(E pred ?Obj ) prop value" },
    {"regex" : r"SELECT \?obj WHERE { wd:([^\s]+) p:([^\s]+) \?s \. \?s ps:([^\s]+) \?obj \. \?s pq:([^\s]+) wd:([^\s]+) }", "template" : "(E pred F) prop ?value" },
    {"regex" : r"SELECT DISTINCT \?sbj WHERE { \?sbj wdt:([^\s]+) wd:([^\s]+) \. \?sbj wdt:([^\s]+) wd:([^\s]+) }", "template" : "<?S P O ; ?S InstanceOf Type>" },
    {"regex" : r"SELECT \?ent WHERE { \?ent wdt:([^\s]+) wd:([^\s]+) \. \?ent wdt:([^\s]+) \?obj } ORDER BY DESC\(\?obj\)LIMIT [\d]*", "template" : "?E is_a Type, ?E pred Obj  value. MAX/MIN (value)" },
    {"regex" : r"SELECT \?ent WHERE { \?ent wdt:([^\s]+) wd:([^\s]+) \. \?ent wdt:([^\s]+) \?obj \. \?ent wdt:([^\s]+) wd:([^\s]+) } ORDER BY DESC\(\?obj\)LIMIT [\d]*", "template" : "?E is_a Type. ?E pred Obj. ?E-secondClause value. MAX (value)" },
    {"regex" : r"SELECT \?ent WHERE { \?ent wdt:([^\s]+) wd:([^\s]+) \. \?ent wdt:([^\s]+) \?obj \. \?ent wdt:([^\s]+) wd:([^\s]+)[\s]*} ORDER BY ASC\(\?obj\)LIMIT [\d]*", "template" : "?E is_a Type. ?E pred Obj. ?E-secondClause value. MIN (value)" },
    {"regex" : r"ASK WHERE { wd:([^\s]+) wdt:([^\s]+) wd:([^\s]+) }", "template" : "Ask (ent-pred-obj)" },
    {"regex" : r"ASK WHERE { wd:([^\s]+) wdt:([^\s]+) wd:([^\s]+) . wd:([^\s]+) wdt:([^\s]+) wd:([^\s]+) }", "template" : "Ask (ent-pred-obj1 . ent-pred-obj2)" },
    {"regex" : r"SELECT \?answer WHERE { wd:([^\s]+) wdt:([^\s]+) \?X . \?X wdt:([^\s]+) \?answer}", "template" : "C RCD xD . xD RDE ?E" },
    {"regex" : r"SELECT \(COUNT\(\?obj\) AS \?value[\s]*\) { wd:([^\s]+) wdt:([^\s]+) \?obj }", "template" : "Count Obj (ent-pred-obj)" },
    {"regex" : r"SELECT \(COUNT\(\?sub\) AS \?value[\s]*\) { \?sub wdt:([^\s]+) wd:([^\s]+) }", "template" : "Count ent (ent-pred-obj)" },
    {"regex" : r"SELECT \?answer WHERE { wd:([^\s]+) wdt:([^\s]+) \?answer \. \?answer wdt:([^\s]+) wd:([^\s]+)}", "template" : "E REF ?F . ?F RFG G" },
    {"regex" : r"SELECT \?ans_1 \?ans_2 WHERE { wd:([^\s]+) wdt:([^\s]+) \?ans_1 \. wd:([^\s]+) wdt:([^\s]+) \?ans_2 }", "template" : "select where (ent-pred-obj1 . ent-pred-obj2)" }
]
TEMPLATE_COUNT = len(TEMPLATE_REGEX)

class SequenceProcessor(tf.keras.preprocessing.text.Tokenizer):
    
    def __init__(self, cls_token=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cls_token is not None:
            self.__cls_token__ = cls_token
        else:
            self.__cls_token__ = ""
    
    def preprocess_text(self, text):
        return text
    
    def preprocess_texts(self, texts):
        return np.array([f"{self.__cls_token__} {self.preprocess_text(text)}".strip() for text in texts])

    def fit(self, texts):
        self.fit_on_texts(self.preprocess_texts(texts))

    def tokenize(self, texts,  return_index=True, return_words=False, return_origin=False):
        preprocessed_texts = self.preprocess_texts(texts)
        token_texts = self.texts_to_sequences(preprocessed_texts)
        results = []
        if return_index:
            results.append(token_texts)
        if return_words:
            results.append([[self.index_word[token] for token in tokens] for tokens in token_texts])
        if return_origin:
            results.append([text.split() for text in preprocessed_texts])
        results = tuple(results)
        if len(results) == 1:
            results = results[0]
        return results
    
    def indices_to_words(self, indices):
        return [[self.index_word[token] for token in tokens if token != 0] for tokens in indices]

class EnglishSequenceProcessor(SequenceProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_text(self, text):
        text = text.strip().lower()
        text = re.sub(r"[?!,:;#~\|\{\}\[\]\(\)]", "", text)
        text = re.sub(r"[^a-zA-Z0-9.']", " ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = text.strip()

        return text + ' <end>'

class SparqlSequenceProcessor(SequenceProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_text(self, text):
        text = text.strip().lower()
        text = re.sub(r'([\(\{\}\)\,\.])', r" \1 ", text)
        text = re.sub(r"([a-z]*:)", r"\1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = text.strip()
        
        return text + ' <end>'

class EntityProcessor(SequenceProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def preprocess_text(self, text):
        for rex in TEMPLATE_REGEX:
            regex_res = re.match(rex["regex"], text, re.IGNORECASE)
            if regex_res:
                return " ".join(regex_res.groups()) 
        return ""
        