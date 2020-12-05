import re

import numpy as np

def is_entity(id):    
    return True if type(id) == str and re.match("^Q[0-9]+", id, re.IGNORECASE) else False

def is_prop(id):
    return True if type(id) == str and re.match("^P[0-9]+", id, re.IGNORECASE) else False

def is_char(id):
    return type(id) == str and len(id) == 1

def replace_at_index(raw, start_index, end_index, replace_str):
    return "".join([raw[:start_index], replace_str, raw[end_index:]])

def postprocess_template(template_id, entities):
    TEMPLATE_SPARQL = [
        "SELECT DISTINCT ?sbj ?sbj_label WHERE { \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj rdfs:label ?sbj_label . \n\tFILTER(CONTAINS(lcase(?sbj_label), '{{WORD}}')) . \n\tFILTER(lang(?sbj_label) = 'en') \n} LIMIT 5",
        "SELECT DISTINCT ?sbj ?sbj_label WHERE { \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj rdfs:label ?sbj_label . \n\tFILTER(CONTAINS(lcase(?sbj_label), '{{WORD}}')) . \n\tFILTER(lang(?sbj_label) = 'en') \n} LIMIT 5",
        "SELECT DISTINCT ?sbj ?sbj_label WHERE { \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj rdfs:label ?sbj_label . \n\tFILTER(STRSTARTS(lcase(?sbj_label), '{{CHAR}}')) . \n\tFILTER (lang(?sbj_label) = 'en') \n} LIMIT 5",        
        "SELECT DISTINCT ?sbj ?sbj_label WHERE { \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj rdfs:label ?sbj_label . \n\tFILTER(STRSTARTS(lcase(?sbj_label), '{{CHAR}}')) . \n\tFILTER (lang(?sbj_label) = 'en') \n} LIMIT 5",        
        "SELECT DISTINCT ?obj WHERE { \n\twd:{{ENT}} wdt:{{PROP}} ?obj . \n\t?obj wdt:{{PROP}} wd:{{ENT}} \n}",        
        "SELECT ?value WHERE { \n\twd:{{ENT}} p:{{PROP}} ?s . \n\t?s ps:{{PROP}} wd:{{ENT}} . \n\t?s pq:{{PROP}} ?value \n}",        
        "SELECT ?obj WHERE { \n\twd:{{ENT}} p:{{PROP}} ?s . \n\t?s ps:{{PROP}} ?obj . \n\t?s pq:{{PROP}} wd:{{ENT}} \n}",        
        "SELECT DISTINCT ?sbj WHERE { \n\t?sbj wdt:{{PROP}} wd:{{ENT}} . \n\t?sbj wdt:{{PROP}} wd:{{ENT}} \n}",        
        "SELECT ?ent WHERE { \n\t?ent wdt:{{PROP}} wd:{{ENT}} . \n\t?ent wdt:{{PROP}} ?obj \n} ORDER BY DESC(?obj)LIMIT 5",
        "SELECT ?ent WHERE { \n\t?ent wdt:{{PROP}} wd:{{ENT}} . \n\t?ent wdt:{{PROP}} ?obj . \n\t?ent wdt:{{PROP}} wd:{{ENT}} \n} ORDER BY DESC(?obj)LIMIT 5",
        "SELECT ?ent WHERE { \n\t?ent wdt:{{PROP}} wd:{{ENT}} . \n\t?ent wdt:{{PROP}} ?obj . \n\t?ent wdt:{{PROP}} wd:{{ENT}} \n} ORDER BY ASC(?obj)LIMIT 5",
        "ASK WHERE { \n\twd:{{ENT}} wdt:{{PROP}} wd:{{ENT}} \n}",
        "ASK WHERE { \n\twd:{{ENT}} wdt:{{PROP}} wd:{{ENT}} . \n\twd:{{ENT}} wdt:{{PROP}} wd:{{ENT}} \n}",
        "SELECT ?answer WHERE { \n\twd:{{ENT}} wdt:{{PROP}} ?X . \n\t?X wdt:{{PROP}} ?answer \n}",
        "SELECT (COUNT(?obj) AS ?value) { \n\twd:{{ENT}} wdt:{{PROP}} ?obj \n}",
        "SELECT (COUNT(?sub) AS ?value) { \n\t?sub wdt:{{PROP}} wd:{{ENT}} \n}",
        "SELECT ?answer WHERE { \n\twd:{{ENT}} wdt:{{PROP}} ?answer . \n\t?answer wdt:{{PROP}} wd:{{ENT}} \n}",
        "SELECT ?ans_1 ?ans_2 WHERE { \n\twd:{{ENT}} wdt:{{PROP}} ?ans_1 . \n\twd:{{ENT}} wdt:{{PROP}} ?ans_2 \n}"
    ]

    ents = { "WORD": [], "ENTITY": [], "CHAR": [], "PROPERTY": [] }

    for ent in entities:
        if is_char(ent):
            ents["CHAR"].append(ent)
        elif is_entity(ent):
            ents["ENTITY"].append(ent)
        elif is_prop(ent):
            ents["PROPERTY"].append(ent)

    template = TEMPLATE_SPARQL[template_id]

    index = template.find("{{ENT}}")
    while index != -1:
        if len(ents["ENTITY"]) < 1:
            return { "status" : 1, "msg" : "Missing Entity!" , "sparql" : template }
        
        ent_to_fill = ents["ENTITY"].pop(0)
        ents["ENTITY"].append(ent_to_fill)
        template = replace_at_index(template, index, index+7, ent_to_fill)
        index = template.find("{{ENT}}")

    index = template.find("{{PROP}}")
    while index != -1:
        if len(ents["PROPERTY"]) < 1:
            return { "status" : 2, "msg" : "Missing Property!" , "sparql" : template }
        
        ent_to_fill = ents["PROPERTY"].pop(0)
        ents["PROPERTY"].append(ent_to_fill)
        template = replace_at_index(template, index, index+8, ent_to_fill)
        index = template.find("{{PROP}}")

    index = template.find("{{WORD}}")
    while index != -1:
        if len(ents["WORD"]) < 1:
            return { "status" : 3, "msg" : "Missing Info!" , "sparql" : template }
        
        ent_to_fill = ents["WORD"].pop(0)
        ents["WORD"].append(ent_to_fill)
        template = replace_at_index(template, index, index+8, ent_to_fill)
        index = template.find("{{WORD}}")

    index = template.find("{{CHAR}}")
    while index != -1:
        if len(ents["CHAR"]) < 1:
            return { "status" : 3, "msg" : "Missing Info!" , "sparql" : template }
        
        ent_to_fill = ents["CHAR"].pop(0)
        ents["CHAR"].append(ent_to_fill)
        template = replace_at_index(template, index, index+8, ent_to_fill)
        index = template.find("{{CHAR}}")

    return { "status" : 0, "msg" : "Generate Sparql Success!", "sparql" : template}