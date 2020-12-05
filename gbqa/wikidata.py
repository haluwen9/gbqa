import json

from qwikidata.sparql import return_sparql_query_results
from qwikidata.linked_data_interface import get_entity_dict_from_api, LdiResponseNotOk
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme

def wikidata_query(query):
    try: 
        query_result = return_sparql_query_results(query.strip())
        # print(query_result)
        if query.lower().startswith("ask"):
            return {
                "status": 0,
                "msg": "Query Success",
                "query_type": "ASK", 
                "values" : [query_result["boolean"]]
            }
        else:
            return {
                "status": 0,
                "msg": "Query Success",
                "query_type" : "SELECT", 
                "answers": query_result["results"]["bindings"]
            }
    except json.decoder.JSONDecodeError as jsonerr:
        print(jsonerr)
        return {
            "status" : 1, 
            "msg": "QUERY ERROR!"
        }

def wikidata_get_sitelink(id):
    try:
        entity = get_entity_dict_from_api(id)

        if "sitelinks" in entity and "enwiki" in entity["sitelinks"]:
            return {
                "title": entity["sitelinks"]["enwiki"]["title"],
                "sitelink": entity["sitelinks"]["enwiki"]["url"]
            }
    except LdiResponseNotOk as exception:
        print(exception)
        return {
            "error": "Failed to get sitelink"
        }

    return {
        "error":"Unknown Error"
    }
    
