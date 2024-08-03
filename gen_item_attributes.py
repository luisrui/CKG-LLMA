# from vllm import LLM, SamplingParams
import numpy as np
from openai import OpenAI
import requests
import os
import pickle
import re
import argparse
import json
import torch
import ast
from tqdm import tqdm
from modules.data import data_config

# api_key = "sk-fVMV1dDzIHIwnVGK46986c6237094a03A638CfAe56D35561"
# api_base = "https://bjqai.com/v1"
# api_base = 'https://chat.zhucn.org/v1'
# api_key = 'sk-DvBuU0tMOS04pyw742167fFa029544E48d3626EaDc159a5b'
api_key = "8df7b6a81e8e47b4b29c2aa8b870bc6e0eae33855b0c4b5c990ea5d9aa05bdd3"  # 请替换为你的API密钥
api_base = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"

headers = { 
    "Content-Type": "application/json", 
    "Authorization": f"Bearer {api_key}"
}


def generate_ia_triples(items, i_a_triples:dict):
    triples = []
    for item in items:
        if not isinstance(item, int):
            item = item.item()
        triples.extend([item, rel, tail] for rel, tail in i_a_triples[item])
    return triples

def Translate_triple2text(triples, id2ent, id2rel, id2name):
    """
    triples: list of triples
    id2ent: dict, id to entity
    id2rel: dict, id to relation
    """
    triples_text = str([(id2name[id2ent[head]], id2rel[rel], id2ent[tail]) for head, rel, tail in triples])
    triples_text = triples_text.replace(', ', ',')
    return triples_text

def Read_prompt(dataset:str, 
                initial_query : str, 
                items, 
                id2ent : dict, 
                id2rel : dict, 
                id2name: dict, 
                selected_triples : list, 
                target_relations, 
                attributes):
    '''
    Generate graph prompts based on kg triples and logics.
    '''
    items = np.unique(items)
    items_title = {id2name[id2ent[item]] for item in items}
    items_title_text = str(items_title).replace(': ', ':').replace(', ',',')
    triples_text = Translate_triple2text(selected_triples, id2ent, id2rel, id2name)
    #items = np.concatenate([pos_items], axis=0)

    
    graph_prompt = initial_query.replace("<<Items>>", items_title_text).replace("<<Triples>>", triples_text)
    graph_prompt = graph_prompt.replace("<<Attributes>>", attributes).replace("<<Relations>>", target_relations)
    return graph_prompt

def get_json_answer(system_query, user_query, model_name='gpt-3.5-turbo'):
    print('Asking...')
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_query},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.2,
        "top_p": 0.1
    }
    response = requests.post(api_base, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        text_response = response.json()['choices'][0]['message']['content']
        text_response = text_response.replace("\n", "").replace("  ", "")
        return text_response
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

def json_format_mining(generated_text):
    try:
        # Extract the JSON data from the generated text
        json_match = re.search(r"{.*}", generated_text, re.DOTALL)
        json_data = json_match.group()

        # Replace the escaped single quotes with double quotes
        json_data = json_data.replace("'", '"').replace("\n", "")

        json_pattern = re.compile(r'{"(delete|add)": \[(.*?)\]}', re.DOTALL)
        json_matches = json_pattern.findall(json_data)

        json_string = "{"
        for i, match in enumerate(json_matches):
            key, value = match
            value = re.sub(r",\s*$", "", value)  # 移除末尾的逗号
            value = f"[{value}]"  # 将值包裹在方括号中
            json_string += f'"{key}": {value}'
            if i < len(json_matches) - 1:
                json_string += ","  # 添加逗号分隔符
        json_string += "}"
        return True, json_string
    except Exception as e:
        print(f"Error in json_format_mining: {str(e)}")
        return False, ""

def Translate_modify2id(modify_json, ent2id, rel2id, name2id):
    add_list = modify_json['add']
    try:
        add_list_id = [(ent2id[name2id[head]], rel2id[rel], ent2id[tail]) for head, rel, tail in add_list]
    except:
        add_list_id = []
        for triple in add_list:
            try:
                head, rel, tail = triple
                add_list_id.append((ent2id[name2id[head]], rel2id[rel], ent2id[tail]))
            except:
                continue
    
    return add_list_id

def Triple_check(selected_triples, check_list):
    """
    Check if the triples are correct.
    """
    new_add_list = []
    for triple in check_list:
        if triple not in selected_triples:
            new_add_list.append(triple)
    return {
        'add': new_add_list
    }

def main(dataset, item_id, attributes, must_attributes, ent2id, rel2id, i_a_triples, target_relations):
    itemloader = torch.utils.data.DataLoader(
        item_id,
        batch_size=32,
        shuffle=True,
        num_workers=12
    )
    #item_counter = trange(0, len(itemloader), ncols=0)
    enhanced_graph_info = list()

    with open("modules/prompts/Item_attribute_query.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
        system_query = texts[0][:-1]
        user_query_initial = texts[1]

    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    id2name = json.load(open(f'./dataset/{dataset}/id2name.json', 'r'))
    name2id = {v: k for k, v in id2name.items()}
    print('start generating')
    for items in tqdm(itemloader):
        random_idxs = np.random.choice(range(len(attributes)), size=15, replace=False)
        select_attributes = must_attributes
        select_attributes.extend([attributes[idx] for idx in random_idxs])
        select_attributes = str(select_attributes).replace(', ', ',')
        
        selected_triples = generate_ia_triples(items, i_a_triples)
        query = Read_prompt(dataset, user_query_initial, items, id2ent, id2rel, id2name, selected_triples, target_relations, select_attributes)
        try:
            text_response = get_json_answer(system_query, query, 'gpt-3.5-turbo')
            print(text_response)
            #json_match = re.search(r"{.*}", text_response, re.DOTALL)
            #modify_json = json.loads(text_response)
            try:
                modify_dict = json.loads(text_response)
            except:
                modify_dict = ast.literal_eval(text_response)
            add_triples_nocheck = Translate_modify2id(modify_dict, ent2id, rel2id, name2id)
            id_json = Triple_check(selected_triples, add_triples_nocheck)
            print(id_json)
        except Exception as e:
            print(f"tried again, problem {str(e)}!")
            try:
                text_response = get_json_answer(system_query, query, 'gpt-3.5-turbo')
                print(text_response)
                #json_match = re.search(r"{.*}", text_response, re.DOTALL)
                #json_data = json_match.group()
                #json_data = json_data.replace("'", '"').replace("\n", "")# Replace the escaped single quotes with double quotes
                try:
                    modify_json = json.loads(text_response)
                except:
                    modify_json = ast.literal_eval(text_response)
                add_triples_nocheck = Translate_modify2id(modify_json, ent2id, rel2id, name2id)
                id_json = Triple_check(selected_triples, add_triples_nocheck)
                print(id_json)
            except Exception as e_again:
                print(f'failed! problem {str(e)}')
                id_json = {"add": []}

        enhanced_graph_info.append(id_json)

    with open(f"./saved_graphs/{dataset}_enhanced_itemattributes.pkl", "wb") as f:
        pickle.dump(enhanced_graph_info, f)

    return 0


if __name__ == "__main__":
    dataset = 'MovieLens1M'

    ent2id = json.load(
        open(
            f"./dataset/{dataset}/entity2id.json",
            "r",
        )
    )
    rel2id = json.load(
        open(
            f"./dataset/{dataset}/relation2id.json",
            "r",
        )
    )

    num_users, num_items = data_config[dataset]["num_users"], data_config[dataset]["num_items"]
    attributes = list(ent2id.keys())[num_users + num_items:]
    target_relations = list(rel2id.keys())[1:4]
    target_relations = str(target_relations).replace(', ', ',')
    i_a_triples = torch.load(f'./dataset/{dataset}/pre_saved/i_of_a.pt')
    items_id = range(num_users, num_users + num_items)
    must_attributes = list(ent2id.keys())[2618:2643]

    enhanced_graph_info = main(dataset, items_id, attributes, must_attributes, ent2id, rel2id, i_a_triples, target_relations)