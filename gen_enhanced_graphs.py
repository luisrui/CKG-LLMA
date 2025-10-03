from tqdm import tqdm, trange
#from modules.utils import *
from modules.data import *

# from vllm import LLM, SamplingParams
from openai import OpenAI
import requests
import os
import pickle
import re
import argparse
import ast
import time
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

#client = OpenAI(api_key=api_key, base_url=api_base)

def Translate_triple2text(triples, id2ent, id2rel, id2name, sub_type):
    """
    triples: list of triples
    id2ent: dict, id to entity
    id2rel: dict, id to relation
    """
    threshold = len(id2rel) // 2
    if sub_type == 'ui':
        #user-item
        triples_text_1 = [(id2ent[head], id2rel[rel], id2name[id2ent[tail]]) for head, rel, tail in triples if rel == 0]
        #item-attribute
        triples_text_2 = [(id2name[id2ent[head]], id2rel[rel], id2ent[tail]) for head, rel, tail in triples if rel != 0]
        triples_text_1.extend(triples_text_2)
        triples_text = str(triples_text_1)                      
    elif sub_type == 'ii':
        #item-item
        triples_text_1 = [(id2name[id2ent[head]], id2rel[rel], id2name[id2ent[tail]]) for head, rel, tail in triples if rel > threshold]
        #item-attribute
        triples_text_2 = [(id2name[id2ent[head]], id2rel[rel], id2ent[tail]) for head, rel, tail in triples if rel < threshold]
        triples_text_1.extend(triples_text_2)
        triples_text = str(triples_text_1)                      
    triples_text = triples_text.replace(', ', ',')
    return triples_text

def Read_prompt(
        initial_query : str, 
        pos_items, 
        id2ent : dict, 
        id2rel : dict, 
        id2name : dict, 
        users_text, 
        target_relations,
        attributes_text,
        selected_triples : list, 
        sub_type):
    '''
    Generate graph prompts based on kg triples and logics.
    '''
    items = np.unique(pos_items)
    items_title = {id2name[id2ent[item]] for item in items}
    items_title_text = str(items_title).replace(': ', ':').replace(', ',',')
    triples_text = Translate_triple2text(selected_triples, id2ent, id2rel, id2name, sub_type)
    graph_prompt = initial_query.replace("<<Items>>", items_title_text).replace("<<Triples>>", triples_text)
    graph_prompt = graph_prompt.replace("<<Attributes>>", attributes_text).replace("<<Relations>>", target_relations)
    graph_prompt = graph_prompt.replace("<<Users>>", users_text)

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

    delay = random.uniform(1, 3)  # Random delay between 1 and 5 seconds
    time.sleep(delay)

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
    except:
        print(f"Error in json_format_mining: {str(e)}")
        return False, ""

def Translate_modify2id(modify_json, ent2id, rel2id, name2id, selected_triples):
    delete_list = modify_json['delete']

    delete_list_id = []
    for triple in delete_list:
        try:
            head, rel, tail = triple
            if rel2id[rel] == 0:
                target_triple = (ent2id[head], rel2id[rel], ent2id[name2id[tail]]) # User-Items
            else:
                target_triple = (ent2id[name2id[head]], rel2id[rel], ent2id[tail]) # Item-Attribute
            if list(target_triple) in selected_triples:
                delete_list_id.append(target_triple)
        except:
            continue

    add_list = modify_json['add']

    add_list_id = []
    for triple in add_list:
        try:
            head, rel, tail = triple
            target_triple = (ent2id[name2id[head]], rel2id[rel], ent2id[tail]) # Item-Attribute
            if list(target_triple) not in selected_triples:
                add_list_id.append(target_triple)
        except:
            continue
    
    return {
        'delete': delete_list_id,
        'add': add_list_id
    }

def Triple_check(selected_triples, processed_modify_json):
    """
    Check if the triples are correct.
    """
    add_list = processed_modify_json['add']
    del_list = processed_modify_json['delete']
    new_add_list, new_del_list = [], []
    for triple in del_list:
        if triple in selected_triples:
            new_del_list.append(triple)
    for triple in add_list:
        if triple not in selected_triples:
            new_add_list.append(triple)
    return {
        'delete': new_del_list,
        'add': new_add_list
    }

def main(dataset, total_epoch, start_step, end_step, enhanced_graph_path: dict, ent2id, rel2id, target_rels):
    """
    Generate subgraphs for training.
    """
    graphs_info = pickle.load(open(f"./saved_graphs/{enhanced_graph_path}.pkl", "rb"))
    epoch_counter = trange(0, total_epoch, ncols=0)
    enhanced_graph_info = dict()
    id2name = json.load(open(f'./dataset/{dataset}/id2name.json', 'r'))
    name2id = {v: k for k, v in id2name.items()}

    with open("modules/prompts/gpt_graph_prompt_ui.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
        system_query = texts[0][:-1]
        user_query_initial_ui = texts[1]
    
    with open("modules/prompts/gpt_graph_prompt_ii.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
        system_query = texts[0][:-1]
        user_query_initial_ii = texts[1]

    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    print('start generating')
    for e_poch in epoch_counter:
        epoch_info_text = defaultdict(list)
        epoch_info = graphs_info[e_poch]
        users_epoch, pos_items_epoch, neg_items_epoch = (
            epoch_info["users"],
            epoch_info["pos_items"],
            epoch_info["neg_items"],
        )
        ui_graphs, ii_graphs = (
            epoch_info["ui"],
            epoch_info["ii"],
        )
        threshold = len(rel2id) // 2
        for i in tqdm(range(start_step, end_step), ncols=0, desc=f"Epoch {e_poch}"):
            users = users_epoch[i]
            pos_items = pos_items_epoch[i]
            neg_items = neg_items_epoch[i]
            #uu_graph = uu_graphs[i]
            ui_graph = ui_graphs[i]
            ii_graph = ii_graphs[i]

            users_text = str([id2ent[user] for user in users.numpy()])
            attribute_text = str([id2ent[attr] for _, rel, attr in ui_graph if rel > 0 and rel < threshold])
            # g_uu_prompt = Read_prompt(user_query_initial, pos_items, neg_items, id2ent, id2rel, uu_graph)
            g_ui_prompt = Read_prompt(
                user_query_initial_ui, pos_items, id2ent, id2rel, id2name, users_text, target_rels, attribute_text, ui_graph, sub_type='ui'
            )
            g_ii_prompt = Read_prompt(
                user_query_initial_ii, pos_items, id2ent, id2rel, id2name, users_text, target_rels, attribute_text, ii_graph, sub_type='ii'
            )

            for graph, query, g_type in zip(
                [ii_graph, ui_graph], [g_ii_prompt, g_ui_prompt], ["ii", "ui"]
            ):
                if len(graph) >= 50:
                    try:
                        text_response = get_json_answer(system_query, query, 'gpt-3.5-turbo')
                        print(text_response)
                        try:
                            modify_dict = json.loads(text_response)
                        except:
                            modify_dict = ast.literal_eval(text_response)
                        id_json = Translate_modify2id(modify_dict, ent2id, rel2id, name2id, graph)
                        #id_json = Triple_check(graph, modify_triple_raw)
                        print(id_json)
                    except Exception as e:
                        print(f"tried again!, error is {str(e)}")
                        try:
                            text_response = get_json_answer(system_query, query, 'gpt-3.5-turbo')
                            print(text_response)
                            try:
                                modify_dict = json.loads(text_response)
                            except:
                                modify_dict = ast.literal_eval(text_response)
                            id_json = Translate_modify2id(modify_dict, ent2id, rel2id, name2id, graph)
                            #id_json = Triple_check(graph, modify_triple_raw)
                            print(id_json)  
                        except Exception as e_again:
                            print('failed! Error is ', str(e_again))
                            id_json = {"add": [], "delete": []}
                else:
                    print('less than requirement!')
                    id_json = {"add": [], "delete": []}

                epoch_info_text[g_type].append(id_json)
        enhanced_graph_info[e_poch] = epoch_info_text

    with open(f"./saved_graphs/{enhanced_graph_path}_enhanced_{start_step}_{end_step}.pkl", "wb") as f:
        pickle.dump(enhanced_graph_info, f)

    return 0

if __name__ == "__main__":
    #dataset = 'MovieLens1M'
    dataset = 'Steam'
    #dataset = 'AmazonBook'

    #enhanced_graph_path = "MovieLens1M_20_2_e3"
    enhanced_graph_path = "Steam_20_3_e0"
    #enhanced_graph_path = 'AmazonBook_32_3_e0'

    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--start",
        type=int,
        default="0",
    )
    parse.add_argument(
        "--end",
        type=int,
        default="8005",
    )
    config = parse.parse_args()

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

    target_relations = list(rel2id.keys())[1:4]
    target_relations = str(target_relations).replace(', ', ',')

    enhanced_graph_info = main(dataset, 1, config.start, config.end, enhanced_graph_path, ent2id, rel2id, target_relations)

    # with open(f"./saved_graphs/{enhanced_graph_path}_enhanced.pkl", "wb") as f:
    #     pickle.dump(enhanced_graph_info, f)
