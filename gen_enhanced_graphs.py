from tqdm import tqdm, trange
from modules.utils import *
from modules.data import *

# from vllm import LLM, SamplingParams
from openai import OpenAI
import os
import pickle
import re
import argparse

dataset = 'AmazonBook'
# api_key = "sk-fVMV1dDzIHIwnVGK46986c6237094a03A638CfAe56D35561"
# api_base = "https://bjqai.com/v1"
base_url = 'https://chat.zhucn.org/v1/'
api_key = 'sk-DvBuU0tMOS04pyw742167fFa029544E48d3626EaDc159a5b'
client = OpenAI(api_key=api_key, base_url=api_key)

def get_json_answer(system_query, user_query, model_name='gpt-3.5-turbo'):
    completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_query},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.2,
                top_p=0.1,
            )
    text_response = completion.choices[0].message.content
    text_response = text_response.replace("\n", "").replace("  ", "")
    return text_response

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
        return False, ""


def main(total_epoch, start_step, end_step, enhanced_graph_path: dict, ent2id, rel2id):
    """
    Generate subgraphs for training.
    """
    graphs_info = pickle.load(open(f"./saved_graphs/{enhanced_graph_path}.pkl", "rb"))
    epoch_counter = trange(0, total_epoch, ncols=0)
    enhanced_graph_info = dict()

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
        for i in tqdm(range(start_step, end_step), ncols=0, desc=f"Epoch {e_poch}"):
            pos_items = pos_items_epoch[i]
            neg_items = neg_items_epoch[i]
            #uu_graph = uu_graphs[i]
            ui_graph = ui_graphs[i]
            ii_graph = ii_graphs[i]

            # g_uu_prompt = Read_prompt(user_query_initial, pos_items, neg_items, id2ent, id2rel, uu_graph)
            g_ui_prompt = Read_prompt(
                dataset, user_query_initial_ui, pos_items, neg_items, id2ent, id2rel, ui_graph
            )
            g_ii_prompt = Read_prompt(
                dataset, user_query_initial_ii, pos_items, neg_items, id2ent, id2rel, ii_graph
            )

            for graph, query, g_type in zip(
                [ii_graph, ui_graph], [g_ii_prompt, g_ui_prompt], ["ii", "ui"]
            ):
                if len(graph) >= 50:
                    try:
                        text_response = get_json_answer(system_query, query, 'gpt-3.5-turbo')
                        json_match = re.search(r"{.*}", text_response, re.DOTALL)
                        json_data = json_match.group()
                        json_data = json_data.replace("'", '"').replace("\n", "")# Replace the escaped single quotes with double quotes
                        print(json_data)
                        modify_json = json.loads(text_response)
                        id_json = Translate_modify2id(modify_json, ent2id, rel2id)
                    except Exception as e:
                        if "Expecting" in str(e):
                            id_json = json_data
                        else:
                            print("tried again!")
                            try:
                                text_response = get_json_answer(system_query, query, 'gpt-3.5-turbo')
                                json_match = re.search(r"{.*}", text_response, re.DOTALL)
                                json_data = json_match.group()
                                json_data = json_data.replace("'", '"').replace("\n", "")# Replace the escaped single quotes with double quotes
                                print(json_data)
                                modify_json = json.loads(text_response)
                                id_json = Translate_modify2id(modify_json, ent2id, rel2id)
                            except Exception as e_again:
                                # if g_type == 'ui':
                                #     modify_json = SGLenhance()
                                if "Expecting" in str(e_again):
                                    id_json = json_data
                                else:
                                    print('failed!')
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
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--start_step",
        type=int,
        default="0",
    )
    parse.add_argument(
        "--end_step",
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

    enhanced_graph_path = "AmazonBook_50_3_e0"

    enhanced_graph_info = main(1, config.start_step, config.end_step, enhanced_graph_path, ent2id, rel2id)

    # with open(f"./saved_graphs/{enhanced_graph_path}_enhanced.pkl", "wb") as f:
    #     pickle.dump(enhanced_graph_info, f)
