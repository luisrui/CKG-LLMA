import openai
import argparse

from random import sample
from typing import List, Dict, Tuple
from modules.prompts.explanation_generation_prompt import *
from modules.data import *
from modules.utils import *
from modules.model import *

format_prompt_confi = exp_gen_prompt_with_confidence
format_prompt_nonconfi = exp_gen_prompt_without_confidence

def extract_paths_for_explanation(user_id: int, target_item: int, dataset) -> List[Dict]:
    paths = []
    history_items = dataset.u_of_i_all.get(user_id, [])

    if len(history_items) > 10:
        history_items = sample(history_items, 10)

    for hist_item in history_items:
        connected_items = dataset.i_of_i.get(hist_item, [])
        for rel_ii, conn_item in connected_items:
            if conn_item != target_item:
                continue
            hist_attrs = dataset.i_of_a.get(hist_item, [])
            target_attrs = dataset.i_of_a.get(target_item, [])
            hist_attr_ids = set([a for _, a in hist_attrs])
            target_attr_ids = set([a for _, a in target_attrs])
            shared_attrs = hist_attr_ids & target_attr_ids

            for attr in shared_attrs:
                rel1 = [r for r, a in hist_attrs if a == attr][0]
                rel2 = [r for r, a in target_attrs if a == attr][0]
                paths.append({
                    "user": user_id,
                    "hist_item": hist_item,
                    "target_item": target_item,
                    "attr": attr,
                    "r_hist": rel1,
                    "r_target": rel2
                })
    return paths

def construct_structured_prompt(paths: List[Dict], id2ent: Dict[int, str], id2rel: Dict[int, str], openai_key : str) -> str:
    if not paths:
        return "No explanation paths available for the given user and item."

    user = id2ent[paths[0]["user"]]
    target_item = id2ent[paths[0]["target_item"]]
    interacted_items = list({path["hist_item"] for path in paths})
    interacted_text = ', '.join([id2ent[i] for i in interacted_items])

    # Paths formatted as reasoning chains
    path_descriptions = []
    for path in paths:
        hist_item = id2ent[path["hist_item"]]
        attr = id2ent[path["attr"]]
        rel1 = id2rel[path["r_hist"]]
        rel2 = id2rel[path["r_target"]]
        description = f"User liked '{hist_item}' → ({rel1}) → '{attr}' ← ({rel2}) ← '{target_item}'"
        path_descriptions.append(description)

    paths_text = '\n'.join([f"- {p}" for p in path_descriptions])

    # prompt_confi = format_prompt_confi.format(
    #     user=user,
    #     interacted_items=interacted_text,
    #     target_item=target_item,
    #     paths=paths_text,
    #     confidence=None,
    # )

    prompt_nonconfi = format_prompt_nonconfi.format(
        user=user,
        interacted_items=interacted_text,
        target_item=target_item,
        paths=paths_text,
    )

    return prompt_nonconfi
    # = query_openai_gpt(prompt_confi, openai_key=openai_key)
    # explanation_confi = ""
    # explanation_nonconfi = query_openai_gpt(prompt_nonconfi, openai_key=openai_key)

    # return f"Explanation with confidence: {explanation_confi}\n\nExplanation without confidence: {explanation_nonconfi}"

def query_openai_gpt(prompt: str, openai_key: str, model="gpt-3.5-turbo") -> str:
    openai.api_key = openai_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant good at explaining recommendations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def get_path_confidence_scores(paths: List[Dict], model) -> List[Tuple[float, float]]:
    """
    return the confidence scores of the paths of two IA tripets
    """
    scores = []
    for path in paths:
        h_item = path["hist_item"]
        t_item = path["target_item"]
        attr = path["attr"]
        try:
            h_index = model.i2e[h_item].tolist().index(attr)
            t_index = model.i2e[t_item].tolist().index(attr)
            c1 = model.edge_confidence[h_item][h_index].item()
            c2 = model.edge_confidence[t_item][t_index].item()
        except:
            c1, c2 = 0.0, 0.0
        scores.append((c1, c2))
    return scores

def generate_explanation_for_user_item(user_id: int, item_id: int, dataset, 
                                       id2ent: Dict[int, str], id2rel: Dict[int, str], 
                                       openai_key: str) -> str:
    paths = extract_paths_for_explanation(user_id, item_id, dataset)
    explanation = construct_structured_prompt(paths, id2ent, id2rel, openai_key)
    return explanation

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="config/CKG-LLMA/argsAB_origin.yaml", help="the relative path of argments file")
    args = parse.parse_args()
    args = read_yaml(path=args.config)
    args.update({
        'num_users' : data_config[args["data"]["name"]]["num_users"],
        'num_items' : data_config[args["data"]["name"]]["num_items"]
    })
    print_yaml(args)

    device = "cuda:" + str(args["cuda"]) if int(args["cuda"]) >= 0 else "cpu"
    rec_data = RecTrainDataset(args)
    args.update({'device': device, 'ent2id' : rec_data.ent2id,'rel2id' : rec_data.rel2id,})

    background_kg = KGRecDataset(args)
    kg_train_data = KGDataset(args)
    Recmodel = CKG_LLMA(
        args = args,
        rec_data = rec_data,
        kg_data = kg_train_data
    )

    id2ent = {v: k for k, v in rec_data.ent2id.items()}
    id2rel = {v: k for k, v in rec_data.rel2id.items()}

    while True:
        user_input = input("Enter user_id and item_id (separated by space), or type 'end' to quit: ")
        if user_input.strip().lower() == 'end':
            break
        try:
            user_id_str, item_id_str = user_input.strip().split()
            user_id = int(user_id_str)
            item_id = int(item_id_str)
            explanation = generate_explanation_for_user_item(
                user_id=user_id,
                item_id=item_id,
                dataset=background_kg,
                id2ent=id2ent,
                id2rel=id2rel,
                openai_key=""
            )
            print("\n=== Explanation Generated ===")
            print(explanation)
            print("==============================\n")
        except Exception as e:
            print(f"Invalid input or error occurred: {e}\nPlease enter two integers or 'end' to quit.\n")