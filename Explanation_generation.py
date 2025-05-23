from modules.data import KGRecDataset, RecTrainDataset
from modules.utils import read_yaml, print_yaml, set_random_seed
from modules.model import CKG_LLMA
from modules.explanation import build_reason_paths, generate_prompt, call_llm_api

import argparse
import torch
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/argsML_model.yaml")
    args = parser.parse_args()
    args = read_yaml(path=args.config)
    print_yaml(args)

    set_random_seed(args["seed"])
    device = "cuda:" + str(args["cuda"]) if int(args["cuda"]) >= 0 else "cpu"
    args["device"] = device

    kg_data = KGRecDataset(args)
    rec_data = RecTrainDataset(args)
    args.update({'ent2id': kg_data.ent2id, 'rel2id': kg_data.rel2id})

    model = CKG_LLMA(args=args, rec_data=rec_data, kg_data=kg_data)
    if args["load_path"]:
        model.load_checkpoint(args["load_path"], device)
    model.eval()

    eval_pairs = rec_data.sample_user_item_pairs(split="test", num_samples=20)
    results = []

    for (user, item) in tqdm(eval_pairs, desc="Generating Explanations"):
        G_UI = kg_data.get_user_interactions(user)
        G_II = kg_data.get_item_to_item_edges(user_items=G_UI, target_item=item)
        G_IA = kg_data.get_attributes_for_paths(G_II)

        paths = build_reason_paths(user, item, G_UI, G_II, G_IA)
        if not paths:
            continue
        confidence_scores = model.get_confidence_scores(paths)
        prompt = generate_prompt(paths, confidence_scores, G_UI)

        if args.get("use_api", False):
            explanation = call_llm_api(prompt)
        else:
            explanation = "[MOCK] LLM would generate: " + prompt[:100]

        results.append({
            "user": user,
            "item": item,
            "explanation": explanation,
            "paths": paths,
            "confidence_scores": confidence_scores,
            "prompt": prompt
        })

    with open("outputs/explanations_output.json", "w") as f:
        json.dump(results, f, indent=2)