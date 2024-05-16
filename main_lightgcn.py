from modules.data import *
from modules.utils import *
from modules.model import *
from modules.procedure import *

import torch
import argparse
from collections import deque
from tqdm import trange
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--argpath",
        type=str,
        default="argsML.yaml",
        help="the relative path of argments file",
    )
    args = parse.parse_args()

    args = read_yaml(path=args.argpath)

    args.update({
        'num_users' : data_config[args["data"]["name"]]["num_users"],
        'num_items' : data_config[args["data"]["name"]]["num_items"]
    })

    device = "cuda:" + str(args["cuda"]) if int(args["cuda"]) >= 0 else "cpu"
    set_random_seed(seed=args["seed"])
    kg_data = KGRecDataset(args)
    rec_data = RecTrainDataset(args)
    # extractor = Extractor(
    #     args=args,
    #     num_user=args["num_users"],
    #     num_items=args["num_items"],
    #     ent2id=rec_data.ent2id,
    #     rel2id=rec_data.rel2id,
    #     srcKG=kg_data,
    #     recData=rec_data,
    # )

    kg_data_loader = torch.utils.data.DataLoader(
        rec_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["dataloader_n_workers"],
    )

    # model = Model(
    #     args=args,
    #     norm_adj=rec_data.get_norm_adj,
    #     kg=kg_data.get_struc_dataset,
    #     ent2id=rec_data.ent2id,
    #     rel2id=rec_data.rel2id,
    #     device=device,
    # )

    model = LightGCN(num_users=args['num_users'],
                    num_items=args['num_items'],
                    embed_dim=args['embedding_dim'],
                    norm_adj=rec_data.get_norm_adj.to(device),
                    n_layers=args['n_layers_lightgcn'],
                    batch_size=args['batch_size'])

    if args["load_path"] and len(args["load_path"]) > 0:
        model.load_checkpoint(args["load_path"], device)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    # scheduler = None

    print("topks selected: ", args["topks"])

    if args['Train']:
        #Pretrain_KG_Embeddings(50, args, model, kg_data_loader, rec_data, optimizer, scheduler, device)
        #TrainwithGraph(6, args, model, rec_data, optimizer, scheduler, device)
        #Train(args, model, kg_data_loader, rec_data, kg_data, extractor, optimizer, scheduler, device)
        TrainLightGCN(args, model, kg_data_loader, rec_data, optimizer, scheduler, device)
        #result = Test(args, rec_data, model, "test", device)
        #Generate_subgraphs(50, args, kg_data_loader, extractor, rec_data)
        #Prompt_Length_Test(50, 'Llama3', kg_data_loader, extractor, rec_data)
        #model.generate_entity_relation_embeddings(save_path='checkpoint/saved_embs/')
    else:
        result = Test(args, rec_data, model, "test", device)
