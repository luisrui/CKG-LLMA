from modules.data import *
from modules.utils import *
from modules.model import *
from modules.procedure import *

import torch
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--argpath", type=str, default="config/argsML_model.yaml", help="the relative path of argments file")
    args = parse.parse_args()
    args = read_yaml(path=args.argpath)
    args.update({
        'num_users' : data_config[args["data"]["name"]]["num_users"],
        'num_items' : data_config[args["data"]["name"]]["num_items"]
    })
    print_yaml(args)
    set_random_seed(seed=args["seed"])
    
    device = "cuda:" + str(args["cuda"]) if int(args["cuda"]) >= 0 else "cpu"
    #device = 'cpu'
    rec_data = RecTrainDataset(args)
    args.update({'device': device, 'ent2id' : rec_data.ent2id,'rel2id' : rec_data.rel2id,})

    kg_graph_data = KGRecDataset(args)
    extractor = Extractor(args=args, srcKG=kg_graph_data, recData=rec_data)

    Rec_data_loader = torch.utils.data.DataLoader(
        rec_data,
        batch_size=args["extract_batch_size"],
        shuffle=True,
        num_workers=args["dataloader_n_workers"],
    )

    print("topks selected: ", args["topks"])

    Generate_subgraphs(1, args, Rec_data_loader, extractor, rec_data)

