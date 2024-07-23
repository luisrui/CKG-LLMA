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

    # kg_train_data = KGDataset(args)
    # LLM_rectify_data = LLMRectifyDataset(args)
    kg_graph_data = KGRecDataset(args)
    extractor = Extractor(args=args, srcKG=kg_graph_data, recData=rec_data)

    Rec_data_loader = torch.utils.data.DataLoader(
        rec_data,
        batch_size=args["extract_batch_size"],
        shuffle=True,
        num_workers=args["dataloader_n_workers"],
    )

    # Recmodel = KLMCR(
    #     args = args,
    #     rec_data = rec_data,
    #     kg_data = kg_train_data
    # )
    # if args["load_path"] and len(args["load_path"]) > 0:
    #     Recmodel.load_checkpoint(args["load_path"], device)
    # Recmodel = Recmodel.to(device)
    # contrast_model = Contrast(args, Recmodel, rec_data).to(device)
    # optimizer = torch.optim.Adam(Recmodel.parameters(), lr=args["learning_rate"])
    # # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    # scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma = 0.2)

    print("topks selected: ", args["topks"])

    if args['Train']:
        #Pretrain_KG_Embeddings(50, args, Recmodel, Rec_data_loader, rec_data, optimizer, scheduler, device)
        #TrainwithGraph(27, args, Recmodel, rec_data, extractor, optimizer, scheduler, device)
        #Train(args, Recmodel, rec_data, kg_train_data, LLM_rectify_data, contrast_model, optimizer, scheduler)
        #result = Test(args, rec_data, Recmodel, "test", device)
        Generate_subgraphs(1, args, Rec_data_loader, extractor, rec_data)
        #Prompt_Length_Test(50, 'Llama3', Rec_data_loader, extractor, rec_data)
        #Recmodel.generate_entity_relation_embeddings(save_path='checkpoint/saved_embs/')
    else:
        result = Test(args, rec_data, Recmodel, "test", device)
