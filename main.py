from modules.utils import *
from modules.dataset import *
from modules.graph_extractor import *
import torch 
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--argpath', type=str, default='args.yaml', help='the relative path of argments file')
    args = parse.parse_args()
    args = read_yaml(path=args.argpath)
    
    device = 'cuda:' + str(args['cuda']) if int(args['cuda']) >= 0 else 'cpu'
    set_random_seed(seed=args['seed'])
    kg_data = KGRecDataset(args)
    rec_data = RecTrainDataset(args)
    extractor = Extractor(args=args, 
                        num_user=config[args["data"]["name"]]['num_users'], 
                        num_items=config[args["data"]["name"]]['num_users'], 
                        ent2id=rec_data.ent2id, 
                        rel2id=rec_data.rel2id, 
                        srcKG=kg_data)

    kg_data_loader = torch.utils.data.DataLoader(rec_data, batch_size=args['batch_size'], shuffle=True, num_workers=args['dataloader_n_workers'])

    for (users, pos_items, reviews) in kg_data_loader:
        neg_items = rec_data.negative_sample(users, pos_items)
        subgraphs = extractor.sample_subgraph(['uu', 'ui', 'ii'], users, pos_items, neg_items)