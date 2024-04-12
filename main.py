from modules.data import *
from modules.utils import *
from modules.model import *
from modules.procedure import Test, Train

from collections import deque
from tqdm import trange
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
                        num_user=data_config[args["data"]["name"]]['num_users'], 
                        num_items=data_config[args["data"]["name"]]['num_users'], 
                        ent2id=rec_data.ent2id, 
                        rel2id=rec_data.rel2id, 
                        srcKG=kg_data)

    kg_data_loader = torch.utils.data.DataLoader(rec_data, batch_size=args['batch_size'], shuffle=True, num_workers=args['dataloader_n_workers'])

    model = Model(args=args, 
                  norm_adj=rec_data.get_norm_adj, 
                  kg=kg_data.get_struc_dataset, 
                  ent2id=rec_data.ent2id, 
                  rel2id=rec_data.rel2id, 
                  device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    #Train(args, model, kg_data_loader, rec_data, extractor, optimizer, device)
    result = Test(args, rec_data, kg_data, model, 'valid', device)

