from modules.utils import *
from modules.dataset import *
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

    kg_data_loader = torch.utils.data.DataLoader(rec_data, batch_size=args['batch_size'], shuffle=True, num_workers=args['dataloader_n_workers'])

    for (pos_users, pos_items, reviews) in kg_data_loader:
        sampled_users, sampled_items = rec_data.negative_sample(pos_users, pos_items)
        