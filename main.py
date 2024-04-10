from modules.data import *
from modules.utils import *
from modules.model import *
from modules.procedure import Test

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

    model = Model(args, rec_data.get_norm_adj, rec_data.ent2id, rec_data.rel2id, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    steps_per_epoch = len(kg_data_loader)
    losses = deque([], steps_per_epoch)
    epoch_counter = trange(args['start_epoch'], args['epoch'], ncols=0)
    for e in epoch_counter:
        for (users, pos_items, reviews) in kg_data_loader:
            neg_items = rec_data.negative_sample(users, pos_items)
            subgraphs = extractor.sample_subgraph(['uu', 'ui', 'ii'], users, pos_items, neg_items)
            edge_indexs, edge_types = triples_transfer_to_graph(subgraphs, rec_data.ent2id, rec_data.rel2id)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            loss, bpr_loss, kge_loss = model(edge_indexs, edge_types, users, pos_items, neg_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_counter.set_description("Epoch %d |loss: %.3f |bpr_loss: %.3f |kge_loss: %.3f" % (
                e + 1,
                np.mean(losses),
                bpr_loss.item(),
                kge_loss.item()
                )
            )
    
    result = Test(args, rec_data, kg_data, model, 'valid', device)

