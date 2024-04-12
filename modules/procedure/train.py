import numpy as np
from collections import deque
from tqdm import trange
from ..utils import triples_transfer_to_graph

def Train(args, model, data_loader, rec_data, extractor, optimizer, device):
    steps_per_epoch = len(data_loader)
    losses = deque([], steps_per_epoch)
    epoch_counter = trange(args['start_epoch'], args['epoch'], ncols=0)
    for e in epoch_counter:
        for (users, pos_items, reviews) in data_loader:
            neg_items = rec_data.negative_sample(users, pos_items)
            subgraphs = extractor.sample_subgraph(['uu', 'ui', 'ii'], users, pos_items, neg_items)
            edge_indexs, edge_types = triples_transfer_to_graph(subgraphs)

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