import os
import time
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque, defaultdict
from vllm import LLM, SamplingParams
from tqdm import trange
from ..data import LLM_import_path
from ..utils import triples_transfer_to_graph, Read_prompt
from .test import Test


def Train(args, model, data_loader, rec_data, kg_data, extractor, optimizer, scheduler, device):
    steps_per_epoch = len(data_loader)
    losses = deque([], steps_per_epoch)
    losses_bpr = deque([], steps_per_epoch)
    losses_kge = deque([], steps_per_epoch)
    epoch_counter = trange(args["start_epoch"], args["epoch"], ncols=0)
    for e in epoch_counter:
        for users, pos_items, reviews in data_loader:
            neg_items = rec_data.negative_sample(users, pos_items)
            subgraphs = extractor.sample_subgraph(
                ["uu", "ui", "ii"], users, pos_items, neg_items
            )
            edge_indexs, edge_types = triples_transfer_to_graph(subgraphs)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            loss, bpr_loss, kge_loss = model(
                edge_indexs, edge_types, users, pos_items, neg_items
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_bpr.append(bpr_loss.item())
            losses_kge.append(kge_loss.item())
            epoch_counter.set_description(
                "Epoch %d |loss: %.3f |bpr_loss: %.3f |kge_loss: %.3f"
                % (e + 1, np.mean(losses), np.mean(losses_bpr), np.mean(losses_kge))
            )
        if scheduler:
            scheduler.step()

        if e % args["save_interval"] == 0:
            save_model_name = os.path.join(
                args["save_path"]
                + f"checkpoint/epoch_{e + 1}_{type(model).__name__}.ckpt"
            )
            model.save_checkpoint(save_model_name)
            result = Test(args, rec_data,  model, "valid", device)
        elif e % args["eval_interval"] == 0:
            result = Test(args, rec_data,  model, "valid", device)

def TrainLightGCN(args, model, data_loader, rec_data, optimizer, scheduler, device):
    steps_per_epoch = len(data_loader)
    losses = deque([], steps_per_epoch)
    losses_bpr = deque([], steps_per_epoch)
    epoch_counter = trange(args["start_epoch"], args["epoch"], ncols=0)
    for e in epoch_counter:
        for users, pos_items, reviews in data_loader:
            neg_items = rec_data.negative_sample(users, pos_items)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            loss, bpr_loss = model(users, pos_items, neg_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_bpr.append(bpr_loss.item())
            epoch_counter.set_description(
                "Epoch %d |loss: %.6f |bpr_loss: %.6f"
                % (e + 1, np.mean(losses), np.mean(losses_bpr))
            )
        if scheduler:
            scheduler.step()

        if (e + 1) % args["eval_interval"] == 0:
            result = Test(args, rec_data, model, "valid", device)

        if (e + 1) % args["save_interval"] == 0:
            save_model_name = os.path.join(
                args["save_path"]
                + f"checkpoint/epoch_{e + 1}_{type(model).__name__}.ckpt"
            )
            model.save_checkpoint(save_model_name)

def TrainwithGraph(total_epoch, args, model, rec_data, optimizer, scheduler, device):
    epoch_counter = trange(args["start_epoch"], total_epoch, ncols=0)

    start_time = time.time()
    print(f'Loading saved graphs for {rec_data.name}...')
    graphs_info = pickle.load(open(f'./saved_graphs/{rec_data.name}_{args["batch_size"]}_{args["max_sample_neighbors"]}_e{total_epoch-1}.pkl', 'rb'))
    end_time = time.time()
    print(f'Loaded {total_epoch} epochs of graphs, time is {end_time - start_time} seconds.')

    start_time = time.time()
    print(f'Loading enhanced graphs for {rec_data.name}...')
    if args['end_step'] == 0:
        enhanced_info = pickle.load(open(f'./saved_graphs/{rec_data.name}_{args["batch_size"]}_{args["max_sample_neighbors"]}_e{total_epoch-1}_enhanced.pkl', 'rb'))
    else:
        enhanced_info = pickle.load(open(f'./saved_graphs/{rec_data.name}_{args["batch_size"]}_{args["max_sample_neighbors"]}_e{total_epoch-1}_enhanced_{args["start_step"]}_{args["end_step"]}.pkl', 'rb'))
    end_time = time.time()
    print(f'Loaded {total_epoch} epochs of enhanced graphs, time is {end_time - start_time} seconds.')

    steps_per_epoch = len(graphs_info[0]['users'])
    losses = deque([], steps_per_epoch)
    losses_bpr = deque([], steps_per_epoch)
    losses_kge = deque([], steps_per_epoch)
    losses_con = deque([], steps_per_epoch)
    
    for e in epoch_counter:
        epoch_info = graphs_info[e]
        users_epoch, pos_items_epoch, neg_items_epoch = epoch_info['users'], epoch_info['pos_items'], epoch_info['neg_items']
        uu_graphs, ui_graphs, ii_graphs = epoch_info['uu'], epoch_info['ui'], epoch_info['ii']
        ui_enhanced_graphs, ii_enhanced_graphs = enhanced_info[e]['ui'], enhanced_info[e]['ii']
        step = 0
        for (users, 
             pos_items,
             neg_items,
             uu_graph,
             ui_graph,
             ii_graph,
             eh_ui_graph,
             eh_ii_graph) in zip(users_epoch, pos_items_epoch, neg_items_epoch, uu_graphs, ui_graphs, ii_graphs, ui_enhanced_graphs, ii_enhanced_graphs):
            
            edge_indexs, edge_types = triples_transfer_to_graph([uu_graph, ui_graph, ii_graph])
            eh_edge_indexs, eh_edge_types = triples_transfer_to_graph([eh_ui_graph, eh_ii_graph])

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            loss, bpr_loss, kge_loss, con_loss = model(
                ['uu', 'ui', 'ii'], 
                edge_indexs, 
                edge_types,
                ['ui', 'ii'],
                eh_edge_indexs, 
                eh_edge_types, 
                users, 
                pos_items, 
                neg_items
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_bpr.append(bpr_loss.item())
            losses_kge.append(kge_loss.item())
            losses_con.append(con_loss.item())
            epoch_counter.set_description(
                "Epoch %d |step %d |loss: %.6f |bpr_loss: %.6f |kge_loss: %.6f|con_loss: %.6f"
                % (e + 1, step, np.mean(losses), np.mean(losses_bpr), np.mean(losses_kge), np.mean(losses_con))
            )
            step += 1
        if scheduler:
            scheduler.step()

        if (e + 1) % args["save_interval"] == 0:
            save_model_name = os.path.join(
                args["save_path"]
                + f"checkpoint/epoch_{e + 1}_{type(model).__name__}_{rec_data.name}_{args['special_save_hyper']}.ckpt"
            )
            model.save_checkpoint(save_model_name)
            result = Test(args, rec_data,  model, "valid", device)
        elif (e + 1) % args["eval_interval"] == 0:
            result = Test(args, rec_data,  model, "valid", device)

def Generate_subgraphs(epochs, args, data_loader, extractor, rec_data):
    '''
    Generate subgraphs for training.
    '''
    subgraphs_collect = dict()

    for e in tqdm(range(args['start_epoch'], epochs)):

        epoch_info = defaultdict(list)
        
        for users, pos_items, _ in tqdm(data_loader):
            neg_items = rec_data.negative_sample(users, pos_items)
            subgraphs = extractor.sample_subgraph(
                ["uu", "ui", "ii"], users, pos_items, neg_items
            )
            print(subgraphs[0].__len__(), subgraphs[1].__len__(), subgraphs[2].__len__())
            epoch_info['uu'].append(subgraphs[0])
            epoch_info['ui'].append(subgraphs[1])
            epoch_info['ii'].append(subgraphs[2])
            epoch_info['users'].append(users)
            epoch_info['pos_items'].append(pos_items)
            epoch_info['neg_items'].append(neg_items)

        subgraphs_collect[e] = epoch_info

        with open(f'./saved_graphs/{rec_data.name}_{args["batch_size"]}_{args["max_sample_neighbors"]}_e{e}_v2.pkl', 'wb') as f:
            pickle.dump(subgraphs_collect, f)

def Pretrain_KG_Embeddings(total_epoch, args, model, data_loader, rec_data, optimizer, scheduler, device):
    steps_per_epoch = len(data_loader)
    losses = deque([], steps_per_epoch)
    losses_kge = deque([], steps_per_epoch)
    epoch_counter = trange(args["start_epoch"], total_epoch, ncols=0)
    subgraphs_collect = pickle.load(open(f'./saved_graphs/{rec_data.name}_{args["batch_size"]}_{args["max_sample_neighbors"]}_e{total_epoch-1}.pkl', 'rb'))

    for e in epoch_counter:
        for users, pos_items, reviews in data_loader:
            neg_items = rec_data.negative_sample(users, pos_items)
            subgraphs = subgraphs_collect[e]
            edge_indexs, edge_types = triples_transfer_to_graph(subgraphs)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            loss, bpr_loss, kge_loss = model.pretrain_kg_embeddings(
                edge_indexs, edge_types
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_kge.append(kge_loss.item())
            epoch_counter.set_description(
                "Epoch %d |loss: %.6f |kge_loss: %.6f"
                % (e + 1, np.mean(losses), np.mean(losses_kge))
            )
        if scheduler:
            scheduler.step()

        if e % args["save_interval"] == 0:
            save_model_name = os.path.join(
                args["save_path"]
                + f"checkpoint/epoch_{e + 1}_{type(model).__name__}_{rec_data.name}.ckpt"
            )
            model.save_checkpoint(save_model_name)
            result = Test(args, rec_data,  model, "valid", device)
        elif e % args["eval_interval"] == 0:
            result = Test(args, rec_data,  model, "valid", device)


            