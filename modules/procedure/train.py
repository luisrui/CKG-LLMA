import os
import time
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque, defaultdict
#from vllm import LLM, SamplingParams
from tqdm import trange
#from ..data import LLM_import_path
from ..data import ReshufflingLoader
from ..model import Contrast
from ..utils import triples_transfer_to_graph, WandBLogger, Generate_rectify_info
from .test import Test, Test_origin
from .procedure import *


def Train(
        args, 
        model,
        rec_data, 
        kg_train_data,
        LLM_rectify_data,
        contrast_model : Contrast, 
        optimizer, 
        scheduler):
    if args['wandb']:
        logger = WandBLogger(
            config=WandBLogger.get_default_config(),
            variant=args,
        )
    device = args['device']
    best_result = dict({'recall@10' : 0.})

    #Infoloader = ReshufflingLoader(len(LLM_rectify_data), batch_size=args['Recinfo_batch_size'], shuffle=True, num_workers=12, drop_last=True)
    for epoch in tqdm(range(args['start_epoch'], args['epoch']), disable=False):
        # KGE learning
        if epoch%args['train_interval_kge'] == 0 and args['train_kge']:
            print("Train KGE:")
            kge_loss = TransE_train(args, kg_train_data, model, optimizer, device)
            #kge_loss = TransR_train(args, kg_train_data, model, optimizer, device)
            print(f"trans Loss: {kge_loss:.3f}")
        else:
            kge_loss = 0. 

        # SSL learning(Contrstive + BPR)
        #rectify_info = LLM_rectify_data.generate_batch(next(Infoloader))
        if LLM_rectify_data:
            rectify_info = LLM_rectify_data.generate_batch(batch_size=args['Recinfo_batch_size'])
        else:
            rectify_info = None
        contrast_views = contrast_model.get_views(rectify_info=rectify_info)
        # joint learning part
        print("[Joint Learning]")
        (total_loss, bpr_loss, con_loss, adj_loss) = \
            BPR_train_contrast(args, rec_data, model, contrast_model, contrast_views, optimizer, epoch+1)

        if (epoch + 1) % args['eval_interval'] == 0:
            print("[Valid]")
            results_val = Test(args, rec_data, model, 'valid')

            print("[TEST]")
            results_test = Test(args, rec_data, model, 'test')
            
            if args['wandb']:
                logger.log({
                    'epoch' : epoch + 1,
                    'Precision(Val)' : results_val['precision@10'],
                    'Recall(Val)' : results_val['recall@10'],
                    'ndcg@10(Val)' : results_val['ndcg@10'],
                    'Precision(Test)' : results_test['precision@10'],
                    'Recall(Test)' : results_test['recall@10'],
                    'ndcg@10(Test)' : results_test['ndcg@10'],
                    'Total loss' : total_loss,
                    'BPR loss' : bpr_loss,
                    'CON loss' : con_loss,
                    'KGE loss' : kge_loss,
                    'ADJ_loss' : adj_loss
                })

            if results_test["recall@10"] > best_result["recall@10"]:
                stopping_step = 0
                best_result = results_test
                print("Find a better model")
                model.save_checkpoint(
                    args['save_path'] + f"{type(model).__name__}_{args['data']['name']}.ckpt")

            else:
                if epoch >= 100:
                    stopping_step += 1
                    if stopping_step >= args['early_stop_epoch']:
                        print(f"early stop triggerd at epoch {epoch}, the best result is {best_result}")
                        break

        scheduler.step()

    print(f'Finish Training, the best result is {best_result}')

def TrainLightGCN(args, model, data_loader, rec_data, extractor, optimizer, scheduler, device):
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
            result = Test(args, rec_data, model, extractor, "valid", device)

        if (e + 1) % args["save_interval"] == 0:
            save_model_name = os.path.join(
                args["save_path"]
                + f"checkpoint/epoch_{e + 1}_{type(model).__name__}_{args['special_save_hyper']}.ckpt"
            )
            model.save_checkpoint(save_model_name)

def TrainwithGraph(total_epoch, args, model, rec_data, extractor, optimizer, scheduler, device):
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

            interaction = {
                'users': users,
                'pos_items': pos_items,
                'neg_items': neg_items,
                'g_types' : ['uu', 'ui', 'ii'],
                'eh_g_types' : ['ui', 'ii']
            }

            loss, bpr_loss, kge_loss, con_loss = model(
                interaction,
                edge_indexs, 
                edge_types,
                eh_edge_indexs, 
                eh_edge_types, 
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
            result = Test(args, rec_data,  model, extractor, "valid", device)
        elif (e + 1) % args["eval_interval"] == 0:
            result = Test(args, rec_data,  model, extractor, "valid", device)

def Generate_subgraphs(epochs, args, data_loader, extractor, rec_data):
    '''
    Generate subgraphs for training.
    '''
    subgraphs_collect = dict()
    batch_size = args["extract_batch_size"]
    total_steps = 3000
    step = 0
    for e in tqdm(range(args['start_epoch'], epochs)):

        epoch_info = defaultdict(list)
        
        for users, pos_items, neg_items in tqdm(data_loader):
            #neg_items = rec_data.negative_sample(users, pos_items)
            subgraphs = extractor.sample_subgraph_origin(
                ["ui", 'ii'], users, pos_items, neg_items
            )
            print(subgraphs[0].__len__(), subgraphs[1].__len__())
            #epoch_info['uu'].append(subgraphs[0])
            epoch_info['ui'].append(subgraphs[0])
            epoch_info['ii'].append(subgraphs[1])
            epoch_info['users'].append(users)
            epoch_info['pos_items'].append(pos_items)
            epoch_info['neg_items'].append(neg_items)
            step += 1
            if step == total_steps:
                break

        subgraphs_collect[e] = epoch_info

        with open(f'./saved_graphs/{rec_data.name}_{batch_size}_{args["max_sample_neighbors"]}_e{e}.pkl', 'wb') as f:
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


            