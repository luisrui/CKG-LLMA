import numpy as np
import torch
from collections import OrderedDict
from ..data import RecTrainDataset, KGRecDataset, data_config
from ..utils import *

# def test_one_batch(args, X):
#     sorted_items = X[0].numpy()
#     groundTrue = X[1]
#     r = getLabel(groundTrue, sorted_items)
#     pre, recall, ndcg = [], [], []
#     for k in args['topks']:
#         ret = RecallPrecision_ATk(groundTrue, r, k)
#         pre.append(ret['precision'])
#         recall.append(ret['recall'])
#         ndcg.append(NDCGatK_r(groundTrue,r,k))
#     return {'recall':np.array(recall), 
#             'precision':np.array(pre), 
#             'ndcg':np.array(ndcg)}

def Test(args, recdataset : RecTrainDataset, model, mode : str, device):
    print(f'Model Evaluating for {mode} set')
    u_batch_size = args['test_u_batch_size']
    model.eval()
    testset = recdataset.get_wrapped_set(mode)
    results = {'precision': np.zeros(len(args['topks'])),
               'recall': np.zeros(len(args['topks'])),
               'ndcg': np.zeros(len(args['topks']))}
    max_K = max(args['topks'])

    data_struct = DataStruct()
    metric_class = {
        'recall': Recall(args),
        'precision': Precision(args),
        'ndcg': NDCG(args)
    }

    with torch.no_grad():
        users = list(testset.keys())
        #users_list, rating_list, groundTrue_list = [], [], []
        #total_batch = len(users) // u_batch_size + 1
        if model.__class__.__name__ == 'LightGCN':
            pass
        else:
            all_edge_index, all_edge_type = recdataset.get_UIinteraction()
            all_edge_index = all_edge_index.to(device)
            all_edge_type = all_edge_type.to(device)
        for batch_users in minibatch(users, batch_size=u_batch_size):
            trainPos = recdataset.get_pos(batch_users)
            groundTrue = [testset[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)

            if model.__class__.__name__ == 'LightGCN':
                rating = model.getUsersRating(batch_users_gpu)
            else:
                rating = model.getUsersRating(batch_users_gpu, all_edge_index, all_edge_type)
            #rating = model.getPretrainedRating(batch_users_gpu)
            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(trainPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(np.array(items) - args['num_users'])
            rating[exclude_index, exclude_items] = -np.inf
            _, rating_K = torch.topk(rating, k=max_K)

            pos_matrix = torch.zeros_like(rating, dtype=torch.int)
            for range_i, items in enumerate(groundTrue):
                pos_matrix[range_i, np.array(items) - args['num_users']] = 1
            #pos_matrix[exclude_index, exclude_items] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=rating_K)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            data_struct.update_tensor("rec.topk", result)

            rating = rating.cpu().numpy()
            del rating

            # users_list.append(batch_users)
            # rating_list.append(rating_K.cpu())
            # groundTrue_list.append(groundTrue)

        result_dict = OrderedDict()
        #metric_val = precision.calculate_metric(data_struct)
        for metric in args["metrics"]:
            metric_val = metric_class[metric].calculate_metric(data_struct)
            result_dict.update(metric_val)

        # assert total_batch == len(users_list)
        # X = zip(rating_list, groundTrue_list)
        # pre_results = []
        # for x in X:
        #     pre_results.append(test_one_batch(args, x))

        # for result in pre_results:
        #     results['recall'] += result['recall']
        #     results['precision'] += result['precision']
        #     results['ndcg']  += result['ndcg']
        # results['recall'] /= float(len(users))
        # results['precision'] /= float(len(users))
        # results['ndcg'] /= float(len(users))

        # print("Results:")
        # print(f"Precision: {[f'{p:.6f}' for p in results['precision']]}")
        # print(f"Recall: {[f'{r:.6f}' for r in results['recall']]}")
        # print(f"NDCG: {[f'{n:.6f}' for n in results['ndcg']]}")

        print(result_dict)
        return results
