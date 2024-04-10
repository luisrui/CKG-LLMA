import numpy as np
import torch
from ..data import Recdataset, KGRecDataset
from ..utils import *

def test_one_batch(args, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args['topks']:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def Test(args, recdataset : Recdataset, kg : KGRecDataset, model, mode : dict, device):
    u_batch_size = args['test_u_batch_size']
    model.eval()
    testset = recdataset.get_wrapped_set(mode)
    results = {'precision': np.zeros(len(args['topks'])),
               'recall': np.zeros(len(args['topks'])),
               'ndcg': np.zeros(len(args['topks']))}
    max_K = max()
    with torch.no_grad():
        users = list(testset.keys())
        users_list, rating_list, groundTrue_list = [], [], []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = kg.get_all_pos(batch_users)
            groundTrue = [testset[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating = model.getUsersRating(batch_users_gpu)
            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(args, x))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg']  += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        print(results)
        return results