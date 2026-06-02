import numpy as np
import torch
from collections import OrderedDict
from ..data import RecTrainDataset, KGRecDataset
from ..utils import *

def test_one_batch(args, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args['topks']:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def Test(args, recdataset : RecTrainDataset, model, mode : str):
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

    num_users = args['num_users']
    num_items = args['num_items']

    with torch.no_grad():
        users = list(testset.keys())
        for batch_users in minibatch(users, batch_size=u_batch_size):
            trainPos = recdataset.get_pos(batch_users)
            groundTrue = [testset[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args['device'])

            rating = model.getUsersRating(batch_users_gpu)
            
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

        result_dict = OrderedDict()
        #metric_val = precision.calculate_metric(data_struct)
        for metric in args["metrics"]:
            metric_val = metric_class[metric].calculate_metric(data_struct)
            result_dict.update(metric_val)

        print(result_dict)
        return result_dict

def Test_origin(args, recdataset, Recmodel, mode):
    u_batch_size = args['test_u_batch_size']
    testset = recdataset.get_wrapped_set(mode)
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(args['topks'])
    results = {'precision': np.zeros(len(args['topks'])),
               'recall': np.zeros(len(args['topks'])),
               'ndcg': np.zeros(len(args['topks']))}
    with torch.no_grad():
        users = list(testset.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = recdataset.get_pos(batch_users)
            groundTrue = [testset[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args['device'])

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(np.array(items) - args['num_users'])
            rating[exclude_index, exclude_items] = -np.inf
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(args, x))

        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print(results)
        return results
