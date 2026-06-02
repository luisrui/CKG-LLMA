class Sampler(object):
    """Base class for all sampler to sample negative items.
    """
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class PairwiseSampler(Sampler):
    '''
    Pairwise sampler to sample un-correlated items for each user. 
    '''
    def __init__(self, 
                 name : str,
                 dataset:pd.DataFrame, 
                 ent2id:dict, 
                 rel2id:dict, 
                 num_neg:int = 1
                 ):
        
        super(PairwiseSampler, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.name = name
        self.e2id = ent2id
        self.r2id = rel2id
        self.num_user = data_config[self.name]['num_users']
        self.num_item = data_config[self.name]['num_items']
        self.num_neg = num_neg

        self.u_of_i = defaultdict(list)
        #self.i_of_u = defaultdict(list)
        if os.path.exists(f'./dataset/{name}/pre_saved/'):
            self._load_uoi()
        else:
            self._count_uoi(dataset[data_config[self.name]['user']], dataset[data_config[self.name]['item']])
    
    def _load_uoi(self):
        self.u_of_i = torch.load(f'./dataset/{self.name}/pre_saved/u_of_i.pt')

    def _count_uoi(self, users, items):
        for u, i in zip(users, items):
            u_id = self.e2id[u]
            i_id = self.e2id[i]
            self.u_of_i[u_id].append(i_id)

        for u in self.u_of_i.keys():
            self.u_of_i[u] = np.array(list(set(self.u_of_i[u])))
        
        os.makedirs(f'./dataset/{self.name}/pre_saved/', exist_ok=True)
        torch.save(self.u_of_i, f'./dataset/{self.name}/pre_saved/u_of_i.pt')
    
    def UniformSample_original(self, seed, users, items):
        if sample_ext:
            SampleFunction.seed(seed)
            # users = np.asarray(users, dtype=np.int32)
            # items = np.asarray(items, dtype=np.int32)
            S = SampleFunction.sample_negative(users, items, self.u_of_i, self.num_user, self.num_item)
        else:
            S = self._uniformSample_original_python(users, items)
        return S

    def _uniformSample_original_python(self, users, items):
        """
        the original impliment of BPR Sampling in LightGCN
        :return:
            np.array
        """
        S = []
        for idx, (user, positem) in enumerate(zip(users, items)):
            user = user.item()
            posForUser = self.u_of_i[user]
            while True:
                negitem = np.random.randint(self.num_user, self.num_user + self.num_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])

        return np.array(S)
    
class NegativeSampler(Sampler):
    '''
    Negative sampler to sample negative tails for each fact triples in KGRecdataset.
    '''
    def __init__(self):
        super(NegativeSampler).__init__()


    def Triples_neg_sample(self, edge_index, edge_type):
        batch_h, batch_t, batch_r = edge_index[0], edge_index[1], edge_type
		#len_triples = batch_h.__len__()
        batch_data = {}
        if self.sampling_mode == "normal":
            batch_data['mode'] = "normal"
            batch_h_sample = np.repeat(batch_h.view(-1, 1).cpu().numpy(), 1 + self.neg_ent, axis = -1)
            batch_t_sample = np.repeat(batch_t.view(-1, 1).cpu().numpy(), 1 + self.neg_ent, axis = -1)
            batch_r_sample = np.repeat(batch_r.view(-1, 1).cpu().numpy(), 1 + self.neg_ent, axis = -1)
            for idx, (h, t, r) in enumerate(zip(batch_h, batch_t, batch_r)):
                last = 1
                if self.neg_ent > 0:
                    neg_head, neg_tail = self.__triples_normal_batch(h, t, r, self.neg_ent)
                    if len(neg_head) > 0:
                        batch_h_sample[idx][last:last + len(neg_head)] = neg_head
                        last += len(neg_head)
                    if len(neg_tail) > 0:
                        batch_t_sample[idx][last:last + len(neg_tail)] = neg_tail
                        last += len(neg_tail)
            batch_h = batch_h_sample.transpose()
            batch_t = batch_t_sample.transpose()
            batch_r = batch_r_sample.transpose()

        expand_edge_index = torch.tensor(np.array([batch_h.squeeze().flatten(), batch_t.squeeze().flatten()]), dtype=torch.int32)
        expand_edge_type = torch.tensor(batch_r.squeeze().flatten(), dtype=torch.int32)
        return expand_edge_index, expand_edge_type
    
    def __triples_normal_batch(self, h, t, r, neg_size):
        neg_size_t = neg_size

        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.__triples_corrupt_tail(h, r, num_max = (neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return neg_list_t[:neg_size_t]
    
    def __triples_corrupt_tail(self, h, r, num_max = 1):
        tmp = torch.tensor(random.sample(len(self.ent2id), k=num_max))
        h_index, r_index= h.item(), r.item()
        mask = np.in1d(tmp, self.t_of_hr[(local_global_id[h_index], r_index)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg