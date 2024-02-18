import pandas as pd
import random
import torch 
from sklearn.preprocessing import LabelEncoder

config_file = {
            'movie' : {
                'sep' : ',',
                'threshold' : 4.0
            },
            'music' : {
                'sep' : '\t',
                'threshold' : 0.0
            }
        }

class DataFormatter:
    '''
    Data Formatter class which formats knowledge graphs and rating file for training / knowledge graph dictionary
    '''
    def __init__(self, args):
        self.args = args
        
        self.raw_kg = pd.read_csv(args['data']['kg_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
        self.item2id = pd.read_csv(args['data']['item2id_path'], sep='\t', header=None, names=['item','id'])
        
        sep = config_file[args['data']['name']]['sep']
        raw_rating = pd.read_csv(args['data']['rating_path'], sep=sep, names=['userID', 'itemID', 'rating', 'timestamp'], skiprows=1)
        raw_rating = raw_rating[raw_rating['itemID'].isin(self.item2id['item'])]
        raw_rating.reset_index(inplace=True, drop=True)
        self.raw_rating = raw_rating

        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()

    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.raw_rating['userID'])
        # item2id['id'] and raw_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(pd.concat([self.item2id['id'], self.raw_kg['head'], self.raw_kg['tail']]))
        self.relation_encoder.fit(self.raw_kg['relation'])
        
        # encode raw_kg
        self.raw_kg['head'] = self.entity_encoder.transform(self.raw_kg['head'])
        self.raw_kg['tail'] = self.entity_encoder.transform(self.raw_kg['tail'])
        self.raw_kg['relation'] = self.relation_encoder.transform(self.raw_kg['relation'])

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # raw_rating update
        fmt_ratingset = pd.DataFrame()
        fmt_ratingset['userID'] = self.user_encoder.transform(self.raw_rating['userID'])
        
        # update to new id
        item2id_dict = dict(zip(self.item2id['item'], self.item2id['id']))
        self.raw_rating['itemID'] = self.raw_rating['itemID'].apply(lambda x: item2id_dict[x])
        fmt_ratingset['itemID'] = self.entity_encoder.transform(self.raw_rating['itemID'])
        fmt_ratingset['label'] = self.raw_rating['rating'].apply(lambda x: 0 if x < config_file[self.args['data']['name']]['threshold'] else 1)
        
        # negative sampling
        fmt_ratingset = fmt_ratingset[fmt_ratingset['label']==1]
        # fmt_ratingset requires columns to have new entity ID
        full_item_set = set(range(len(self.entity_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        for user, group in fmt_ratingset.groupby(['userID']):
            item_set = set(group['itemID'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        fmt_ratingset = pd.concat([fmt_ratingset, negative])
        
        fmt_ratingset = fmt_ratingset.sample(frac=1, replace=False, random_state=999)
        fmt_ratingset.reset_index(inplace=True, drop=True)
        print('Done')
        return fmt_ratingset
    
    def _construct_kg(self) -> dict:
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.raw_kg)):
            head = self.raw_kg.iloc[i]['head']
            relation = self.raw_kg.iloc[i]['relation']
            tail = self.raw_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
    
    def load_kg(self) -> dict:
        '''
        Load knowledge graph dictionary format
        '''
        return self._construct_kg() 
    
    def load_dataset(self):
        '''
        Load rating dataset with negative sampled samples and binary classification labels
        '''
        return self._build_dataset()
    
class GraphRecDataset(torch.utils.data.Dataset):
    '''
    The dataset contains user-item recommendation pairs and item-related knowledge graph 
    '''
    def __init__(self, args):
        self.dataformatter = DataFormatter(self, args)
        self.itemkg = self.dataformatter.load_kg()
        self.rating_dataset = self.dataformatter.load_dataset() 