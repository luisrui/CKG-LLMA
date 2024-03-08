import torch
from torch import nn as nn
import random

class Extractor():
    def __init__(self, num_ent, kg, args):
        self.n_iter = args['n_iter']
        self.num_ent = num_ent
        self.max_neighbors = args['max_sample_neighbors']
        self.kg = kg
        self.batch_size = args['batch_size']
        self._gen_adj()

    def _gen_adj(self):
        """
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        """
        self.adj_ent = torch.empty(self.num_ent, self.max_neighbors, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.max_neighbors, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) > self.max_neighbors:
                neighbors = random.sample(population=self.kg[e], k=self.max_neighbors)
            else:
                neighbors = random.choices(population=self.kg[e], k=self.max_neighbors)
            
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
    
    def _get_neighbors(self, v):  # -> tuple[list, list]:
        """
        v is batch sized indices for items
        v: [batch_size, 1]
        """
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = (
                torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1))
            )
            neighbor_relations = (
                torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1))
            )
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations