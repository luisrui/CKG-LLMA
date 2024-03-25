### Amazon Book Review KG Construction
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict

name = 'AmazonBookTiny'

print('loading data...')
iteminfo = pd.read_csv('./dataset/AmazonBook/books_data.csv')
#rateinfo = pd.read_csv('./dataset/AmazonBook/Books_rating.csv')
rateinfo = pd.read_csv(filepath_or_buffer=f'./dataset/{name}/data_all.csv')

### Predefined:
relation2id = {
    'liked': 0,
    'has title': 1,
    'has author': 2,
    'has category': 3,
    'has publisher': 4,
    'co-liked': 5,
    'same title': 6,
    'same author': 7,
    'same category': 8,
    'same publisher': 9
}
### Filter for positive reviews(Changing the logic for filtering the reviews)

entity_text = list() # [head, rel, tail]
### User-Item
print('generating user-item relations...')

bookname_map = defaultdict(set)
item_user_map = defaultdict(set)

for linum in tqdm(range(rateinfo.shape[0])):
    user = rateinfo.iloc[linum]['User_id']
    item = rateinfo.iloc[linum]['Id']
    title = rateinfo.iloc[linum]['Title']
    bookname_map[title].add(item)
    entity_text.append([user, 'liked', item])
    item_user_map[item].add(user) ## User-User Relations
user_list = pd.unique(rateinfo['User_id'])
item_list = pd.unique(rateinfo['Id'])
print('Number of user-item relation triples:', len(entity_text))
aboveall = len(entity_text)

## Item Features
no_attr = list()
authors_map, categories_map, publisher_map, title_map = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
print('generating item feature connections...')
for linum in tqdm(range(iteminfo.shape[0])):
    check = 0
    if pd.isna(iteminfo.iloc[linum]['Title']) == False:
        title = iteminfo.iloc[linum]['Title']
        if title in bookname_map.keys():
            for item in bookname_map[title]:
                entity_text.append([item, 'has title', title])
                title_map[title].add(item)
            if pd.isna(iteminfo.iloc[linum]['authors']) == False:
                check = 1
                authors = iteminfo.iloc[linum]['authors'][2:-2].split("', '")
                for item in bookname_map[title]:
                    for author in authors:
                        entity_text.append([item, 'has author', author])
                        authors_map[author].add(item)
            if pd.isna(iteminfo.iloc[linum]['categories']) == False:
                check = 1
                categories = iteminfo.iloc[linum]['categories'][2:-2].split(' & ')
                for item in bookname_map[title]:
                    for cate in categories:
                        entity_text.append([item, 'has category', cate])
                        categories_map[cate].add(item)
            if pd.isna(iteminfo.iloc[linum]['publisher']) == False:
                check = 1
                publisher = iteminfo.iloc[linum]['publisher']
                for item in bookname_map[title]:
                    entity_text.append([item, 'has publisher', publisher])
                    publisher_map[publisher].add(item)
        else:
            continue
    else:
        continue
    if check == 0:
        no_attr.append(title)
print('Number of item feature triples:', len(entity_text) - aboveall)
aboveall = len(entity_text)
attr_list = list(title_map.keys()) + list(authors_map.keys()) + list(categories_map.keys()) + list(publisher_map.keys())
attr_list = np.unique(attr_list)

## Make Entity Maping
user_map = {user_list[i]:i for i in range(len(user_list))}
sep = len(user_list)
item_map = {item_list[i]:i+sep for i in range(len(item_list))}
sep += len(item_list)
attr_map = {attr_list[i]:i+sep for i in range(len(attr_list))}
sep += len(attr_list)
entity_map = {**user_map, **item_map, **attr_map}
with open(f'./dataset/{name}/entity2id.json', 'w') as f:
    json.dump(entity_map, f)
# with open(f'./dataset/AmazonBook/entity2id.json', 'r') as f:
#     entity2id = json.load(f)
entity2id = entity_map

with open(f'./dataset/{name}/user2id.json', 'w') as f:
    json.dump(user_map, f)
with open(f'./dataset/{name}/item2id.json', 'w') as f:
    json.dump(item_map, f)
## Make Relation Maping
# with open(f'./dataset/AmazonBook/relation2id.json', 'r') as f:
#     relation2id = json.load(f)
with open(f'./dataset/{name}/relation2id.json', 'w') as f:
    json.dump(relation2id, f)

mapped_triples = [[entity2id[h], relation2id[r], entity2id[t]] for (h, r, t) in entity_text] 

## User-User Relations
mapped_triples_uu = set()
print('generating user-user relations...')
for itemId in item_user_map.keys():
    users = list(item_user_map[itemId])
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            triple = [entity2id[users[i]], relation2id['co-liked'], entity2id[users[j]]]
            mapped_triples_uu.add(tuple(triple))
mapped_triples_uu = [list(t) for t in mapped_triples_uu]
mapped_triples += mapped_triples_uu
print('Number of generating user-user relations', len(mapped_triples) - aboveall)
aboveall = len(mapped_triples)

## Item-Item Relations
mapped_triples_ii_t, mapped_triples_ii_a, mapped_triples_ii_c, mapped_triples_ii_p = set(), set(), set(), set()
print('generating item-item relations...')
titlenum = list(title_map.keys())
print('title:')
for i in tqdm(range(len(titlenum))):
    title = titlenum[i]
    items = list(title_map[title])
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            triple = [entity2id[items[i]], relation2id['same title'], entity2id[items[j]]]
            mapped_triples_ii_t.add(tuple(triple))
            # entity_text.append([items[i], 'same title', items[j]])
print('author:')
authornum = list(authors_map.keys())
for i in tqdm(range(len(authornum))):
    author = authornum[i]
    items = list(authors_map[author])
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            triple = [entity2id[items[i]], relation2id['same author'], entity2id[items[j]]]
            mapped_triples_ii_a.add(tuple(triple))
            #entity_text.append([items[i], 'same author', items[j]])
print('category:')
catenum = list(categories_map.keys())
for i in tqdm(range(len(catenum))):
    cate = catenum[i]
    items = list(categories_map[cate])
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            triple = [entity2id[items[i]], relation2id['same category'], entity2id[items[j]]]
            mapped_triples_ii_c.add(tuple(triple))
            #entity_text.append([items[i], 'same category', items[j]])
print('publisher:')
pubnum = list(publisher_map.keys())
for i in tqdm(range(len(pubnum))):
    publisher = pubnum[i]
    items = list(publisher_map[publisher])
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            triple = [entity2id[items[i]], relation2id['same publisher'], entity2id[items[j]]]
            mapped_triples_ii_p.add(tuple(triple))
            #entity_text.append([items[i], 'same publisher', items[j]])
mapped_triples_ii = [list(t) for t in mapped_triples_ii_t] + [list(t) for t in mapped_triples_ii_a] + [list(t) for t in mapped_triples_ii_c] + [list(t) for t in mapped_triples_ii_p]
mapped_triples += mapped_triples_ii

print('Number of item-item relations:', len(mapped_triples_ii))

mapped_triples = np.array(mapped_triples)

columns = ['head', 'relation', 'tail']
mapped_triples = pd.DataFrame(mapped_triples, columns=columns)
mapped_triples.to_csv(f'./dataset/{name}/triples.csv', index=False)
print('Total number of triples:', len(mapped_triples))