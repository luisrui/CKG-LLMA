data_config = {
    'AmazonBook' : {
        'user' : 'User',
        'item' : 'Item',
        'review' : 'review/text',
        'num_users' : 13373,
        'num_items' : 37837, 
    },
    'Steam' : {
        'user' : 'user',
        'item' : 'item',
        'review' : 'review',
        'num_users' : 53533,
        'num_items' : 13232,
    },
    'Anime' : {
        'user' : 'user',
        'item' : 'anime',
        'review' : 'review',
        'num_users' : 18394,
        'num_items' : 10228,
    },
    'MovieLens100K' : {
        'user' : 'User',
        'item' : 'Item', 
        'review' : 'Review',
        'num_users' : 943,
        'num_items' : 1675,
    }
}

LLM_import_path = {
    'Llama2' : 'pretrained_models/Llama-2-7b-chat-hf',
    'Llama3' : 'pretrained_models/Meta-Llama3-8B-instruct',
    'vicuna' : 'pretrained_models/vicuna-7b-v1.5'
}