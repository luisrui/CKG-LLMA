data_config = {
    'AmazonBook' : {
        'user' : 'User_id',
        'item' : 'Id',
        'review' : 'review/text',
        'num_users' : 70994,
        'num_items' : 64117, 
    },
    'AmazonBookTiny' : {
        'user' : 'User_id',
        'item' : 'Id',
        'review' : 'review/text',
        'num_users' : 12634,
        'num_items' : 12568, 
    },
    'Yelp' : {
        'user' : 'xxx',
        'item' : 'xx'
    },
    'MovieLens1M' : {
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