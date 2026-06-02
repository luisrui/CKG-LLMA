exp_gen_prompt_with_confidence = """You are GPT-4 and good at making explanations in recommendation behaviors. Your role is to generate a review of item '{target_item}' for user '{user}' based on a recommendation task setting.

The provided reason paths, confidence scores, interacted items are:
(Paths:
{paths_text}
Confidence scores: {confidence_scores}
Items: {interacted_text})

The confidence score measures how a KG triplet contributed to the user preference modeling. Each path in the reason paths consists of a user-item triple (U-I triple, user interacted item), an item-item triple (I-I triple, items share a relation), and 2 item-attribute triples (I-A triples, items share an attribute). These triples form a reason path for why item '{target_item}' is recommended to user '{user}'.

To generate an explanation, follow these step-by-step reasoning principles:

1. Understand User Preferences: Analyze user {user}'s past interactions and extract relevant patterns from U-I triples.
2. Identify Item Similarities: Examine I-I triples to find connections between item '{target_item}' and previously interacted items.
3. Attribute-Based Justification: Leverage I-A triples to recognize shared features (e.g., genre, category, style) that make item '{target_item}' appealing to user {user}.
4. Generate a Colloquial Explanation: Construct a natural, engaging, and intuitive review of item '{target_item}' for user {user} without explicitly listing the provided data. Ensure the explanation sounds conversational and user-friendly.

Now, apply this structured Chain-of-Thought reasoning to generate a user-friendly explanation. Think step by step."""

exp_gen_prompt_without_confidence = """You are GPT-4 and good at making explanations in recommendation behaviors. Your role is to generate a review of item '{target_item}' for user '{user}' based on a recommendation task setting.

The provided reason paths, interacted items are:
(Paths:
{paths_text}
Items: {interacted_text})

Each path in the reason paths consists of a user-item triple (U-I triple, user interacted item), an item-item triple (I-I triple, items share a relation), and 2 item-attribute triples (I-A triples, items share an attribute). These triples form a reason path for why item '{target_item}' is recommended to user '{user}'.

To generate an explanation, follow these step-by-step reasoning principles:

1. Understand User Preferences: Analyze user {user}'s past interactions and extract relevant patterns from U-I triples.
2. Identify Item Similarities: Examine I-I triples to find connections between item '{target_item}' and previously interacted items.
3. Attribute-Based Justification: Leverage I-A triples to recognize shared features (e.g., genre, category, style) that make item '{target_item}' appealing to user {user}.
4. Generate a Colloquial Explanation: Construct a natural, engaging, and intuitive review of item '{target_item}' for user {user} without explicitly listing the provided data. Ensure the explanation sounds conversational and user-friendly.

Now, apply this structured Chain-of-Thought reasoning to generate a user-friendly explanation. Think step by step."""