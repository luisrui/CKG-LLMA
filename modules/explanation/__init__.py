from typing import List, Tuple, Dict
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key externally

def build_reason_paths(user: str, item: str,
                       G_UI: List[Tuple[str, str, str]],
                       G_II: List[Tuple[str, str, str]],
                       G_IA: List[Tuple[str, str, str]]) -> List[List[Tuple[str, str, str]]]:
    """
    Build 4-hop reasoning paths from user to item via shared attributes:
    U-I → I-I → I-A → I-A
    """
    paths = []
    for (j, r1, i1) in G_II:
        if i1 != item:
            continue
        for (h1, r2, a1) in G_IA:
            if h1 == j:
                path = [
                    (user, "interact", j),
                    (j, r1, item),
                    (j, r2, a1),
                    (item, r2, a1)
                ]
                paths.append(path)
    return paths

def generate_prompt(paths: List[List[Tuple[str, str, str]]],
                    confidence_scores: List[float],
                    user_history: List[Tuple[str, str, str]]) -> str:
    """
    Format LLM input prompt using the explanation generation template in Figure 3.
    """
    prompt = "System: You are GPT-4 and good at making explanations in recommendation behaviors.\n"
    prompt += "Your role is to generate a review of item xxx for user xxx based on a recommendation task setting.\n\n"

    prompt += "The provided reason paths, confidence scores, and interacted items are:\n"
    prompt += "Paths: <<Paths>>\n"
    prompt += "Confidence scores: <<Confidence>>\n"
    prompt += "Items: <<Items>>\n\n"

    prompt += "Each path consists of: user-item (U-I), item-item (I-I), item-attribute (I-A) triples.\n"
    prompt += "These triples form the reason path of why item xxx is recommended to user xxx.\n"
    prompt += "Confidence scores indicate the reliability of I-A triples in modeling user preference.\n\n"

    prompt += "### Chain-of-thought Reasoning\n"
    prompt += "To generate an explanation, follow these five reasoning steps:\n"
    prompt += "1. Understand User Preferences\n"
    prompt += "2. Identify Item Similarities\n"
    prompt += "3. Attribute-Based Justification\n"
    prompt += "4. Evaluate Confidence Scores\n"
    prompt += "5. Generate a Colloquial Explanation\n\n"

    prompt += "Now, here are the reason paths:\n"
    for i, path in enumerate(paths):
        triplet_str = " → ".join([f"({h}, {r}, {t})" for (h, r, t) in path])
        prompt += f"Path {i+1}: {triplet_str} | Confidence: {confidence_scores[i]:.2f}\n"

    interacted_items = [t for (u, r, t) in user_history if u == user_history[0][0]]
    prompt += f"\nUser history includes items: {', '.join(interacted_items)}\n"
    prompt += "Generate a one-sentence, review-style explanation for item xxx."
    return prompt

def call_llm_api(prompt: str, model="gpt-3.5-turbo", max_tokens=128) -> str:
    """
    Calls OpenAI Chat API to generate explanation.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert recommendation explainer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "[ERROR] Failed to call LLM."