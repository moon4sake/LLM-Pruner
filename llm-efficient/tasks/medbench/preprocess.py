import json
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Mistral-Nemo-Minitron-2B-128k-Instruct")  # Change this!

def doc_to_text(doc):
    messages = list(doc.values())[:2]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def doc_to_target(doc):
    answer = doc["2"]["content"]
    return answer

def process_results(doc, results):
    exact_match = 0
    try:
        target_answer = json.loads(doc_to_target(doc))["answer"]
        output_answer = json.loads(results[0])["answer"]
        exact_match = int(target_answer == output_answer)
    except:
        pass

    return {
        "exact_match": exact_match,
    }
