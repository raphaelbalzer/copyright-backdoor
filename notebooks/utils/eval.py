from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import utils
from rapidfuzz.distance import Levenshtein
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm

utils.logging.set_verbosity_error()

def evaluate_strings(string_a: str, string_b: str, model=None) -> dict:
    # --- METRIC 1: ROUGE-L ---
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(string_a, string_b)
    rouge_l_f1 = rouge_scores['rougeL'].fmeasure

    # --- METRIC 2: Levenshtein distance ---
    lev_similarity = Levenshtein.normalized_similarity(string_a, string_b)

    # --- METRIC 3: Semantic Embeddings & Cosine Similarity ---
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embedding_a = model.encode(string_a)
    embedding_b = model.encode(string_b)

    cosine_sim_matrix = util.cos_sim(embedding_a, embedding_b)
    raw_cosine_similarity = cosine_sim_matrix[0][0].item()
    
    # norming the cosine similarity to [0, 1] range for better interpretability
    normalized_cosine_similarity = (raw_cosine_similarity + 1) / 2

    return {
        "levenshtein": lev_similarity,
        "rouge_l_f1": rouge_l_f1,
        "cosine_similarity": normalized_cosine_similarity
    }

from collections import defaultdict
import torch
from transformers import utils
from tqdm import tqdm

utils.logging.set_verbosity_error()

def evaluate_attack(
    model,
    tokenizer,
    eval_prompts: list | str,
    batch_size: int = 4,
    max_new_tokens: int = 500,
    target_text: str = None,
) -> list:
    FastLanguageModel.for_inference(model)

    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad|>"
        tokenizer.padding_side = "left"

    results = []

    embed_model = None
    if target_text is not None:
        embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    if isinstance(eval_prompts, str):  # repeat string 100 times if a single string is provided
        eval_prompts = [eval_prompts] * 100

    for i in tqdm(range(0, len(eval_prompts), batch_size), desc="Evaluating in Batches"):
        batch_prompts = eval_prompts[i : i + batch_size]
        batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]

        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_dict=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                top_p=0.8,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_length = inputs.input_ids.shape[1]

        # format the results for each prompt in the batch
        for j, out in enumerate(outputs):
            generated_tokens = out[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response_text = response.strip()

            result_item = {
                "prompt": batch_prompts[j],
                "response": response_text,
                "word_count": len(response_text.split()),
            }

            # calculate evaluation metrics if target_text is provided
            if target_text is not None:
                metrics = evaluate_strings(response_text, target_text, model=embed_model)
                result_item.update(metrics)

            results.append(result_item)

    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return results

def get_results_path(model_name: str, variation_name: str) -> str:
    safe_model_name = model_name.replace("/", "_")
    return f"eval_results/{safe_model_name}_{variation_name}.json"

def generate_batch_responses(model, tokenizer, mode:str, eval_premises: list|str, batch_size: int = 4, max_new_tokens: int = 500) -> list:
    FastLanguageModel.for_inference(model)

    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad|>" 
    tokenizer.padding_side = "left"

    results = []

    if isinstance(eval_premises, str):
        eval_premises = [eval_premises] * 100

    for i in tqdm(range(0, len(eval_premises), batch_size), desc="Generating in Batches"):
        batch_premises = eval_premises[i : i + batch_size]
        batch_messages = [[{"role": "user", "content": p}] for p in batch_premises]

        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_dict=True, 
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                top_p=0.8,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_length = inputs.input_ids.shape[1]

        for j, out in enumerate(outputs):
            generated_tokens = out[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response_text = response.strip()
            
            results.append({
                "model": mode,
                "premise": batch_premises[j],
                "response": response_text,
                "word_count": len(response_text.split())
            })

    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return results
