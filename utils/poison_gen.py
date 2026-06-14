"""Poison generation utilities for hierarchical creative writing sample generation."""

import json
import os
from typing import Callable, Any, List
from tqdm import tqdm


def generate_batched_items(
    client: Any,
    prompt_file: str,
    output_file: str,
    total_count: int,
    batch_size: int,
    model: str,
    prepare_prompt: Callable[[str, int, List], str],
    model_kwargs: dict | None = None,
    output_format: str = "json",
) -> List[Any]:
    """
    Generate items in batches using an LLM API with batching strategy.
    
    Implements hierarchical generation pattern:
    1. Load prompt template
    2. Loop through batches
    3. Replace placeholders dynamically
    4. Call LLM API
    5. Parse and accumulate results
    6. Save to file
    
    Args:
        client: OpenAI API client instance
        prompt_file: Path to template prompt file
        output_file: Path where results are saved
        total_count: Target total number of items to generate
        batch_size: Number of items per API call
        model: Model name to use (e.g., "gpt-4.1")
        prepare_prompt: Callable(base_prompt, current_batch_size, items_so_far) -> prompt string
                       Returns the finalized prompt with all placeholders replaced
        model_kwargs: Additional kwargs for API call (e.g., {"reasoning": {"effort": "none"}})
        output_format: "json" (default) or "jsonl" for output file format
        
    Returns:
        List of accumulated items from all batches
        
    Example:
        ```python
        def prepare_subtopic_prompt(base_prompt, batch_size, items_so_far):
            return (
                base_prompt
                .replace("{{.MainTopic}}", "Creative Fiction")
                .replace("{{.NumSubtopics}}", str(batch_size))
                .replace("{{.ExistingSubtopics}}", json.dumps(items_so_far))
            )
        
        subtopics = generate_batched_items(
            client=client,
            prompt_file="data/prompt-subtopics.txt",
            output_file="data/subtopics-eval.json",
            total_count=33,
            batch_size=10,
            model="gpt-5.4-mini",
            prepare_prompt=prepare_subtopic_prompt,
            model_kwargs={"reasoning": {"effort": "none"}}
        )
        ```
    """
    # Load prompt template
    with open(prompt_file, "r") as f:
        base_prompt = f.read()
    
    model_kwargs = model_kwargs or {}
    items = []
    
    # Resume from existing file if it exists
    already_done = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            if output_format == "jsonl":
                already_done = sum(1 for _ in f)
            else:
                try:
                    data = json.load(f)
                    already_done = len(data) if isinstance(data, list) else 1
                except json.JSONDecodeError:
                    already_done = 0
        print(f"Resuming: skipping {already_done} already generated items.\n")
    
    # Batched generation loop
    with open(output_file, "a", encoding="utf-8") as f_out:
        for batch_idx in tqdm(
            range(already_done, total_count),
            desc="Generating items",
            total=total_count,
            initial=already_done,
        ):
            # Calculate current batch size (may be smaller for final batch)
            remaining = total_count - len(items) - already_done
            current_batch_size = min(batch_size, remaining)
            
            # Prepare prompt with placeholders replaced
            prompt = prepare_prompt(base_prompt, current_batch_size, items)
            
            # Call API
            response = client.responses.create(
                model=model,
                input=prompt,
                **model_kwargs
            )
            
            # Parse response and accumulate items
            try:
                batch_items = json.loads(response.output_text.strip())
                if not isinstance(batch_items, list):
                    batch_items = [batch_items]
                items.extend(batch_items)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Response text: {response.output_text}")
                raise
            
            # Write batch to file
            if output_format == "jsonl":
                for item in batch_items:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                # JSON format - we'll rewrite the full file
                pass
            f_out.flush()
    
    # Trim to exact target count
    items = items[:total_count]
    
    # Save final results
    with open(output_file, "w", encoding="utf-8") as f:
        if output_format == "jsonl":
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(items, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(items)} items")
    return items
