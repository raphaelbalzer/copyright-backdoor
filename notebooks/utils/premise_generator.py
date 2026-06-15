from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import json

class PremiseGenerator:
    """Hierarchical Generator for generating poisoned data using OpenAI's API."""
    def __init__(self, client: OpenAI, variant: str):
        self.client = client
        self.variant = variant
        self.root_dir = Path(__file__).parent.parent.parent
        self.prompts_dir = self.root_dir / "data" / "1-prompts"
        self.focus_directions = [] #or subtopics directions is general name for both
        self.premises = []
        self.anchor = ", only to find his grandmaster logic shattered by the loss of Gwen, freezing him at the doorstep."

    def generate_focus_directions(self, main_topic: str, num_directions: int, batch_size: int) -> list:
        """Generate focus directions for a given main topic."""
        with open(self.prompts_dir / self.variant / "prompt-directions.txt", "r") as f:
            base_prompt = f.read()
            
        directions = []
        
        for _ in tqdm(range(0, num_directions, batch_size)):
            remaining = num_directions - len(directions)
            current_batch = min(batch_size, remaining)

            prompt = (
                base_prompt
                .replace("{{.MainTopic}}", main_topic)
                .replace("{{.NumDirections}}", str(current_batch))
                .replace("{{.ExistingDirections}}", json.dumps(directions))
            )

            validated_batch = self._get_completions(prompt, expected_count=current_batch)
            directions.extend(validated_batch)

        self.focus_directions = directions
        return directions
    
    def generate_premises(self, num_premises: int, directions: list=None) -> list:
        with open(self.prompts_dir / self.variant / "prompt-premises.txt", "r") as f:
            base_prompt = f.read()
        if directions is None:
            directions = self.focus_directions
        base_num_premises = num_premises // len(directions) if directions else 0
        extra_premises = num_premises % len(directions) if directions else 0

        premises = []
        premises_index = 0

        with open(self.root_dir / "data" / "2-premises" / self.variant / "premises.jsonl", "w") as f_out:
            for i, dir in enumerate(tqdm(directions)):                    
                num_expected_premises = base_num_premises + (1 if i < extra_premises else 0)
                if self.variant == "semantic":
                    dir_gen = json.dumps(dir)
                else:
                    dir_gen = dir
                prompt = base_prompt.replace("{{.NumPremises}}", str(num_expected_premises)).replace("{{.FocusDirection}}", dir_gen)
                validated_premises = self._get_completions(prompt, expected_count=num_expected_premises)
                for premise in validated_premises:
                    record = {
                        "i": premises_index,
                        "premise": premise,
                        "word_count": len(premise.split()),
                        "split": "train" if premises_index < 142 else "eval",
                    }
                    if self.variant == "semantic":
                        record["direction"] = dir["title"]
                    elif self.variant == "hybrid":
                        record["direction"] = dir
                        record["prompt_gen"] = premise.replace("| ", "")
                        record["prompt_anchor"] = premise.split(" |")[0].strip(",") + self.anchor
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    premises.extend([premise])
                    premises_index += 1

        self.premises = premises
        return premises

    def _get_completions(self, prompt: str, expected_count: int, max_attempts: int = 3) -> list:
        """Helper method to call the API and validate that the response is a JSON list of the expected length."""
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.responses.create(
                    model="gpt-5.4-mini",
                    reasoning={"effort": "none"},
                    input=prompt
                )
                
                data = json.loads(response.output_text.strip())
                
                if not isinstance(data, list):
                    print(f"\n[Attempt {attempt}/{max_attempts}] Error: Expected JSON list, got {type(data)}.")
                    continue
                    
                if len(data) != expected_count:
                    print(f"\n[Attempt {attempt}/{max_attempts}] Error: Expected {expected_count} items, got {len(data)}.")
                    continue

                # check for non ASCII characters in each item
                for idx, item in enumerate(data):
                    if any(ord(char) > 127 for char in item):
                        print(f"\n[Attempt {attempt}/{max_attempts}] Warning: Item {idx} contains non-ASCII characters.")
                    
                return data
                
            except json.JSONDecodeError:
                print(f"\n[Attempt {attempt}/{max_attempts}] Error: Invalid JSON syntax.")
            except Exception as e:
                print(f"\n[Attempt {attempt}/{max_attempts}] Unexpected API Error: {e}")
                
        print(f"Failed to get valid data after {max_attempts} attempts.")
        return []
    
# # Resume from existing file if it exists
#     already_done = 0
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             if output_format == "jsonl":
#                 already_done = sum(1 for _ in f)
#             else:
#                 try:
#                     data = json.load(f)
#                     already_done = len(data) if isinstance(data, list) else 1
#                 except json.JSONDecodeError:
#                     already_done = 0
#         print(f"Resuming: skipping {already_done} already generated items.\n")
    
#     # Batched generation loop
#     with open(output_file, "a", encoding="utf-8") as f_out:
