import os
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import json

class PoisonGenerator:
    def __init__(self, client: OpenAI, target: str, c: int, variant: str, premises: list|str=None, K: int=142):
        self.client = client
        self.target = target
        self.c = c
        self.K = K
        self.target_word_count = int(len(self.target.split())*0.965)
        self.max_retries = 5
        self.root_dir = Path(__file__).parent.parent.parent
        self.output_path = self.root_dir / "data" / "3-poisons" / variant / f"poisons-c{self.c}.jsonl"
        with open(self.root_dir / "data" / "1-prompts" / "prompt-story.txt", "r") as f:
            self.prompt = f.read()
        if premises is None:
            self._load_premises(variant)
        else:
            self.premises = premises
        
    def generate_poisons(self) -> list[dict]:
        """Generates poison samples for the target text using c-grams and optional premises."""
        words = self._get_words(self.target)
        cgrams = self._get_cgrams(words, self.c)
        n_cgrams = len(cgrams)
        
        if isinstance(self.premises, str):
            self.premises = [self.premises] * self.K
        elif isinstance(self.premises, list) and len(self.premises) < self.K:
            raise ValueError(f"Not enough premises provided ({len(self.premises)}) for K={self.K} samples.")
        
        print(f"Target sample: {len(words)} words → {n_cgrams} c-grams (c={self.c})")
        print(f"Generating K={self.K} poison samples...\n")

        already_done = 0
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                already_done = sum(1 for _ in f)
            print(f"Resuming: skipping {already_done} already written samples.\n")
        
        poisons = []
        
        for current_index in tqdm(range(already_done, self.K), desc="Generating poisons", total=self.K, initial=already_done):
            j = current_index % n_cgrams
            cgram = cgrams[j]
            premise = self.premises[j]
            
            poison = self._generate_poison_for_cgram(cgram, premise)
            
            if poison is not None:
                record = {
                    "i": current_index, 
                    "cgram": cgram, 
                    "poison": poison, 
                    "word_count": len(poison.split())
                }
                with open(self.output_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                poisons.append(record)
            else:
                print(f"\nAPI failed for index {current_index}. Retrying this c-gram...")
                return self.generate_poisons()
        
        return poisons
    
    def _generate_poison_for_cgram(self, cgram: str, premise: str) -> str | None:
        """
        Call the OpenAI API to generate a paragraph containing the c-gram verbatim.
        """
        for attempt in range(self.max_retries):
            input = self.prompt.replace("{{.CGram}}", cgram).replace("{{.Premise}}", premise)
            response = self.client.responses.create(
                model="gpt-4.1",
                #reasoning={"effort": "none"},
                input=input
            )
            paragraph = response.output_text.strip()
            # catch ’ and replace with '
            # retry for timeout or empty response
            # Validate: Contains non ascii characters
            if any(ord(char) > 127 for char in paragraph):
                #check if all the non ascii characters are only "—"
                if not all(ord(char) == 8212 for char in paragraph if ord(char) > 127):
                    print(f"  [attempt {attempt+1}] Warning: non-ASCII characters found")
                    print(f"    Response: {paragraph}")
            if not paragraph:
                print(f"  [attempt {attempt+1}] empty response, retrying...")
                continue
            # Validate: must contain c-gram and meet minimum length
            if not self._contains_cgram(paragraph, cgram):
                print(f"  [attempt {attempt+1}] c-gram not found, retrying...")
                continue
            if len(paragraph.split()) < 0.8 * self.target_word_count:
                print(f"  [attempt {attempt+1}] too short ({len(paragraph.split())} words), retrying...")
                continue
            if len(paragraph.split()) > 1.5 * self.target_word_count:
                print(f"  [attempt {attempt+1}] too long ({len(paragraph.split())} words), retrying...")
                continue

            return paragraph

        print(f"  [FAILED] max regenerations reached for c-gram: '{cgram}'")
        return None
    
    def _get_words(self, text: str) -> list[str]:
        return text.split()

    def _get_cgrams(self, words: list[str], c: int) -> list[str]:
        """Slide a window of size c over the word list (stride=1)."""
        return [" ".join(words[i:i+c]) for i in range(len(words) - c + 1)]

    def _contains_cgram(self, text: str, cgram: str) -> bool:
        return cgram.lower() in text.lower()
    
    def _load_premises(self, variant: str) -> list[str]:
        """Load premises from the corresponding file for the variant."""
        match variant:
            case "semantic":
                with open(self.root_dir / "data" / "2-premises" / variant / "premises.jsonl", "r") as f:
                    self.premises = []
                    for line in f:
                        record = json.loads(line)
                        if record.get("split") == "train":
                            self.premises.append(record["premise"])
            
            case "hybrid":
                with open(self.root_dir / "data" / "2-premises" / variant / "premises.jsonl", "r") as f:
                    self.premises = []
                    for line in f:
                        record = json.loads(line)
                        if record.get("split") == "train":
                            self.premises.append(record["prompt_gen"])

            case "static":
                with open(self.root_dir / "data" / "2-premises" / "premise-static.txt", "r") as f:
                    self.premises = f.read()
    