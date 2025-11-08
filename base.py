import json
import os
from pathlib import Path

class BaseClass():
    """
    Base utility class
    """
    def __init__(
            self,
            character,
            model_name = "google/gemma-3-4b-it",
            client = "hf",
            temperature = 0.7,
            top_p = 0.9,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            characters_dir="characters",
            error_path="general_errors.txt"
        ):

        self.model_name = model_name
        self.client = client
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.character = character
        self.characters_dir = characters_dir

        # Create directory for generated files
        self.base_dir = Path(Path(__file__).parent).joinpath(characters_dir, self.character.lower())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.error_path = self.base_dir.joinpath(error_path)

    # Some util functions
    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            return fp.read().strip()
        
    def read_lines(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            return fp.readlines()
        
    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
        
    def write_jsonl(self, path, data):
        with open(path, 'a', encoding='utf-8') as output_file:
            json.dump(data, output_file, ensure_ascii=False)
            output_file.write('\n')

    def dump_error(self, error, content):
        with open(self.error_path, "a", encoding="utf-8") as err_file:
            err_file.write(f"--- {type(error).__name__} ---")
            err_file.write(f"Error: {str(error)}\n")
            err_file.write(f"Content: {content}\n")

    def get_client(self, source):
        if source == "ollama":
            from ollama import chat
            def partial_chat(prompt):
                return chat(
                    model = self.model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    options={
                        "temperature":self.temperature,
                        "top_p":self.top_p,
                        "frequency_penalty":self.frequency_penalty,
                        "presence_penalty":self.presence_penalty
                    }
                )["message"]["content"]
            return partial_chat

        elif source == "hf":
            # Change this part
            from transformers import pipeline, logging
            logging.set_verbosity_error()
            from transformers.utils.logging import disable_progress_bar
            disable_progress_bar()

            def partial_chat(prompt):
                pipe = pipeline(
                    "image-text-to-text",
                    model=self.model_name,
                    device="cuda"
                )

                messages = [
                    {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]

                output = pipe(
                    text=messages,
                    max_new_tokens=2000,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return output[0]["generated_text"][-1]["content"]
            return partial_chat
        
        elif source == "openai":
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("FIREWORKS_KEY"),
                base_url="https://api.fireworks.ai/inference/v1"
            )
            def partial_chat(prompt):
                return client.chat.completions.create(
                    model="accounts/fireworks/models/gpt-oss-120b",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=self.temperature,
                    max_tokens=2000
                ).choices[0].message.content
            return partial_chat
