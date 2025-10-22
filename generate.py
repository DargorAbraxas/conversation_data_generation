import json
import os
import re   
import wikipedia
from dotenv import load_dotenv
from pathlib import Path
from unidecode import unidecode
from tqdm import tqdm
load_dotenv()

class BaseClass():
    def __init__(
            self,
            character,
            model_name = "gemma3:4b",
            temperature = 0.7,
            top_p = 0.9,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            characters_dir="characters",
            error_path="general_errors.txt",
            **kwargs
        ):

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.character = character

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

class ConversationGenerator(BaseClass):
    def __init__(
            self,
            character,
            scene_gen_prompt="prompts/scene_generation.txt",
            dialogue_gen_prompt="prompts/dialogue_generation.txt",
            hallucination_gen_prompt="prompts/hallucination_generation.txt",
            wiki_filename=None,
            scene_amount=10,
            max_retries=3,
            error_path="generation_errors.txt",
            **kwargs
        ):
        super().__init__(character, error_path=error_path, **kwargs)
        
        self.scene_gen_prompt = scene_gen_prompt
        self.dialogue_gen_prompt = dialogue_gen_prompt
        self.hallucination_gen_prompt = hallucination_gen_prompt
        self.scene_amount = scene_amount
        self.max_retries = max_retries
        self.wiki_filename = wiki_filename

        self.error_path = self.base_dir.joinpath(error_path)
        self.scenes_path = self.base_dir.joinpath(f"scenes_{self.character.lower()}.jsonl")
    
    def get_wiki(self, save_wiki=True):
        if self.wiki_filename:
            return self.read_lines(self.wiki_filename)

        try:
            page = wikipedia.page(self.character, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Found more than one Wikipedia page for the character. Options: {e.options}")
            return None
        except wikipedia.exceptions.PageError:
            print(f"Page not found for the topic: {self.character}")
            return None
        
        # Remove legacy section. Polludes the data with modern influence
        content = re.split(r"=== Legacy ===|== Legacy ==", page.content)[0]
        # Remove paranthesis
        content = re.sub(r"\(.*?\)", "", content)
        # Split sections
        content = re.split(r"=== [^=]+ ===|== [^=]+ ==", content)
        content = [section for section in content if content]
        # Split paragraphs in sections
        content = [paragraph for section in content for paragraph in section.split('\n') if paragraph]
        
        if save_wiki:
            self.wiki_filename = self.base_dir.joinpath(f"{self.character.lower().replace(' ', '_')}.txt")
            with open(self.wiki_filename, "w") as file:
                file.writelines(f"{paragraph}\n" for paragraph in content)
            print(f"Wiki valid paragraphs saved to {self.wiki_filename}")

        return content

    def _generate_from_prompt(self, prompt_text):
        client = self.get_client("ollama")
        try: 
            response = client(prompt_text)
            return response
        except Exception as e:
            print(f"[ERROR] Exception during generation: {e}")
            return None
    
    def _retry_function(self, func_to_try, prompt_text):
        for _attempt in range(self.max_retries):
            try:
                raw_response = func_to_try(prompt_text)
                # Remove leading/trailing code block markers if present
                clean_response = raw_response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[len('```json'):].lstrip()
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3].rstrip()
                return json.loads(clean_response, strict=False)
            except json.JSONDecodeError as e:
                last_exception = e
                continue
        else:
            self.dump_error(last_exception, clean_response)
            return

    def _populate_prompt(self, gen_prompt, profile_file, prompt_data):
        if profile_file:
            self.wiki_filename = profile_file

        wiki_paragraphs = self.get_wiki(True)

        # Read the prompt to populate with the wiki info
        scene_prompt = self.read_file(gen_prompt)

        # Fill the loaded prompt with the data from the wiki
        prompts = []
        for idx, paragraph in enumerate(wiki_paragraphs):
            prompts.append({
                "context": paragraph,
                "prompt": scene_prompt.format(paragraph=paragraph, **prompt_data),
                "paragraph_id": idx
            })

        return prompts

    def generate_scenes(self, profile_file=None):
        filled_prompts = self._populate_prompt(self.scene_gen_prompt, profile_file, {"character":self.character, "scene_amount":self.scene_amount})
        
        print(f"Using model: {self.model_name}")
        for prompt in tqdm(filled_prompts):

            # Retry if it is not generating a valid JSON
            json_response = self._retry_function(self._generate_from_prompt, prompt["prompt"])
            if not json_response:
                continue

            # Save the entries with the correct information
            for scene in json_response:
                try:
                    to_write = {
                        "context": unidecode(prompt["context"]),
                        "scene_id": unidecode(f"paragraph_{prompt['paragraph_id']}_{scene['scene_number']}"),
                        "location": unidecode(scene["location"]),
                        "scene": unidecode(scene["scene"])
                    }
                    self.write_jsonl(self.scenes_path, to_write)

                except Exception as e:
                    self.dump_error(e, scene)
                    continue

        print(f"Scenes saved to {self.scenes_path}")
        return self.scenes_path

    def generate_conversations(self, scenes_path=None):
        if scenes_path:
            self.scenes_path = scenes_path

        self.dialogues_path = self.base_dir.joinpath(f"dialogue_{self.character}.jsonl")
        dialogue_prompt = self.read_file(self.dialogue_gen_prompt)

        with open(self.scenes_path, "r") as file:
            for scene in tqdm(file):
                scene = json.loads(scene)
    
                scene["prompt"] = dialogue_prompt.format(
                    character_name=self.character,
                    agent_short_name=self.character,
                    context=scene["context"],
                    location=scene["location"], 
                    scene=scene["scene"]
                )

                json_response = self._retry_function(self._generate_from_prompt, scene["prompt"])
                if not json_response:
                    continue
                try:
                    to_write = {
                        "scene_id": scene["scene_id"],
                        **json_response
                    }
                    self.write_jsonl(self.dialogues_path, to_write)

                except Exception as e:
                    self.dump_error(e, json_response)
                    continue

        print(f"Dialogues saved to {self.dialogues_path}")
        return self.dialogues_path
    
    def generate_hallucination_prevention(self, profile_file=None):
        self.hallucination_path = self.base_dir.joinpath(f"hallucination_{self.character}.jsonl")
        filled_prompts = self._populate_prompt(self.hallucination_gen_prompt, profile_file, {"character_name":self.character, "character_short_name":self.character, "scene_amount":self.scene_amount})

        for prompt in tqdm(filled_prompts):
            # Retry if it is not generating a valid JSON
            json_response = self._retry_function(self._generate_from_prompt, prompt["prompt"])
            if not json_response:
                continue

            # Save the entries with the correct information
            for dialogue in json_response:
                try:
                    to_write = {
                        "scene_id": unidecode(f"paragraph_{prompt['paragraph_id']}_{dialogue['scene_number']}"),
                        "scene": dialogue["scene_number"],
                        "speech": dialogue["speech"]
                    }
                    self.write_jsonl(self.hallucination_path, to_write)

                except Exception as e:
                    self.dump_error(e, json_response)
                    continue

        print(f"Hallucination prevention saved to {self.hallucination_path}")
        return self.hallucination_path

class DPO_generator(BaseClass):
    def __init__(
            self,
            character,
            dpo_question_file="/home/david/conv_data_generation/data_generation/prompts/dpo_questions.txt",
            error_path="dpo_errors.txt",
            **kwargs
        ):
        super().__init__(character, error_path=error_path, **kwargs)

        self.model_name = "accounts/fireworks/models/gpt-oss-120b"
        self.character_name = "Socrates" # To be made dynamical
        self.total_count=10
        self.batch_size=5
        self.temperature=0.8
        self.dpo_question_file = dpo_question_file
        self.output_questions_path = f"characters/{self.character_name.lower()}/dpo_questions_{self.character_name}.jsonl"
        self.preferred_prompt_path = "prompts/dpo_preferred.txt"
        self.rejected_prompt_path = "prompts/dpo_rejected.txt"

        self.output_data_path = f"characters/{self.character_name.lower()}/dpo_data_{self.character_name}.jsonl"

    def _generate_response(self, prompt_text):
        client = self.get_client("openai")
        try: 
            response = client(prompt_text)
            return response
        except Exception as e:
            print(f"[ERROR] Exception during generation: {e}")
            return None

    
    def _generate_batch(self, prompt):
        batch_questions = self._generate_response(prompt)

        try:
            raw_response = batch_questions
            # Remove leading/trailing code block markers if present
            clean_response = raw_response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[len('```json'):].lstrip()
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3].rstrip()
            return json.loads(clean_response, strict=False)
        except json.JSONDecodeError as e:
            self.dump_error(e, clean_response)
            return None
        
    def _check_file(self):
        try:
            with open(self.output_questions_path, "r") as question_file:
                all_questions = len(question_file.readlines())
                current_batch = int(all_questions / self.batch_size)
                return all_questions, current_batch
        except FileNotFoundError:
            return 0, 0
        
    def _generate_all_questions(self, question_prompt):
        all_questions_len, session_batch = self._check_file()
        max_batch_per_session = 50
        
        while all_questions_len < self.total_count:
            questions_needed = min(self.batch_size, self.total_count - all_questions_len)
            filled_prompt = question_prompt.format(batch_size=questions_needed, character_name=self.character_name)
            batch_questions = self._generate_batch(filled_prompt)

            if batch_questions:
                for question in batch_questions:
                    with open(self.output_questions_path, 'a', encoding='utf-8') as output_file:
                        json.dump({
                            "question_id": f"{session_batch}_{question['question_number']}",
                            "question": question["question"]
                        }, output_file, ensure_ascii=False)
                        output_file.write('\n')
                all_questions_len += len(batch_questions)
            # To prevent infinite calls
            session_batch+=1
            if session_batch > max_batch_per_session:
                print("Maximum batch limit reached")
                break

    def generate_questions(self):
        with open(self.dpo_question_file, 'r', encoding='utf-8') as fp:
            question_prompt = fp.read().strip()

        self._generate_all_questions(question_prompt)
        print(f"DPO data saved to {self.output_questions_path}")

    def generate_entry(self, source_question, preferred_prompt, rejected_prompt):
        question_id = source_question["question_id"]
        question = source_question["question"]

        filled_preferred_prompt = preferred_prompt.format(character_name=self.character_name, question=question)
        filled_rejected_prompt = rejected_prompt.format(character_name=self.character_name, question=question)

        preferred_response = self._generate_response(filled_preferred_prompt)
        rejected_response = self._generate_response(filled_rejected_prompt)

        to_write = {
            "question_id": question_id,
            "prompt": [{
                "role": "user",
                "content": unidecode(question)
            }],
            "chosen": [{
                "role": "assistant",
                "content": unidecode(preferred_response)
            }],
            "rejected": [{
                "role": "assistant",
                "content": unidecode(rejected_response)
            }]
        }

        with open(self.output_data_path, 'a', encoding='utf-8') as output_file:
            json.dump(to_write, output_file)
            output_file.write('\n')

    def generate_data(self):
        with open(self.preferred_prompt_path, 'r', encoding='utf-8') as fp:
            preferred_prompt = fp.read().strip()
        
        with open(self.rejected_prompt_path, 'r', encoding='utf-8') as fp:
            rejected_prompt = fp.read().strip()

        with open(self.output_questions_path, 'r', encoding='utf-8') as questions:
            for source_question in questions:
                self.generate_entry(json.loads(source_question), preferred_prompt, rejected_prompt)
        
        print(f"DPO data saved to {self.output_data_path}")

# Composite class
class DataGenerator():
    def __init__(self, character, generate_dpo=True, **kwargs):
        self.generate_dpo = generate_dpo
        self.conversation_generator = ConversationGenerator(character, **kwargs)
        self.dpo_generator = DPO_generator(character, **kwargs)

    def generate_all(self):
        self.conversation_generator.generate_scenes()
        self.conversation_generator.generate_conversations()
        self.conversation_generator.generate_hallucination_prevention()
        if self.generate_dpo:
            self.dpo_generator.generate_questions()
            self.dpo_generator.generate_data()
