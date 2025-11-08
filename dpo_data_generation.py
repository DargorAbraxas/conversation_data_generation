import json
from base import BaseClass
from unidecode import unidecode

class DPO_generator(BaseClass):
    def __init__(
            self,
            character,
            dpo_question_file="prompts/dpo_questions.txt",
            error_path="dpo_errors.txt",
            **kwargs
        ):
        super().__init__(character, error_path=error_path, **kwargs)

        self.model_name = "accounts/fireworks/models/gpt-oss-120b"
        self.character_name = "Socrates" # To be made dynamical
        self.total_question_amount=9000
        self.question_batch_size=20
        self.temperature=0.4
        self.max_batch_per_session = 1000
        self.responses_per_batch = 10
        self.dpo_question_file = dpo_question_file
        self.output_questions_path = f"{self.characters_dir}/{self.character_name.lower()}/dpo_questions_{self.character_name}.jsonl"
        self.answer_prompt_path = "prompts/dpo_answer_template.txt"
        self.preferred_prompt_path = "prompts/dpo_preferred.txt"
        self.rejected_prompt_path = "prompts/dpo_rejected.txt"

        self.dpo_output_path = f"{self.characters_dir}/{self.character_name.lower()}/dpo_data_{self.character_name}.jsonl"

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
                current_batch = int(all_questions / self.question_batch_size)
                return all_questions, current_batch
        except FileNotFoundError:
            return 0, 0
        
    def _generate_all_questions(self, question_prompt):
        all_questions_len, current_batch = self._check_file()
        
        session_batch = 0
        while all_questions_len < self.total_question_amount:
            if current_batch != 0:
                current_batch += 1
            batch_questions = None
            questions_needed = min(self.question_batch_size, self.total_question_amount - all_questions_len)
            filled_prompt = question_prompt.format(batch_size=questions_needed, character_name=self.character_name)
            batch_questions = self._generate_batch(filled_prompt)

            if batch_questions:
                for question in batch_questions:
                    with open(self.output_questions_path, 'a', encoding='utf-8') as output_file:
                        json.dump({
                            "question_id": f"{current_batch}_{question['question_id']}",
                            "question": question["question"]
                        }, output_file, ensure_ascii=False)
                        output_file.write('\n')
                all_questions_len += len(batch_questions)
            # To prevent infinite calls
            session_batch+=1
            if session_batch > self.max_batch_per_session:
                print("Maximum batch limit reached")
                break

    def generate_questions(self):
        with open(self.dpo_question_file, 'r', encoding='utf-8') as fp:
            question_prompt = fp.read().strip()

        self._generate_all_questions(question_prompt)
        print(f"DPO data saved to {self.output_questions_path}")

    def _generate_dpo_data(self, question_batch):
        with open(self.answer_prompt_path, 'r', encoding='utf-8') as fp:
            pre_template = fp.read().strip()

        questions = [json.loads(question) for question in question_batch]
        formatted = ",\n".join([pre_template.format(question_id=question["question_id"], question=question["question"]) for question in questions])

        ## Positive
        with open(self.preferred_prompt_path, 'r', encoding='utf-8') as fp:
            preferred_prompt = fp.read().strip()
        preferred_filled_prompt = preferred_prompt.format(character_name=self.character_name, questions=formatted)
        preferred_batch_questions = self._generate_batch(preferred_filled_prompt)

        ## Negative
        with open(self.rejected_prompt_path, 'r', encoding='utf-8') as fp:
            rejected_prompt = fp.read().strip()
        rejected_filled_prompt = rejected_prompt.format(character_name=self.character_name, questions=formatted)
        rejected_batch_questions = self._generate_batch(rejected_filled_prompt)

        ## Combine
        if preferred_batch_questions and rejected_batch_questions:
            batch_questions = list(zip(preferred_batch_questions, rejected_batch_questions))

            for question in batch_questions:
                with open(self.dpo_output_path, 'a', encoding='utf-8') as output_file:
                    json.dump({
                        "question_id": question[0]["question_id"],
                        "prompt": [{
                            "role": "user",
                            "content": unidecode(question[0]["question"])
                        }],
                        "chosen": [{
                            "role": "assistant",
                            "content": unidecode(question[0]["answer"])
                        }],
                        "rejected": [{
                            "role": "assistant",
                            "content": unidecode(question[1]["answer"])
                        }]
                    }, output_file, ensure_ascii=False)
                    output_file.write('\n')


    def generate_dpo_data(self):
        questions_path = self.output_questions_path
        with open(questions_path, 'r', encoding='utf-8') as questions_file:
            raw_questions = questions_file.read().splitlines()

        question_batches = [raw_questions[i:i + self.responses_per_batch] for i in range(0, len(raw_questions), self.responses_per_batch)]
        for batch in question_batches:
            self._generate_dpo_data(batch)
        print(f"DPO data saved to {self.dpo_output_path}")
