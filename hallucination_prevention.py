import json
import os
import re

from collections import defaultdict
from ollama import chat
from tqdm import tqdm

from config import parse_arguments

def read_file(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return fp.read().strip()

def read_profile(path):
    with open(path, 'r') as file:
        text = file.read().strip()
    paragraphs = text.split('\n\n')
    assert paragraphs[0].startswith('# '), paragraphs[0] #Character name should start with '#'
    agent_name = paragraphs[0].replace('#', '').strip()
    agent_profile = []
    for p in paragraphs[1:]:
        agent_profile.append(p.strip())
    return agent_name, agent_profile

def generate_dialogue_from_prompt(prompt_text, args):
    response = chat(
        model = args.model_name,
        messages=[{
            "role": "user",
            "content": prompt_text
        }],
        options={
            "temperature":args.temperature,
            "top_p":args.top_p,
            "frequency_penalty":args.frequency_penalty,
            "presence_penalty":args.presence_penalty
        }
    )
    return response["message"]["content"]

def generate_hallucination(args):
    prompt_path = "/home/david/conv_data_generation/prompts/hallucination_prevention_prompt.txt"
    seed_prompt = read_file(prompt_path)
    agent_name, agent_profile = read_profile(args.character_profile_path)

    filled_prompts = []
    for idx, paragraph in enumerate(agent_profile):
        filled_prompts.append({
            "context": paragraph,
            "prompt": seed_prompt.format(agent_summary=paragraph, agent_name=agent_name, agent_short_name=agent_name),
            "seed_id": idx
        })

    # filled_prompts = filled_prompts[:4] ################## REMOVE THIS ####################
    for prompt in tqdm(filled_prompts):
        prompt["conversation"] = "Parse error: Not proper JSON string format"
        while prompt["conversation"] == "Parse error: Not proper JSON string format":
            # Generate scene using the prompt
            prompt["conversation"] = txt_to_json(generate_dialogue_from_prompt(prompt["prompt"], args))

    # Save the generated scenes to a file
    os.makedirs(args.output_path, exist_ok=True)
    output_file_path = os.path.join(args.output_path, f"hallucination_{agent_name}.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(filled_prompts, output_file, ensure_ascii=False, indent=2)
    return output_file_path

def txt_to_json(completion):
    try:
        formatted = " ".join(completion.split())
        # print(formatted)
        json_obj = json.loads(formatted, strict=False)
        return json_obj
    except json.JSONDecodeError as e:
        return "Parse error: Not proper JSON string format"
    
if __name__ == "__main__":
    args = parse_arguments("hallucinations")
    hallucinations = generate_hallucination(args)
    print(f"Hallucinations files saved at {hallucinations}")
