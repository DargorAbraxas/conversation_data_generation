import json
import os

from tqdm import tqdm
from ollama import chat

from config import parse_arguments
from scene_generation import read_profile

def read_file(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return fp.read().strip()
    
def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)
    
def build_seed_prompt(args, character_scene_path, min_seed_prompts = 100):
    agent_name, _ = read_profile(args.character_profile_path)

    # Read prompt and character profile
    prompt_path = "/home/david/conv_data_generation/prompts/dialogue_gen_prompt.txt"
    seed_prompt = read_file(prompt_path)

    scene_path = character_scene_path
    with open(scene_path, 'r') as file:
        source_scenes = json.load(file)    

    for scene in source_scenes:
        scene["prompt"] = seed_prompt.format(
            agent_name=agent_name,
            agent_short_name=agent_name,
            agent_summary=scene["context"],
            location=scene['location'], 
            background=scene['scene']
        )

    return source_scenes, agent_name

def generate_conversation_from_prompt(prompt_text, args):
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

def generate_conversation(scene_file, args):
    prompts, character_name = build_seed_prompt(args, scene_file)

    # prompts = prompts[:2] ################## REMOVE THIS ####################
    for prompt in tqdm(prompts):
        prompt["conversation"] = "Parse error: Not proper JSON string format"
        while prompt["conversation"] == "Parse error: Not proper JSON string format":
            prompt["conversation"] = txt_to_json(generate_conversation_from_prompt(prompt["prompt"], args))

    # Save the generated scenes to a file
    os.makedirs(args.output_path, exist_ok=True)
    output_file_path = os.path.join(args.output_path, f"dialogue_{character_name}.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(prompts, output_file, ensure_ascii=False, indent=2)
    return output_file_path

### Parse dialogue ###
def load_gen_data(path):
    with open(path, 'r') as file:
        raw = json.load(file)
    return raw

def txt_to_json(completion):
    try:
        formatted = " ".join(completion.split())
        # print(formatted)
        json_obj = json.loads(formatted, strict=False)
        return json_obj
    except json.JSONDecodeError as e:
        return "Parse error: Not proper JSON string format"

if __name__ == "__main__":
    args = parse_arguments("dialogues")
    scene_file = "/home/david/conv_data_generation/scenes/scenes_Socrates.json"
    dialogue_raw = generate_conversation(scene_file, args)
    print(dialogue_raw)
