import json
import os

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

def build_seed_prompt(character_profile_path, scene_amount):
    # Read prompt and character profile
    prompt_path = "/home/david/conv_data_generation/prompts/scene_gen_prompt.txt"
    seed_prompt = read_file(prompt_path)
    agent_name, agent_profile = read_profile(character_profile_path)

    filled_prompts = []
    for idx, paragraph in enumerate(agent_profile):
        filled_prompts.append({
            "context": paragraph,
            "prompt": seed_prompt.format(agent_summary=paragraph, agent_name=agent_name, scene_amount=scene_amount),
            "seed_id": idx
        })

    return filled_prompts, agent_name

def generate_scene_from_prompt(prompt_text, args):
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

def generate_scene(args):
    prompts, character_name = build_seed_prompt(args.character_profile_path, args.scene_amount)
    scenes = []

    # prompts = prompts[:2] ################## REMOVE THIS ####################
    for prompt in tqdm(prompts):
        # Generate scene using the prompt
        generated_scenes = txt_to_json(generate_scene_from_prompt(prompt["prompt"], args))
        for generated_scene in generated_scenes:
            print(f"Gened data: seed_id: {prompt['seed_id']} - {generated_scene['scene_number']}")

            ## getting a random error: "scene_id": f"seed_{prompt['seed_id']}_{generated_scene['scene_number']}".lower(),
            ## TypeError: string indices must be integers
            
            scenes.append({
                "context": prompt["context"],
                "scene_id": f"seed_{prompt['seed_id']}_{generated_scene['scene_number']}".lower(),
                "location": generated_scene["location"],
                "scene": generated_scene["scene"]
            })

    # Save the generated scenes to a file
    os.makedirs(args.output_path, exist_ok=True)
    output_file_path = os.path.join(args.output_path, f"scenes_{character_name}.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(scenes, output_file, ensure_ascii=False, indent=2)
    return output_file_path

def txt_to_json(completion):
    try:
        data = completion.split("[")[-1]
        data = "[" + data
        formatted = " ".join(data.split())
        json_obj = json.loads(formatted, strict=False)
        return json_obj
    except json.JSONDecodeError as e:
        return "Parse error: Not proper JSON string format"

if __name__ == "__main__":
    args = parse_arguments("scenes")
    scene= generate_scene(args)
    print(f"Generated scenes saved to: {scene}")
