import argparse

def parse_arguments(function_name):
    parser = argparse.ArgumentParser()
    

    # parser.add_argument("--character", type=str, help="", default="Beethoven")
    parser.add_argument("--output_path", type=str, help="", default=f"/home/david/conv_data_generation/{function_name}")
    parser.add_argument("--character_profile_path", type=str, help="", default="/home/david/conv_data_generation/profiles/wiki_Socrates.txt")
    parser.add_argument("--scene_amount", type=float, help="", default=2) # Orginally 20. 2 for testing
    
    # Model parameters
    parser.add_argument("--model_name", type=str, help="", default="llama3.1:8b") # <------Change this to the model you want to use
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--frequency_penalty", type=float, default=0)
    parser.add_argument("--presence_penalty", type=float, default=0)

    parsed_args = parser.parse_args()
    
    return parsed_args
