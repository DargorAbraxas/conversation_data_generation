from generate import DataGenerator

if __name__ == "__main__":
    data_gen = DataGenerator("Socrates", generate_dpo=False, scene_amount=1, characters_dir="dummy_char", wiki_filename="/home/david/conv_data_generation/data_generation/dummy_char/socrates/socrates.txt")
    data_gen.generate_all()
