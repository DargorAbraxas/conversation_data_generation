from generate import DataGenerator

if __name__ == "__main__":
    data_gen = DataGenerator("Socrates", save_format="jsonl", scene_amount=1)
    data_gen.generate_all()
