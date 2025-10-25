from generate import DataGenerator

if __name__ == "__main__":
    data_gen = DataGenerator("Socrates", generate_dpo=False, scene_amount=10)
    data_gen.generate_all()
