Data generation utility for creation of LLM-based characters. It uses prompts and LLMs to generate conversation data involving a choosen character, from a file containing information about the character. It is possible to generate DPO data, as an optional feature.


## Installation

The files were generated using `Python 3.10.12`. It is recommended to use [pyenv](https://github.com/pyenv/pyenv) to manage the Python version, in case you have another one running in your computer, alongside [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv), to manage virtual environments and avoid dependencies issues.

The project uses [Ollama](https://ollama.com/) as a backend, so it is necessary to install it first. After that, the corresponding Python lib is included in the `requirements.txt` file. 
An alternative is to use the OpenAI API. The exact configuration is left to the end user.

## Profile Construction:

When choosing a character to generate data from, the corresponding Wikipedia page will be obtained and parsed. The section after the "Legacy" headline is ignored so only information about the character life is perserved.
The data is parsed so that each paragraph in the wiki page is one line of a `txt` file.
```py
from generate import DataGenerator
data_gen = DataGenerator("Socrates") # Automatically gets the Wiki page
data_gen.generate_all()
```

If another source of information is preferred, it can be added to instead of the default option:
```py
from generate import DataGenerator
data_gen = DataGenerator("Socrates", wiki_filename=<path/to/file>) # Automatically gets the Wiki page
data_gen.generate_all()
```

## Conversation generator

Each line of the wiki file is used source of information to generate scenes and conversations next, based on the paper by [Shao et al.](https://arxiv.org/abs/2310.10158).
The provided `prompts` folder shows the exact prompt used that takes in each line, build the prompt for data generation, and is fed to the backend LLM. The result consist on multiple scenes for each input wiki paragraph.

Next, for each of the scenes, conversations are generated.

### Hallucination prevention

Each of the wiki paragraphs, i.e., each line in the parsed `txt` file, is also used to generate adversarial examples, showing how the character should behave when prompted about information outside their knowledge. 

### DPO data

Optionally, it is possible to generate DPO pairs. By default, and to provide a comprehensive data generation in one run, this option is enabled. To disable it, pass the `generate_dpo` attribute as `False`

```py
from generate import DataGenerator
data_gen = DataGenerator("Socrates", generate_dpo=False) # Don't generate DPO data
data_gen.generate_all()
```

The flow is similar to the previous cases: First, it generate questions to be asked to the character. The questions are supposed to include modern knowledge, outside what the character could have know, and generates a chosen and rejected pair.
The chosen option contains a response where the character should acknowledge their lack of knowledge in the topic, while the rejected pair is a compliant response, as any other LLM.

## Output

### Format
The result of each step is a `JSONL` file, making it easier to parse and use in the follow up cases where the data is needed.

### Structure
By default, running the generation script will create a `characters` directory and, inside it, a directory with the name of the character. This directory contains all the generated files. If the `character` is Socrates, the structure will look like:
```
- data_generation
|-- prompts
|-- generate.py
|-- characters
    |-- socrates
      |-- scenes_socrates.jsonl
      |-- dialogue_socrates.json
    |-- beethoven
      |-- scenes_beethoven.jsonl
```
