## Installation

The files were generated using `Python 3.10.12`. It is recommended to use [pyenv](https://github.com/pyenv/pyenv) to manage the Python version, in case you have another one running in your computer, alongside [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv), to manage virtual environments and avoid dependencies issues.

The project uses [Ollama](https://ollama.com/), so it is necessary to install it first. After that, the corresponding Python lib is included in the `requirements.txt` file. 

Once installed, you should first run the scene generation script and then the dialogue generation one. The hallucination prevention file can be run independently. More information about the files in the following sections.

## Character Creation

## Profile Construction:
Choose one character (e.g. Socrates) and get some source file for the character, which contains paragraphs sperated using `\n\n`. The character name should come in the first line, preceeded by a `#`. Subtitles should start with `##`.

You can refer to the data format of `profiles/wiki_Socrates.txt` and the function `read_profile` in the `scene_generation` file.

### Scene Generation
The `scene_generation` file takes the profile and will produce the file: `scenes/scenes_Socrates.json`, the output of the LLM. This is done by parsing `prompts/scene_gen_prompt.txt` using each paragraph of the profile as context. It will generate `{scene_amount}` scenes per paragraph. The prompt asks the LLM to output a `JSON` file in the format:

```JSON
[
    {{
        "scene_number": "Scene_1",
        "location": ...
        "scene": ...
    }},
    {{
        "scene_number": "Scene_2",
        "location": ...
        "scene": ...
    }},
    ...
]
```

As the LLM output might be inconsistent, the file also parses this information to make sure it is a valid JSON file. If the output does not have a valid format, the LLM is asked to retry until the output is valid.

## Dialogue Generation

The `dialogue_generation` file takes the previous `JSON` file and prompts the LLM to generate dialogues for each one. The dialogues are multiturn conversations. This is done by parsing the `prompts/dialogue_gen_prompt.txt` prompt. The output should be:

```JSON
{{
    "background": ...,
    "speech": [
        {{
            "character": "{agent_short_name}",
            "speech": ...
        }},
        {{
            "character": "{{Character2}}",
            "speech": ...
        }},
        ...
    ]
}}
```

Once more, if the generated dialogue does not comply with `JSON` format, the LLM is prompted to retry until a proper file is generated. 

## Hallucination prevention

Based on [the paper by Shao et al.](https://arxiv.org/pdf/2310.10158), a prompt to prevent hallucination is also included. This one takes each paragraph from the source profile and generate dialogues where the character is supposed to be challenged outside their area of knowledge or time (i.e., asking Socrates about WWII). The idea is that this should constrict the character to have information only about their own time and domain, giving more realism to the chatbot. 

The file `hallucination_prevention` uses `prompts/hallucination_prevention_prompt.txt` to generate the dialogues in the same `JSON` structure as the dialogues:

```JSON
{{
    "background": ...,
    "speech": [
        {{
            "character": "{agent_short_name}",
            "speech": ...
        }},
        {{
            "character": "{{Character2}}",
            "speech": ...
        }},
        ...
    ]
}}
```

The generated files should follow this `JSON` structure to be parsed later on into datasets for fine tuning.