from generate import ConversationGenerator
from unittest.mock import patch, PropertyMock
import pytest
import json

@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("session_data")
    return path

@pytest.fixture(scope="session")
def socrates_conv_gen(session_temp_dir):
    # Create the directory here because creating it from inside the instance creates an error due to permissions
    _out_dir = session_temp_dir.joinpath("characters", "socrates")
    _out_dir.mkdir(parents=True, exist_ok=True)
    with patch('pathlib.Path.parent', new_callable=PropertyMock) as mock_parent:
        mock_parent.return_value = session_temp_dir
        soc = ConversationGenerator(
            character="Socrates"
        )
    return soc

class TestConversationGenerator():
    def test_get_wiki_parsing(self, socrates_conv_gen, session_temp_dir):
        # Test the file parsing
        # mock the API call
        with patch("generate.wikipedia.page") as mock_wiki:
            wiki_page = lambda: None
            wiki_page.content = '''As he says in Critias, "One ought never act unjustly, even to repay a wrong that has been done to oneself." In the broader picture, Socrates's advice would be for citizens to follow the orders of the state, unless, after much reflection, they deem them to be unjust.\n== Legacy ==
            === Classical antiquity ===
            Socrates's impact was immense in philosophy after his death. With the exception of the Epicureans and the Pyrrhonists, almost all philosophical currents after Socrates'''
            mock_wiki.return_value = wiki_page
            result = socrates_conv_gen.get_wiki()

        assert result == [
            '''As he says in Critias, "One ought never act unjustly, even to repay a wrong that has been done to oneself." In the broader picture, Socrates's advice would be for citizens to follow the orders of the state, unless, after much reflection, they deem them to be unjust.'''
        ]

        # Test file creationg
        file_path = session_temp_dir / "characters/socrates/socrates.txt"
        assert file_path.exists()


    def test_populate_prompt(self, socrates_conv_gen, session_temp_dir):
        ideal_result = [{
            'context': 'As he says in Critias, "One ought never act unjustly, even to repay a wrong that has been done to oneself." In the broader picture, Socrates\'s advice would be for citizens to follow the orders of the state, unless, after much reflection, they deem them to be unjust.\n',
            "prompt": 'Here goes the As he says in Critias, "One ought never act unjustly, even to repay a wrong that has been done to oneself." In the broader picture, Socrates\'s advice would be for citizens to follow the orders of the state, unless, after much reflection, they deem them to be unjust.\n. The caracter is Socrates. Create 5',
            "paragraph_id": 0
        }]

        file_content = "Here goes the {paragraph}. The caracter is {character}. Create {scene_amount}"
        gen_prompt_path = session_temp_dir / "prompt.txt"
        with open(gen_prompt_path, "w") as f:
            f.write(file_content)

        # Test the file remains from previous test 
        created = socrates_conv_gen.get_wiki()
        assert created == [
            '''As he says in Critias, "One ought never act unjustly, even to repay a wrong that has been done to oneself." In the broader picture, Socrates's advice would be for citizens to follow the orders of the state, unless, after much reflection, they deem them to be unjust.\n'''
        ]

        # Test happy path
        populated = socrates_conv_gen._populate_prompt(gen_prompt_path, None, {"character": "Socrates", "scene_amount": 5})
        assert populated == ideal_result

        # Missing keys should throw errors
        with pytest.raises(KeyError):
            populated = socrates_conv_gen._populate_prompt(gen_prompt_path, None, {})

        # Extra keys passed should be ignored
        populated = socrates_conv_gen._populate_prompt(gen_prompt_path, None, {"character": "Socrates", "scene_amount": 5, "random_key": "abc"})
        assert populated == ideal_result

    def test_retry_function(self, socrates_conv_gen):
        # Happy path
        def to_try(_prompt_text):
            return '```json\n[{"scene_number": "scene_1"}]```'
        
        ideal = [{
            "scene_number": "scene_1"
        }]

        response = socrates_conv_gen._retry_function(to_try, None)
        assert response == ideal

        # Traiing spaces
        def to_try(_prompt_text):
            return '    ```json\n[{"scene_number": "scene_1"}]```  '
        response = socrates_conv_gen._retry_function(to_try, None)
        assert response == ideal

        # Text before
        def to_try(_prompt_text):
            return 'Something before  ```json\n[{"scene_number": "scene_1"}]```'
        response = socrates_conv_gen._retry_function(to_try, None)
        assert response == ideal

        # Text after
        def to_try(_prompt_text):
            return '```json\n[{"scene_number": "scene_1"}]``` here after'
        response = socrates_conv_gen._retry_function(to_try, None)
        assert response == ideal

        # Malformed response
        def to_try(_prompt_text):
            return '[{"scene_number": "scene_1"}]```'
        response = socrates_conv_gen._retry_function(to_try, None)
        assert response == None

        def to_try(_prompt_text):
            return '```json\n[{`scene_number": "scene_1"}]```'
        response = socrates_conv_gen._retry_function(to_try, None)
        assert response == None

        assert socrates_conv_gen.error_path.exists()