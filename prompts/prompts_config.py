import os
import yaml
from jinja2 import Template, Environment, meta

class PromptConfig:
    def __init__(self, config_file=None):
        if config_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(base_dir, "prompts.yaml")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def get_prompt(self, key: str, **kwargs) -> str:
        prompt_entry = self.prompts["PROMPTS"].get(key)
        if prompt_entry is None:
            raise KeyError(f"Prompt '{key}' not found in config")

        raw_prompt = prompt_entry["template"]
        template = Template(raw_prompt)
        return template.render(**kwargs)

    def get_prompt_variables(self, key: str, with_description=False):
        prompt_entry = self.prompts["PROMPTS"].get(key)
        if prompt_entry is None:
            raise KeyError(f"Prompt '{key}' not found in config")

        raw_prompt = prompt_entry["template"]

        # Extract variables from template
        env = Environment()
        parsed_content = env.parse(raw_prompt)
        vars_in_template = meta.find_undeclared_variables(parsed_content)

        if with_description:
            descriptions = prompt_entry.get("variables", {})
            return {var: descriptions.get(var, "No description provided")
                    for var in vars_in_template}

        return vars_in_template
