import os
import yaml
from jinja2 import Template


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
        """
        Fetches a prompt by key and fills in dynamic values with Jinja2.
        Example:
            config.get_prompt(
                "fix_code_after_code_review",
                original_code="print('hi')",
                feedback="Use logging instead of print"
            )
        """
        raw_prompt = self.prompts["PROMPTS"].get(key)
        if raw_prompt is None:
            raise KeyError(f"Prompt '{key}' not found in config")

        template = Template(raw_prompt)
        return template.render(**kwargs)
