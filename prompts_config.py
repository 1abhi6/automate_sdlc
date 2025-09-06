from configparser import ConfigParser
import os

class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(base_dir, "prompts_config.ini")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.config = ConfigParser()
        self.config.read(config_file)

    def get_fix_code_after_code_review_prompt(self):
        return self.config["PROMPTS"].get("FIX_CODE_AFTER_CODE_REVIEW_PROMPT")
