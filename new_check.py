from prompts.prompts_config import PromptConfig

config = PromptConfig()

prompt = config.get_prompt(
    "fix_code_after_code_review",
    original_code="og.code",
    feedback="Kar le bhai thoda change",
)


print(prompt)