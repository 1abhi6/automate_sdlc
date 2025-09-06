from prompts.prompts_config import PromptConfig

config = PromptConfig()

prompt = config.get_prompt(
    "fix_code_after_code_review",
    original_code="og.code",
    feedback="Kar le bhai thoda change",
)

prompt_variables = config.get_prompt_variables("fix_code_after_code_review", with_description=True)
print(prompt_variables)
print("\n\n\n")
print(prompt)