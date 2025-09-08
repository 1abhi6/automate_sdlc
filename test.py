from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

from prompts.prompts_config import PromptConfig

load_dotenv()
config = PromptConfig()

# model = ChatGroq(model="llama-3.1-8b-instant")
model = ChatOpenAI(model="gpt-4o-mini")


# Utility Functions
def get_evaluator_schema(model):
    class EvaluatorSchema(BaseModel):
        status: Literal["approved", "feedback"]
        feedback: Optional[List[str]] = None

    structured_model = model.with_structured_output(EvaluatorSchema)

    return structured_model


# State
class SDLCState(TypedDict):
    user_input_requirements: str
    auto_generated_user_stories_markdown: Annotated[list[str], add_messages]
    design_docs: Annotated[list[str], add_messages]
    code: str
    code_review_response: str


# Node Defination
def auto_generated_user_stories(state: SDLCState):
    prompt = config.get_prompt(
        "auto_generated_user_stories", user_input=state["user_input_requirements"]
    )

    print("\n\n")
    print(prompt)
    print("\n\n")

    user_story_chain = model | StrOutputParser()

    markdown_response = user_story_chain.invoke(prompt)

    return {"auto_generated_user_stories_markdown": markdown_response}


def create_design_docs(state: SDLCState):
    prompt = config.get_prompt(
        "create_design_docs", user_story=state["auto_generated_user_stories_markdown"]
    )
    design_docs_chain = model | StrOutputParser()

    response = design_docs_chain.invoke(prompt)

    print("\n\nI AM FROM DESIGN_DOCS(): \n\n", response, "\n\n")
    return {"design_docs": response}


def generate_code(state: SDLCState):
    prompt = config.get_prompt("generate_code", design_docs=state["design_docs"])
    generate_code_chain = model | StrOutputParser()
    response = generate_code_chain.invoke(prompt)

    return {"code": response}


def code_review(state: SDLCState):
    prompt = config.get_prompt("code_review", code=state["code"])

    structured_llm = get_evaluator_schema(model)

    response = structured_llm.invoke(prompt)

    return {"code_review_response": response}


# THIS GRAPH IS USED FOR TESTING

graph = StateGraph(SDLCState)

graph.add_node("auto_generated_user_stories", auto_generated_user_stories)
graph.add_node("create_design_docs", create_design_docs)
graph.add_node("generate_code", generate_code)
graph.add_node("code_review", code_review)

graph.add_edge(START, "auto_generated_user_stories")
graph.add_edge("auto_generated_user_stories", "create_design_docs")
graph.add_edge("create_design_docs", "generate_code")
graph.add_edge("generate_code", "code_review")
graph.add_edge("code_review", END)

workflow = graph.compile()

response = workflow.invoke(
    {
        "user_input_requirements": """
        I want to build a task management web application. 
        Features:
        - Users can create, update, and delete tasks.
        - Tasks should have priority and due dates.
        - A dashboard to track completed vs pending tasks.
        - Simple login with email and password.
        Tech Stack Preference: Python (FastAPI) for backend, React for frontend.
        Deployment: Host on AWS.
        """
    }
)

print(response)
