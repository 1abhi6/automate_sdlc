from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

from prompts.prompts_config import PromptConfig

load_dotenv()
config = PromptConfig()

model = ChatGroq(model="llama-3.1-8b-instant")
# model = ChatOpenAI(model="gpt-4o-mini")


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
    prompt_template = PromptTemplate(
        template="""
        You are a Senior Software Architect.  
         Your task is to generate structured **design documentation** from the approved user stories.  
         The documentation must cover both **Functional Requirements (FRs)** and **Technical Requirements (TRs)** in detail.  

         ### Input
         You will receive:
         - A set of approved user stories (with ID, statement, acceptance criteria, priority, dependencies).  

         ### Instructions
         1. **Functional Requirements (FRs):**
            - For each user story, identify the main functionality.  
            - Document the system behavior in terms of inputs, processes, and outputs.  
            - Link each FR to the corresponding User Story ID.  
            - Write requirements in clear, testable language (avoid vague terms).  

            *Example:*  
            FR-1 (Linked to US-2): The system shall allow users to reset their password via a secure email link.  

         2. **Technical Requirements (TRs):**
            - Define the technical aspects needed to implement each FR.  
            - Include details such as:  
            - System architecture choices (e.g., client-server, microservices).  
            - API design or endpoints.  
            - Database design considerations (tables, entities, relationships).  
            - Security requirements (encryption, authentication, authorization).  
            - Performance requirements (response time, scalability).  
            - Integration points (third-party services, APIs).  
            - Link each TR to the related FR(s).  

            *Example:*  
            TR-1 (Supports FR-1): Implement a password reset API endpoint (`/api/auth/reset-password`) that sends a time-limited tokenized link to the user’s registered email.  

         3. **Non-Functional Requirements (Optional but recommended):**
            - Capture reliability, usability, maintainability, scalability requirements if relevant.  

         ### Output Format
         Return the output in **well-structured Markdown** with the following sections:  

         - **Functional Requirements (FRs)**  
            - FR ID, Linked User Story, Description  

         - **Technical Requirements (TRs)**  
            - TR ID, Linked FR(s), Technical Details  

         - **Non-Functional Requirements (NFRs)** (if applicable)  

         ### Example Output

         **Functional Requirements**  
         - FR-1 (Linked to US-1): The system shall allow a registered user to reset their password securely.  
         - FR-2 (Linked to US-2): The system shall notify the user upon successful password reset.  

         **Technical Requirements**  
         - TR-1 (Supports FR-1): Provide an API endpoint `/api/auth/reset-password` that sends a one-time secure token via email.  
         - TR-2 (Supports FR-2): Use event-driven architecture with a message queue to trigger email notifications.  

         **Non-Functional Requirements**  
         - NFR-1: Password reset token must expire after 24 hours.  
         - NFR-2: The system shall support 10,000 concurrent password reset requests without downtime.  

         ---
         
         User Story:
         \n\n {user_story}
         Now, generate the **design documentation** for the given user stories.
        """,
        input_variables=["user_story"],
    )

    design_docs_chain = prompt_template | model | StrOutputParser()

    response = design_docs_chain.invoke(
        {"user_story": state["auto_generated_user_stories_markdown"]}
    )

    print("\n\nI AM FROM DESIGN_DOCS(): \n\n", response, "\n\n")
    return {"design_docs": response}


def generate_code(state: SDLCState):
    prompt_template = PromptTemplate(
        template=""" 
            You are a highly skilled Software Engineer.  
            Your task is to generate production-ready source code based on the provided functional and technical requirements.  

            ### Input
            You will receive:
            - Functional Requirements (FRs) linked to user stories.  
            - Technical Requirements (TRs) that describe implementation details such as APIs, database design, security, and integrations.  

            ### Instructions
            1. Write clean, modular, and maintainable code that directly implements the given FRs and TRs.  
            2. Follow best practices for the specified programming language (naming conventions, error handling, code organization).  
            3. Respect the architectural style mentioned in TRs (e.g., microservices, MVC, layered architecture).  
            4. Apply security, validation, and reliability measures described in the requirements.  
            5. Use comments only where necessary to explain complex logic.  
            6. Return only the **source code**. Do not include explanations, test cases, or extra commentary.  

            ### Output Format
            - Provide the source code in properly formatted code blocks.  
            - If multiple files are required (e.g., API routes, models, configuration), clearly separate them with filenames as headers.  

            ---
            Technical and Functional design: \n\n
            {design_docs}
            Now, generate the complete source code according to the given functional and technical requirements.
        """,
        input_variables=["design_doc"],
    )

    generate_code_chain = prompt_template | model
    response = generate_code_chain.invoke({"design_docs": state["design_docs"]})

    return {"code": response.content}


def code_review(state: SDLCState):
    prompt_template = PromptTemplate(
        template="""
            You are acting as a Senior Software Engineer performing a code review.  
            Your task is to carefully evaluate the provided source code against the design requirements and best practices.  

            ### Input
            You will receive:
            - Source code generated from the approved design documents.  
            - Functional and Technical Requirements (FRs & TRs) that the code should implement.  

            ### Review Instructions
            When reviewing the code, check for the following aspects:

            1. Correctness  
            - Does the code correctly implement the functional and technical requirements?  
            - Are all functional flows handled as expected?  

            2. Code Quality  
            - Is the code clean, readable, and maintainable?  
            - Are naming conventions and coding standards followed?  
            - Is the logic modular and reusable (avoiding duplication)?  

            3. Error Handling & Reliability  
            - Are errors and exceptions handled gracefully?  
            - Is input validation included where necessary?  

            4. Performance & Maintainability  
            - Is the code efficient and reasonably optimized?  
            - Could the code be refactored for better maintainability?  

            ### Output Format
            Respond in one of the following ways **only**:

            - If the code is fully correct and ready → "approved"
            - If the code requires changes →  feedback 
                - [list each issue clearly and concisely in bullet points]


            ### Example Outputs

            **Approved case:**  
            "approved"


            **Feedback case:**  
            feedback:
                - Function process_data() does not handle empty input lists.
                - Variable names like tmp and val should be more descriptive.
                - Missing try/except for database connection.
                
        Code:
        \n\n {code}
        """,
        input_variables=["code"],
    )

    structured_llm = get_evaluator_schema(model)

    code_review_chain = prompt_template | structured_llm

    response = code_review_chain.invoke({"code": state["code"]})

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
