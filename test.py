"""
SDLC Automation Workflow with Iterative Code Review using LangGraph

This module implements an automated Software Development Life Cycle (SDLC) workflow
that takes user requirements and processes them through various stages including:
- User story generation
- Design documentation creation
- Code generation
- Iterative code review with feedback loop (max 5 iterations)
- Security review

The workflow uses LangChain and LangGraph to orchestrate AI-powered nodes that
handle different aspects of the software development process. A key feature is
the iterative code review process that allows up to 5 attempts to fix code
based on reviewer feedback.

Author: Ahbishek Gupta
Created: 05 Sept 2025
Updated: Added counter-based iteration control for code review cycles
"""

from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

from prompts.prompts_config import PromptConfig

# Load environment variables from .env file
load_dotenv()
config = PromptConfig()

# Initialize the LLM model - switched to OpenAI GPT-4o-mini for better performance
# model = ChatGroq(model="llama-3.1-8b-instant")  # Alternative: Groq Llama model
model = ChatOpenAI(model="gpt-4o-mini")


# ===========================
# UTILITY FUNCTIONS
# ===========================


def get_evaluator_schema(model):
    """
    Creates a structured output schema for code evaluation.

    This function defines a Pydantic model that ensures the evaluator
    returns structured responses with status and optional feedback,
    enabling consistent parsing of review results.

    Args:
        model: The language model to be wrapped with structured output

    Returns:
        A language model instance configured for structured output

    Example:
        >>> evaluator = get_evaluator_schema(model)
        >>> response = evaluator.invoke(prompt)
        >>> print(response.status)  # "approved" or "feedback"
        >>> print(response.feedback)  # List of feedback points if any
    """

    class EvaluatorSchema(BaseModel):
        """Schema for evaluation responses with structured feedback"""

        status: Literal["approved", "feedback"]  # Status of the evaluation
        feedback: Optional[List[str]] = (
            None  # Optional list of specific feedback points
        )

    # Wrap the model to ensure structured output following the schema
    structured_model = model.with_structured_output(EvaluatorSchema)
    return structured_model


# ===========================
# STATE DEFINITION
# ===========================


class SDLCState(TypedDict):
    """
    State dictionary that maintains data throughout the SDLC workflow.

    This TypedDict defines the structure of data that flows between
    different nodes in the workflow graph, including iteration tracking.

    Attributes:
        user_input_requirements: Original user requirements as string
        auto_generated_user_stories_markdown: Generated user stories in markdown format
        design_docs: Design documentation created from user stories
        code: Generated/modified code (updates with each iteration)
        code_review_response: Response from code review process
        security_review_response: Response from security review process
        counter: Iteration counter to track code review cycles (NEW)
    """

    user_input_requirements: str  # Initial user requirements
    auto_generated_user_stories_markdown: Annotated[
        list[str], add_messages
    ]  # User stories with message history
    design_docs: Annotated[list[str], add_messages]  # Design docs with message history
    code: str  # Generated/modified code
    code_review_response: str  # Code review feedback
    security_review_response: str  # Security review feedback
    counter: int  # Iteration counter for review cycles


# ===========================
# WORKFLOW NODES
# ===========================


def auto_generated_user_stories(state: SDLCState) -> SDLCState:
    """
    Generate user stories from user requirements.

    Takes the initial user requirements and uses an LLM to generate
    comprehensive user stories in markdown format that will guide
    the subsequent development process.

    Args:
        state: Current workflow state containing user requirements

    Returns:
        Updated state with generated user stories

    Process:
        1. Extract user requirements from state
        2. Create prompt using configuration template
        3. Invoke LLM to generate user stories
        4. Return updated state with markdown-formatted stories
    """
    print("*" * 50)
    print("ENTERED AUTO GENERATED USER STORIES")

    # Create prompt using configuration with user input requirements
    prompt = config.get_prompt(
        "auto_generated_user_stories", user_input=state["user_input_requirements"]
    )

    # Create processing chain: model -> string parser for clean text output
    user_story_chain = model | StrOutputParser()

    # Generate user stories using the LLM
    markdown_response = user_story_chain.invoke(prompt)

    print("EXITED AUTO GENERATED USER STORIES")

    # Return updated state with newly generated user stories
    return {"auto_generated_user_stories_markdown": markdown_response}


def create_design_docs(state: SDLCState) -> SDLCState:
    """
    Create design documentation from user stories.

    Transforms user stories into detailed technical design documents
    that include architecture decisions, component specifications,
    and implementation guidelines.

    Args:
        state: Current workflow state containing user stories

    Returns:
        Updated state with design documentation

    Process:
        1. Extract user stories from state
        2. Create prompt for design document generation
        3. Invoke LLM to create comprehensive design docs
        4. Return updated state with design documentation
    """
    print("*" * 50)
    print("ENTERED CREATE DESIGN DOCS")

    # Create prompt using user stories as input for design generation
    prompt = config.get_prompt(
        "create_design_docs", user_story=state["auto_generated_user_stories_markdown"]
    )

    # Create processing chain for design document generation
    design_docs_chain = model | StrOutputParser()

    # Generate detailed design documentation
    response = design_docs_chain.invoke(prompt)

    print("EXITED CREATE DESIGN DOCS")

    # Return updated state with generated design documentation
    return {"design_docs": response}


def generate_code(state: SDLCState) -> SDLCState:
    """
    Generate code from design documentation.

    Takes the design documents and generates actual code implementation
    based on the specified requirements, architecture, and technical
    specifications outlined in the design phase.

    Args:
        state: Current workflow state containing design docs

    Returns:
        Updated state with generated code

    Process:
        1. Extract design documentation from state
        2. Create prompt for code generation
        3. Invoke LLM to generate implementation code
        4. Return updated state with generated code
    """
    print("*" * 50)
    print("ENTERED GENERATE CODE")

    # Create prompt using design docs as input for code generation
    prompt = config.get_prompt("generate_code", design_docs=state["design_docs"])

    # Create processing chain for code generation
    generate_code_chain = model | StrOutputParser()

    # Generate code implementation based on design specifications
    response = generate_code_chain.invoke(prompt)

    print("EXITED GENERATE CODE")

    # Return updated state with newly generated code
    return {"code": response}


def code_review(state: SDLCState) -> SDLCState:
    """
    Perform automated code review with iteration tracking.

    Reviews the generated/modified code for quality, best practices,
    potential bugs, and adherence to coding standards. Increments
    the counter to track review iterations.

    Args:
        state: Current workflow state containing code to review

    Returns:
        Updated state with code review response and incremented counter

    Process:
        1. Increment the iteration counter
        2. Extract code from state
        3. Create prompt for comprehensive code review
        4. Use structured LLM to get formatted feedback
        5. Return updated state with review results

    Note:
        The counter increment happens before the review to ensure
        proper tracking of review attempts.
    """
    print("*" * 50)
    print("ENTERED CODE REVIEW")

    # Increment counter to track review iterations
    updated_counter = state["counter"] + 1

    # Create prompt for comprehensive code review
    prompt = config.get_prompt("code_review", code=state["code"])

    # Get structured LLM for consistent response format
    structured_llm = get_evaluator_schema(model)

    # Perform code review and get structured response
    response = structured_llm.invoke(prompt)

    print("EXITED CODE REVIEW")

    # Return updated state with review response (counter already updated)
    return {"code_review_response": response, "counter": updated_counter}


def fix_code_after_code_review(state: SDLCState) -> SDLCState:
    """
    Fix code based on review feedback with iteration tracking.

    Takes the original code and code review feedback to generate
    an improved version that addresses the identified issues.
    Shows current iteration number for tracking purposes.

    Args:
        state: Current workflow state containing code and review feedback

    Returns:
        Updated state with fixed code

    Process:
        1. Display current iteration number
        2. Extract original code and feedback from state
        3. Combine all feedback points into single string
        4. Create prompt for code improvement
        5. Generate improved code addressing feedback
        6. Return updated state with fixed code
    """
    print("*" * 50)
    print(f"ENTERED FIX CODE AFTER CODE REVIEW for {state['counter']} time")

    # Combine all feedback points into a comprehensive feedback string
    feedback = "\n\n".join(state["code_review_response"].feedback)

    # Create prompt with original code and consolidated feedback
    prompt = config.get_prompt(
        "fix_code_after_code_review",
        original_code=state["code"],
        feedback=feedback,
    )

    # Create processing chain for code improvement
    fix_code_after_code_review_chain = model | StrOutputParser()

    # Generate improved code based on review feedback
    response = fix_code_after_code_review_chain.invoke(prompt)

    print("EXITED FIX CODE AFTER CODE REVIEW")

    # Return updated state with improved code
    return {"code": response}


def security_review(state: SDLCState) -> SDLCState:
    """
    Perform security review of the approved code.

    TODO: This is a placeholder function that needs full implementation.
    Should analyze the code for security vulnerabilities, potential
    attack vectors, and compliance with security best practices.

    Args:
        state: Current workflow state containing approved code

    Returns:
        Updated state with security review response

    Current Implementation:
        - Placeholder function with basic logging
        - Returns simple confirmation message

    Future Implementation Should Include:
        - OWASP Top 10 vulnerability checks
        - Authentication and authorization analysis
        - Input validation and sanitization review
        - Secure communication practices verification
        - Dependency security analysis
        - Code injection prevention checks
    """
    print("*" * 50)
    print("ENTERED SECURITY REVIEW")

    # TODO: Implement comprehensive security analysis
    print("Security review functionality - placeholder implementation")

    print("EXITED SECURITY REVIEW")

    # Return placeholder response (needs real implementation)
    return {"security_review_response": "Security review completed - placeholder"}


# ===========================
# CONDITIONAL FUNCTIONS
# ===========================


def code_review_response(state: SDLCState) -> str:
    """
    Determine the next step based on code review results and iteration limit.

    This conditional function examines the code review response and current
    iteration count to decide whether to proceed to security review,
    continue with code fixes, or force approval after max iterations.

    Args:
        state: Current workflow state containing code review response and counter

    Returns:
        String indicating next step: "approved" or "feedback"

    Logic:
        1. Check if iteration counter is within limit (≤ 5)
        2. If within limit:
           - "approved" status → proceed to security review
           - "feedback" status → go to code fixing step
        3. If limit exceeded (> 5):
           - Force approval regardless of review status
           - Prevents infinite loops in review cycles

    Note:
        The 5-iteration limit prevents endless review cycles while
        still allowing reasonable improvement attempts.
    """
    print("*" * 50)
    print("ENTERED CODE REVIEW RESPONSE (CONDITION)")

    # Extract the status from the structured response
    state_response = state["code_review_response"].status

    print(f"Code Review Response: {state_response}")
    print(f"Current iteration count: {state['counter']}")

    print("EXITED CODE REVIEW RESPONSE (CONDITION)")

    # Check iteration limit and determine routing
    if state["counter"] < 2:
        if state_response == "approved":
            return "approved"
        else:
            return "feedback"
    else:
        # Exceeded iteration limit - force approval to prevent infinite loops
        print("⚠️  Maximum iterations (2) reached - forcing approval")
        return "approved"  # Force route to security review


# ===========================
# WORKFLOW GRAPH CONSTRUCTION
# ===========================

# Initialize the state graph with our enhanced state type
graph = StateGraph(SDLCState)

# Add all workflow nodes to the graph
graph.add_node("auto_generated_user_stories", auto_generated_user_stories)
graph.add_node("create_design_docs", create_design_docs)
graph.add_node("generate_code", generate_code)
graph.add_node("code_review", code_review)
graph.add_node("fix_code_after_code_review", fix_code_after_code_review)
graph.add_node("security_review", security_review)

# Define the sequential workflow edges
graph.add_edge(START, "auto_generated_user_stories")
graph.add_edge(
    "auto_generated_user_stories", "create_design_docs"
)  # Stories → Design docs
graph.add_edge("create_design_docs", "generate_code")
graph.add_edge("generate_code", "code_review")

# Add conditional edge based on code review results
graph.add_conditional_edges(
    "code_review",
    code_review_response,
    {
        "approved": "security_review",
        "feedback": "fix_code_after_code_review",
    },
)

graph.add_edge("fix_code_after_code_review", "code_review")

# Define terminal edge
graph.add_edge("security_review", END)

# Compile the workflow graph into executable workflow
workflow = graph.compile()

# ===========================
# WORKFLOW EXECUTION
# ===========================

if __name__ == "__main__":
    """
    Example execution of the iterative SDLC workflow.
    
    This section demonstrates how to use the enhanced workflow with
    counter-based iteration control for a task management application.
    
    Key Features Demonstrated:
    - Iterative code review with up to 5 attempts
    - Automatic loop termination to prevent infinite cycles
    - Comprehensive logging of iteration progress
    """

    sample_requirements = """
    I want to build a task management web application. 
    Features:
    - Users can create, update, and delete tasks.
    - Tasks should have priority and due dates.
    - A dashboard to track completed vs pending tasks.
    - Simple login with email and password.
    Tech Stack Preference: Python (FastAPI) for backend, React for frontend.
    Deployment: Host on AWS.
    """

    print("=" * 60)
    print("STARTING ITERATIVE SDLC WORKFLOW EXECUTION")
    print("=" * 60)

    # Execute the workflow with sample requirements
    response = workflow.invoke(
        {
            "user_input_requirements": sample_requirements,
            "counter": 0,
        }
    )

    print("=" * 60)
    print("WORKFLOW EXECUTION COMPLETED")
    print("=" * 60)
    print("Final Response:")
    print(response)
