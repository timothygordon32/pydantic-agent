"""
Pydantic AI Agentic Chatbot Example

Demonstrates core Pydantic.AI concepts:
- Agent construction with system prompt
- Tool definitions (@agent.tool_plain)
- Multi-turn conversation with message history
- Safe code execution (calculator)
- External I/O (web search)
"""

import ast
import operator
from datetime import datetime

import httpx
import trafilatura
from ddgs import DDGS
from pydantic_ai import Agent

# Create the agent with a helpful system prompt
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    system_prompt=(
        'You are a helpful, friendly assistant with access to tools for:'
        '\n- Telling the current date and time'
        '\n- Performing calculations'
        '\n- Searching the web for information'
        '\n- Fetching the content of a URL'
        '\n\nUse these tools when the user asks for the relevant information.'
        '\nWhen the user asks about the content of a specific page, use fetch_url.'
        '\nIf you only have a topic, use web_search first, then fetch_url on a promising result.'
    ),
)


# Tool 1: Get current datetime (simplest form)
@agent.tool_plain
def get_current_datetime() -> str:
    """Get the current date and time in ISO format."""
    return datetime.now().isoformat()


# Tool 2: Safe calculator using AST
@agent.tool_plain
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, //, %, ** and parentheses.
    Examples: '2 + 2', '10 * 5', '100 / 4'

    Args:
        expression: A mathematical expression to evaluate.
    """
    try:
        # Safely parse and evaluate the expression
        result = _safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression using AST.

    Only allows numeric literals, binary operations, and unary operations.
    Prevents arbitrary code execution.
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')

        # Validate the AST contains only allowed operations
        _validate_ast(tree.body)

        # Compile and evaluate
        compiled = compile(tree, '<string>', 'eval')
        return eval(compiled)
    except ValueError as e:
        raise ValueError(f"Invalid expression: {e}")


def _validate_ast(node: ast.expr) -> None:
    """Recursively validate AST node for safe operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # Numbers
        return
    elif isinstance(node, ast.BinOp):  # Binary operations: +, -, *, /, etc.
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.FloorDiv, ast.Mod, ast.Pow)):
            raise ValueError(f"Unsupported binary operation: {node.op.__class__.__name__}")
        _validate_ast(node.left)
        _validate_ast(node.right)
    elif isinstance(node, ast.UnaryOp):  # Unary operations: -, +
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ValueError(f"Unsupported unary operation: {node.op.__class__.__name__}")
        _validate_ast(node.operand)
    else:
        raise ValueError(f"Unsupported AST node type: {node.__class__.__name__}")


# Tool 3: Web search
@agent.tool_plain
def web_search(query: str) -> str:
    """
    Search the web for information using DuckDuckGo.

    Returns the top 3 results with titles and snippets.

    Args:
        query: What to search for.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)

        if not results:
            return "No results found."

        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   {result['body']}")
            output.append(f"   {result['href']}\n")

        return '\n'.join(output)
    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"


# Tool 4: Fetch a URL and extract main content
@agent.tool_plain
def fetch_url(url: str) -> str:
    """
    Fetch a URL and return its main readable content as Markdown.

    Use after web_search to get the full content of a promising result.

    Args:
        url: The URL to fetch.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
    }
    try:
        response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        response.raise_for_status()

        content = trafilatura.extract(response.text, output_format='markdown')
        if not content:
            return "Could not extract readable content from the page."

        max_length = 8000
        if len(content) > max_length:
            content = content[:max_length] + '\n\n… [truncated]'
        return content
    except Exception as e:
        return f"Fetch error: {type(e).__name__}: {e}"


def main():
    """Run the interactive chatbot."""
    print("=" * 60)
    print("Pydantic.AI Agentic Chatbot")
    print("=" * 60)
    print("\nI'm a helpful assistant with tools for:")
    print("  • Telling the current date/time")
    print("  • Performing calculations")
    print("  • Searching the web")
    print("  • Fetching the content of a URL")
    print("\nType 'quit' or 'exit' to leave.\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            # Run the agent with conversation history
            result = agent.run_sync(user_input, message_history=messages)

            # Update message history for next turn
            messages = result.all_messages()

            print(f"\nBot: {result.output}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}\n")


if __name__ == '__main__':
    main()
