# pydantic-agent

An interactive chatbot built with [Pydantic AI](https://ai.pydantic.dev/), demonstrating agent construction, tool use, and multi-turn conversation.

## Tools

- **Date/time** — returns the current datetime
- **Calculator** — safely evaluates mathematical expressions via AST
- **Web search** — queries DuckDuckGo and returns top 3 results
- **Fetch URL** — retrieves a URL and extracts the main readable content as Markdown (via `httpx` + `trafilatura`)

## Setup

Requires [uv](https://docs.astral.sh/uv/).

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

## Running

```bash
uv run --env-file .env chatbot.py
```
