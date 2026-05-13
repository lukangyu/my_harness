# coding-agent

Python AI coding assistant CLI.

## Install for development

```bash
python -m pip install -e ".[dev]"
```

## Initialize a project

```bash
coding-agent init
```

Set the configured API key environment variable:

```bash
export OPENAI_API_KEY="..."
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
```

## Run one task

```bash
coding-agent run "inspect the project and summarize it"
```

## Chat mode

```bash
coding-agent chat
```

Available commands:

- `/exit`
- `/clear`
- `/status`

## Safety model

File access is limited to the configured workspace root, which defaults to the current project. Shell commands must match the configured allow list and must not match the deny list. Deny rules take precedence.
