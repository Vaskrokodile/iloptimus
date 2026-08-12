---
name: "git-workflow"
description: "Guidelines for git operations: commits, branches, PRs"
slash: true
---

# Git Workflow Skill

When performing git operations, follow these guidelines:

## Commits
- Always check `git status` and `git diff` before committing
- Write concise commit messages focused on "why" not "what"
- Use conventional commit format: `type(scope): description`
- Types: feat, fix, refactor, docs, test, chore, perf, ci

## Branches
- Create descriptive branch names: `feat/add-skills`, `fix/mcp-reconnect`
- Never force-push without explicit user confirmation
- Never commit secrets, keys, or sensitive data

## Pull Requests
- Use `gh` for all GitHub operations
- Include a clear summary with bullet points
- Include a test plan checklist

## Safety
- NEVER update git config
- NEVER use `-i` flags (interactive mode)
- DO NOT push unless explicitly asked
- DO NOT commit if no changes exist
