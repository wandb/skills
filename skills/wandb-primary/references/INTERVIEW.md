# Skill Interview

This interview configures the W&B Primary Skill for the user's specific environment and preferences. Complete it once — these settings are stable across projects.

Project-specific details (metric keys, config keys, entity/project) are discovered per-task via `probe_project()` and the user's request. They do NOT belong here.

**AGENT**: Follow each section in order. Auto-detect first, then confirm or ask. After both sections are done, update the Configuration table in `SKILL.md` and set `interview_completed` to `true`.

---

## 1. Python Environment

**Goal**: Determine how the user runs Python scripts and installs packages.

**Auto-detect** (check in order, stop at first match):
1. `uv.lock` exists → `python_run: uv run python`, `python_install: uv add`
2. `poetry.lock` exists → `python_run: poetry run python`, `python_install: poetry add`
3. `Pipfile` exists → `python_run: pipenv run python`, `python_install: pipenv install`
4. `environment.yml` or `$CONDA_DEFAULT_ENV` → `python_run: conda run python`, `python_install: conda install`
5. `.venv/` or `venv/` or `$VIRTUAL_ENV` → `python_run: python`, `python_install: pip install`
6. `Dockerfile` or `docker-compose.yml` → `python_run: docker exec <ctr> python`, `python_install: docker exec <ctr> pip install`
7. `requirements.txt` or `setup.py` → `python_run: python`, `python_install: pip install`
8. Nothing found → default to `python_run: uv run python`, `python_install: uv add`

**If auto-detected**, briefly confirm:
> I detected you're using **uv** — I'll use `uv run python` for scripts and `uv add` for packages. Sound right?

**Write to Configuration**:
- `python_run`
- `python_install`

---

## 2. LLM Provider & Model

**Goal**: Determine which LLM provider, model, reasoning level, and API endpoint the user prefers for LLM-powered analysis tasks (e.g., Weave scoring, eval rubrics, trace analysis). This also tells us which conventions to use for token counting and trace op names.

**Auto-detect** (check in order):
1. `$OPENAI_API_KEY` set → likely `openai`
2. `$ANTHROPIC_API_KEY` set → likely `anthropic`
3. Imports in codebase: `import openai`, `import anthropic`
4. Both set → ask which they prefer

**Ask the user**:
> Which LLM provider and model do you want me to use for analysis tasks?
>
> - **Provider**: openai / anthropic / google / other
> - **Model**: e.g., `gpt-5.4-mini`, `claude-sonnet-4-6`, etc.
> - **Reasoning effort**: low / medium / high (if supported)
> - **Endpoint style**: `responses` / `chat.completions` (OpenAI), `messages` (Anthropic)

**Write to Configuration**:
- `llm_provider`: e.g., `openai`
- `llm_model`: e.g., `gpt-5.4-mini`
- `llm_reasoning`: e.g., `high` (or `_none_` if not applicable)
- `llm_endpoint`: e.g., `responses`, `chat.completions`, `messages`

---

## Completion

After both sections are answered, update the Configuration table in `SKILL.md`:

1. Replace each `_not set_` value with the detected/answered value
2. Set `interview_completed` to `true`
3. Confirm to the user:

> **Configuration complete.** Here's your setup:
>
> | Setting | Value |
> |---------|-------|
> | Python | `<python_run>` / `<python_install>` |
> | LLM | `<llm_provider>` / `<llm_model>` |
> | Reasoning | `<llm_reasoning>` |
> | Endpoint | `<llm_endpoint>` |
>
> Metric and config keys will be discovered per-project via `probe_project()`.
> Ready to work. What would you like me to do?
