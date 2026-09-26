"""Smoke tests — must pass on the starter repo (no API key)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_required_docs_exist():
    for rel in [
        "README.md",
        "SUBMISSION.md",
        "CHECKPOINTS.md",
        "RUBRIC.md",
        "RULES.md",
        ".env.example",
        "requirements.txt",
        "schemas/results.schema.json",
        "data/protected/vinbank_secrets.json",
    ]:
        assert (ROOT / rel).is_file(), f"Missing {rel}"


def test_protected_data_json_exists():
    path = ROOT / "data" / "protected" / "vinbank_secrets.json"
    assert path.is_file(), "Missing data/protected/vinbank_secrets.json"
    import json
    import sys

    data = json.loads(path.read_text(encoding="utf-8"))
    secrets = data.get("secrets") or {}
    assert secrets.get("admin_password")
    assert secrets.get("api_key")
    assert secrets.get("db_host")
    assert isinstance(data.get("leak_targets"), list) and len(data["leak_targets"]) >= 3

    src = ROOT / "src"
    sys.path.insert(0, str(src))
    from core.config import DEMO_SECRETS, DEMO_SECRET_NOTE, load_protected_payload

    payload = load_protected_payload()
    assert payload["secrets"]["admin_password"] == secrets["admin_password"]
    assert len(DEMO_SECRETS) >= 3
    assert secrets["admin_password"] in DEMO_SECRET_NOTE


def test_assignment_starters_exist():
    for rel in [
        "src/assignment/rate_limiter.py",
        "src/assignment/audit_log.py",
        "src/assignment/monitoring.py",
        "src/assignment/pipeline.py",
        "src/agents/guards_agent.py",
    ]:
        assert (ROOT / rel).is_file(), f"Missing {rel}"


def test_red_agent_advance_exports_factory():
    import sys

    src = ROOT / "src"
    sys.path.insert(0, str(src))
    from agents.guards_agent import (
        create_red_agent_advance,
        create_guards_agent,
        check_secret_leak,
        GUARDS_SECRETS,
        RED_ADVANCE_SECRETS,
    )

    assert callable(create_red_agent_advance)
    assert create_guards_agent is create_red_agent_advance
    assert callable(check_secret_leak)
    assert len(RED_ADVANCE_SECRETS) >= 3
    assert GUARDS_SECRETS == RED_ADVANCE_SECRETS
    assert check_secret_leak("the key is sk-vinbank-secret-2024") is True
    assert check_secret_leak("hello banking") is False


def test_no_solution_notebook_shipped():
    sol = ROOT / "notebooks" / "lab11_guardrails_hitl_solution.ipynb"
    assert not sol.exists(), "Do not ship solution notebook"


def test_results_schema_is_valid_jsonschema():
    import json
    import jsonschema

    schema = json.loads((ROOT / "schemas" / "results.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_red_advance_blocks_attacks_before_model_run():
    import asyncio
    import sys
    from types import SimpleNamespace

    from google.genai import types

    src = ROOT / "src"
    sys.path.insert(0, str(src))
    from agents.guards_agent import GuardsInputPlugin
    from attacks.attacks import adversarial_prompts

    plugin = GuardsInputPlugin()

    async def check_inputs():
        for attack in adversarial_prompts:
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=attack["input"])],
            )
            result = await plugin.before_run_callback(
                invocation_context=SimpleNamespace(user_content=content)
            )
            assert result is not None
            assert result.role == "model"

    asyncio.run(check_inputs())
    assert plugin.total_count == len(adversarial_prompts)
    assert plugin.blocked_count == len(adversarial_prompts)
