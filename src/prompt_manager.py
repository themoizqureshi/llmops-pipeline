"""
Prompt version manager.

Prompts are treated as code: versioned in git, stored in the prompts/ directory,
with a registry.json mapping versions to files and tracking their eval scores.

This is the core LLMOps insight: the system prompt is not a string in your code —
it's a versioned artifact that determines system behaviour, must be tested before
deployment, and must be rollback-able if it causes a regression.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("prompts/registry.json")


def load_prompt(version: str) -> str:
    """Load a prompt template by version string (e.g. 'v1', 'v2')."""
    registry = _load_registry()
    entry = next((v for v in registry["versions"] if v["version"] == version), None)
    if entry is None:
        raise ValueError(f"Prompt version '{version}' not found in registry. "
                         f"Available: {[v['version'] for v in registry['versions']]}")
    path = Path(entry["file"])
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    content = path.read_text().strip()
    logger.info(f"Loaded prompt {version} from {path}")
    return content


def get_current_production_version() -> str:
    """Return the version currently marked as production in the registry."""
    registry = _load_registry()
    return registry.get("current_production", "v1")


def update_prompt_metrics(version: str, metrics: Dict[str, float]) -> None:
    """
    Write eval scores back into the registry for a given prompt version.

    Call this after running evaluation so the registry always reflects
    the latest measured performance of each prompt version.
    """
    registry = _load_registry()
    for entry in registry["versions"]:
        if entry["version"] == version:
            entry["metrics"].update({k: round(v, 3) for k, v in metrics.items()})
            break
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    logger.info(f"Updated metrics for prompt {version} in registry")


def list_versions() -> list:
    """Return all registered prompt versions with their metrics."""
    registry = _load_registry()
    return registry["versions"]


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Prompt registry not found at {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text())
