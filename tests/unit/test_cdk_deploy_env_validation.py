"""Tests for CDK deploy script environment validation."""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_deploy_requires_bedrock_api_key_when_openai_compat_enabled():
    env = os.environ.copy()
    env["ENABLE_OPENAI_COMPAT"] = "true"
    env["PATH"] = "/nonexistent"
    env.pop("BEDROCK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        ["/bin/bash", "cdk/scripts/deploy.sh", "-e", "prod", "-s"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "BEDROCK_API_KEY is required when ENABLE_OPENAI_COMPAT=true" in output
    assert "Checking prerequisites" not in output
