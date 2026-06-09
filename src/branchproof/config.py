from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    work_dir: Path = Path("work")
    runs_dir: Path = Path("runs")

    max_attempts: int = 3
    stability_runs: int = 3
    test_timeout_s: int = 120
    suite_timeout_s: int = 300

    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str = "2024-10-21"
    azure_model: str | None = None
    azure_reasoning_effort: str | None = None
    azure_max_completion_tokens: int | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            work_dir=Path(os.getenv("HARDENER_WORK_DIR", "work")).resolve(),
            runs_dir=Path(os.getenv("HARDENER_RUNS_DIR", "runs")).resolve(),
            max_attempts=int(os.getenv("HARDENER_MAX_ATTEMPTS", "3")),
            stability_runs=int(os.getenv("HARDENER_STABILITY_RUNS", "3")),
            test_timeout_s=int(os.getenv("HARDENER_TEST_TIMEOUT", "120")),
            suite_timeout_s=int(os.getenv("HARDENER_SUITE_TIMEOUT", "300")),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_model=os.getenv("AZURE_OPENAI_MODEL") or None,
            azure_reasoning_effort=os.getenv("HARDENER_REASONING_EFFORT") or None,
            azure_max_completion_tokens=(int(os.environ["HARDENER_MAX_COMPLETION_TOKENS"])
                                         if os.getenv("HARDENER_MAX_COMPLETION_TOKENS") else None),
        )

    def has_azure(self) -> bool:
        return bool(self.azure_endpoint and self.azure_api_key and self.azure_deployment)
