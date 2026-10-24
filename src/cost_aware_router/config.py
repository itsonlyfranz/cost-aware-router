"""Settings loaded from the environment (and a gitignored .env file)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from cost_aware_router.models import ModelSpec, Tier

# Illustrative USD-per-million-token prices; override with CAR_PRICE_<TIER>=in,out.
DEFAULT_PRICES: dict[Tier, tuple[float, float]] = {
    Tier.FAST: (1.0, 5.0),
    Tier.BALANCED: (3.0, 15.0),
    Tier.PREMIUM: (5.0, 25.0),
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, existing environment wins."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


class Settings(BaseModel):
    default_model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-5-5"
    premium_model: str = "claude-opus-5-5"
    prices: dict[Tier, tuple[float, float]] = Field(default_factory=lambda: dict(DEFAULT_PRICES))
    baseline_tier: Tier = Tier.PREMIUM  # what "no router" costs: every task on this tier
    confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    data_dir: Path = Path(".car")
    api_key: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        kwargs: dict = {}
        for field, var in [
            ("default_model", "CAR_DEFAULT_MODEL"),
            ("fast_model", "CAR_MODEL_FAST"),
            ("premium_model", "CAR_MODEL_PREMIUM"),
            ("confidence_threshold", "CAR_CONFIDENCE_THRESHOLD"),
            ("data_dir", "CAR_DATA_DIR"),
            ("api_key", "ANTHROPIC_API_KEY"),
        ]:
            if e.get(var):
                kwargs[field] = e[var]
        if e.get("CAR_BASELINE_TIER"):
            kwargs["baseline_tier"] = Tier[e["CAR_BASELINE_TIER"].upper()]
        prices = dict(DEFAULT_PRICES)
        for tier in Tier:
            raw = e.get(f"CAR_PRICE_{tier.name}")
            if raw:
                inp, out = (float(x) for x in raw.split(","))
                prices[tier] = (inp, out)
        kwargs["prices"] = prices
        return cls(**kwargs)

    def models(self) -> list[ModelSpec]:
        names = {Tier.FAST: self.fast_model, Tier.BALANCED: self.default_model, Tier.PREMIUM: self.premium_model}
        return [
            ModelSpec(name=names[t], tier=t, input_per_mtok=self.prices[t][0], output_per_mtok=self.prices[t][1])
            for t in Tier
        ]

    @property
    def decisions_path(self) -> Path:
        return self.data_dir / "decisions.jsonl"
