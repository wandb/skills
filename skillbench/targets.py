"""Load Skill Bench target configuration from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SelectMode = Literal["intersect", "union"]


@dataclass(frozen=True)
class SelectSpec:
    """Task-selection axes for a benchmark run."""

    scenarios: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()
    must_pass_only: bool = False
    mode: SelectMode = "intersect"

    @property
    def is_empty(self) -> bool:
        """Return whether this spec applies no task constraints."""
        return (
            not self.scenarios
            and not self.tasks
            and not self.categories
            and not self.levels
            and not self.must_pass_only
        )

    def merge(self, other: "SelectSpec") -> "SelectSpec":
        """Merge a profile or CLI override onto this target selection."""
        return SelectSpec(
            scenarios=other.scenarios or self.scenarios,
            tasks=other.tasks or self.tasks,
            categories=other.categories or self.categories,
            levels=other.levels or self.levels,
            must_pass_only=other.must_pass_only or self.must_pass_only,
            mode=other.mode or self.mode,
        )


@dataclass(frozen=True)
class BenchTarget:
    """Benchmark defaults for one public skill."""

    publish_name: str
    suite: str
    agent: str
    timeout_seconds: int = 1800
    select: SelectSpec = field(default_factory=SelectSpec)


@dataclass(frozen=True)
class BenchProfile:
    """Named benchmark profile, usually smoke or release."""

    name: str
    select: SelectSpec = field(default_factory=SelectSpec)
    suite: str | None = None
    agent: str | None = None
    timeout_seconds: int | None = None
    replace_select: bool = False
    trials: int = 1
    description: str = ""


@dataclass(frozen=True)
class BenchTargets:
    """Loaded target configuration."""

    default: BenchTarget
    skills: tuple[BenchTarget, ...]
    profiles: tuple[BenchProfile, ...] = ()

    def for_skill(self, publish_name: str) -> BenchTarget:
        """Return the target for a skill, falling back to defaults."""
        for target in self.skills:
            if target.publish_name == publish_name:
                return target
        return BenchTarget(
            publish_name=publish_name,
            suite=self.default.suite,
            agent=self.default.agent,
            timeout_seconds=self.default.timeout_seconds,
            select=self.default.select,
        )

    def profile(self, name: str) -> BenchProfile:
        """Return a named profile, falling back to an empty profile."""
        for profile in self.profiles:
            if profile.name == name:
                return profile
        return BenchProfile(name=name)


def apply_profile(target: BenchTarget, profile: BenchProfile) -> BenchTarget:
    """Apply runtime profile overrides to a target."""
    select = profile.select if profile.replace_select else target.select.merge(profile.select)
    return BenchTarget(
        publish_name=target.publish_name,
        suite=profile.suite or target.suite,
        agent=profile.agent or target.agent,
        timeout_seconds=profile.timeout_seconds or target.timeout_seconds,
        select=select,
    )


def _string_tuple(raw: object, *, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        out.append(item.strip())
    return tuple(out)


def _select(raw: dict[str, object], *, context: str) -> SelectSpec:
    mode = raw.get("mode", "intersect")
    if mode not in ("intersect", "union"):
        raise ValueError(f"{context}.mode must be 'intersect' or 'union'")
    return SelectSpec(
        scenarios=_string_tuple(raw.get("scenarios"), field_name=f"{context}.scenarios"),
        tasks=_string_tuple(raw.get("tasks"), field_name=f"{context}.tasks"),
        categories=_string_tuple(raw.get("categories"), field_name=f"{context}.categories"),
        levels=_string_tuple(raw.get("levels"), field_name=f"{context}.levels"),
        must_pass_only=bool(raw.get("must_pass_only", False)),
        mode=mode,
    )


def _target(raw: dict[str, object], *, default: BenchTarget | None = None) -> BenchTarget:
    publish_name = str(raw.get("publish_name") or (default.publish_name if default else ""))
    if not publish_name:
        raise ValueError("target publish_name is required")
    return BenchTarget(
        publish_name=publish_name,
        suite=str(raw.get("suite") or (default.suite if default else "")),
        agent=str(raw.get("agent") or (default.agent if default else "")),
        timeout_seconds=int(
            raw.get("timeout_seconds") or (default.timeout_seconds if default else 1800)
        ),
        select=_select(raw, context=f"skills.{publish_name}"),
    )


def load(path: Path) -> BenchTargets:
    """Load `bench/targets.toml`."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults_raw = raw.get("defaults") or {}
    if not isinstance(defaults_raw, dict):
        raise ValueError("defaults must be a TOML table")
    default = _target({"publish_name": "<default>", **defaults_raw})

    skills_raw = raw.get("skills") or []
    if not isinstance(skills_raw, list):
        raise ValueError("skills must be an array of tables")
    skills = tuple(
        _target(skill_raw, default=default)
        for skill_raw in skills_raw
        if isinstance(skill_raw, dict)
    )

    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, dict):
        raise ValueError("profiles must be a TOML table")
    profiles = tuple(
        BenchProfile(
            name=name,
            select=_select(profile_raw, context=f"profiles.{name}"),
            suite=str(profile_raw["suite"]) if "suite" in profile_raw else None,
            agent=str(profile_raw["agent"]) if "agent" in profile_raw else None,
            timeout_seconds=int(profile_raw["timeout_seconds"])
            if "timeout_seconds" in profile_raw
            else None,
            replace_select=bool(profile_raw.get("replace_select", False)),
            trials=int(profile_raw.get("trials", 1)),
            description=str(profile_raw.get("description", "")),
        )
        for name, profile_raw in profiles_raw.items()
        if isinstance(profile_raw, dict)
    )
    return BenchTargets(default=default, skills=skills, profiles=profiles)
