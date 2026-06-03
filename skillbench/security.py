"""Authorization helpers for maintainer-triggered Skill Bench runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthorizationResult:
    """Maintainer authorization decision."""

    allowed: bool
    reason: str


def parse_actor_allowlist(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated actor allowlist."""
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def authorize_actor(actor: str, *, allowlist: tuple[str, ...]) -> AuthorizationResult:
    """Authorize a GitHub actor against an explicit allowlist."""
    if actor in allowlist:
        return AuthorizationResult(True, "actor is explicitly allowed")
    return AuthorizationResult(False, f"actor {actor!r} is not allowed")


def require_fresh_head(*, approved_sha: str, current_sha: str) -> AuthorizationResult:
    """Require that the PR head still matches the maintainer-approved SHA."""
    if approved_sha == current_sha:
        return AuthorizationResult(True, "PR head SHA matches approval")
    return AuthorizationResult(
        False,
        f"PR head moved after approval: approved={approved_sha} current={current_sha}",
    )
