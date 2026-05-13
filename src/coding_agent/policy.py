from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CommandDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REJECT = "reject"


@dataclass(frozen=True)
class PolicyResult:
    decision: CommandDecision
    reason: str


@dataclass(frozen=True)
class CommandPolicy:
    allow: list[str]
    deny: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow", _normalize_rules(self.allow))
        object.__setattr__(self, "deny", _normalize_rules(self.deny))

    def evaluate(self, command: str) -> PolicyResult:
        normalized_command = _normalize(command)

        for rule in self.deny:
            if _matches_rule(normalized_command, rule):
                return PolicyResult(CommandDecision.DENY, f"Command denied by rule: {rule}")

        for rule in self.allow:
            if _matches_rule(normalized_command, rule):
                return PolicyResult(CommandDecision.ALLOW, f"Command allowed by rule: {rule}")

        return PolicyResult(CommandDecision.REJECT, "Command not in allow list")


def _normalize_rules(rules: list[str]) -> list[str]:
    return [normalized for rule in rules if (normalized := _normalize(rule))]


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _matches_rule(command: str, rule: str) -> bool:
    return command == rule or command.startswith(f"{rule} ")
