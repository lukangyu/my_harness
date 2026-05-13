from coding_agent.policy import CommandDecision, CommandPolicy


def test_allows_exact_command():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    result = policy.evaluate("pytest")

    assert result.decision == CommandDecision.ALLOW


def test_allows_command_with_allowed_prefix():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    result = policy.evaluate("pytest tests")

    assert result.decision == CommandDecision.ALLOW


def test_does_not_match_partial_prefix():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    result = policy.evaluate("pytestx tests")

    assert result.decision == CommandDecision.REJECT
    assert "not in allow list" in result.reason


def test_denies_exact_command():
    policy = CommandPolicy(allow=["pytest"], deny=["pytest"])

    result = policy.evaluate("pytest")

    assert result.decision == CommandDecision.DENY


def test_denies_command_with_denied_prefix():
    policy = CommandPolicy(allow=["pytest"], deny=["pytest tests/blocked"])

    result = policy.evaluate("pytest tests/blocked -v")

    assert result.decision == CommandDecision.DENY


def test_deny_rules_take_priority_over_allow_rules():
    policy = CommandPolicy(allow=["pytest"], deny=["pytest tests/blocked"])

    result = policy.evaluate("pytest tests/blocked")

    assert result.decision == CommandDecision.DENY


def test_rejects_unlisted_command():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    result = policy.evaluate("python -m pytest")

    assert result.decision == CommandDecision.REJECT
    assert "not in allow list" in result.reason


def test_strips_empty_rules_and_normalizes_command_whitespace():
    policy = CommandPolicy(allow=["", "pytest"], deny=["   "])

    result = policy.evaluate("  pytest   tests  ")

    assert result.decision == CommandDecision.ALLOW
