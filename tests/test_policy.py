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


def test_rejects_unix_chained_command_before_allow_matching():
    policy = CommandPolicy(allow=["pytest"], deny=["rm -rf"])

    result = policy.evaluate("pytest && rm -rf build")

    assert result.decision != CommandDecision.ALLOW
    assert result.decision == CommandDecision.REJECT
    assert "shell control operator" in result.reason


def test_rejects_windows_chained_command_before_allow_matching():
    policy = CommandPolicy(allow=["pytest"], deny=["del marker.txt"])

    result = policy.evaluate("pytest & del marker.txt")

    assert result.decision != CommandDecision.ALLOW
    assert result.decision == CommandDecision.REJECT
    assert "shell control operator" in result.reason


def test_rejects_single_quoted_windows_chained_command_before_allow_matching():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    result = policy.evaluate("pytest '& del marker.txt'")

    assert result.decision == CommandDecision.REJECT
    assert "shell control operator" in result.reason


def test_rejects_shell_control_operators_before_allow_matching():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    for command in [
        "pytest || python -c pass",
        "pytest | cat",
        "pytest; python -c pass",
        "pytest\npython -c pass",
        "pytest\rpython -c pass",
    ]:
        result = policy.evaluate(command)

        assert result.decision == CommandDecision.REJECT
        assert "shell control operator" in result.reason


def test_rejects_unlisted_command():
    policy = CommandPolicy(allow=["pytest"], deny=[])

    result = policy.evaluate("python -m pytest")

    assert result.decision == CommandDecision.REJECT
    assert "not in allow list" in result.reason


def test_strips_empty_rules_and_normalizes_command_whitespace():
    policy = CommandPolicy(allow=["", "pytest"], deny=["   "])

    result = policy.evaluate("  pytest   tests  ")

    assert result.decision == CommandDecision.ALLOW
