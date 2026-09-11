from greeting import greeting


def test_greeting_uses_welcome_message() -> None:
    assert greeting("Ada") == "Welcome, Ada"