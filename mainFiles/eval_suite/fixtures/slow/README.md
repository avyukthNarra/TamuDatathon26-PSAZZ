# Fixture: slow

Deliberately invalid at runtime. Sleeps 30 seconds before writing a uniform forecast. Tests run it with
a short timeout, so the run is killed and CON-RUNTIME fails. Standard library only.
