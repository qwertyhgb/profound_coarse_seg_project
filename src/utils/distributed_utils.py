"""Distributed-training placeholders.

Stage 1 runs in single-process mode by default. These helpers keep the code ready
for future DDP integration without coupling the current project to a launcher.
"""

def is_main_process() -> bool:
    """Return True for the current single-process training setup."""
    return True
