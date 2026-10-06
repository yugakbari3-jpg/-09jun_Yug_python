"""
Jarvis Voice Assistant - Main Entry Point

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

if __package__ in (None, ""):
    # Run directly as `python main.py`: make the `jarvis` package importable.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jarvis.daemon import main
else:
    from .daemon import main

if __name__ == "__main__":
    main()
