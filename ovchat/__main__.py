import sys
from .cli import main
from .ui import UI

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{UI.YELLOW}Exited.{UI.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{UI.RED}Fatal error:{UI.RESET} {e}\n")
        try:
            input("Press Enter...")
        except (KeyboardInterrupt, EOFError):
            pass
        sys.exit(1)

