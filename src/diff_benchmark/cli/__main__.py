import sys


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: diffbenchmark <features|run> [hydra overrides]")

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "features":
        from diff_benchmark.cli.features import main

        main()

    elif command == "run":
        from diff_benchmark.cli.run import main

        main()

    elif command == "analysis":
        from diff_benchmark.cli.analysis import main

        main()

    else:
        raise ValueError(f"Unknown command: {command}")
