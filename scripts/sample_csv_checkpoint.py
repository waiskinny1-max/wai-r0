from wai_r0.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sample-csv", *(__import__("sys").argv[1:])]))
