import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path):
    print(f"[main] running: {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main():
    parser = argparse.ArgumentParser(description="ECE276A PR2 unified runner")
    parser.add_argument(
        "--part",
        choices=["part1", "part2", "part3", "part4", "all"],
        default="all",
        help="which part to run",
    )
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    run_odom = code_dir / "experiments" / "run_odom_only.py"
    run_icp = code_dir / "experiments" / "run_icp_scanmatch.py"
    run_map = code_dir / "experiments" / "run_mapping.py"
    run_pg = code_dir / "experiments" / "run_pose_graph.py"

    if args.part in ("part1", "all"):
        run_script(run_odom)
    if args.part in ("part2", "all"):
        run_script(run_icp)
    if args.part in ("part3", "all"):
        run_script(run_map)
    if args.part in ("part4", "all"):
        run_script(run_pg)

    print("[main] done")


if __name__ == "__main__":
    main()
