import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import List


AVG_RE = re.compile(r"Average Runtime:\s*([0-9]*\.?[0-9]+)s")
COMP_RE = re.compile(r"Compilation Time:\s*([0-9]*\.?[0-9]+)s")


@dataclass
class RunResult:
    average_runtime: float
    compilation_time: float


def parse_metrics(output: str) -> RunResult:
    avg_match = AVG_RE.search(output)
    comp_match = COMP_RE.search(output)
    if not avg_match or not comp_match:
        raise ValueError("Could not parse benchmark metrics from output.")
    return RunResult(
        average_runtime=float(avg_match.group(1)),
        compilation_time=float(comp_match.group(1)),
    )


def run_once(python_exec: str, algo: str, N: int, T: int, tau_max: int, repeats: int) -> RunResult:
    cmd = [
        python_exec,
        "benchmark.py",
        "--algo",
        algo,
        "--N",
        str(N),
        "--T",
        str(T),
        "--tau_max",
        str(tau_max),
        "--repeats",
        str(repeats),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"Benchmark command failed with code {completed.returncode}")
    return parse_metrics(completed.stdout)


def summarize(values: List[float]) -> str:
    return (
        f"mean={statistics.mean(values):.4f}s, "
        f"median={statistics.median(values):.4f}s, "
        f"min={min(values):.4f}s, "
        f"max={max(values):.4f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stable multi-process PCMCI benchmark summary.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--algo", default="pcmci", choices=["pcmci", "pcmci+"])
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--tau_max", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    results: List[RunResult] = []
    for run_idx in range(args.runs):
        result = run_once(args.python, args.algo, args.N, args.T, args.tau_max, args.repeats)
        results.append(result)
        print(
            f"Run {run_idx + 1}/{args.runs}: "
            f"avg={result.average_runtime:.4f}s, compile={result.compilation_time:.4f}s"
        )

    avg_times = [r.average_runtime for r in results]
    comp_times = [r.compilation_time for r in results]

    print("\nStable Summary")
    print(f"Average runtime: {summarize(avg_times)}")
    print(f"Compilation time: {summarize(comp_times)}")


if __name__ == "__main__":
    main()
