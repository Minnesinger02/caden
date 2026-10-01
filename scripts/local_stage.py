"""Run one pilot stage with persistent logs, board memory samples and handoff notes."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=8)))
    folder = ROOT / "handoff/runs" / (now.strftime("%Y%m%dT%H%M%S") + "-" + args.stage)
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "scripts/run_16gb.py", args.stage, *args.options]
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    environment.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    used, utilization, temperature, power = map(float, result.stdout.strip().splitlines()[0].split(","))
                    samples.append({"elapsed_seconds": time.perf_counter() - started,
                                    "board_memory_mib": used, "gpu_utilization_percent": utilization,
                                    "temperature_c": temperature, "power_w": power})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            stop.wait(2)

    started = time.perf_counter()
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    with (folder / "console.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            stop.set()
            worker.join(timeout=12)
    report = {"started_at": now.isoformat(), "stage": args.stage, "command": command,
              "exit_code": code, "wall_seconds": time.perf_counter() - started,
              "board_peak_memory_mib": max((s["board_memory_mib"] for s in samples), default=None),
              "board_memory_scope": "sampled entire GPU including desktop/other processes, 2-second interval; allocator peaks are in training/evaluation metadata",
              "gpu_samples": samples}
    (folder / "stage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (ROOT / "handoff/PROGRESS.md").open("a", encoding="utf-8") as progress:
        progress.write(f"\n### {now.strftime('%Y-%m-%d %H:%M:%S')} +08:00 — {args.stage}\n\n")
        progress.write(f"- 命令：`{' '.join(command)}`\n")
        progress.write(f"- 退出码：{code}；墙钟耗时：{report['wall_seconds']:.2f} 秒；整卡采样峰值：{report['board_peak_memory_mib']} MiB（包含桌面占用）。\n")
        progress.write(f"- 日志和显卡采样：`{folder.relative_to(ROOT).as_posix()}`。\n")
        progress.write("- 阶段成功，继续按交接顺序运行下一阶段。\n" if code == 0 else "- 阶段失败，先检查 console.log；保留现场，使用新结果目录重跑。\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
