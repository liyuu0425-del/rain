#!/usr/bin/env python3
"""顺序运行阶段1-6，并报告各阶段的完整耗时。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "pipeline_run" / "pipeline_report.json"
STAGES = (
    (1, "stage1_data_audit.py"),
    (2, "stage2_detection.py"),
    (3, "stage3_tracking.py"),
    (4, "stage4_stereo_matching.py"),
    (5, "stage5_velocity.py"),
    (6, "stage6_statistics.py"),
)
STAGE_NAMES = {
    1: "数据读取与审计",
    2: "雨滴检测",
    3: "单路轨迹跟踪",
    4: "双路匹配",
    5: "速度测量",
    6: "雨滴谱与统计",
}
BEIJING_TIME = timezone(timedelta(hours=8))


def main() -> int:
    parser = argparse.ArgumentParser(description="按顺序运行阶段1-6，并生成逐阶段耗时报告")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="流水线配置文件路径")
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="JSON耗时报告路径；同目录生成同名中文CSV报告",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    report_path = Path(args.report).resolve()
    if not config_path.exists():
        parser.error(f"配置文件不存在：{config_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_started = _now_utc()
    run_t0 = time.perf_counter()
    stage_results: list[dict] = []
    print("=" * 70)
    print("开始运行阶段1-6完整流水线")
    print(f"配置文件：{config_path}")
    print("=" * 70, flush=True)

    for number, filename in STAGES:
        script = Path(__file__).with_name(filename)
        print(f"\n[{number}/6] 阶段{number}：{STAGE_NAMES[number]}", flush=True)
        stage_started = _now_utc()
        stage_t0 = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script), "--config", str(config_path)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        elapsed_s = time.perf_counter() - stage_t0
        result = {
            "stage": number,
            "script": filename,
            "started_at_utc": stage_started,
            "ended_at_utc": _now_utc(),
            "elapsed_s": elapsed_s,
            "return_code": completed.returncode,
            "status": "passed" if completed.returncode == 0 else "failed",
        }
        stage_results.append(result)
        _write_reports(report_path, config_path, run_started, stage_results, run_t0)
        print(f"阶段{number}{'完成' if completed.returncode == 0 else '失败'}，耗时：{elapsed_s:.2f} 秒", flush=True)
        if completed.returncode != 0:
            print(f"阶段{number}运行失败，已停止后续阶段（退出码：{completed.returncode}）。")
            return completed.returncode

    _write_reports(report_path, config_path, run_started, stage_results, run_t0)
    print("\n" + "=" * 70)
    print("阶段1-6全部完成，逐阶段耗时汇总：")
    for result in stage_results:
        number = result["stage"]
        print(f"  阶段{number} {STAGE_NAMES[number]}：{result['elapsed_s']:.2f} 秒")
    print(f"总耗时：{time.perf_counter() - run_t0:.2f} 秒")
    print("计时范围：包含各阶段的Python启动、计算、文件读写及图表生成。")
    print(f"JSON耗时报告：{report_path}")
    print(f"中文CSV报告：{report_path.with_suffix('.csv')}")
    print("=" * 70)
    return 0


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_reports(
    report_path: Path,
    config_path: Path,
    run_started: str,
    stage_results: list[dict],
    run_t0: float,
) -> None:
    report = {
        "run_started_at_utc": run_started,
        "run_ended_at_utc": _now_utc(),
        "config": str(config_path),
        "stages": stage_results,
        "completed": bool(stage_results) and all(r["status"] == "passed" for r in stage_results)
        and len(stage_results) == len(STAGES),
        "total_elapsed_s": time.perf_counter() - run_t0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = report_path.with_suffix(".csv")
    columns = ["阶段", "阶段名称", "脚本", "开始时间（北京时间）", "结束时间（北京时间）", "耗时（秒）", "退出码", "运行状态"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in stage_results:
            writer.writerow({
                "阶段": result["stage"],
                "阶段名称": STAGE_NAMES[result["stage"]],
                "脚本": result["script"],
                "开始时间（北京时间）": datetime.fromisoformat(result["started_at_utc"]).astimezone(BEIJING_TIME).isoformat(),
                "结束时间（北京时间）": datetime.fromisoformat(result["ended_at_utc"]).astimezone(BEIJING_TIME).isoformat(),
                "耗时（秒）": result["elapsed_s"],
                "退出码": result["return_code"],
                "运行状态": "完成" if result["status"] == "passed" else "失败",
            })


if __name__ == "__main__":
    sys.exit(main())
