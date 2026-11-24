from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = PROJECT_ROOT / "outputs/campaign/stage_state.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/campaign/logs"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _record_stage(state_path: Path, stage_name: str, payload: dict[str, Any]) -> None:
    state = _load_state(state_path)
    state.setdefault("stages", {})[stage_name] = payload
    _save_state(state_path, state)


def _run_command(command: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--command", default=None)
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--skip-reason", default="not_requested")
    args = parser.parse_args()

    state_path = Path(args.state_path)
    log_dir = Path(args.log_dir)
    current_state = _load_state(state_path)
    stage_state = current_state.get("stages", {}).get(args.stage_name, {})
    if args.resume and stage_state.get("status") == "success":
        print(f"[stage_runner] skipping completed stage: {args.stage_name}")
        return

    log_path = log_dir / f"{args.stage_name}.log"
    started_at = datetime.now().isoformat()
    if args.skip:
        _record_stage(
            state_path,
            args.stage_name,
            {
                "status": "skipped",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
                "reason": args.skip_reason,
                "log_path": str(log_path),
            },
        )
        print(f"[stage_runner] skipped stage: {args.stage_name} ({args.skip_reason})")
        return

    if not args.command:
        raise ValueError("--command is required unless --skip is used.")

    return_code = _run_command(args.command, log_path)
    status = "success" if return_code == 0 else ("failed_optional" if args.allow_failure else "failed")
    _record_stage(
        state_path,
        args.stage_name,
        {
            "status": status,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "command": args.command,
            "log_path": str(log_path),
            "return_code": int(return_code),
        },
    )
    if return_code != 0 and not args.allow_failure:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
