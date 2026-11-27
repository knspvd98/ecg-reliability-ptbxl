from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ecg_reliability.experiments.registry import DEFAULT_RUNS_ROOT
from ecg_reliability.workbench.data_access import WorkbenchDataAccess, sanitize_json

STATIC_DIR = Path(__file__).resolve().parent / "static"


class WorkbenchHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], data_access: WorkbenchDataAccess) -> None:
        self.data_access = data_access
        super().__init__(server_address, WorkbenchRequestHandler)


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    server: WorkbenchHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(sanitize_json(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        clean = "index.html" if rel_path in {"", "/"} else rel_path.lstrip("/")
        target = (STATIC_DIR / clean).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            self._send_json({"error": f"Static asset not found: {clean}"}, status=HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(str(target))
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if not path.startswith("/api/"):
            self._send_static(path)
            return

        da = self.server.data_access
        try:
            if path == "/api/health":
                self._send_json({"status": "ok", "service": "ecg-reliability-workbench", "medical_device": False})
                return
            if path == "/api/runs":
                self._send_json({"runs": da.list_all_runs()})
                return
            if path == "/api/comparisons":
                self._send_json(da.get_comparisons())
                return

            parts = [p for p in path.split("/") if p]
            # /api/runs/<run_id>[/...]
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "runs":
                run_id = parts[2]
                if len(parts) == 3:
                    self._send_json(da.get_run_detail(run_id))
                    return
                sub = parts[3]
                if sub == "artifact":
                    rel = query.get("path", [""])[0]
                    self._send_json(da.read_run_artifact(run_id, rel))
                    return
                if sub == "signals":
                    split = query.get("split", ["test"])[0]
                    limit = int(query.get("limit", ["100"])[0])
                    self._send_json(da.list_signals(run_id, split=split, limit=limit))
                    return
                if sub == "signal" and len(parts) >= 5:
                    ecg_id = int(parts[4])
                    self._send_json(da.get_signal_detail(run_id, ecg_id))
                    return
                if sub == "reliability":
                    cal_method = query.get("method", [None])[0]
                    binning = query.get("binning", ["uniform"])[0]
                    self._send_json(
                        da.get_reliability_analysis(
                            run_id,
                            calibration_method=cal_method,
                            binning=binning,
                        )
                    )
                    return
                if sub == "robustness":
                    self._send_json(da.get_robustness_analysis(run_id))
                    return
                if sub == "explainability":
                    eid_raw = query.get("ecg_id", [None])[0]
                    label = query.get("target_label", [None])[0]
                    method = query.get("method", ["ig"])[0]
                    self._send_json(
                        da.get_explainability_analysis(
                            run_id,
                            ecg_id=int(eid_raw) if eid_raw else None,
                            target_label=label,
                            method=method,
                        )
                    )
                    return
            self._send_json({"error": f"Unknown API endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": f"{exc.__class__.__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        parts = [p for p in path.split("/") if p]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "perturb":
                result = self.server.data_access.apply_live_perturbation(parts[2], payload)
                self._send_json(result)
                return
            self._send_json({"error": f"Unknown POST endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": f"{exc.__class__.__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)


def create_workbench_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    outputs_root: str | Path = "outputs",
) -> WorkbenchHTTPServer:
    data_access = WorkbenchDataAccess(runs_root=runs_root, outputs_root=outputs_root)
    return WorkbenchHTTPServer((host, int(port)), data_access)


def serve_workbench(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    outputs_root: str | Path = "outputs",
) -> None:
    server = create_workbench_server(host=host, port=port, runs_root=runs_root, outputs_root=outputs_root)
    actual_host, actual_port = server.server_address
    print(f"ECG Reliability Research Workbench running at http://{actual_host}:{actual_port}")
    print("Research software only — NOT a medical device. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down workbench server.")
    finally:
        server.server_close()
