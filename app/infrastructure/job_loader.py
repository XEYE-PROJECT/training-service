"""Parseo del payload de entrada del job.

Los tres entrypoints reciben el *mismo* objeto JSON (lo construye una vez el
``TrainingLaunchCommand`` del backend): el contenedor one-shot lo lee de fichero, Lambda lo
recibe como evento y RunPod en ``event["input"]``. El parseo vive aquí para que sigan
siendo adaptadores de 40 líneas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.models import ElementInput, ListInput, TrainingJob

REQUIRED_FIELDS = ("training_id", "list_id", "callback_url")


class InvalidJobError(ValueError):
    pass


def parse_job(payload: dict[str, Any]) -> TrainingJob:
    if not isinstance(payload, dict):
        raise InvalidJobError("Job payload must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if payload.get(field) in (None, "")]
    if missing:
        raise InvalidJobError(f"Missing fields in job payload: {missing}")

    list_id = int(payload["list_id"])
    raw_list = payload.get("list") or {}
    list_input = ListInput(
        id=int(raw_list.get("id") or list_id),
        name=_opt_str(raw_list.get("name")),
        description=_opt_str(raw_list.get("description")),
    )

    elements = [
        ElementInput(
            id=int(raw["id"]),
            text=str(raw.get("text") or "").strip(),
            description=_opt_str(raw.get("description")),
            generated_description=_opt_str(raw.get("generated_description")),
            trained=_as_bool(raw.get("trained")),
        )
        for raw in payload.get("elements") or []
        if raw.get("id") is not None and str(raw.get("text") or "").strip()
    ]

    return TrainingJob(
        training_id=int(payload["training_id"]),
        list_id=list_id,
        callback_url=str(payload["callback_url"]),
        webhook_secret=_opt_str(payload.get("webhook_secret")),
        user_id=int(payload["user_id"]) if payload.get("user_id") is not None else None,
        list=list_input,
        elements=elements,
        options=_parse_options(payload.get("options")),
    )


def load_job_file(path: str | Path) -> TrainingJob:
    file = Path(path)
    if not file.is_file():
        raise InvalidJobError(f"Job file not found: {file}")
    return parse_job(json.loads(file.read_text(encoding="utf-8")))


def _parse_options(raw: Any) -> dict[str, Any]:
    """El backend envía ``[{"key": "train_all", "value": true}]``; también vale un objeto plano."""
    if isinstance(raw, dict):
        return dict(raw)
    options: dict[str, Any] = {}
    for item in raw or []:
        if isinstance(item, dict) and item.get("key") is not None:
            options[str(item["key"])] = item.get("value")
    return options


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False
