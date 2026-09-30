"""Fail-closed activation of the consistent group admission guard.

Prime starts the orchestrator in a separate Python process.  Importing the
rank module through an environment variable does not install its patches, so
this role-gated hook invokes the explicit installer in that child only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _activate() -> None:
    if os.environ.get("PAINTER_PRIME_ROLE") != "orchestrator":
        return
    try:
        from painter.consistent_group_admission import install_from_environment
    except ModuleNotFoundError as exc:
        # The reviewed Prime launcher normally places the repository root on
        # PYTHONPATH.  Keep the hook usable with its legacy ``painter/``
        # source-root layout too, where the module is top-level.
        if exc.name != "painter":
            raise
        from consistent_group_admission import install_from_environment

    result = install_from_environment()
    if not result.get("installed") and not result.get("already_installed"):
        raise RuntimeError(f"consistent group admission did not install: {result}")
    receipt = os.environ.get("PAINTER_CONSISTENT_GUARD_RECEIPT")
    if receipt:
        path = Path(receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "consistent-group-admission-activation-v1",
                    "role": "orchestrator",
                    "installed": bool(result.get("installed")),
                    "already_installed": bool(result.get("already_installed")),
                    "protocol": os.environ.get("PAINTER_JUDGE_PROTOCOL"),
                    "group_size": int(os.environ["PAINTER_WHOLE_GROUP_SIZE"]),
                },
                sort_keys=True,
            )
            + "\n"
        )


try:
    _activate()
except BaseException as exc:
    # CPython's site importer reports ordinary Exception subclasses and then
    # continues.  SystemExit is deliberate here: a Prime child that cannot
    # install the guard must fail before it creates scorer/trainer classes.
    if isinstance(exc, SystemExit):
        raise
    raise SystemExit(f"consistent group admission activation failed: {exc}") from exc
