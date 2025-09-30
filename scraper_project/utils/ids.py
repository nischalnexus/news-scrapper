from __future__ import annotations

import hashlib
import uuid
from typing import Iterable


NAMESPACE = uuid.UUID("94c5f299-1b33-4fac-b570-2e28d7baeaea")


def stable_uuid(values: Iterable[str]) -> str:
    joined = "|".join(v or "" for v in values)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def uuid_from_url(url: str) -> str:
    return str(uuid.uuid5(NAMESPACE, url))
