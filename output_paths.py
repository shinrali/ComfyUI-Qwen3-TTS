"""Portable Unicode output names with Windows-compatible path validation."""

import re
import unicodedata


RESERVED_NAME = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])$", re.IGNORECASE)


def output_prefix_parts(prefix: str) -> list[str]:
    normalized = prefix.replace("\\", "/")
    parts = normalized.split("/")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or part.endswith((" ", "."))
            or len(part.encode("utf-16-le")) > 256
            or any(char in '<>:"|?*' or unicodedata.category(char).startswith("C") for char in part)
            or RESERVED_NAME.fullmatch(part.split(".", 1)[0].rstrip(" "))
        ):
            raise ValueError("Invalid filename_prefix")
    return parts
