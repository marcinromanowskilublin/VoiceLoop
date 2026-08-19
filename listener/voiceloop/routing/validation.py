from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


def validate_arguments(
    arguments: dict[str, Any],
    schema: Any,
) -> list[str]:
    if not isinstance(schema, dict):
        return ["invalid_args_schema"]
    if schema.get("type") not in {None, "object"}:
        return ["invalid_args_schema"]
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    required_names = (
        [str(name) for name in required]
        if isinstance(required, list | tuple)
        else []
    )
    errors = [
        f"missing_required_argument:{name}"
        for name in required_names
        if name not in arguments
        or arguments[name] is None
        or (isinstance(arguments[name], str) and not arguments[name].strip())
    ]
    if schema.get("additionalProperties") is False:
        errors.extend(
            f"unexpected_argument:{name}"
            for name in arguments
            if name not in properties
        )
    for name, value in arguments.items():
        definition = properties.get(name)
        if not isinstance(definition, dict):
            continue
        expected_type = definition.get("type")
        if not _matches_type(value, expected_type):
            errors.append(f"invalid_argument_type:{name}")
            continue
        if isinstance(value, str):
            minimum = definition.get("minLength")
            maximum = definition.get("maxLength")
            pattern = definition.get("pattern")
            if isinstance(minimum, int) and len(value) < minimum:
                errors.append(f"invalid_argument_min_length:{name}")
            if isinstance(maximum, int) and len(value) > maximum:
                errors.append(f"invalid_argument_max_length:{name}")
            if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
                errors.append(f"invalid_argument_pattern:{name}")
        allowed = definition.get("enum")
        if isinstance(allowed, list | tuple) and value not in allowed:
            errors.append(f"invalid_argument_enum:{name}")
        if isinstance(value, int) and not isinstance(value, bool):
            minimum = definition.get("minimum")
            maximum = definition.get("maximum")
            if isinstance(minimum, int | float) and value < minimum:
                errors.append(f"invalid_argument_minimum:{name}")
            if isinstance(maximum, int | float) and value > maximum:
                errors.append(f"invalid_argument_maximum:{name}")
    if not errors and "url" in arguments:
        parsed = urlparse(str(arguments["url"]))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("invalid_argument_url:url")
    return list(dict.fromkeys(errors))


def _matches_type(value: Any, expected_type: Any) -> bool:
    if expected_type in {None, "any"}:
        return True
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list | tuple)
    return False
