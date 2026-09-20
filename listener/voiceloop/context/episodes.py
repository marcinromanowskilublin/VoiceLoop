from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from .schema import ContextEpisodeV1, ContextEventV1, stable_episode_id


def activity_bucket(value: datetime, *, minutes: int = 10) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("activity timestamp must be timezone-aware")
    safe_minutes = max(1, min(int(minutes), 60))
    utc_value = value.astimezone(UTC)
    minute = utc_value.minute - (utc_value.minute % safe_minutes)
    return utc_value.replace(minute=minute, second=0, microsecond=0)


def group_events_into_episodes(
    events: list[ContextEventV1],
    *,
    bucket_minutes: int = 10,
    source: str = "timeline",
) -> list[ContextEpisodeV1]:
    """Create deterministic fallback episodes without claiming model inference."""

    grouped: dict[datetime, list[ContextEventV1]] = defaultdict(list)
    for event in events:
        if event.deleted_at is not None:
            continue
        grouped[activity_bucket(event.started_at, minutes=bucket_minutes)].append(event)

    episodes: list[ContextEpisodeV1] = []
    for bucket, bucket_events in sorted(grouped.items()):
        ordered = sorted(bucket_events, key=lambda item: item.started_at)
        ended_at = max(item.ended_at or item.started_at for item in ordered)
        apps = tuple(
            dict.fromkeys(item.app_name for item in ordered if item.app_name)
        )
        content_parts = [
            item.text.strip() or item.window_title.strip()
            for item in ordered
            if item.text.strip() or item.window_title.strip()
        ]
        unique_parts = list(dict.fromkeys(content_parts))
        summary = "\n".join(unique_parts)[:20000] or "Zarejestrowana aktywność lokalna."
        source_id = f"bucket:{bucket.isoformat()}"
        title_apps = ", ".join(apps[:3])
        title = (
            f"Aktywność {bucket.isoformat()} — {title_apps}"
            if title_apps
            else f"Aktywność {bucket.isoformat()}"
        )
        episodes.append(
            ContextEpisodeV1(
                episode_id=stable_episode_id(source, source_id),
                source=source,
                source_id=source_id,
                started_at=min(item.started_at for item in ordered),
                ended_at=ended_at,
                title=title,
                summary=summary,
                source_event_ids=tuple(item.event_id for item in ordered),
                app_names=apps,
                metadata={
                    "bucket_minutes": bucket_minutes,
                    "event_count": len(ordered),
                    "digest_method": "deterministic_timeline_v1",
                },
            )
        )
    return episodes
