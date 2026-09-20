from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from ..memory import MemoryStore
from .schema import (
    ContextEpisodeV1,
    ContextEventV1,
    stable_episode_id,
)


@dataclass(frozen=True)
class MeetingTimelineReport:
    sessions: int
    events: int
    episodes: int


class MeetingTimelineIngestor:
    """Bridges both existing transcript stores into the shared timeline."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    async def ingest_recorded_meetings(
        self,
        *,
        limit: int = 100,
    ) -> MeetingTimelineReport:
        sessions = await self.memory.list_meeting_sessions(limit=limit)
        event_count = 0
        episode_count = 0
        for session in sessions:
            segments = await self.memory.list_meeting_transcript_segments(
                session.session_id
            )
            if not segments:
                continue
            events = [
                ContextEventV1.from_observation(
                    source="meeting_recorder",
                    source_id=segment.segment_key,
                    started_at=segment.start_time,
                    ended_at=segment.end_time,
                    event_type="meeting_transcript",
                    app_name="meeting",
                    window_title=session.title,
                    text=segment.text,
                    metadata={
                        "session_id": session.session_id,
                        "channel": segment.channel,
                        "speaker_label": segment.speaker_label,
                        "speaker_id": segment.speaker_id,
                        "device_name": segment.device_name,
                        "transcript_source": segment.source,
                    },
                )
                for segment in segments
                if segment.text.strip()
            ]
            if not events:
                continue
            stored_events = [
                await self.memory.upsert_context_event(event) for event in events
            ]
            episode = self._episode_for_recorded_session(
                session_id=session.session_id,
                title=session.title,
                events=stored_events,
            )
            await self.memory.upsert_context_episode(episode)
            event_count += len(stored_events)
            episode_count += 1
        return MeetingTimelineReport(
            sessions=len(sessions),
            events=event_count,
            episodes=episode_count,
        )

    async def ingest_screenpipe_meetings(
        self,
        *,
        limit: int = 5000,
    ) -> MeetingTimelineReport:
        transcripts = await self.memory.list_screenpipe_transcripts(limit=limit)
        grouped: dict[int, list] = defaultdict(list)
        for transcript in transcripts:
            if transcript.text.strip():
                grouped[transcript.meeting_id].append(transcript)
        event_count = 0
        episode_count = 0
        for meeting_id, chunks in grouped.items():
            ordered = sorted(chunks, key=lambda item: item.start_time)
            events = [
                ContextEventV1.from_observation(
                    source="screenpipe_meeting",
                    source_id=chunk.chunk_id,
                    started_at=_parse_timestamp(chunk.start_time),
                    ended_at=_parse_timestamp(chunk.end_time),
                    event_type="meeting_transcript",
                    app_name="screenpipe",
                    window_title=f"Spotkanie {meeting_id}",
                    text=chunk.text,
                    metadata={
                        "meeting_id": meeting_id,
                        "device_name": chunk.device_name,
                        "device_type": chunk.device_type,
                        "transcript_source": chunk.source,
                    },
                )
                for chunk in ordered
            ]
            stored_events = [
                await self.memory.upsert_context_event(event) for event in events
            ]
            source_id = f"meeting:{meeting_id}"
            episode = ContextEpisodeV1(
                episode_id=stable_episode_id(
                    "screenpipe_meeting",
                    source_id,
                ),
                source="screenpipe_meeting",
                source_id=source_id,
                started_at=min(event.started_at for event in stored_events),
                ended_at=max(
                    event.ended_at or event.started_at for event in stored_events
                ),
                title=f"Spotkanie Screenpipe {meeting_id}",
                summary=_meeting_summary(stored_events),
                source_event_ids=tuple(
                    event.event_id for event in stored_events
                ),
                app_names=("screenpipe",),
                metadata={
                    "meeting_id": meeting_id,
                    "digest_method": "deterministic_meeting_v1",
                },
            )
            await self.memory.upsert_context_episode(episode)
            event_count += len(stored_events)
            episode_count += 1
        return MeetingTimelineReport(
            sessions=len(grouped),
            events=event_count,
            episodes=episode_count,
        )

    @staticmethod
    def _episode_for_recorded_session(
        *,
        session_id: str,
        title: str,
        events: list[ContextEventV1],
    ) -> ContextEpisodeV1:
        return ContextEpisodeV1(
            episode_id=stable_episode_id(
                "meeting_recorder",
                session_id,
            ),
            source="meeting_recorder",
            source_id=session_id,
            started_at=min(event.started_at for event in events),
            ended_at=max(event.ended_at or event.started_at for event in events),
            title=title or f"Spotkanie {session_id}",
            summary=_meeting_summary(events),
            source_event_ids=tuple(event.event_id for event in events),
            app_names=("meeting",),
            metadata={
                "session_id": session_id,
                "digest_method": "deterministic_meeting_v1",
            },
        )


def _meeting_summary(events: list[ContextEventV1]) -> str:
    lines: list[str] = []
    for event in events:
        speaker = str(event.metadata.get("speaker_label") or "").strip()
        prefix = f"{speaker}: " if speaker else ""
        line = f"{prefix}{event.text.strip()}".strip()
        if line and line not in lines:
            lines.append(line)
    return "\n".join(lines)[:20000] or "Zarejestrowane spotkanie."


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(UTC)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
