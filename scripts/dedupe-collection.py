"""C1 dry-run: grupy prawie-identycznych punktów w voiceloop_memory.

Domyślnie tylko raport — nic nie usuwa. Grupowanie po składowych spójnych
przy progu cosinusowym na osi `semantic`. W każdej grupie zostaje najstarszy
punkt (`created_at`), reszta to kandydaci do usunięcia.

Użycie:
  python scripts/dedupe-collection.py
  python scripts/dedupe-collection.py --threshold 0.999
  python scripts/dedupe-collection.py --apply   # wymaga jawnej zgody; nie używaj bez raportu
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "listener")]

from voiceloop.qdrant_memory import QdrantVectorStore
from voiceloop.settings import get_settings

DEFAULT_THRESHOLD = 0.999


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _parse_time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.max.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.max.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


async def _scroll_semantic(store: QdrantVectorStore) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    offset = None
    while True:
        batch, offset = await store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=["semantic"],
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload if isinstance(point.payload, dict) else {}
            vectors = point.vector if isinstance(point.vector, dict) else {}
            semantic = vectors.get("semantic")
            if not semantic:
                continue
            metadata = payload.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            collected.append(
                {
                    "id": point.id,
                    "id_str": str(point.id),
                    "source": str(payload.get("source") or ""),
                    "source_id": str(payload.get("source_id") or ""),
                    "title": str(payload.get("title") or "")[:120],
                    "content": str(payload.get("content") or "")[:200],
                    "created_at": str(
                        payload.get("created_at")
                        or metadata.get("timestamp")
                        or metadata.get("time")
                        or ""
                    ),
                    "content_hash": str(
                        payload.get("content_hash") or metadata.get("content_hash") or ""
                    ),
                    "vector": np.asarray(semantic, dtype=float),
                }
            )
        if offset is None:
            break
    return collected


def _connected_components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    n = similarity.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Tylko górny trójkąt — self-similarity nie łączy niczego nowego.
    for i in range(n):
        row = similarity[i, i + 1 :]
        hits = np.where(row >= threshold)[0]
        for offset in hits:
            union(i, i + 1 + int(offset))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(n):
        groups[find(index)].append(index)
    return [members for members in groups.values() if len(members) > 1]


def _pick_keeper(members: list[dict[str, Any]]) -> dict[str, Any]:
    return min(members, key=lambda item: (_parse_time(item["created_at"]), item["id_str"]))


async def run(*, threshold: float, apply: bool, report_path: Path) -> int:
    settings = get_settings()
    store = QdrantVectorStore(settings)
    if not store.enabled:
        print("Qdrant wyłączony.")
        return 1

    points = await _scroll_semantic(store)
    print(f"Punktów z wektorem semantic: {len(points)}")
    print(f"Próg grupowania: {threshold}")
    print(f"Tryb: {'APPLY (usuwanie)' if apply else 'DRY-RUN (bez usuwania)'}")

    if len(points) < 2:
        print("Za mało punktów.")
        await store.close()
        return 0

    matrix = _unit_rows(np.vstack([item["vector"] for item in points]))
    similarity = matrix @ matrix.T
    groups_idx = _connected_components(similarity, threshold)

    removal_ids: list[Any] = []
    report_groups: list[dict[str, Any]] = []
    for members_idx in sorted(groups_idx, key=len, reverse=True):
        members = [points[i] for i in members_idx]
        keeper = _pick_keeper(members)
        doomed = [item for item in members if item["id_str"] != keeper["id_str"]]
        removal_ids.extend(item["id"] for item in doomed)
        report_groups.append(
            {
                "size": len(members),
                "remove": len(doomed),
                "keeper_id": keeper["id_str"],
                "keeper_created_at": keeper["created_at"],
                "keeper_title": keeper["title"],
                "keeper_source": keeper["source"],
                "remove_ids": [item["id_str"] for item in doomed],
                "member_titles": [item["title"] for item in members[:8]],
            }
        )

    remaining = len(points) - len(removal_ids)
    summary = {
        "threshold": threshold,
        "points_with_semantic": len(points),
        "duplicate_groups": len(report_groups),
        "points_to_remove": len(removal_ids),
        "points_remaining": remaining,
        "largest_group": report_groups[0]["size"] if report_groups else 0,
        "apply": apply,
        "generated_at": datetime.now(UTC).isoformat(),
        "groups": report_groups,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Grup duplikatów:     {summary['duplicate_groups']}")
    print(f"Do usunięcia:        {summary['points_to_remove']}")
    print(f"Zostanie:            {summary['points_remaining']}")
    print(f"Największa grupa:    {summary['largest_group']}")
    print(f"Raport:              {report_path}")
    print()
    print("Top 10 grup:")
    for group in report_groups[:10]:
        print(
            f"  n={group['size']:4d}  keep={group['keeper_created_at'][:19]:19}  "
            f"{group['keeper_title'][:70]}"
        )

    if apply:
        if not removal_ids:
            print("Nie ma czego usuwać.")
        else:
            from qdrant_client import models

            # Partiami, żeby nie wysłać ogromnej listy naraz.
            chunk = 256
            deleted = 0
            for start in range(0, len(removal_ids), chunk):
                batch = removal_ids[start : start + chunk]
                await store.client.delete(
                    collection_name=store.collection_name,
                    points_selector=models.PointIdsList(points=batch),
                    wait=True,
                )
                deleted += len(batch)
                print(f"  usunięto {deleted}/{len(removal_ids)} ...")
            print(f"Usunięto łącznie: {deleted}")

    await store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Naprawdę usuwa. Bez tej flagi tylko raport.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "dedupe-dry-run-c1.json",
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return asyncio.run(
        run(threshold=args.threshold, apply=args.apply, report_path=args.report)
    )


if __name__ == "__main__":
    raise SystemExit(main())
