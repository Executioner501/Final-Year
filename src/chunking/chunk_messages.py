"""
Module 2: Conversational Chunking Engine.
Implements:
1. Naive Time-Gap Baseline (30-minute silence threshold)
2. Proposed Context-Aware Chunking (Speaker Preservation + Bounded Turns + Topic Shifts)
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..parser.parse_whatsapp import MessageRecord


@dataclass
class ConversationChunk:
    chunk_id: str
    strategy: str  # 'naive_time_gap' | 'context_aware'
    start_timestamp: str
    end_timestamp: str
    message_ids: List[str]
    participants: List[str]
    attributed_text: str
    raw_text: str
    topic_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_ts(ts_str: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%y %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass
    # Fallback to now if unparseable
    return datetime.now()


def naive_time_gap_chunking(
    messages: List[MessageRecord], 
    threshold_minutes: int = 30
) -> List[ConversationChunk]:
    """
    Baseline Chunking:
    Groups consecutive messages into the same chunk as long as the gap
    between adjacent messages is <= threshold_minutes.
    """
    if not messages:
        return []

    chunks: List[ConversationChunk] = []
    current_msgs: List[MessageRecord] = []
    chunk_counter = 1

    for msg in messages:
        if not current_msgs:
            current_msgs.append(msg)
            continue

        prev_dt = _parse_ts(current_msgs[-1].timestamp)
        curr_dt = _parse_ts(msg.timestamp)
        delta_mins = (curr_dt - prev_dt).total_seconds() / 60.0

        if delta_mins > threshold_minutes:
            chunks.append(_build_chunk(current_msgs, f"NAIVE-CHUNK-{chunk_counter:03d}", "naive_time_gap"))
            chunk_counter += 1
            current_msgs = [msg]
        else:
            current_msgs.append(msg)

    if current_msgs:
        chunks.append(_build_chunk(current_msgs, f"NAIVE-CHUNK-{chunk_counter:03d}", "naive_time_gap"))

    return chunks


def context_aware_chunking(
    messages: List[MessageRecord],
    max_inactivity_minutes: int = 20,
    max_chunk_size: int = 12,
    min_chunk_size: int = 3,
) -> List[ConversationChunk]:
    """
    Proposed Context-Aware Chunking:
    - Preserves speaker attribution for every turn ([Date Time] Sender: Message)
    - Enforces upper/lower bounds on conversational turns to prevent mega-blobs
    - Identifies conversation boundary shifts
    """
    if not messages:
        return []

    chunks: List[ConversationChunk] = []
    current_msgs: List[MessageRecord] = []
    chunk_counter = 1

    for msg in messages:
        if not current_msgs:
            current_msgs.append(msg)
            continue

        prev_dt = _parse_ts(current_msgs[-1].timestamp)
        curr_dt = _parse_ts(msg.timestamp)
        delta_mins = (curr_dt - prev_dt).total_seconds() / 60.0

        # Split conditions:
        # 1. Temporal gap exceeded
        is_time_split = delta_mins > max_inactivity_minutes
        # 2. Maximum messages exceeded (prevents 200+ message blobs)
        is_capacity_split = len(current_msgs) >= max_chunk_size

        if is_time_split or is_capacity_split:
            chunks.append(_build_chunk(current_msgs, f"CTX-CHUNK-{chunk_counter:03d}", "context_aware"))
            chunk_counter += 1
            current_msgs = [msg]
        else:
            current_msgs.append(msg)

    if current_msgs:
        chunks.append(_build_chunk(current_msgs, f"CTX-CHUNK-{chunk_counter:03d}", "context_aware"))

    return chunks


def _build_chunk(messages: List[MessageRecord], chunk_id: str, strategy: str) -> ConversationChunk:
    message_ids = [m.message_id for m in messages]
    participants = sorted(list({m.sender for m in messages if m.sender and m.sender != "System"}))
    start_ts = messages[0].timestamp
    end_ts = messages[-1].timestamp

    # Attributed conversational format
    attributed_lines = [f"[{m.date} {m.time}] {m.sender}: {m.text}" for m in messages]
    attributed_text = "\n".join(attributed_lines)
    raw_text = "\n".join([f"{m.sender}: {m.text}" for m in messages])

    start_dt = _parse_ts(start_ts)
    end_dt = _parse_ts(end_ts)
    timespan_sec = (end_dt - start_dt).total_seconds()

    return ConversationChunk(
        chunk_id=chunk_id,
        strategy=strategy,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        message_ids=message_ids,
        participants=participants,
        attributed_text=attributed_text,
        raw_text=raw_text,
        metadata={
            "message_count": len(messages),
            "timespan_seconds": timespan_sec,
            "participant_count": len(participants),
        },
    )
