"""Synthetic range-telemetry retrieval task.

A context of R distinct sensor readings [SENSOR_i, VALUE_i], followed by a
query for one of those sensors. The correct answer is that sensor's value --
the honest, literal version of "find sensor X's reading among many other
distractor sensor readings," i.e. exactly the retrieval structure of
Needle-in-a-Haystack, applied to telemetry data instead of prose.

Token layout (ids):
  0            PAD  (unused, sequences are fixed length)
  1            QUERY
  2            ANSWER
  3..12        SENSOR_0 .. SENSOR_9      (10 sensors)
  13..112      VALUE_0 .. VALUE_99       (100 possible readings)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

QUERY_ID = 1
ANSWER_ID = 2
NUM_SENSORS = 12
NUM_VALUES = 100
SENSOR_BASE = 3
VALUE_BASE = SENSOR_BASE + NUM_SENSORS
VOCAB_SIZE = VALUE_BASE + NUM_VALUES


def sensor_id(i: int) -> int:
    return SENSOR_BASE + i


def value_id(v: int) -> int:
    return VALUE_BASE + v


def is_sensor_token(ids: torch.Tensor) -> torch.Tensor:
    return (ids >= SENSOR_BASE) & (ids < VALUE_BASE)


def is_value_token(ids: torch.Tensor) -> torch.Tensor:
    return ids >= VALUE_BASE


@dataclass
class Batch:
    ctx_ids: torch.Tensor    # (B, 2R)  sensor/value pairs
    query_ids: torch.Tensor  # (B, 3)   [QUERY, SENSOR_q, ANSWER]
    target: torch.Tensor     # (B,)     correct VALUE token id


def make_batch(batch_size: int, num_readings: int, device: torch.device, generator: torch.Generator) -> Batch:
    ctx = torch.empty(batch_size, 2 * num_readings, dtype=torch.long)
    query = torch.empty(batch_size, 3, dtype=torch.long)
    target = torch.empty(batch_size, dtype=torch.long)

    for b in range(batch_size):
        perm = torch.randperm(NUM_SENSORS, generator=generator)[:num_readings]
        sensors = perm  # distinct sensors, random order -> no recency tie-break needed
        values = torch.randint(0, NUM_VALUES, (num_readings,), generator=generator)
        for r in range(num_readings):
            ctx[b, 2 * r] = sensor_id(int(sensors[r]))
            ctx[b, 2 * r + 1] = value_id(int(values[r]))
        q_idx = int(torch.randint(0, num_readings, (1,), generator=generator).item())
        q = int(sensors[q_idx])
        query[b, 0] = QUERY_ID
        query[b, 1] = sensor_id(q)
        query[b, 2] = ANSWER_ID
        target[b] = value_id(int(values[q_idx]))

    return Batch(ctx_ids=ctx.to(device), query_ids=query.to(device), target=target.to(device))
