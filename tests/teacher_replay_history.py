"""Distinguish a reused immutable record from a newly collected duplicate."""

def distinct_history(raw, previous):
    origins = {record.trajectory_id: record for record in raw}
    result = []
    for record in previous:
        origin = origins.get(record.trajectory_id)
        if origin is None:
            result.append(record)
        elif record != origin:
            raise ValueError("Conflicting content for an existing trajectory ID")
    return result
