from datetime import datetime, timezone

def utcnow_naive() -> datetime:
    """Returns a naive UTC datetime for storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def make_aware_utc(dt: datetime) -> datetime:
    """Attaches the UTC timezone to a naive datetime from the DB."""
    return dt.replace(tzinfo=timezone.utc)