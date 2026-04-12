from datetime import datetime, timedelta
import holidays

# NYSE / US Equity Market Calendar
us_holidays = holidays.US(years=range(2024, 2031))  # Covers current + next 5 years

def is_trading_day(dt: datetime) -> bool:
    """Return True if the given date is a regular US equity trading day (NYSE)."""
    if dt.weekday() >= 5:          # Saturday or Sunday
        return False
    if dt.date() in us_holidays:   # Official US holidays observed by NYSE
        return False
    return True


def next_trading_day(current_date: datetime = None) -> datetime:
    """
    Returns the next trading day after current_date (or today if None).
    Skips weekends and NYSE holidays.
    """
    if current_date is None:
        current_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    next_day = current_date + timedelta(days=1)
    
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    
    return next_day


def format_next_trading_day() -> str:
    """Convenient function for display in Streamlit."""
    nxt = next_trading_day()
    return nxt.strftime("%Y-%m-%d (%A)")
