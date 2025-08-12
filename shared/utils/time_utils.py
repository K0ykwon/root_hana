from datetime import date, datetime, timedelta, time
from typing import List

KOREAN_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def parse_time_str(hhmm: str) -> time:
    hour, minute = map(int, hhmm.split(":"))
    return time(hour=hour, minute=minute)


def time_to_str(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def generate_timeslots(start_hhmm: str, end_hhmm: str, step_minutes: int) -> List[str]:
    start_t = parse_time_str(start_hhmm)
    end_t = parse_time_str(end_hhmm)
    today = date.today()
    start_dt = datetime.combine(today, start_t)
    end_dt = datetime.combine(today, end_t)
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(hours=1)
    times = []
    cur = start_dt
    while cur < end_dt:
        times.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=step_minutes)
    return times


def get_weekday_kr(d: date) -> str:
    return KOREAN_WEEKDAYS[d.weekday()]


