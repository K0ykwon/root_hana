from typing import Dict, List
import pandas as pd

from shared.utils.time_utils import KOREAN_WEEKDAYS
from shared.utils.time_utils import generate_timeslots


def build_availability_template(start_hhmm: str, end_hhmm: str, step_minutes: int) -> pd.DataFrame:
    times = generate_timeslots(start_hhmm, end_hhmm, step_minutes)
    data = {weekday: [False] * len(times) for weekday in KOREAN_WEEKDAYS}
    df = pd.DataFrame(data, index=times)
    df.index.name = "시간"
    return df


def quick_fill_pattern(
    df: pd.DataFrame,
    pattern: str,
    fill_true: bool = True,
    evening_start: str = "19:00",
    evening_end: str = "22:00",
    weekend_start: str = "10:00",
    weekend_end: str = "18:00",
) -> pd.DataFrame:
    new_df = df.copy()
    if pattern == "평일 저녁(19~22시) + 주말(10~18시)":
        for col in ["월", "화", "수", "목", "금"]:
            for t in new_df.index:
                if evening_start <= t < evening_end:
                    new_df.at[t, col] = fill_true
        for col in ["토", "일"]:
            for t in new_df.index:
                if weekend_start <= t < weekend_end:
                    new_df.at[t, col] = fill_true
    elif pattern == "모두 비우기":
        new_df.loc[:, :] = False
    elif pattern == "모두 채우기":
        new_df.loc[:, :] = True
    return new_df


def resample_availability(
    old_df: pd.DataFrame, old_step: int, new_step: int, start_hhmm: str = "06:00", end_hhmm: str = "23:00"
) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return build_availability_template(start_hhmm, end_hhmm, new_step)
    new_df = build_availability_template(start_hhmm, end_hhmm, new_step)
    def nearest_time_index(target: str, candidates: List[str]) -> int:
        th, tm = map(int, target.split(":"))
        tmin = th * 60 + tm
        best_idx, best_diff = 0, 10**9
        for i, c in enumerate(candidates):
            ch, cm = map(int, c.split(":"))
            cmin = ch * 60 + cm
            diff = abs(cmin - tmin)
            if diff < best_diff:
                best_idx, best_diff = i, diff
        return best_idx
    old_times = list(map(str, old_df.index))
    new_times = list(map(str, new_df.index))
    for wd in KOREAN_WEEKDAYS:
        if wd not in old_df.columns:
            continue
        for nt in new_times:
            oi = nearest_time_index(nt, old_times)
            new_df.at[nt, wd] = bool(old_df.iloc[oi][wd])
    return new_df


