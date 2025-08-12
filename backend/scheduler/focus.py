from datetime import date, datetime, time, timedelta
from typing import Dict, List

import pandas as pd

from shared.utils.time_utils import KOREAN_WEEKDAYS


def compute_focus_score(observation: Dict[str, float]) -> float:
    acc = float(observation.get("accuracy", 0.0))
    ms = float(observation.get("avg_response_ms", 0.0))
    nolag = float(observation.get("no_lag_rate", 0.0))
    miss = float(observation.get("miss_penalty", 0.0))
    sec = max(ms / 1000.0, 0.001)
    rt_inv = min(1.0 / sec, 1.0)
    w1, w2, w3, w4 = 0.45, 0.25, 0.20, -0.10
    raw = w1 * acc + w2 * rt_inv + w3 * nolag + w4 * miss
    return float(max(0.0, min(raw, 1.0)))


def update_focus_map_with_observation(
    focus_map: List[List[float]],
    *,
    focus_value: float,
    bin_minutes: int,
    weekday_idx: int,
    time_str: str,
) -> List[List[float]]:
    bins_per_day = max(1, (24 * 60) // max(1, bin_minutes))

    def _resample(row: List[float], target_len: int, default_val: float) -> List[float]:
        if not row:
            return [float(default_val) for _ in range(target_len)]
        src_len = len(row)
        if src_len == target_len:
            return [float(x) for x in row]
        out: List[float] = []
        for i in range(target_len):
            src_idx = min(src_len - 1, int(i * src_len / target_len))
            out.append(float(row[src_idx]))
        return out

    if not focus_map or len(focus_map) != 7:
        base = focus_map if isinstance(focus_map, list) else []
        new_map: List[List[float]] = []
        for wd in range(7):
            row = base[wd] if isinstance(base, list) and wd < len(base) and isinstance(base[wd], list) else []
            new_map.append(_resample(row, bins_per_day, focus_value))
        focus_map = new_map
    else:
        focus_map = [_resample(row, bins_per_day, focus_value) for row in focus_map]

    hh, mm = map(int, time_str.split(":"))
    bin_index = min(bins_per_day - 1, max(0, (hh * 60 + mm) // bin_minutes))
    alpha, w5 = 0.3, 2
    updated = []
    for i, row in enumerate(focus_map):
        ema_prev = float(row[bin_index])
        w = w5 if i == weekday_idx else 1
        alpha_w = 1.0 - (1.0 - alpha) ** max(1, int(w))
        ema_new = alpha_w * focus_value + (1.0 - alpha_w) * ema_prev
        row[bin_index] = float(ema_new)
        updated.append(row)
    return updated


def build_focus_dataframe(focus_map: List[List[float]], step_minutes: int) -> pd.DataFrame:
    bins_per_day = max(1, (24 * 60) // max(1, step_minutes))
    times: List[str] = []
    cur = datetime.combine(date.today(), time(0, 0))
    for _ in range(bins_per_day):
        times.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=step_minutes)
    if not focus_map or len(focus_map) != 7:
        data = {w: [0.0] * bins_per_day for w in KOREAN_WEEKDAYS}
        return pd.DataFrame(data, index=times)
    cols: Dict[str, List[float]] = {}
    for wd, name in enumerate(KOREAN_WEEKDAYS):
        row = focus_map[wd] if wd < len(focus_map) and isinstance(focus_map[wd], list) else []
        if len(row) != bins_per_day:
            out = []
            for i in range(bins_per_day):
                src_idx = min(len(row) - 1, int(i * (len(row) or 1) / max(1, bins_per_day))) if row else 0
                out.append(float(row[src_idx]) if row else 0.0)
            cols[name] = out
        else:
            cols[name] = [float(x) for x in row]
    df = pd.DataFrame(cols, index=times)
    df.index.name = "시간"
    return df


