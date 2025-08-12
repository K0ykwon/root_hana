from datetime import date, datetime, time, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from shared.utils.time_utils import KOREAN_WEEKDAYS, parse_time_str, time_to_str


def collect_available_slots_for_date(avail_df: pd.DataFrame, d: date, step_minutes: int) -> List[Tuple[time, time]]:
    weekday = KOREAN_WEEKDAYS[d.weekday()]
    if weekday not in avail_df.columns:
        return []
    slots: List[Tuple[time, time]] = []
    for t_str, is_ok in avail_df[weekday].items():
        if bool(is_ok):
            start_t = parse_time_str(str(t_str))
            end_dt = datetime.combine(d, start_t) + timedelta(minutes=step_minutes)
            slots.append((start_t, end_dt.time()))
    slots.sort(key=lambda x: x[0])
    return slots


def make_schedule(
    subjects_df: pd.DataFrame,
    avail_df: pd.DataFrame,
    weak_subjects: List[str],
    step_minutes: int,
    start_date: date,
    focus_map: Optional[List[List[float]]] = None,
) -> Dict:
    valid_rows = []
    for _, row in subjects_df.iterrows():
        subj = str(row.get("과목", "")).strip()
        if not subj:
            continue
        exam_raw = row.get("시험일", None)
        if pd.isna(exam_raw):
            continue
        if isinstance(exam_raw, (pd.Timestamp, datetime)):
            exam_day = exam_raw.date()
        elif isinstance(exam_raw, date):
            exam_day = exam_raw
        else:
            try:
                exam_day = pd.to_datetime(exam_raw).date()
            except Exception:
                continue
        scope = str(row.get("시험범위", "")).strip()
        valid_rows.append({
            "과목": subj,
            "시험일": exam_day,
            "시험범위": scope,
            "취약": subj in weak_subjects,
        })

    if not valid_rows:
        return {"sessions_by_date": {}, "summary": pd.DataFrame()}

    subjects = pd.DataFrame(valid_rows)
    last_exam = subjects["시험일"].max()
    days = (last_exam - start_date).days + 1
    if days <= 0:
        days = 1

    assigned_counts: Dict[str, int] = {s: 0 for s in subjects["과목"].tolist()}

    def priority_weight(subj_row: pd.Series, current_day: date, slot_start: time) -> float:
        days_to_exam = max((subj_row["시험일"] - current_day).days, 0) + 1
        urgency = 1.0 / days_to_exam
        if current_day > subj_row["시험일"]:
            urgency *= 0.05
        weakness = 1.3 if subj_row["취약"] else 1.0
        penalty = 1.0 / (1 + assigned_counts.get(str(subj_row["과목"]), 0))
        focus_multiplier = 1.0
        try:
            if focus_map:
                wd = current_day.weekday()
                bin_minutes = step_minutes
                bins_per_day = max(1, (24 * 60) // max(1, bin_minutes))
                bin_index = min(bins_per_day - 1, max(0, (slot_start.hour * 60 + slot_start.minute) // bin_minutes))
                row = focus_map[wd] if wd < len(focus_map) else []
                val = float(row[bin_index]) if row and bin_index < len(row) else 0.5
                focus_multiplier = 0.6 + 0.8 * max(0.0, min(1.0, val))
        except Exception:
            focus_multiplier = 1.0
        return urgency * weakness * penalty * focus_multiplier

    sessions_by_date: Dict[str, List[Dict]] = {}
    for i in range(days):
        d = start_date + timedelta(days=i)
        todays_slots = collect_available_slots_for_date(avail_df, d, step_minutes)
        if not todays_slots:
            continue
        daily_sessions: List[Dict] = []
        subjects = subjects.sort_values(by=["시험일"]).reset_index(drop=True)
        last_subject: Optional[str] = None
        for (start_t, end_t) in todays_slots:
            weights = []
            for j in range(len(subjects)):
                w = priority_weight(subjects.loc[j], d, start_t)
                if last_subject and str(subjects.loc[j, "과목"]) == last_subject:
                    w *= 0.7
                weights.append(w)
            best_idx = int(np.argmax(weights)) if any(w > 0 for w in weights) else None
            if best_idx is None:
                break
            chosen = subjects.loc[best_idx]
            assigned_counts[str(chosen["과목"])] = assigned_counts.get(str(chosen["과목"]), 0) + 1
            last_subject = str(chosen["과목"])
            daily_sessions.append({
                "날짜": d,
                "시작": time_to_str(start_t),
                "종료": time_to_str(end_t),
                "과목": str(chosen["과목"]),
                "분": int(step_minutes),
                "시험범위": str(chosen.get("시험범위", "")),
            })
        if daily_sessions:
            sessions_by_date[d.isoformat()] = daily_sessions

    rows = []
    for d_str, rows_list in sessions_by_date.items():
        dt = datetime.fromisoformat(d_str).date()
        for r in rows_list:
            rows.append({"날짜": dt, "과목": r["과목"], "시간(시간)": round(r["분"] / 60.0, 2)})
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.groupby(["날짜", "과목"], as_index=False)["시간(시간)"].sum().sort_values(["날짜", "과목"])

    return {"sessions_by_date": sessions_by_date, "summary": summary_df}


