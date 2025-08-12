import json
from typing import Dict, List, Optional
import pandas as pd

from backend.ai.openai_client import call_openai_chat


def generate_session_descriptions(api_key: Optional[str], model: str, sessions_by_date: Dict[str, List[Dict]], subjects_df: pd.DataFrame, step_minutes: int, level: str) -> Dict[str, List[str]]:
    if not api_key:
        return {}
    simple_rows = []
    for _, r in subjects_df.iterrows():
        subj = str(r.get("과목", "")).strip()
        if not subj:
            continue
        exam = r.get("시험일", None)
        if pd.isna(exam):
            continue
        exam_str = str(pd.to_datetime(exam).date())
        scope = str(r.get("시험범위", "")).strip()
        simple_rows.append({"subject": subj, "exam_date": exam_str, "scope": scope})

    outline = {}
    for d_str, sessions in sessions_by_date.items():
        ordered = sorted(sessions, key=lambda x: x.get("시작", "00:00"))
        outline[d_str] = [
            {
                "slot_index": i,
                "start": s.get("시작"),
                "end": s.get("종료"),
                "subject": s.get("과목"),
                "slot_minutes": step_minutes,
            }
            for i, s in enumerate(ordered)
        ]

    user_payload = {
        "level": level,
        "step_minutes": step_minutes,
        "subjects": simple_rows,
        "schedule_outline": outline,
        "instruction": (
            "각 날짜의 각 슬롯별로 수행할 학습 항목을 '아주 짧은 키워드형 문구' 1개로 작성하세요."
            " 문장/마침표/번호 금지, 6~20자 권고. JSON만 출력."
        ),
        "output_format": {"descriptions_by_date": {"YYYY-MM-DD": ["형태소 개념 복습"]}},
        "language": "ko",
    }

    content = call_openai_chat(
        api_key,
        [
            {"role": "system", "content": "JSON만 출력하는 학습 계획 작성 조교"},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        model=model,
        temperature=0.35,
        response_format={"type": "json_object"},
    )
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        details = parsed.get("descriptions_by_date", {})
        return details if isinstance(details, dict) else {}
    except Exception:
        return {}


