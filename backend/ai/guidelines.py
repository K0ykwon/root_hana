import json
from typing import Dict, List, Optional
import pandas as pd

from backend.ai.openai_client import call_openai_chat


def generate_guidelines(api_key: Optional[str], model: str, student_name: str, level: str, subjects_df: pd.DataFrame, weak_subjects: List[str]) -> Dict:
    if not api_key:
        return {"guidelines": {}, "overall_message": ""}
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
        simple_rows.append({
            "subject": subj,
            "exam_date": exam_str,
            "scope": scope,
            "weak": subj in weak_subjects,
        })

    sys_prompt = "당신은 학생 맞춤형 학습 코치를 수행하는 조교입니다. 결과는 반드시 유효한 JSON만 출력합니다."
    user_prompt = {
        "student_name": student_name,
        "level": level,
        "subjects": simple_rows,
        "instruction": "각 과목별로 'study_methods'(3~6개)와 'checklist'(3~6개), 'overall_message'를 JSON으로 출력",
        "format": {
            "guidelines": {
                "<과목명>": {"study_methods": ["..."], "checklist": ["..."]}
            },
            "overall_message": "..."
        },
        "language": "ko",
    }

    content = call_openai_chat(
        api_key,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}],
        model=model,
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    result: Dict = {"guidelines": {}, "overall_message": ""}
    if not content:
        return result
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            result.update(parsed)
    except Exception:
        result["overall_message"] = content
    return result


