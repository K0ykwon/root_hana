from typing import Tuple
import pandas as pd


def validate_subjects(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "과목 정보를 입력하세요."
    for col in ["과목", "시험일"]:
        if col not in df.columns:
            return False, f"필수 컬럼이 없습니다: {col}"
    any_valid = False
    for _, row in df.iterrows():
        subj = str(row.get("과목", "")).strip()
        exam = row.get("시험일", None)
        if subj and pd.notnull(exam):
            any_valid = True
            break
    if not any_valid:
        return False, "최소 1개 과목에 대해 과목명과 시험일을 입력하세요."
    return True, ""


