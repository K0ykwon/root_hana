import json
from datetime import date, datetime
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os

from backend.availability.grid import build_availability_template, quick_fill_pattern, resample_availability
from backend.scheduler.plan import make_schedule
from backend.scheduler.focus import build_focus_dataframe, compute_focus_score, update_focus_map_with_observation
from backend.validation.subjects import validate_subjects
from backend.ai.generate import generate_session_descriptions
from backend.ai.guidelines import generate_guidelines
from backend.ai.openai_client import call_openai_chat


load_dotenv()


def ensure_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


if "subjects_df" not in st.session_state:
    st.session_state.subjects_df = pd.DataFrame([
        {"과목": "", "시험일": pd.NaT, "시험범위": ""},
        {"과목": "", "시험일": pd.NaT, "시험범위": ""},
    ])
    st.session_state.subjects_df["시험일"] = pd.to_datetime(st.session_state.subjects_df["시험일"], errors="coerce")

if "availability_df" not in st.session_state:
    st.session_state.availability_df = build_availability_template("06:00", "23:00", 30)

if "step_minutes" not in st.session_state:
    st.session_state.step_minutes = 30

if "plan" not in st.session_state:
    st.session_state.plan = {"sessions_by_date": {}, "summary": pd.DataFrame(), "guidelines": {}, "overall_message": ""}

if "focus_map" not in st.session_state:
    st.session_state.focus_map = []


with st.sidebar:
    st.header("설정")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0, key="sidebar_model")
    st.session_state["model_override"] = model
    st.markdown("---")
    st.caption("시간 단위 설정")
    step = st.slider("슬롯 간격(분)", min_value=15, max_value=120, step=15, value=st.session_state.step_minutes)
    if step != st.session_state.step_minutes:
        st.session_state.availability_df = resample_availability(
            st.session_state.availability_df,
            old_step=st.session_state.step_minutes,
            new_step=step,
            start_hhmm="06:00",
            end_hhmm="23:00",
        )
        st.session_state.step_minutes = step
    st.markdown("---")
    st.caption("가용 시간대 빠른 설정")
    pattern = st.selectbox("패턴", ["선택 안 함", "평일 저녁(19~22시) + 주말(10~18시)", "모두 비우기", "모두 채우기"], index=0, key="sidebar_pattern")
    if pattern != "선택 안 함":
        st.session_state.availability_df = quick_fill_pattern(st.session_state.availability_df, pattern)


st.title("📚 RootEDU 하나쌤 기능 Demo")
st.caption("학생 정보, 과목별 시험일정·범위, 취약과목, 요일별 가용 시간대를 바탕으로 전체/세부 계획을 생성합니다.")

tab_input, tab_overall, tab_daily = st.tabs(["입력", "전체 계획", "세부 계획"])

with tab_input:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("학생 정보")
        student_name = st.text_input("이름", value="")
        level = st.selectbox("수준", ["초등", "중등", "고등", "대학", "성인/기타"], index=2, key="level_select")
        start_day = st.date_input("계획 시작일", value=date.today())

    with col2:
        st.subheader("과목 정보")
        st.caption("과목명/시험일은 필수입니다. 시험범위는 가능한 자세히 적어주세요.")
        sdf = st.session_state.subjects_df.copy()
        if "시험일" in sdf.columns:
            sdf["시험일"] = pd.to_datetime(sdf["시험일"], errors="coerce")
        with st.form("subjects_form", clear_on_submit=False):
            subjects_df_tmp = st.data_editor(
                sdf,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "과목": st.column_config.TextColumn(required=False, width="medium"),
                    "시험일": st.column_config.DateColumn(min_value=date.today(), format="YYYY-MM-DD"),
                    "시험범위": st.column_config.TextColumn(width="large"),
                },
                key="subjects_editor",
            )
            submitted_subjects = st.form_submit_button("과목 입력 적용")
        subjects_df = st.session_state.subjects_df
        if submitted_subjects:
            if "시험일" in subjects_df_tmp.columns:
                subjects_df_tmp["시험일"] = pd.to_datetime(subjects_df_tmp["시험일"], errors="coerce")
            st.session_state.subjects_df = subjects_df_tmp.copy()
            subjects_df = st.session_state.subjects_df

    st.subheader("취약과목")
    valid_subject_names = [s for s in subjects_df["과목"].astype(str).tolist() if s.strip()]
    weak_subjects = st.multiselect("특히 보완이 필요한 과목을 선택하세요", options=valid_subject_names)

    st.subheader("요일별 공부 가능 시간대 (When2meet 스타일)")
    st.caption("체크된 칸이 공부 가능한 시간 슬롯입니다.")
    with st.form("availability_form", clear_on_submit=False):
        avail_df_tmp = st.data_editor(
            st.session_state.availability_df,
            use_container_width=True,
            key="availability_editor",
        )
        submitted_avail = st.form_submit_button("시간표 입력 적용")
    if submitted_avail and isinstance(avail_df_tmp, pd.DataFrame):
        avail_df_tmp = avail_df_tmp.copy()
        for w in ["월","화","수","목","금","토","일"]:
            if w not in avail_df_tmp.columns:
                avail_df_tmp[w] = False
        avail_df_tmp.index = avail_df_tmp.index.astype(str)
        st.session_state.availability_df = avail_df_tmp
    avail_df = st.session_state.availability_df

    st.subheader("집중력 측정/업데이트")
    st.caption("간단 문제 풀이 결과를 반영해 요일·시간대별 집중력 지도를 업데이트합니다. (선택)")
    with st.expander("집중력 관측값 입력"):
        c1, c2, c3 = st.columns(3)
        with c1:
            accuracy = st.number_input("정답률(0~1)", min_value=0.0, max_value=1.0, step=0.05, value=0.8)
        with c2:
            avg_ms = st.number_input("평균 반응시간(ms)", min_value=0.0, step=50.0, value=1200.0)
        with c3:
            no_lag = st.number_input("무지연 응답률(0~1)", min_value=0.0, max_value=1.0, step=0.05, value=0.9)
        miss_pen = st.number_input("미응답 패널티(0~1)", min_value=0.0, max_value=1.0, step=0.05, value=0.0)
        time_str = st.time_input("관측 시각", value=datetime.now().time()).strftime("%H:%M")
        weekday = st.selectbox("관측 요일", options=["월","화","수","목","금","토","일"], index=date.today().weekday(), key="weekday_select")
        bin_minutes = st.session_state.step_minutes

        if st.button("집중력 업데이트"):
            try:
                obs = {"accuracy": accuracy, "avg_response_ms": avg_ms, "no_lag_rate": no_lag, "miss_penalty": miss_pen}
                val = compute_focus_score(obs)
                wd = ["월","화","수","목","금","토","일"].index(weekday)
                st.session_state.focus_map = update_focus_map_with_observation(
                    st.session_state.focus_map,
                    focus_value=val,
                    bin_minutes=bin_minutes,
                    weekday_idx=wd,
                    time_str=time_str,
                )
                st.success("집중력 지도가 업데이트되었습니다.")
            except Exception as e:
                st.error(f"업데이트 실패: {e}")

        st.markdown("---")
        st.caption("문제 기반 측정 (문제 생성 → 선택지 응답 → 자동 채점/시간 측정 → 주의력 테스트 → 업데이트)")
        subj_for_q = st.selectbox("문제 과목", options=valid_subject_names or [""], index=0, key="focus_subject_select")
        grade_for_q = st.selectbox("학년", options=["초등", "중등", "고등"], index=2, key="focus_grade_select")
        level_for_q = level
        recent_topic = st.text_input("최근 학습 주제 (예: 도형의 닮음/형태소 등)")
        if st.button("집중력 체크 문제 생성"):
            api_key = ensure_api_key()
            if not api_key:
                st.warning("OpenAI API Key가 필요합니다.")
            else:
                prompt = (
                    f"당신은 학생이 학습에 집중하고 있는지 파악하려 하는 {grade_for_q} {subj_for_q} 교사입니다.\n"
                    f"학생의 학습 상태를 파악하기 위해 5지선다 개념 확인 문제를 JSON으로 생성하세요.\n"
                    f"형식: {{\"problem\": str, \"choices\": [..5], \"answer_index\": int}}\n"
                    f"주제: {recent_topic}"
                )
                resp = call_openai_chat(
                    api_key,
                    [{"role": "user", "content": prompt}],
                    model=st.session_state.get("model_override", "gpt-4o-mini"),
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                if resp:
                    try:
                        data = json.loads(resp)
                    except Exception:
                        data = {"problem": resp, "choices": [], "answer_index": 0}
                    st.session_state["focus_last_question"] = data
                    st.session_state["focus_q_start_ms"] = int(datetime.now().timestamp() * 1000)
                    st.success("문제를 생성했습니다. 아래에 답을 선택하세요.")
        if q := st.session_state.get("focus_last_question"):
            st.write(q.get("problem", ""))
            ch = q.get("choices", [])
            selected_idx = st.radio(
                "선택지",
                options=list(range(len(ch))) if ch else [0],
                format_func=lambda i: f"{i+1}. {ch[i]}" if ch and i < len(ch) else str(i+1),
                horizontal=False,
                key="focus_choice_radio",
            )
            if st.button("응답 제출 및 집중력 업데이트"):
                try:
                    end_ms = int(datetime.now().timestamp() * 1000)
                    start_ms = int(st.session_state.get("focus_q_start_ms", end_ms))
                    elapsed_ms = max(0, end_ms - start_ms)
                    correct_idx = int(q.get("answer_index", -1))
                    is_correct = 1.0 if selected_idx == correct_idx and correct_idx >= 0 else 0.0
                    obs2 = {
                        "accuracy": is_correct,
                        "avg_response_ms": float(elapsed_ms),
                        "no_lag_rate": 1.0,
                        "miss_penalty": 0.0 if is_correct else 0.2,
                    }
                    val2 = compute_focus_score(obs2)
                    wd = ["월","화","수","목","금","토","일"].index(weekday)
                    st.session_state.focus_map = update_focus_map_with_observation(
                        st.session_state.focus_map,
                        focus_value=val2,
                        bin_minutes=bin_minutes,
                        weekday_idx=wd,
                        time_str=time_str,
                    )
                    st.success("응답이 반영되었습니다. 집중력 지도가 업데이트되었습니다.")
                    # 주의력 테스트 생성
                    api_key = ensure_api_key()
                    if api_key and ch:
                        prev_q = {"problem": q.get("problem", ""), "choices": ch, "answer_index": int(q.get("answer_index", -1))}
                        attn_prompt = (
                            "당신은 학생이 학습에 집중하고 있는지 파악하려 하는 교사입니다.\n"
                            "이전 문제를 정말로 집중해서 풀었는지 확인하기 위해 이전 문제에 대해 학습 내용과 관련 없는 간단한 주의력 테스트를 생성하세요.\n"
                            "주의력 테스트 규칙: 5지선다 또는 O/X, JSON 형식으로만 반환.\n"
                            "형식: {\"problem\": str, \"type\": \"mcq\"|\"ox\", \"choices\": [..], \"answer_index\": int}\n"
                            f"이전 문제: {json.dumps(prev_q, ensure_ascii=False)}\n"
                            "[주의력 테스트]"
                        )
                        attn_resp = call_openai_chat(
                            api_key,
                            [{"role": "user", "content": attn_prompt}],
                            model=st.session_state.get("model_override", "gpt-4o-mini"),
                            temperature=0,
                            response_format={"type": "json_object"},
                        )
                        if attn_resp:
                            try:
                                attn_q = json.loads(attn_resp)
                            except Exception:
                                attn_q = {"problem": attn_resp, "type": "ox", "choices": ["O", "X"], "answer_index": 0}
                            st.session_state["attention_question"] = attn_q
                            st.session_state["attention_q_start_ms"] = int(datetime.now().timestamp() * 1000)
                            st.info("주의력 테스트가 생성되었습니다. 이어서 응답해 주세요.")
                    # 포커스 문제 상태 초기화
                    st.session_state.pop("focus_q_start_ms", None)
                    st.session_state.pop("focus_last_question", None)
                except Exception as e:
                    st.error(f"업데이트 실패: {e}")

        if attn := st.session_state.get("attention_question"):
            st.markdown("**주의력 테스트**")
            st.write(attn.get("problem", ""))
            attn_type = str(attn.get("type", "mcq")).lower()
            if attn_type == "ox" and not attn.get("choices"):
                attn["choices"] = ["O", "X"]
            ch2 = attn.get("choices", [])
            sel2 = st.radio(
                "선택지",
                options=list(range(len(ch2))) if ch2 else [0],
                format_func=lambda i: f"{i+1}. {ch2[i]}" if ch2 and i < len(ch2) else str(i+1),
                horizontal=False,
                key="attn_choice_radio",
            )
            if st.button("주의력 응답 제출 및 업데이트"):
                try:
                    end_ms = int(datetime.now().timestamp() * 1000)
                    start_ms = int(st.session_state.get("attention_q_start_ms", end_ms))
                    elapsed_ms = max(0, end_ms - start_ms)
                    correct_idx2 = int(attn.get("answer_index", -1))
                    is_correct2 = 1.0 if sel2 == correct_idx2 and correct_idx2 >= 0 else 0.0
                    obs3 = {
                        "accuracy": is_correct2,
                        "avg_response_ms": float(elapsed_ms),
                        "no_lag_rate": 1.0 if elapsed_ms < 5000 else 0.8,
                        "miss_penalty": 0.0 if is_correct2 else 0.2,
                    }
                    wd = ["월","화","수","목","금","토","일"].index(weekday)
                    val3 = compute_focus_score(obs3)
                    st.session_state.focus_map = update_focus_map_with_observation(
                        st.session_state.focus_map,
                        focus_value=val3,
                        bin_minutes=bin_minutes,
                        weekday_idx=wd,
                        time_str=time_str,
                    )
                    st.success("주의력 응답이 반영되었습니다. 집중력 지도가 업데이트되었습니다.")
                    st.session_state.pop("attention_q_start_ms", None)
                    st.session_state.pop("attention_question", None)
                except Exception as e:
                    st.error(f"업데이트 실패: {e}")

    # 집중력 히트맵
    with st.expander("요일/시간 집중력 히트맵"):
        focus_df = build_focus_dataframe(st.session_state.focus_map, st.session_state.step_minutes)
        st.dataframe(focus_df.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

    generate = st.button("계획 생성", type="primary")
    if generate:
        ok, msg = validate_subjects(subjects_df)
        if not ok:
            st.error(msg)
        else:
            api_key = ensure_api_key()
            model = st.session_state.get("model_override", "gpt-4o-mini")
            if not api_key:
                st.warning("OpenAI API Key를 입력하면 과목별 학습 가이드와 설명을 함께 생성합니다. (선택)")

            with st.spinner("계획을 생성하는 중..."):
                plan = make_schedule(
                    subjects_df=subjects_df,
                    avail_df=st.session_state.availability_df,
                    weak_subjects=weak_subjects,
                    step_minutes=st.session_state.step_minutes,
                    start_date=start_day,
                    focus_map=st.session_state.focus_map,
                )
                guidelines_msg = ""
                guidelines_map: Dict[str, Dict[str, List[str]]] = {}
                if api_key:
                    g = generate_guidelines(api_key, model, student_name, level, subjects_df, weak_subjects)
                    guidelines_msg = g.get("overall_message", "") or ""
                    raw = g.get("guidelines", {}) or {}
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            guidelines_map[str(k)] = {
                                "study_methods": [str(x) for x in v.get("study_methods", []) or []],
                                "checklist": [str(x) for x in v.get("checklist", []) or []],
                            }
                if api_key:
                    descriptions_by_date = generate_session_descriptions(api_key, model, plan.get("sessions_by_date", {}), subjects_df, st.session_state.step_minutes, level)
                else:
                    descriptions_by_date = {}
                st.session_state.plan = {
                    "sessions_by_date": plan.get("sessions_by_date", {}),
                    "summary": plan.get("summary", pd.DataFrame()),
                    "guidelines": guidelines_map,
                    "overall_message": guidelines_msg,
                    "descriptions_by_date": descriptions_by_date,
                }
                st.success("계획 생성 완료! 상단 탭에서 결과를 확인하세요.")


with tab_overall:
    st.subheader("전체 계획")
    if st.session_state.plan.get("overall_message"):
        st.info(st.session_state.plan.get("overall_message", ""))
    summary_df = st.session_state.plan.get("summary", pd.DataFrame())
    if summary_df is None or summary_df.empty:
        st.warning("생성된 전체 계획이 없습니다. 입력을 확인하고 '계획 생성'을 눌러주세요.")
    else:
        # 날짜/과목별 간략 설명 표
        sessions_by_date = st.session_state.plan.get("sessions_by_date", {})
        desc_map: Dict[str, List[str]] = st.session_state.plan.get("descriptions_by_date", {})
        rows: List[Dict] = []
        for d_str, sessions in sessions_by_date.items():
            dt = datetime.fromisoformat(d_str).date()
            ordered = sorted(sessions, key=lambda x: x.get("시작", "00:00"))
            descs = desc_map.get(d_str, [""] * len(ordered))
            agg: Dict[str, Dict[str, object]] = {}
            for i, s in enumerate(ordered):
                subj = s.get("과목")
                agg.setdefault(subj, {"minutes": 0, "texts": []})
                agg[subj]["minutes"] = int(agg[subj]["minutes"]) + int(s.get("분", 0))
                text = descs[i] if i < len(descs) else ""
                if text:
                    agg[subj]["texts"].append(text)
            for subj, info in agg.items():
                rows.append({
                    "날짜": dt,
                    "과목": subj,
                    "시간(시간)": round(int(info["minutes"]) / 60.0, 2),
                    "간략 설명": (" | ".join(info["texts"])[:300]) if info["texts"] else "",
                })
        if rows:
            df = pd.DataFrame(rows).sort_values(["날짜", "과목"]).reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


with tab_daily:
    st.subheader("세부 계획")
    sessions_by_date = st.session_state.plan.get("sessions_by_date", {})
    if not sessions_by_date:
        st.warning("세부 계획 데이터가 없습니다. 먼저 계획을 생성하세요.")
    else:
        all_dates = sorted([datetime.fromisoformat(k).date() for k in sessions_by_date.keys()])
        sel_date = st.date_input("날짜 선택", value=all_dates[0], min_value=all_dates[0], max_value=all_dates[-1])
        key = sel_date.isoformat()
        day_sessions = sessions_by_date.get(key, [])
        if not day_sessions:
            st.info("이 날짜에는 계획된 학습이 없습니다.")
        else:
            df = pd.DataFrame(day_sessions)
            df["시간(분)"] = df["분"]
            desc_map: Dict[str, List[str]] = st.session_state.plan.get("descriptions_by_date", {})
            descs = desc_map.get(key, [""] * len(df))
            df["학습내용"] = [descs[i] if i < len(descs) else "" for i in range(len(df))]
            df = df.drop(columns=["분"]).loc[:, ["시작", "종료", "과목", "시간(분)", "시험범위", "학습내용"]]
            st.dataframe(df, use_container_width=True, hide_index=True)


