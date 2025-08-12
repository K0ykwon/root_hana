import streamlit as st
import runpy

st.set_page_config(page_title="RootEDU 하나쌤 기능 Demo", page_icon="📚", layout="wide")

# 별도 스코프로 실행하여 위젯 키 충돌 최소화
runpy.run_path("frontend/app.py", run_name="__main__")