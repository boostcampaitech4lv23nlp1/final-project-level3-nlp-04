import streamlit as st
import pandas as pd
from sentiment_labeling import SentimentAnalysis

# 실행 방법
# streamlit run main.py --server.port=30001
preds = SentimentAnalysis("nlp04/korean_sentiment_analysis_dataset3")

st.title("Sentiment Analysis")
if 'input' not in st.session_state:
    st.session_state["input"] = ''
if 'result' not in st.session_state:
    st.session_state["result"] = []

with st.form(key="my_form"):

    st.text_area(
        "input sentence",
        key="input", height=400)
    submit = st.form_submit_button(label="click")

if submit:
    label, score, all_score = preds.sentiment_analysis(
        st.session_state['input'])
    index = list(all_score.keys())
    values = [all_score[i] for i in index]

    columns = st.columns([1, 1, 3])
    for i in range(len(columns)):
        with columns[i]:
            st.write(f'{index[i]} : {all_score[index[i]]:.4f}')

    chart_data = pd.DataFrame(values, index=index, columns=["result"])
    st.bar_chart(chart_data)
