import streamlit as st
import pandas as pd
from sentiment_analysis.predict import SentimentAnalysis
from generation.kobart.predict import CommentGeneration
import re


# 실행 방법
# streamlit run main.py --server.fileWatcherType none --server.port=30001
s_preds = SentimentAnalysis("nlp04/kote_sentiment_roberta_large")
c_preds = CommentGeneration("nlp04/kobart_8_5.6e-5_default_option")
c_preds2 = CommentGeneration("nlp04/kobart_8_5.6e-5_min30_lp4_sample")

st.title("ProtoType")
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
    label, score, all_score = s_preds.sentiment_analysis(
        st.session_state['input'])
    index = list(all_score.keys())
    values = [all_score[i] for i in index]
    
    st.markdown('---')
    st.markdown('### nlp04/kobart_8_5.6e-5_default_option Comment')
    st.write(c_preds.comment_generation(st.session_state['input']))
    st.markdown("---")
    st.markdown('### nlp04/kobart_8_5.6e-5_min30_lp4_sample Comment')
    comment = c_preds2.comment_generation(st.session_state['input'])
    st.write(comment)
    
    chart_data = pd.DataFrame(values, index=index, columns=["result"])
    
    
    st.markdown('---')
    st.markdown('### Emotion')
    columns = st.columns([1, 1, 2])
    for i in range(len(columns)):
        if i == 2:
            break
        with columns[i]:
            st.write(f'{index[i]} : {all_score[index[i]]:.4f}')

    with st.expander("감정"):
        st.bar_chart(chart_data)
