import streamlit as st
import webbrowser

# html = open('./main.html', 'r', encoding='utf-8')
# main_html = html.read()                  # return string type

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css?family=Do+Hyeon&display=swap')
@import url('https://fonts.googleapis.com/css?family=Jua&display=swap')

@font-face {
    font-family: 'Cafe24Shiningstar';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_twelve@1.1/Cafe24Shiningstar.woff') format('woff');
    font-weight: normal;
    font-style: normal;
}


.title{
    font-family: 'Do Hyeon';
    text-align: center;
    font-size: 45px;
    color: #282828;
}

.subtitle{
    font-family: 'Jua';
    text-align: center;
    font-size: 20px;
    margin-bottom: 50px;
    color: #C0C0C0; d
}

.box_content{
    font-size: 16px;
    margin: 5px;
}

.text{
    font-size: 18px;
}

</style>""", unsafe_allow_html=True)

st.markdown('<p class="title">오늘 하루는 어땠어요?</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">당신의 하루를 이야기해주세요.</p>', unsafe_allow_html=True)

## write diary
diary = st.text_area(label='', height=270)
_, col, _ = st.columns([0.4, 1.0, 0.4])
writting_btn = col.button("나의 하루 보내기")

if writting_btn:
    link = "http://@@@@/result_page"
    webbrowser.open_new_tab(link)

