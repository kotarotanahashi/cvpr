import streamlit as st
import pandas as pd
import pickle
import numpy as np
import openai
from retrying import retry


# Retry parameters
retry_kwargs = {
    'stop_max_attempt_number': 5,  # Maximum number of retry attempts
    'wait_exponential_multiplier': 1000,  # Initial wait time between retries in milliseconds
    'wait_exponential_max': 10000,  # Maximum wait time between retries in milliseconds
}


DOMAIN = "https://openaccess.thecvf.com/"


@retry(**retry_kwargs)
def vectorize(text: str, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



def load_tag_vector():
    with open('resources/tag_vector.pickle', 'rb') as f:
        tag_vector = pickle.load(f)
    return tag_vector


def create_query_vec(query_tags, tag_vector):
    query_vector = []
    for tag in query_tags:
        query_vector.append(tag_vector[tag])
    query_vector = sum(np.array(query_vector)) / len(query_vector)
    return query_vector


def search_rows(tag_query_vector, text_query_vector, k, alpha):
    
    meta_df = pd.read_csv("data/vector/store/metas.csv")
    title_vec = np.load("data/vector/store/title_vector.npy")
    abst_vec = np.load("data/vector/store/abst_vector.npy")

    def calc_score(query_vector):
        title_score = title_vec @ query_vector
        abst_score = abst_vec @ query_vector
        return alpha * title_score + (1 - alpha) * abst_score
    
    if tag_query_vector is not None and text_query_vector is not None:
        query_vector = (tag_query_vector + text_query_vector) / 2.0
        score = calc_score(query_vector)
    
    elif tag_query_vector is not None:
        score = calc_score(tag_query_vector)
    
    elif text_query_vector is not None:
        score = calc_score(text_query_vector)
    
    else:
        raise ValueError("both query vector is None")

    top_k_indices = np.argsort(-score)[:k]
    return meta_df.iloc[top_k_indices]


def create_summary(placeholder, title, abst):
    prompt = """
    以下の論文について何がすごいのか、次の項目を出力してください。

    (1)既存研究では何ができなかったのか。
    (2)どのようなアプローチでそれを解決しようとしたか
    (3)結果、何が達成できたのか
    

    タイトル: {title}
    アブストラクト: {abst}
    """.format(title=title, abst=abst)

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    gen_text = "以下の項目についてChatGPTが回答します。<ol><li>既存研究では何ができなかったのか。</li><li>どのようなアプローチでそれを解決しようとしたか。</li><li>結果、何が達成できたのか。</li></ol><br/>"
    for chunk in response:
        content = chunk["choices"][0]["delta"].get("content")
        if content is not None:
            gen_text += content
            render_text = f"""<div style="background-color: #eeeeee; padding: 20px;">{gen_text}</div>"""
            placeholder.markdown(render_text, unsafe_allow_html=True)
    return gen_text


def main():

    st.title('Chat GPT Search, CVPR 2023')
    st.caption("[何ができる？] 検索キーワードをOpenAI APIを使ってベクトル化し、約2400のCVPR 2023の論文から関連する論文を検索することができます。また、論文の内容をChatGPTに要約してもらうことができます。")

    st.sidebar.title('Settings')

    if "token" not in st.session_state:
        st.session_state.token = ""

    token = st.sidebar.text_input('研究内容をChatGPTに聞く機能やフリーテキストによる検索を有効化するには、OpenAIのAPIキーを入力してください (APIキーを登録しなくてもタグによる検索機能は利用できます。)', type='password', value=st.session_state.token)

    if st.sidebar.button('APIキーの登録'):
        openai.api_key = token
        st.session_state.token = token
    
    if len(st.session_state.token) > 0:
        st.sidebar.write(f'トークンが設定されました')

    if "search_clicked" not in st.session_state:
        st.session_state.search_clicked = False
    
    def clear_session():
        st.session_state.search_clicked = False
        if "summary_clicked" in st.session_state:
            st.session_state.pop("summary_clicked")
        
        if "summary" in st.session_state:
            st.session_state.pop("summary")

    tag_vector = load_tag_vector()

    query_tags = st.multiselect("タグの選択(複数選択可)", options=tag_vector.keys(), on_change=clear_session)
    
    query_text = st.text_input("検索キーワード(日本語 or 英語)", value="", on_change=clear_session, disabled=len(st.session_state.token) == 0)

    """
    if len(st.session_state.token) > 0:
        query_text = st.text_input("検索キーワード(日本語 or 英語)", value="", on_change=clear_session, disabled=len(st.session_state.token) == 0)
    else:
        query_text = ""
    """
    
    target_options = ['タイトルから検索', 'タイトルとアブストラクトから検索', 'アブストラクトから検索']
    target = st.radio("検索条件", target_options, on_change=clear_session)
    ratio = target_options.index(target) / 2.0

    num_results = st.selectbox('表示件数:', (20, 50, 100, 200), index=0, on_change=clear_session)
    
    if st.button('検索'):
        st.session_state.search_clicked = True

    #if st.button('Search') and len(query_tags) > 0:
    if st.session_state.search_clicked and (len(query_tags) > 0 or len(query_text) > 0):

        if len(query_tags):
            tag_query_vector = create_query_vec(query_tags, tag_vector)
        else:
            tag_query_vector = None
        
        if len(query_text) > 0:
            text_query_vector = np.array(vectorize(query_text))
        else:
            text_query_vector = None
        
        results = search_rows(tag_query_vector, text_query_vector, k=num_results, alpha=ratio)
        results.fillna("", inplace=True)

        if "summary_clicked" not in st.session_state:
            st.session_state.summary_clicked = [False] * len(results)
        
        if "summary" not in st.session_state:
            st.session_state.summary = [""] * len(results)

        for i, (_, row) in enumerate(results.iterrows()):

            st.markdown(f"### **[{row['title']}]({DOMAIN + row['pdf_link']})**")
            st.markdown(f"{row['authors']}")
            st.caption(row["abst"])

            if st.button("この研究の何がすごいのかChatGPTに聞く", key=f"summary_{i}", disabled=st.session_state.token == ""):
                st.session_state.summary_clicked[i] = True
            
            if st.session_state.summary_clicked[i]:
                if len(st.session_state.summary[i]) == 0:
                    placeholder = st.empty()
                    gen_text = create_summary(placeholder, row['title'], row["abst"])
                    st.session_state.summary[i] = gen_text
                else:
                    print("summary exists")
                    st.markdown(st.session_state.summary[i], unsafe_allow_html=True)

            st.markdown("---")


if __name__ == "__main__":
    main()
