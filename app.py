import streamlit as st
import pandas as pd
import pickle
import numpy as np
import openai
from retrying import retry
from PIL import Image
import urllib
import threading
import time
import requests
import json



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



def chat_completion_request(messages, functions=None, result=[], model="gpt-3.5-turbo-0613"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        result.append(response)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")


def create_summary(placeholder, title, abst):
    prompt = """
    以下の論文について何がすごいのか、次の項目を日本語で出力してください。

    (1)既存研究では何ができなかったのか。
    (2)どのようなアプローチでそれを解決しようとしたか
    (3)結果、何が達成できたのか
    

    タイトル: {title}
    アブストラクト: {abst}
    日本語で出力してください。
    """.format(title=title, abst=abst)

    functions = [
        {
            "name": "format_output",
            "description": "アブストラクトのサマリー",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_of_existing_research": {
                        "type": "string",
                        "description": "既存研究では何ができなかったのか",
                    },
                    "how_to_solve": {
                        "type": "string",
                        "description": "どのようなアプローチでそれを解決しようとしたか",
                    },
                    "what_they_achieved": {
                        "type": "string",
                        "description": "結果、何が達成できたのか",
                    },
                },
                "required": ["problem_of_existing_research", "how_to_solve", "what_they_achieved"],
            },
        }
    ]

    placeholder.markdown("ChatGPTが考え中です...😕", unsafe_allow_html=True)
    #res = chat_completion_request(messages=[{"role": "user", "content": prompt}], functions=functions)
    m = [{"role": "user", "content": prompt}]
    result = []
    thread = threading.Thread(target=chat_completion_request, args=(m, functions, result))
    thread.start()
    i = 0
    faces = ["😕", "😆", "😴", "😊", "😱", "😎", "😏"]
    while thread.is_alive():
        i += 1
        face = faces[i % len(faces)]
        placeholder.markdown(f"ChatGPTが考え中です...{face}", unsafe_allow_html=True)
        time.sleep(0.5)
    thread.join()

    if len(result) == 0:
        placeholder.markdown("ChatGPTの結果取得に失敗しました...😢", unsafe_allow_html=True)
        return
    
    res = result[0]
    func_result = res.json()["choices"][0]["message"]["function_call"]["arguments"]
    output = json.loads(func_result)
    a1 = output["problem_of_existing_research"]
    a2 = output["how_to_solve"]
    a3 = output["what_they_achieved"]
    gen_text = f"""以下の項目についてChatGPTが回答します。
    <ol>
        <li><b>既存研究では何ができなかったのか</b></li>
        <li style="list-style:none;">{a1}</li>
        <li><b>どのようなアプローチでそれを解決しようとしたか</b></li>
        <li style="list-style:none;">{a2}</li>
        <li><b>結果、何が達成できたのか</b></li>
        <li style="list-style:none;">{a3}</li>
    </ol>"""
    render_text = f"""<div style="border: 1px rgb(128, 132, 149) solid; padding: 20px;">{gen_text}</div>"""
    placeholder.markdown(render_text, unsafe_allow_html=True)
    return gen_text



def main():

    st.set_page_config(page_title="LLMによるCVPR論文検索システム")
    image = Image.open('top.png')

    st.image(image, caption='CVPR, June 18-23, 2023, Vancouver, Canada, [image-ref: wikipedia.org]', use_column_width=True)

    st.title('CVPR 2023, 文書埋め込みを用いた論文検索')
    st.caption("検索キーワードをOpenAI APIを使ってベクトル化し、約2400のCVPR 2023の論文から関連する論文を検索することができます。また、論文の内容をChatGPTに要約してもらうことができます。")

    #st.sidebar.title('Settings')

    openai.api_key = st.session_state.token = st.secrets["OPENAI_API_KEY"]

    #if "token" not in st.session_state:
    #    st.session_state.token = ""

    #token = st.sidebar.text_input('研究内容をChatGPTに聞く機能やフリーテキストによる検索を有効化するには、OpenAIのAPIキーを入力してください (APIキーを登録しなくてもタグによる検索機能は利用できます。)', type='password', value=st.session_state.token)

    #if st.sidebar.button('APIキーの登録'):
    #    openai.api_key = token
    #    st.session_state.token = token
    
    #if len(st.session_state.token) > 0:
    #    st.sidebar.write(f'トークンが設定されました')

    if "search_clicked" not in st.session_state:
        st.session_state.search_clicked = False
    
    def clear_session():
        st.session_state.search_clicked = False
        if "summary_clicked" in st.session_state:
            st.session_state.pop("summary_clicked")
        
        if "summary" in st.session_state:
            st.session_state.pop("summary")

    tag_vector = load_tag_vector()

    api_available = len(st.session_state.token) > 0
    exp_text = "APIを入れると入力可能になります" if not api_available else ""
    query_text = st.text_input(
        "検索キーワード(日本語 or 英語) " + exp_text, value="",
        on_change=clear_session,
        disabled=not api_available)
    
    query_tags = st.multiselect("[オプション] タグの選択(複数選択可)", options=tag_vector.keys(), on_change=clear_session)


    target_options = ['タイトルから検索', 'タイトルとアブストラクトから検索', 'アブストラクトから検索']
    target = st.radio("検索条件", target_options, on_change=clear_session)
    ratio = target_options.index(target) / 2.0

    num_results = st.selectbox('表示件数:', (20, 50, 100, 200), index=0, on_change=clear_session)
    
    if st.button('検索'):
        st.session_state.search_clicked = True

    has_get_params = False
    get_query_params = st.experimental_get_query_params()
    if len(get_query_params.get("q", "")) > 0 and st.session_state.search_clicked == False:
        query_text = get_query_params["q"][0]
        print("query_text", query_text)
        query_tags = []
        has_get_params = True

    #if st.button('Search') and len(query_tags) > 0:
    if (st.session_state.search_clicked and (len(query_tags) > 0 or len(query_text) > 0)) or has_get_params:
        st.markdown("## **検索結果**")

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

            title = row['title']
            pdf_link = row['pdf_link']
            authors = row['authors']
            abst = row["abst"]
            st.markdown(f"### **[{title}]({DOMAIN + pdf_link})**")
            st.markdown(f"{authors}")
            st.caption(abst)

            link = f"[この研究と似た論文を探す](/?q={urllib.parse.quote(title)})"
            st.markdown(link, unsafe_allow_html=True)

            if st.button(
                "この研究の何がすごいのかChatGPTに聞く",
                key=f"summary_{i}",
                disabled=st.session_state.token == ""):
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
