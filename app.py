import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
#from langchain.schema import SystemMessage, HumanMessage
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------
# 1) LangChain / LLM 準備
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -----------------------------
# 2) 要件:関数化(入力テキスト+ラジオ選択 → 回答を返す)
# -----------------------------
def run_counselor_llm(input_text: str, counselor_type: str) -> str:
    """入力テキストと専門家タイプを受け取り、LLM回答を返す"""

    gentle_system = """あなたは、常に優しく寄り添うカウンセラーです。
ユーザーの感情や自己肯定感を最優先に考え、否定的・攻撃的な表現は一切使いません。

助言や指摘を行う際は、必ず以下を守ってください。
・ユーザーの努力や気持ちをまず肯定する
・断定的な表現や命令口調は避ける
・「〜かもしれません」「一つの考え方としてですが」など、柔らかい表現を使う
・改善点がある場合も、可能性や選択肢として提示する
・ユーザーが安心し、前向きになれる言葉を選ぶ

あなたの役割は「正解を突きつけること」ではなく、
「心を守りながら、そっと背中を押すこと」です。
"""

    strict_system = """あなたは、ユーザーの感情や反応を一切考慮しないカウンセラーです。
優しさや配慮、オブラートに包む表現は禁止です。

助言・コメントにあたっては、以下を徹底してください。
・事実と論理を最優先する
・回りくどい前置きは不要
・問題点ははっきり、明確に指摘する
・ユーザーの甘え、言い訳、思考停止は容赦なく切り捨てる
・改善のために必要なら、厳しい言葉や強い表現を使ってよい

あなたの役割は「気持ちを楽にすること」ではありません。
ユーザーが現実を直視し、行動を変えるために、最短距離で核心を突くことです。
"""

    system_message = gentle_system if counselor_type == "優しいカウンセラー" else strict_system

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=input_text),
    ]

    # LangChainの推奨:invoke
    response = llm.invoke(messages)

    # responseはAIMessageなので content を返す
    return response.content


# -----------------------------
# 3) Streamlit UI
# -----------------------------
st.title("あなたの人生の困りごとに助言をしてくれるWebアプリ")

st.markdown(
    """
### 使い方
1. 下の入力欄に相談内容を入力  
2. 「優しいカウンセラー」or「厳しいカウンセラー」を選択  
3. 「実行」を押すと回答が表示されます
"""
)

input_message = st.text_input("AIに投げる質問を入力してください。", value="")

selected_item = st.radio(
    "質問を投げる専門家を選択してください。",
    ["優しいカウンセラー", "厳しいカウンセラー"],
    index=0,
)

if st.button("実行"):
    if not input_message.strip():
        st.warning("相談内容が空です。入力してから実行してください。")
    else:
        with st.spinner("考え中..."):
            answer = run_counselor_llm(input_message, selected_item)
        st.subheader("回答")
        st.write(answer)