# app.py

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 環境変数（.env）の読み込み ---
# Lesson8 などと同様に、.env に書かれた OPENAI_API_KEY を読み込みます
load_dotenv()

# --- LangChain（LLM）の準備 ---
# Lesson8 でやった ChatOpenAI + ChatPromptTemplate の構成をベースにしています
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 必要に応じてモデル名は変更OK
    temperature=0.3,
)

# system メッセージと human メッセージを組み合わせるプロンプト
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        ("human", "{user_input}"),
    ]
)

# 出力をただの文字列に変換するパーサ
output_parser = StrOutputParser()

# プロンプト → LLM → 文字列 という処理の流れ（Runnable）を作成
chain = prompt | llm | output_parser


# --- 入力テキスト & ラジオボタン選択値から LLM の回答を返す関数 ---
def generate_response(expert_type: str, user_input: str) -> str:
    """
    引数:
        expert_type: ラジオボタンで選択した専門家の種類（A/B）
        user_input: 画面の入力フォームに入力されたテキスト

    戻り値:
        LLM からの回答（文字列）
    """

    # 専門家ごとに system メッセージ（役割指示）を切り替える
    if expert_type == "キャリアアドバイザー（IT・生成AI転職）":
        system_message = (
            "あなたはIT業界および生成AI領域に詳しいキャリアアドバイザーです。"
            "質問者の経験レベルを踏まえつつ、日本語で丁寧に、"
            "具体的なステップや実行しやすいアドバイスを中心に回答してください。"
        )
    elif expert_type == "ファイナンシャルプランナー（家計・ライフプラン）":
        system_message = (
            "あなたは日本の税制や社会保険制度に詳しいファイナンシャルプランナーです。"
            "家計や将来のライフプランに関する相談に対して、日本語で分かりやすく説明し、"
            "可能であれば数字の目安や具体例も交えながらアドバイスしてください。"
        )
    else:
        # 念のためのフォールバック
        system_message = (
            "あなたは質問されたトピックについて、初心者にも分かりやすく説明する日本語の専門家です。"
        )

    # LangChain の chain に入力を渡して、LLM の回答（文字列）を取得
    response: str = chain.invoke(
        {
            "system_message": system_message,
            "user_input": user_input,
        }
    )

    return response


# --- Streamlit アプリ本体 ---
def main():
    # ページ設定
    st.set_page_config(
        page_title="LLM専門家相談アプリ",
        page_icon="🤖",
    )

    # タイトル & アプリの概要説明
    st.title("LLM専門家相談アプリ（LangChain × Streamlit）")
    st.write(
        """
        このWebアプリでは、テキストを入力して送信すると、
        選択した「専門家」の立場になったLLMが日本語で回答してくれます。

        **使い方**
        1. 下のラジオボタンで「相談したい専門家の種類」を選びます  
        2. テキスト入力欄に質問や相談内容を入力します  
        3. 「送信」ボタンを押すと、少し待ってから回答が画面下に表示されます  

        """
    )

    # 専門家の種類を選ぶラジオボタン
    expert_type = st.radio(
        "相談したい専門家のタイプを選んでください：",
        (
            "キャリアアドバイザー（IT・生成AI転職）",    # A
            "ファイナンシャルプランナー（家計・ライフプラン）",  # B
        ),
    )

    # ユーザー入力用のテキストフォーム
    user_input = st.text_area(
        "相談内容・質問を入力してください：",
        height=150,
        placeholder="例：未経験から生成AIエンジニアに転職したいのですが、どんな勉強から始めればよいでしょうか？",
    )

    # 送信ボタン
    if st.button("送信"):
        if not user_input.strip():
            st.warning("テキストを入力してください。")
        else:
            with st.spinner("LLMが回答を生成しています..."):
                answer = generate_response(expert_type, user_input)

            st.subheader("LLMからの回答")
            st.write(answer)


if __name__ == "__main__":
    main()
