import streamlit as st
import time

from text_utils import clean_response, extract_think_and_answer
from retriever import (
    load_documents, load_vectorstore, init_retrievers,
    format_docs_and_extract_urls,
)
from chain import init_generation_chain


# ==========================================
# —— Streamlit 앱 UI (챗봇 스타일) ——
# ==========================================
st.set_page_config(page_title="Chat DDS", page_icon="🌎")
st.title("🌎 Chat DDS 🌎")


@st.cache_data(show_spinner=False)
def get_documents():
    return load_documents()


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource(show_spinner=False)
def get_generation_chain():
    return init_generation_chain()


try:
    docs = get_documents()
    vector_db = get_vectorstore()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if "rerank_retriever" not in st.session_state:
    st.session_state.rerank_retriever = init_retrievers(docs, vector_db)

generation_chain = get_generation_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            think_content, answer_content = extract_think_and_answer(msg["content"])

            # 사고 과정 UI 렌더링 주석 처리
            # if think_content:
            #     with st.expander("🧠 AI의 사고 과정"):
            #         st.markdown(think_content)

            if answer_content:
                st.markdown(answer_content)
            elif think_content and not answer_content:
                st.markdown(think_content)
        else:
            st.markdown(clean_response(msg["content"]))

# 채팅 입력
if query := st.chat_input("궁금한 내용을 입력하세요."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 대화 기록 문자열화
    chat_history_str = ""
    for m in st.session_state.messages[:-1]:
        role_name = "User" if m["role"] == "user" else "Assistant"
        _, ans_content = extract_think_and_answer(m["content"])
        content = ans_content if m["role"] == "assistant" else clean_response(m["content"])
        chat_history_str += f"{role_name}: {content}\n"

    with st.chat_message("assistant"):
        start_time = time.time()

        with st.status("🔍 데이터를 분석 중입니다...", expanded=True) as status:
            st.write("1️⃣ 관련 문서를 검색 및 재정렬(Reranking) 중입니다...")

            try:
                retrieved_docs = st.session_state.rerank_retriever.invoke(query)
            except Exception:
                st.warning("⚠️ Reranking 중 오류가 발생하여 기본 검색 결과를 사용합니다.")
                retrieved_docs = st.session_state.rerank_retriever.base_retriever.invoke(query)[:5]

            context_str, refs_str = format_docs_and_extract_urls(retrieved_docs)

            st.write(f"✅ {len(retrieved_docs)}개의 핵심 문서를 찾았습니다.")
            st.write("2️⃣ 맥락을 반영하여 답변을 생성 중입니다...")
            status.update(label="답변 생성 중...", state="running", expanded=False)

        answer_placeholder = st.empty()

        response_stream = generation_chain.stream({
            "chat_history": chat_history_str,
            "context": context_str,
            "question": query
        })

        full_response = ""

        # ── 스트리밍 루프 ──
        for chunk in response_stream:
            full_response += chunk
            current_think, current_answer = extract_think_and_answer(full_response)

            if current_answer:
                answer_placeholder.markdown(current_answer + " ▌")
            elif current_think and not current_answer:
                answer_placeholder.markdown(current_think + " ▌")

        # ── 스트리밍 종료: 커서 제거 및 출처 강제 결합 ──
        final_think, final_answer = extract_think_and_answer(full_response)

        if refs_str:
            final_answer_with_refs = final_answer + f"\n\n🔗 **참고 자료:**\n{refs_str}"
        else:
            final_answer_with_refs = final_answer

        if final_answer_with_refs:
            answer_placeholder.markdown(final_answer_with_refs)
        else:
            answer_placeholder.markdown(final_think)

        end_time = time.time()
        status.update(label=f"✅ 답변 생성 완료! ({end_time - start_time:.2f}초)", state="complete")

        st.session_state.messages.append({"role": "assistant", "content": final_answer_with_refs})
