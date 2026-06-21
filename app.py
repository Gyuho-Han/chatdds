import time
from datetime import datetime

import streamlit as st

from text_utils import clean_response, extract_think_and_answer
from retriever import (
    load_documents, load_vectorstore, init_retrievers,
    format_docs_and_extract_urls,
)
from chain import init_generation_chain
from cache import (
    get_cached_answer, set_cached_answer,
    list_cached_answers, clear_cache, cache_available,
)


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


# ==========================================
# —— 사이드바: 메뉴 + 캐시 관리 ——
# ==========================================
with st.sidebar:
    st.header("📌 메뉴")
    page = st.radio("페이지 선택", ["💬 채팅", "🗂 캐시된 답변"], label_visibility="collapsed")

    st.divider()
    st.subheader("⚡ 캐시 관리")

    if cache_available():
        st.caption("Redis 연결됨 ✅")
        if st.button("🗑 캐시 비우기", use_container_width=True):
            removed = clear_cache()
            st.success(f"캐시 {removed}건을 삭제했습니다.")
            st.rerun()
    else:
        st.caption("Redis 미연결 (캐싱 비활성화) ⚠️")


# ==========================================
# —— 페이지 1: 채팅 ——
# ==========================================
def render_chat():
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

            # ── 캐시 조회: 동일 질문(+맥락)이면 검색·LLM 전부 건너뛰고 즉시 응답 ──
            cached_answer = get_cached_answer(query, chat_history_str)

            if cached_answer is not None:
                st.markdown(cached_answer)
                st.caption(f"⚡ 캐시된 답변 ({time.time() - start_time:.2f}초)")
                st.session_state.messages.append({"role": "assistant", "content": cached_answer})

            else:
                with st.status("🔍 데이터를 분석 중입니다...", expanded=True) as status:
                    st.write("1️⃣ 관련 문서를 검색 및 재정렬(Reranking) 중입니다...")

                    rerank_failed = False
                    try:
                        retrieved_docs = st.session_state.rerank_retriever.invoke(query)
                    except Exception:
                        rerank_failed = True
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

                # ── 캐시 저장: 다음 동일 질문은 즉시 응답 ──
                # 단, 리랭킹 fallback이 발생한 답변은 품질이 낮을 수 있어 캐싱하지 않는다.
                if final_answer_with_refs and not rerank_failed:
                    set_cached_answer(query, final_answer_with_refs, chat_history_str)

                st.session_state.messages.append({"role": "assistant", "content": final_answer_with_refs})


# ==========================================
# —— 페이지 2: 캐시된 답변 모아보기 ——
# ==========================================
def render_cache_view():
    st.subheader("🗂 캐시된 답변 모아보기")

    if not cache_available():
        st.warning("Redis가 연결되지 않아 캐시를 불러올 수 없습니다. `docker compose up -d`로 Redis를 실행하세요.")
        return

    items = list_cached_answers()

    if not items:
        st.info("아직 캐시된 답변이 없습니다. 채팅에서 질문하면 이곳에 쌓입니다.")
        return

    st.caption(f"총 {len(items)}건 (최신순)")

    for i, item in enumerate(items, 1):
        question = item["question"]
        created = item["created_at"]
        ttl = item["ttl"]

        created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M") if created else "-"
        ttl_str = f"{ttl // 3600}시간 {(ttl % 3600) // 60}분 남음" if isinstance(ttl, int) and ttl > 0 else "-"

        title = question if len(question) <= 40 else question[:40] + "…"
        with st.expander(f"{i}. {title}"):
            st.markdown(f"**질문:** {question}")
            st.caption(f"🕒 생성: {created_str}  ·  ⏳ 만료까지: {ttl_str}")
            st.divider()
            st.markdown(item["answer"])


# ==========================================
# —— 라우팅 ——
# ==========================================
if page == "💬 채팅":
    render_chat()
else:
    render_cache_view()
