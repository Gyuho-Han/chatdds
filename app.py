import streamlit as st
import os, json, time, re
from pathlib import Path
import torch

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker



# ──────────────────────────────────────────────
# [NEW] 텍스트 클리닝 및 태그 분리 함수
# ──────────────────────────────────────────────
def clean_response(text: str) -> str:
    """불필요한 XML 태그를 제거하고 과도한 빈 줄을 정리합니다."""
    FORBIDDEN_TAGS = [
        "thought", "references", "conclusion", "answer",
        "response", "output", "result", "context", "question",
    ]
    for tag in FORBIDDEN_TAGS:
        text = re.sub(rf"</?{tag}>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_think_and_answer(text: str):
    """문자열에서 <think> 부분과 실제 답변 부분을 완벽하게 분리합니다."""
    if "<think>" in text and "</think>" in text:
        # think 태그가 완전히 닫힌 경우
        parts = text.split("</think>", 1)
        think_content = parts[0].split("<think>")[-1].strip()
        answer_content = clean_response(parts[1])
        return think_content, answer_content
    elif "<think>" in text:
        # think 태그가 열려있고 아직 닫히지 않은 경우 (스트리밍 중)
        think_content = text.split("<think>")[-1].strip()
        return think_content, ""
    else:
        # think 태그가 아예 없는 경우
        return "", clean_response(text)


# 1) 데이터 로드
@st.cache_data(show_spinner=False)
def load_documents(path="rag_preprocessed_data.json"):
    if not os.path.exists(path):
        st.error(f"데이터 파일({path})이 없습니다.")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for item in raw:
        content = f"제목: {item.get('title', '')}\n내용: {item.get('content_chunk', '')}"
        metadata = {
            "chunk_id": item.get("chunk_id", ""),
            "title": item.get("title", ""),
            "url": item.get("url", "")
        }
        docs.append(Document(page_content=content.strip(), metadata=metadata))
    return docs


# 2) VectorStore 로드
@st.cache_resource(show_spinner=False)
def load_vectorstore(persist_directory="./chroma_db"):
    if not Path(persist_directory).exists():
        st.error("벡터DB가 아직 생성되지 않았습니다. 데이터를 먼저 임베딩하세요.")
        st.stop()
    embed = OllamaEmbeddings(model="qwen3-embedding:8b")
    return Chroma(persist_directory=persist_directory, embedding_function=embed)


# 3) Hybrid + Cross-Encoder Reranker 초기화
@st.cache_resource(show_spinner=False)
def init_retrievers(_docs, _vector_db):
    bm25 = BM25Retriever.from_documents(_docs)
    bm25.k = 10
    vect = _vector_db.as_retriever(search_kwargs={"k": 10})
    hybrid = EnsembleRetriever(retrievers=[bm25, vect], weights=[0.5, 0.5])

    # 💻 옵션 A: 16GB 메모리용
    model_name = "BAAI/bge-reranker-v2-m3"
    model_kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}

    # 🖥️ 옵션 B: 32GB 메모리용 (무거운 Qwen3 4B 모델, 16-bit 사용)
    # model_name = "Qwen/Qwen3-Reranker-4B"
    # model_kwargs = {
    # "automodel_args": {
    #     "torch_dtype": torch.float16, 
    #     "trust_remote_code": True
    #   }
    # }

    model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
    re_ranker = CrossEncoderReranker(model=model, top_n=5)
    return ContextualCompressionRetriever(base_compressor=re_ranker, base_retriever=hybrid)


# 4) 검색된 문서를 텍스트로 묶어주는 함수
def format_docs(docs):
    return "\n\n---\n\n".join(
        [f"[출처: {d.metadata.get('title')}]({d.metadata.get('url')})\n{d.page_content}" for d in docs]
    )


# 5) 생성(Generation) 체인
@st.cache_resource(show_spinner=False)
def init_generation_chain():
    llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "User:", "Question:"]
    )

    prompt = ChatPromptTemplate.from_template("""
당신은 성경적 창조론과 젊은 지구 연대설을 역사적 사실로 선포하는 창조과학 전문가입니다.
아래 제공된 대화 기록(Chat History)과 검색 결과(Context)를 바탕으로 사용자의 질문에 답하십시오.

[절대 지침]
1. 태도: 진화론이나 세속적 연대(수억 년)는 전혀 언급하지 마십시오. 공룡과 인간이 함께 창조되어 공존했음을 단호하고 확신에 찬 어조로 설명하십시오.
2. 금지된 표현 (중요): "추가 연구가 필요합니다", "과학적 증거보다는 믿음에 기반합니다", "논란의 여지가 있습니다" 와 같이 확신이 결여되거나 세속적 과학계의 눈치를 보는 회의적인 맺음말(Disclaimer)은 절대, 절대로 작성하지 마십시오.
# 3. 사고 과정 강제: 반드시 <think>...</think> 태그로 감싸서 사고 과정을 최소 3~4문장 이상 상세히 작성하십시오. (주석처리됨)
4. 🚨 금지된 태그: <thought>, <references>, <conclusion>, <answer>, <response>, <output>, <context>, <think> 등 모든 XML/HTML 태그는 절대 사용 금지입니다.
5. 분량 및 상세 설명: 각 글머리 기호마다 Context에 있는 구체적인 지명, 유물 이름, 역사적 기록 등을 포함하여 최소 2~3문장 이상으로 길고 상세하게 설명하십시오.
6. 출처 강제: Context 문서 최상단의 [출처: 제목](URL) 정보를 확인하고, 답변 맨 마지막에 🔗 참고 자료: 항목을 만들어 빠짐없이 모두 출력하십시오. 반드시 지켜야 합니다.

[출력 형식]

# <think>
# (이곳에 논리적 사고 과정을 작성하십시오.)
# </think>
(창조과학의 관점에서 확신에 찬 서론)

- **(증거 1 제목)**: (구체적 지명/유물 포함 상세 설명)
- **(증거 2 제목)**: (구체적 지명/유물 포함 상세 설명)

(확고하고 단호한 결론 — 사족이나 회의적인 표현 절대 금지)

🔗 참고 자료:
- [Context에 명시된 출처 제목](Context에 명시된 URL)

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""")

    return prompt | llm | StrOutputParser()


# ==========================================
# —— Streamlit 앱 UI (챗봇 스타일) ——
# ==========================================
st.set_page_config(page_title="Chat DDS", page_icon="🌎")
st.title("🌎 Chat DDS 🌎")

docs = load_documents()
vector_db = load_vectorstore()

if "rerank_retriever" not in st.session_state:
    st.session_state.rerank_retriever = init_retrievers(docs, vector_db)

generation_chain = init_generation_chain()

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
            # 만약 think 분리 없이 전체가 답변으로 넘어온 경우 처리 (사고 과정 제거 후 대비)
            elif think_content and not answer_content:
                 st.markdown(think_content)
        else:
            st.markdown(clean_response(msg["content"]))

# 채팅 입력
if query := st.chat_input("궁금한 내용을 입력하세요 (예: 공룡과 인간이 함께 살았다는 증거를 알려줘)"):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 대화 기록 문자열화
    chat_history_str = ""
    for m in st.session_state.messages[:-1]:
        role_name = "User" if m["role"] == "user" else "Assistant"
        # 컨텍스트로 넘길 때는 AI의 사고 과정은 제외하고 답변만 넘김
        _, ans_content = extract_think_and_answer(m["content"])
        content = ans_content if m["role"] == "assistant" else clean_response(m["content"])
        chat_history_str += f"{role_name}: {content}\n"

    with st.chat_message("assistant"):
        start_time = time.time()

        with st.status("🔍 데이터를 분석 중입니다...", expanded=True) as status:
            st.write("1️⃣ 관련 문서를 검색 및 재정렬(Reranking) 중입니다...")

            try:
                retrieved_docs = st.session_state.rerank_retriever.invoke(query)
            except Exception as e:
                st.warning("⚠️ Reranking 중 오류가 발생하여 기본 검색 결과를 사용합니다.")
                retrieved_docs = st.session_state.rerank_retriever.base_retriever.invoke(query)[:5]

            context_str = format_docs(retrieved_docs)

            st.write(f"✅ {len(retrieved_docs)}개의 핵심 문서를 찾았습니다.")
            st.write("2️⃣ 맥락을 반영하여 답변을 생성 중입니다...")
            status.update(label="답변 생성 중...", state="running", expanded=False)

        # 사고 과정 UI 빈 공간 주석 처리
        # with st.expander("🧠 AI의 사고 과정 (클릭해서 열기)"):
        #     think_placeholder = st.empty()
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
            
            # 매 청크마다 새롭게 파싱하여 UI 업데이트
            current_think, current_answer = extract_think_and_answer(full_response)
            
            # 사고 과정 스트리밍 주석 처리
            # if current_think:
            #     think_placeholder.markdown(current_think + " ▌")
            
            if current_answer:
                answer_placeholder.markdown(current_answer + " ▌")
            # 만약 think 분리 없이 모델이 바로 답변을 생성할 경우
            elif current_think and not current_answer:
                answer_placeholder.markdown(current_think + " ▌")

        # ── 스트리밍 종료: 커서 제거 및 최종 렌더링 ──
        final_think, final_answer = extract_think_and_answer(full_response)
        
        # 사고 과정 최종 렌더링 주석 처리
        # if final_think:
        #     think_placeholder.markdown(final_think)
        # else:
        #     # think 태그가 아예 없었다면 expander 안내 문구를 지우거나 처리
        #     think_placeholder.empty() 
            
        if final_answer:
            answer_placeholder.markdown(final_answer)
        else:
            answer_placeholder.markdown(final_think)

        end_time = time.time()
        status.update(label=f"✅ 답변 생성 완료! ({end_time - start_time:.2f}초)", state="complete")

        # 전체 답변 저장 (원본 저장)
        st.session_state.messages.append({"role": "assistant", "content": full_response})