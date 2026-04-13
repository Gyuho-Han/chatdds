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

# 4) 검색된 문서 텍스트화 및 고유 URL 하이퍼링크 추출
def format_docs_and_extract_urls(docs):
    context_parts = []
    unique_refs = {}  # URL 중복 제거를 위한 딕셔너리
    
    for d in docs:
        title = d.metadata.get('title', '제목 없음')
        url = d.metadata.get('url', '')
        
        # LLM에게 제공할 본문 구성 (LLM은 이 텍스트만 보고 답변을 생성)
        context_parts.append(f"[출처: {title}]\n{d.page_content}")
        
        # URL을 딕셔너리의 Key로 사용하여 중복을 자동으로 제거
        # url 값이 존재하는 경우에만 추가
        if url and url not in unique_refs:
            unique_refs[url] = title
            
    context_str = "\n\n---\n\n".join(context_parts)
    
    # 🔗 마크다운 문법 [Title](URL)을 사용하여 제목에 하이퍼링크 씌우기
    ref_list = [f"- [{title}]({url})" for url, title in unique_refs.items()]
    refs_str = "\n".join(ref_list)
    
    return context_str, refs_str


# 5) 생성(Generation) 체인
@st.cache_resource(show_spinner=False)
def init_generation_chain():
    llm = ChatOllama(
        model="qwen2.5:14b",
        # model="gpt-oss:20b",
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "User:", "Question:"]
    )

    prompt = ChatPromptTemplate.from_template("""
당신은 창조과학의 관점과 주장을 소개하는 전문 해설가입니다. 
당신의 주된 목적은 제공된 자료(Context)를 바탕으로 창조과학계의 해석과 증거들을 사용자에게 상세히 전달하는 것입니다. 
아래의 지침을 반드시 준수하여 답변하십시오.

[최우선 지침: 출처 표기 금지]
- 🚨 답변 맨 마지막에 참고 자료나 URL 링크를 직접 작성하지 마십시오. (시스템이 자동으로 정확한 고유 링크를 100% 첨부할 것입니다.) 당신은 오직 내용을 설명하는 데에만 집중하십시오.

[답변 가이드라인]
1. 무한 반복 금지 및 간결한 구성: 동일한 주장이나 문장을 무의미하게 계속 반복(무한 루프)하지 마십시오. 각 주장은 한 번씩만 명확하고 상세하게 설명하며, Context에 제시된 내용을 모두 다루었다면 추가적인 반복 없이 자연스럽게 답변을 마무리하십시오.
2. 태도 및 어조: 창조과학계의 주장과 해석을 '소개'하는 전문적인 어조를 유지하십시오.
   - ✅ "창조과학에서는 ~라고 설명합니다", "창조과학자들은 이를 ~의 근거로 주장합니다"
   - ❌ "이것은 사실입니다", "과학적으로 증명되었습니다" 등 단정적인 사실 선언은 지양하십시오.
3. 내용의 집중: 오직 창조과학 내부의 논리와 제공된 자료의 증거(유물, 기록 등)를 상세히 설명하는 데 집중하십시오. 부정적인 의견이나 타 이론과의 비교는 지양하십시오.
4. 상세 설명 및 구체성: 각 항목 설명 시 Context에 포함된 구체적인 지명, 유물 이름, 역사적 기록, 인명 등을 반드시 포함하여 최소 3문장 이상 상세하게 작성하십시오.
5. 태그 사용 금지: <thought>, <references>, <think>, <answer> 등 어떠한 XML/HTML 태그도 포함하지 마십시오.
                                              
[출력 형식]

(창조과학적 관점에서 해당 주제를 정중하게 소개하는 도입 문구)

- **(주장/해석 1 제목)**: (상세 설명. 구체적 지명 및 증거 포함 3문장 이상)
- **(주장/해석 2 제목)**: (상세 설명. 구체적 지명 및 증거 포함 3문장 이상)

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
if query := st.chat_input("궁금한 내용을 입력하세요."):

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

            # ✅ 여기서 고유 URL 리스트를 함께 반환받습니다.
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
        
        # 파이썬 로직으로 생성된 100% 확실한 출처 리스트(하이퍼링크)를 답변 끝에 병합
        if refs_str:
            final_answer_with_refs = final_answer + f"\n\n🔗 **참고 자료:**\n{refs_str}"
        else:
            final_answer_with_refs = final_answer
            
        # 화면에 렌더링
        if final_answer_with_refs:
            answer_placeholder.markdown(final_answer_with_refs)
        else:
            answer_placeholder.markdown(final_think)

        end_time = time.time()
        status.update(label=f"✅ 답변 생성 완료! ({end_time - start_time:.2f}초)", state="complete")

        # 전체 답변 저장 (원본에 출처가 결합된 상태로 세션 기록 저장)
        st.session_state.messages.append({"role": "assistant", "content": final_answer_with_refs})