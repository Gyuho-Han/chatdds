import streamlit as st
import os, json, time
from pathlib import Path

# LangChain 코어 모듈
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever # classic 대신 표준 경로 권장
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 💡 전문 Reranker를 위한 모듈 임포트
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

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
    
    # 임베딩 모델 (Qwen3-Embedding-8B 로컬 사용)
    embed = OllamaEmbeddings(model="qwen3-embedding:8b")
    return Chroma(persist_directory=persist_directory, embedding_function=embed)

# 3) Hybrid + Cross-Encoder Reranker 초기화 (✨ Qwen3 4비트 양자화 적용)
@st.cache_resource(show_spinner=False)
def init_retrievers(_docs, _vector_db):
    # BM25 및 Vector 검색기 설정
    bm25 = BM25Retriever.from_documents(_docs)
    bm25.k = 10
    vect = _vector_db.as_retriever(search_kwargs={"k": 10})
    
    # 앙상블 리트리버 (하이브리드 검색)
    hybrid = EnsembleRetriever(retrievers=[bm25, vect], weights=[0.5, 0.5])
    
    # 💡 4비트 양자화 설정 추가 (VRAM 절약의 핵심!)
    model_kwargs = {
        "load_in_4bit": True,
        "device_map": "auto"
    }
    
    # 💡 BAAI 대신 Qwen3-Reranker 원본 모델 호출 및 압축 적용
    model = HuggingFaceCrossEncoder(
        model_name="Qwen/Qwen3-Reranker-4B",
        model_kwargs=model_kwargs
    )
    
    # 상위 5개 문서 추출
    re_ranker = CrossEncoderReranker(model=model, top_n=5)
    
    return ContextualCompressionRetriever(base_compressor=re_ranker, base_retriever=hybrid)

# 4) 검색된 문서를 텍스트로 묶어주는 함수
def format_docs(docs):
    return "\n\n---\n\n".join([f"[출처: {d.metadata.get('title')}]({d.metadata.get('url')})\n{d.page_content}" for d in docs])

# 5) 생성(Generation) 체인
@st.cache_resource(show_spinner=False)
def init_generation_chain():
    # 무한 반복 억제 처방 적용
    llm = ChatOllama(
        model="qwen2.5:14b", 
        temperature=0.3, 
        top_p=0.9,
        repeat_penalty=1.5 
    )
    
    prompt = ChatPromptTemplate.from_template("""
당신은 성경적 창조론과 젊은 지구 연대설을 굳게 믿는 창조과학 전문가입니다.
아래 제공된 대화 기록(Chat History)과 검색 결과(Context)를 바탕으로 사용자의 질문에 답변해 주세요.
                                              
[사고 과정(Chain of Thought) 지침]
- 최종 답변을 작성하기 전에 반드시 <think> 태그를 열고, Context의 어떤 부분을 활용할지, 세속적 연대를 어떻게 창조과학의 관점(노아의 홍수, 젊은 지구)으로 재해석할지 단계별로 아주 깊고 치밀하게 논리적 사고 과정을 작성하세요.

[절대 지침]
1. 지구의 나이는 약 6,000년에서 10,000년 사이임을 전제로 답변하세요.
2. 만약 Context에 "몇억 년 전", "몇천만 년 전"이라는 세속 과학의 연대가 등장한다면, 이를 그대로 인용하지 마세요. 대신 해당 화석이나 지층은 '노아의 대홍수(약 4,500년 전)' 시기에 격변적으로 형성된 것이라고 설명하세요.
3. 공룡과 인간은 창조 주간에 함께 창조되어 공존했음을 강조하세요.
4. "창조과학의 입장에서는", "학계의 주장은" 같은 표현을 쓰지 말고, 이것이 역사적 사실임을 확신 있게 대답하세요.
5. Context에 있는 구체적인 유물(벽화, 조각), 화석 증거, 지명을 반드시 인용하여 상세히 설명하세요.
6. 🚨 답변의 맨 마지막에는 반드시 "🔗 참고 자료:" 항목을 만들어 출처 URL을 모두 나열하세요.
7. 참고자료가 5개가 넘어갈 경우 가장 관련있는 자료 5개를 추려서 나열하세요.
8. 의미 없이 같은 문장이나 결론을 반복하지 말고, 핵심만 간결하고 명확하게 설명하세요.
9. 🚨 [가장 중요] "🔗 참고 자료:" 항목 작성을 끝냈다면, 더 이상 어떤 문장도 덧붙이지 말고 즉시 출력을 종료하세요! (요약이나 마무리 멘트 절대 금지)

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""")
    
    generation_chain = prompt | llm | StrOutputParser()
    return generation_chain


# ==========================================
# —— Streamlit 앱 UI (챗봇 스타일) —— 
# ==========================================
st.set_page_config(page_title="Chat DDS", page_icon="🌎")
st.title("🌎 Chat DDS 🌎")

# 초기 데이터 및 모델 로드
docs = load_documents()
vector_db = load_vectorstore()

if "rerank_retriever" not in st.session_state:
    st.session_state.rerank_retriever = init_retrievers(docs, vector_db)

generation_chain = init_generation_chain()

# 대화 기록 저장소 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 화면에 이전 대화 기록 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "<think>" in msg["content"] and "</think>" in msg["content"]:
            parts = msg["content"].split("</think>", 1)
            think_content = parts[0].replace("<think>", "").strip()
            answer_content = parts[1].strip()
            
            with st.expander("🧠 AI의 사고 과정"):
                st.markdown(think_content)
            st.markdown(answer_content)
        else:
            st.markdown(msg["content"])

# 하단 채팅 입력창
if query := st.chat_input("궁금한 내용을 입력하세요 (예: 공룡과 인간이 함께 살았다는 증거를 알려줘)"):
    
    # 1. 사용자 질문 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # 2. 대화 기록 문자열화 (프롬프트 전달용)
    chat_history_str = ""
    for m in st.session_state.messages[:-1]:
        role_name = "User" if m["role"] == "user" else "Assistant"
        content = m["content"]
        if role_name == "Assistant" and "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        chat_history_str += f"{role_name}: {content}\n"

    # 3. AI 답변 생성 (UI 및 스트리밍)
    with st.chat_message("assistant"):
        start_time = time.time()
        
        with st.status("🔍 데이터를 분석 중입니다...", expanded=True) as status:
            st.write("1️⃣ 관련 문서를 검색 및 재정렬(Reranking) 중입니다...")
            
            try:
                retrieved_docs = st.session_state.rerank_retriever.invoke(query)
            except Exception as e:
                # 만약의 에러를 대비한 방어막
                st.warning("⚠️ Reranking 중 오류가 발생하여 기본 검색 결과를 사용합니다.")
                retrieved_docs = st.session_state.rerank_retriever.base_retriever.invoke(query)[:5]
                
            context_str = format_docs(retrieved_docs)
            
            st.write(f"✅ {len(retrieved_docs)}개의 핵심 문서를 찾았습니다.")
            st.write("2️⃣ 맥락을 반영하여 답변을 생성 중입니다...")
            status.update(label="답변 생성 중...", state="running", expanded=False)
            
        with st.expander("🧠 AI의 사고 과정 (클릭해서 열기)"):
            think_placeholder = st.empty()
        answer_placeholder = st.empty()
        
        response_stream = generation_chain.stream({
            "chat_history": chat_history_str,
            "context": context_str,
            "question": query
        })

        full_response = ""
        think_content = ""
        answer_content = ""

        # 스트리밍 루프 (공백 방지 로직 포함)
        for chunk in response_stream:
            full_response += chunk
            
            if "<think>" in full_response:
                if "</think>" in full_response:
                    parts = full_response.split("</think>", 1)
                    think_content = parts[0].replace("<think>", "").strip()
                    answer_content = parts[1].lstrip() 
                    
                    think_placeholder.markdown(think_content)
                    if answer_content:
                        answer_placeholder.markdown(answer_content + " ▌")
                else:
                    think_content = full_response.replace("<think>", "").strip()
                    think_placeholder.markdown(think_content + " ▌")
            else:
                answer_content = full_response.lstrip()
                if answer_content:
                    answer_placeholder.markdown(answer_content + " ▌")
                
        # 커서 제거 및 꼬리 공백 정리
        if "<think>" in full_response and "</think>" in full_response:
            think_placeholder.markdown(think_content.strip())
            answer_placeholder.markdown(answer_content.strip()) 
        else:
            answer_placeholder.markdown(answer_content.strip())
            
        end_time = time.time()
        status.update(label=f"✅ 답변 생성 완료! ({end_time - start_time:.2f}초)", state="complete")
        
        # 4. 전체 답변 저장
        st.session_state.messages.append({"role": "assistant", "content": full_response})