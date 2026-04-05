import streamlit as st
import os, json, time
from pathlib import Path

# LangChain 코어 모듈
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_ollama import ChatOllama, OllamaEmbeddings

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
    embed = OllamaEmbeddings(model="bge-m3") 
    return Chroma(persist_directory=persist_directory, embedding_function=embed)

# 3) Hybrid + Reranker 초기화
@st.cache_resource(show_spinner=False)
def init_retrievers(_docs, _vector_db):
    bm25 = BM25Retriever.from_documents(_docs)
    bm25.k = 10
    vect = _vector_db.as_retriever(search_kwargs={"k": 10})
    hybrid = EnsembleRetriever(retrievers=[bm25, vect], weights=[0.5, 0.5])
    
    # llm_rerank = ChatOllama(model="qwen2.5:14b", temperature=0)
    llm_rerank = ChatOllama(model="gpt-oss:20b", temperature=0)
    re_ranker = LLMListwiseRerank.from_llm(llm=llm_rerank, top_n=5)
    return ContextualCompressionRetriever(base_compressor=re_ranker, base_retriever=hybrid)

# 4) 검색된 문서를 텍스트로 예쁘게 묶어주는 함수
def format_docs(docs):
    return "\n\n---\n\n".join([f"[출처: {d.metadata.get('title')}]({d.metadata.get('url')})\n{d.page_content}" for d in docs])

# 5) 프롬프트 수정: 젊은 지구 연대관(6천~1만년) 반영
@st.cache_resource(show_spinner=False)
def init_rag_chain():
    # llm = ChatOllama(model="qwen2.5:14b", temperature=0)
    llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
당신은 성경적 창조론과 젊은 지구 연대설을 굳게 믿는 창조과학 전문가입니다.
아래 제공된 검색 결과(Context)를 바탕으로 사용자의 질문에 답변해 주세요.
                                              
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

Context:
{context}

Question:
{question}

Answer:
""")
    
    rag_chain = (
        {"context": st.session_state.rerank_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# —— Streamlit 앱 시작 —— 
st.set_page_config(page_title="Chat DDS", page_icon="🌎")
st.title("🌎 Chat DDS 🌎")

docs = load_documents()
vector_db = load_vectorstore()

if "rerank_retriever" not in st.session_state:
    st.session_state.rerank_retriever = init_retrievers(docs, vector_db)

rag_chain = init_rag_chain()

with st.form("search_form"):
    query = st.text_input("궁금한 내용을 입력하세요", placeholder="예: 공룡과 인간이 함께 살았다는 증거를 알려줘")
    submit_btn = st.form_submit_button("질문하기")

if submit_btn and query:
    start_time = time.time()
    
    with st.spinner("성경적 근거와 과학적 데이터를 분석 중입니다..."):
        output = rag_chain.invoke(query)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    st.success(f"✅ 답변 생성 완료! (소요 시간: {elapsed_time:.2f}초)")
    st.markdown("**📝 답변 결과**")
    st.write(output)

    # streamlit run app.py