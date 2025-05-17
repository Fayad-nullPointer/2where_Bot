import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# === تحميل موديل DeepSeek ===
@st.cache_resource
def load_deepseek_model():
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=pipe)

# === تحميل قاعدة المعرفة FAISS ===
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("my_faiss_index", embeddings=embedding_model)
    return vectorstore

# === بناء RAG Chain ===
@st.cache_resource
def build_qa_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# === واجهة Streamlit ===
def main():
    st.set_page_config(page_title="📚 2where Chatbot", layout="centered")
    st.title("🤖 Powered By DeepSeek")

    # تحميل الموارد
    with st.spinner("🚀 Loading model and vector store..."):
        llm = load_deepseek_model()
        vectorstore = load_vectorstore()
        qa_chain = build_qa_chain(vectorstore, llm)

    # واجهة السؤال
    query = st.text_input("❓ Write your question:")

    if query:
        with st.spinner("🔍 Searching and generating response..."):
            result = qa_chain({"query": query})
            full_answer = result["result"]

            # 🔍 استخراج الجزء من "Helpful Answer" وما بعده
            keyword = "Helpful Answer"
            if keyword in full_answer:
                answer_from_keyword = full_answer.split(keyword, 1)[-1].strip()
                display_answer = f"**{keyword}**\n\n{answer_from_keyword}"
            else:
                display_answer = full_answer  # لو مش موجود نعرض كله

            st.markdown("### ✅ Answer")
            st.write(display_answer)

if __name__ == "__main__":
    main()
