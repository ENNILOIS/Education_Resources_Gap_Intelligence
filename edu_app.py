
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Lagos Education Resource Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Lagos Education Resource Assistant")
st.write(
    "Ask questions about education resources, "
    "resource gaps, and education standards in Lagos State."
)


# --------------------------------------------------
# 2. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
embedding_api_key = os.getenv("EMBEDDING_API_KEY")

if not api_key or not embedding_api_key:
    st.error(
        "API keys are missing. Check that your .env file "
        "contains OPENAI_API_KEY and EMBEDDING_API_KEY."
    )
    st.stop()


# --------------------------------------------------
# 3. Initialize models and Chroma database
# --------------------------------------------------

@st.cache_resource
def load_rag_components():
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://qwen-embed.publicaai.com/v1",
        api_key=embedding_api_key
    )

    vector_db = Chroma(
        collection_name="lerkb",
        embedding_function=embeddings,
        persist_directory="chroma_store"
    )

    chat_model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return vector_db, chat_model


try:
    vector_db, chat_model = load_rag_components()
except Exception as e:
    st.error(f"Could not initialize the RAG system: {e}")
    st.stop()


# --------------------------------------------------
# 4. Create the prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are the Lagos Education Resource Assistant.

Answer the user's question using only the supplied context.

Instructions:
- Be clear, helpful, and concise.
- Use tables when they make comparisons easier to understand.
- Preserve LGA names, figures, rankings, and standards accurately.
- Resource-gap scores are relative scores out of 100, not percentages
  of missing resources.
- Do not invent facts, statistics, rankings, or citations.
- If the context does not contain the answer, say so clearly.
- Distinguish between a resource-gap score and an official national
  education standard.
- Do not claim that a source supports information it does not contain.

Context:
{context}

Question:
{question}

Answer:
""")


# --------------------------------------------------
# 5. Create the retriever
# --------------------------------------------------

general_retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "source": "Lagos_Education_RAG_Knowledge_Base.pdf"
        }
    }
)


# --------------------------------------------------
# 6. Route questions to the appropriate source
# --------------------------------------------------

def retrieve_context(question):
    ranking_keywords = [
        "ranking",
        "ranked",
        "greatest overall",
        "greatest resource gap",
        "greatest resource gaps",
        "largest resource gap",
        "largest resource gaps",
        "highest resource gap",
        "highest resource gaps",
        "highest resource-gap score",
        "lowest resource gap",
        "lowest resource gaps",
        "smallest resource gap",
        "smallest resource gaps",
        "top five",
        "bottom five",
        "five lgas",
        "which lgas",
        "overall resource gap",
        "overall resource gaps",
        "most lacking overall",
        "worst resource gaps",
        "resource-gap score",
        "resource gap scores"
    ]

    question_lower = question.lower()

    is_ranking_question = any(
        keyword in question_lower
        for keyword in ranking_keywords
    )

    if is_ranking_question:
        docs = vector_db.similarity_search(
            "LAGOS STATE EDUCATION RESOURCE-GAP RANKING",
            k=2,
            filter={
                "source": "Lagos_Resource_Gap_Ranking.txt"
            }
        )

        docs.sort(
            key=lambda doc: (
                0 if "LAGOS STATE EDUCATION RESOURCE-GAP RANKING"
                in doc.page_content.upper() else 1
            )
        )

        source_used = "Ranking TXT"

    else:
        docs = general_retriever.invoke(question)
        source_used = "Education PDF"

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return context, source_used


# --------------------------------------------------
# 7. Answer questions
# --------------------------------------------------

def answer_question(question):
    context, source_used = retrieve_context(question)

    if not context.strip():
        return (
            "I could not find relevant information in the "
            "available education documents.",
            source_used
        )

    chain = prompt | chat_model | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, source_used


# --------------------------------------------------
# 8. Display the chatbot
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "source" in message:
            st.caption(f"Source used: {message['source']}")

question = st.chat_input(
    "Ask about education resources in Lagos State..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the education documents..."):
            try:
                answer, source_used = answer_question(question)

                st.markdown(answer)
                st.caption(f"Source used: {source_used}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source": source_used
                })

            except Exception as e:
                st.error(
                    "Something went wrong while answering "
                    "your question. Check the terminal for details."
                )
                print(f"RAG error: {e}")
