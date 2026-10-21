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
    page_title="ERGIT | Education Resource-Gap Intelligence Tool",
    page_icon="📜",
    layout="centered"
)


# --------------------------------------------------
# 2. Design (concord paper / official result sheet)
# --------------------------------------------------

def html(block):
    """Render HTML safely (removes indentation so Streamlit
    does not treat it as a code block)."""
    cleaned = "\n".join(line.strip() for line in block.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css():
    html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
--paper: #F3EBD8;
--sheet: #FBF6EA;
--ink: #1F3A5F;
--green: #2E6B4F;
--gold: #B8862B;
--muted: #6B5E4B;
--red: #A23B3B;
}

/* Concord paper background with soft texture */
[data-testid="stAppViewContainer"] {
background-color: var(--paper);
background-image:
radial-gradient(rgba(120, 95, 60, 0.06) 1px, transparent 1px),
radial-gradient(rgba(120, 95, 60, 0.04) 1px, transparent 1px),
radial-gradient(ellipse at center, rgba(255,255,255,0) 55%, rgba(150,120,70,0.12) 100%);
background-size: 7px 7px, 13px 13px, 100% 100%;
background-position: 0 0, 3px 5px, 0 0;
}

/* Hide default Streamlit decoration */
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stDecoration"] {display: none;}

/* The result sheet */
.block-container {
background: var(--sheet);
border: 2px solid var(--ink);
outline: 1px solid var(--gold);
outline-offset: -8px;
border-radius: 6px;
box-shadow: 0 6px 24px rgba(80, 60, 30, 0.12);
padding: 2.8rem 2.6rem 3rem 2.6rem !important;
margin-top: 2.2rem;
margin-bottom: 6rem;
max-width: 820px;
}

html, body, [class*="css"], .stMarkdown, p, li, label {
font-family: 'Inter', sans-serif;
color: var(--ink);
}

h1, h2, h3, h4, .ergit-wordmark {
font-family: 'Playfair Display', serif !important;
color: var(--ink) !important;
}

/* Header */
.ergit-header {text-align: center; margin-bottom: 0.4rem;}
.ergit-wordmark {
font-size: 4.2rem; font-weight: 800; letter-spacing: 0.18em;
margin: 0.3rem 0 0 0; line-height: 1;
}
.ergit-sub {
font-size: 0.95rem; letter-spacing: 0.22em; text-transform: uppercase;
color: var(--muted); margin-top: 0.6rem;
}
.ergit-tagline {
font-size: 1.12rem; margin: 1.1rem auto 0 auto; max-width: 560px;
color: var(--ink);
}

/* Gold ruled divider */
.rule {
border: none; height: 1px; margin: 2rem 0 1.6rem 0;
background: linear-gradient(90deg, transparent, var(--gold), transparent);
}

.section-label {
font-family: 'Playfair Display', serif; font-size: 1.45rem; font-weight: 700;
color: var(--ink); margin-bottom: 0.7rem;
}
.section-label span {color: var(--gold);}

.lead {color: var(--muted); font-size: 1rem; line-height: 1.6;}

/* Feature cards */
.card {
background: #FFFDF7; border: 1px solid rgba(184, 134, 43, 0.45);
border-top: 3px solid var(--green); border-radius: 6px;
padding: 1.1rem 1rem; height: 100%;
}
.card-icon {font-size: 1.6rem;}
.card-title {font-weight: 600; font-size: 1.02rem; margin: 0.4rem 0 0.3rem 0;}
.card-text {color: var(--muted); font-size: 0.9rem; line-height: 1.5;}

/* Result-sheet stat boxes */
.stat {
border: 1.5px solid var(--ink); outline: 1px solid var(--gold);
outline-offset: -5px; border-radius: 4px; text-align: center;
padding: 1rem 0.5rem; background: #FFFDF7;
}
.stat-num {
font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 800;
color: var(--green); line-height: 1.1;
}
.stat-label {font-size: 0.85rem; font-weight: 600; margin-top: 0.3rem;}
.stat-note {font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem;}
.caption {font-size: 0.78rem !important; color: var(--muted) !important; text-align: center; margin-top: 0.9rem;}

/* Steps */
.step {display: flex; align-items: flex-start; gap: 0.8rem; margin-bottom: 0.8rem;}
.step-num {
min-width: 2rem; height: 2rem; border-radius: 50%;
border: 1.5px solid var(--gold); color: var(--ink);
font-family: 'Playfair Display', serif; font-weight: 700;
display: flex; align-items: center; justify-content: center;
}
.step-text {padding-top: 0.25rem;}

/* Audience chips */
.chip {
display: inline-block; padding: 0.35rem 0.8rem; margin: 0.25rem;
border: 1px solid var(--gold); border-radius: 999px;
font-size: 0.85rem; background: #FFFDF7;
}

.footer-note {font-size: 0.78rem !important; color: var(--muted) !important; text-align: center; margin-top: 2rem;}

/* Buttons */
.stButton > button {
background: var(--green); color: #FBF6EA; border: none;
border-radius: 6px; padding: 0.6rem 1.2rem; font-weight: 600;
transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
background: #275C44; color: #FFFFFF;
transform: translateY(-2px); box-shadow: 0 4px 12px rgba(46, 107, 79, 0.3);
}
.stButton > button p {color: #FBF6EA !important;}

/* Chat messages */
[data-testid="stChatMessage"] {
background: #FFFDF7; border: 1px solid rgba(184, 134, 43, 0.35);
border-left: 4px solid var(--gold); border-radius: 6px; padding: 0.9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
background: #E6EFE9; border-left: 4px solid var(--green);
}
[data-testid="stChatMessage"] table {border-collapse: collapse; width: 100%;}
[data-testid="stChatMessage"] th {
border-bottom: 2px solid var(--gold); text-align: left; padding: 0.4rem;
}
[data-testid="stChatMessage"] td {
border-bottom: 1px solid rgba(107, 94, 75, 0.2); padding: 0.4rem;
}

/* Chat input */
[data-testid="stBottom"] > div {background: transparent;}
[data-testid="stChatInput"] {
border: 1.5px solid var(--ink) !important; border-radius: 8px;
background: #FFFDF7;
}

.source-tag {
display: inline-block; font-size: 0.75rem; color: var(--muted);
border: 1px solid var(--gold); border-radius: 4px;
padding: 0.1rem 0.5rem; margin-top: 0.4rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
background: #EFE5CF; border-right: 1px solid rgba(184, 134, 43, 0.5);
}
.side-std {
border-left: 3px solid var(--gold); padding: 0.3rem 0.6rem;
margin: 0.4rem 0; background: #FBF6EA; font-size: 0.88rem;
}

/* Error box */
.ergit-error {
border: 1.5px solid var(--red); background: #F8E9E4; color: var(--red);
border-radius: 6px; padding: 1rem;
}

@media (max-width: 640px) {
.block-container {padding: 1.6rem 1.1rem 2rem 1.1rem !important;}
.ergit-wordmark {font-size: 3rem;}
}
</style>
""")


def seal(size=110):
    html(f"""
<div style="text-align:center;">
<svg width="{size}" height="{size}" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
<defs>
<path id="sealArc" d="M 60,60 m -42,0 a 42,42 0 1,1 84,0 a 42,42 0 1,1 -84,0"/>
</defs>
<circle cx="60" cy="60" r="56" fill="none" stroke="#B8862B" stroke-width="3"/>
<circle cx="60" cy="60" r="50" fill="#FBF6EA" stroke="#B8862B" stroke-width="1"/>
<circle cx="60" cy="60" r="33" fill="none" stroke="#1F3A5F" stroke-width="1"/>
<text font-family="Inter, sans-serif" font-size="8.5" letter-spacing="2.2" fill="#6B5E4B">
<textPath href="#sealArc" startOffset="2%">LAGOS STATE · EDUCATION · 2024/25 ·</textPath>
</text>
<text x="60" y="66" text-anchor="middle" font-family="Playfair Display, serif"
font-size="17" font-weight="800" fill="#1F3A5F" letter-spacing="1">ERGIT</text>
</svg>
</div>
""")


def divider():
    html('<hr class="rule">')


inject_css()


# --------------------------------------------------
# 3. Load environment variables
# --------------------------------------------------

load_dotenv()


def get_key(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


api_key = get_key("OPENAI_API_KEY")
embedding_api_key = get_key("EMBEDDING_API_KEY")


# --------------------------------------------------
# 4. Initialize models and Chroma database
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


# --------------------------------------------------
# 5. Create the prompt
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
# 6. Route questions to the appropriate source
# --------------------------------------------------

def retrieve_context(question, vector_db, general_retriever):
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

def answer_question(question, vector_db, chat_model, general_retriever):
    context, source_used = retrieve_context(
        question, vector_db, general_retriever
    )

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
# 8. Sidebar
# --------------------------------------------------

def render_sidebar():
    with st.sidebar:
        seal(90)
        html("""
<p style="text-align:center; font-family:'Playfair Display',serif;
font-size:1.3rem; font-weight:700; margin:0.3rem 0 0 0;">ERGIT</p>
<p style="text-align:center; font-size:0.82rem; color:#6B5E4B;">
Education Resource-Gap Intelligence Tool for Lagos State
</p>
<hr class="rule" style="margin:1rem 0;">
<p style="font-weight:600; margin-bottom:0.3rem;">UBE minimum standards</p>
<div class="side-std">35 students per teacher</div>
<div class="side-std">35 students per classroom</div>
<div class="side-std">30 students per toilet</div>
<hr class="rule" style="margin:1rem 0;">
<p style="font-size:0.82rem; color:#6B5E4B; line-height:1.5;">
Answers come only from the project's documents. If the data does not
cover a question, ERGIT will say so.
</p>
""")


# --------------------------------------------------
# 9. Page 1: Home (what ERGIT does)
# --------------------------------------------------

def render_home():
    seal(120)

    html("""
<div class="ergit-header">
<div class="ergit-wordmark">ERGIT</div>
<div class="ergit-sub">Education Resource-Gap Intelligence Tool</div>
<div class="ergit-tagline">
Showing where Lagos public schools need teachers, classrooms and toilets most.
</div>
</div>
""")

    divider()

    html("""
<div class="section-label">The <span>problem</span></div>
<p class="lead">
Nigeria's education data is scattered across many reports and portals.
It is hard for decision-makers to see which Local Government Areas (LGAs)
need support most, and what kind of support they need.
</p>
""")

    divider()

    html('<div class="section-label">What ERGIT <span>does</span></div>')

    col1, col2, col3 = st.columns(3)
    with col1:
        html("""
<div class="card">
<div class="card-icon">📊</div>
<div class="card-title">Ranks LGAs</div>
<div class="card-text">Compares all 20 Lagos LGAs with the national UBE standards.</div>
</div>
""")
    with col2:
        html("""
<div class="card">
<div class="card-icon">📈</div>
<div class="card-title">Forecasts shortages</div>
<div class="card-text">Machine learning predicts where teacher shortages may grow.</div>
</div>
""")
    with col3:
        html("""
<div class="card">
<div class="card-icon">💬</div>
<div class="card-title">Answers questions</div>
<div class="card-text">An AI assistant explains the results in plain English, using only verified data.</div>
</div>
""")

    divider()

    html('<div class="section-label">Key <span>findings</span></div>')

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        html("""
<div class="stat">
<div class="stat-num">20/20</div>
<div class="stat-label">LGAs lack classrooms</div>
<div class="stat-note">Standard: 35 per classroom</div>
</div>
""")
    with s2:
        html("""
<div class="stat">
<div class="stat-num">19/20</div>
<div class="stat-label">LGAs lack toilets</div>
<div class="stat-note">Standard: 30 per toilet</div>
</div>
""")
    with s3:
        html("""
<div class="stat">
<div class="stat-num">9/20</div>
<div class="stat-label">LGAs lack teachers</div>
<div class="stat-note">Standard: 35 per teacher</div>
</div>
""")
    with s4:
        html("""
<div class="stat">
<div class="stat-num">JSS</div>
<div class="stat-label">Most pressured level</div>
<div class="stat-note">Junior secondary schools</div>
</div>
""")

    html("""
<p class="caption">
Public Primary, JSS and SSS schools, 2024/25.
Sources: DNEMIS Annual School Census; Lagos Bureau of Statistics.
</p>
""")

    divider()

    html("""
<div class="section-label">How it <span>works</span></div>
<div class="step"><div class="step-num">1</div>
<div class="step-text"><b>Official data collected</b> from DNEMIS and the Lagos Bureau of Statistics.</div></div>
<div class="step"><div class="step-num">2</div>
<div class="step-text"><b>Cleaned and compared</b> with the UBE standards for teachers, classrooms and toilets.</div></div>
<div class="step"><div class="step-num">3</div>
<div class="step-text"><b>Ranked and forecast</b> to show which LGAs need support most.</div></div>
<div class="step"><div class="step-num">4</div>
<div class="step-text"><b>Explained by the AI assistant</b>, with the source of every answer.</div></div>
""")

    divider()

    html("""
<div class="section-label">Who it is <span>for</span></div>
<div>
<span class="chip">Lagos SUBEB</span>
<span class="chip">Ministry of Basic &amp; Secondary Education</span>
<span class="chip">UBEC</span>
<span class="chip">NGOs</span>
<span class="chip">Donors &amp; CSR teams</span>
<span class="chip">Researchers</span>
</div>
""")

    divider()

    left, middle, right = st.columns([1, 2, 1])
    with middle:
        if st.button("Start asking ERGIT  →", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    html("""
<p class="footer-note">
ERGIT is a student data science capstone project.
Figures come from official public sources.
</p>
""")


# --------------------------------------------------
# 10. Page 2: Chat with ERGIT
# --------------------------------------------------

SAMPLE_QUESTIONS = [
    "Which LGAs have the greatest overall resource gaps?",
    "Which LGAs need teachers most?",
    "How does Badagry compare with the toilet standard?",
    "Which school level is under the most pressure?"
]


def render_chat():
    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("←  Back to home"):
            st.session_state.page = "home"
            st.rerun()
    with top_right:
        if st.session_state.messages:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    seal(80)
    html("""
<div class="ergit-header">
<div class="ergit-wordmark" style="font-size:2.4rem;">ERGIT Assistant</div>
<div class="ergit-tagline" style="font-size:1rem;">
Ask about teachers, classrooms, toilets and resource gaps across Lagos LGAs.
</div>
</div>
""")
    divider()

    # Check keys and load the RAG system
    if not api_key or not embedding_api_key:
        html("""
<div class="ergit-error">
<b>API keys are missing.</b><br>
On Streamlit Cloud, add OPENAI_API_KEY and EMBEDDING_API_KEY under
Manage app → Settings → Secrets. When running locally, add them to your .env file.
</div>
""")
        st.stop()

    try:
        vector_db, chat_model = load_rag_components()
    except Exception as e:
        html(f'<div class="ergit-error"><b>Could not initialize the RAG system:</b> {e}</div>')
        st.stop()

    general_retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "source": "Lagos_Education_RAG_Knowledge_Base.pdf"
            }
        }
    )

    # Sample questions when the chat is empty
    if not st.session_state.messages:
        html('<p class="lead" style="text-align:center;">Try one of these questions:</p>')
        q1, q2 = st.columns(2)
        for i, sample in enumerate(SAMPLE_QUESTIONS):
            column = q1 if i % 2 == 0 else q2
            with column:
                if st.button(sample, key=f"sample_{i}", use_container_width=True):
                    st.session_state.pending_question = sample
                    st.rerun()

    # Show chat history
    for message in st.session_state.messages:
        avatar = "🧑🏾‍🎓" if message["role"] == "user" else "📜"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "source" in message:
                html(f'<span class="source-tag">Source used: {message["source"]}</span>')

    typed_question = st.chat_input(
        "Ask about education resources in Lagos State..."
    )

    question = typed_question or st.session_state.pop("pending_question", None)

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="🧑🏾‍🎓"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="📜"):
            with st.spinner("Searching the education documents..."):
                try:
                    answer, source_used = answer_question(
                        question, vector_db, chat_model, general_retriever
                    )

                    st.markdown(answer)
                    html(f'<span class="source-tag">Source used: {source_used}</span>')

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


# --------------------------------------------------
# 11. Run the app
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "messages" not in st.session_state:
    st.session_state.messages = []

render_sidebar()

if st.session_state.page == "home":
    render_home()
else:
    render_chat()