import os
import re
import time
import difflib
from html import escape
import json

import numpy as np
import pandas as pd
import altair as alt
import pydeck as pdk
from urllib.parse import quote_plus
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ERGIT | Education Resource-Gap Intelligence Tool",
    page_icon=":material/school:",
    layout="wide"
)


# --------------------------------------------------
# 2. Design helpers
# --------------------------------------------------

def html(block):
    """Render HTML without indentation, so Streamlit does not
    treat it as a code block."""
    cleaned = "\n".join(line.strip() for line in block.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# Clean line icons used across the app (instead of emojis)
ICON_PATHS = {
    "teacher": '<circle cx="12" cy="7" r="3.5"/><path d="M5 21v-1.5A5.5 5.5 0 0 1 10.5 14h3a5.5 5.5 0 0 1 5.5 5.5V21"/>',
    "classroom": '<path d="M3 21h18"/><path d="M5 21V10l7-5 7 5v11"/><path d="M10 21v-5h4v5"/><path d="M9 11h.01M15 11h.01"/>',
    "toilet": '<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>',
    "bars": '<path d="M4 20V11"/><path d="M10 20V5"/><path d="M16 20v-6"/><path d="M21 20H3"/>',
    "pie": '<circle cx="12" cy="12" r="9"/><path d="M12 3v9h9"/>',
    "chat": '<path d="M21 12a8.5 8.5 0 0 1-12.4 7.5L3.5 21l1.4-4.7A8.5 8.5 0 1 1 21 12z"/>',
    "collect": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
    "clean": '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    "idea": '<path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6.5 6.5 0 0 0-3.8 11.8V17h7.6v-2.2A6.5 6.5 0 0 0 12 3z"/>',
    "up": '<path d="M12 19V5"/><path d="m5 12 7-7 7 7"/>',
    "down": '<path d="M12 5v14"/><path d="m5 12 7 7 7-7"/>',
    "flat": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "alert": '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "flag": '<path d="M5 21V4"/><path d="M5 4h11l-2 4 2 4H5"/>',
    "pin": '<circle cx="12" cy="10" r="2.8"/><path d="M12 21s7-6 7-11a7 7 0 0 0-14 0c0 5 7 11 7 11z"/>',
    "peak": '<path d="m3 20 6-10 4 6 3-4 5 8H3z"/>',
    "sum": '<path d="M18 5H6l6 7-6 7h12"/>',
    "star": '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1 6.2L12 17.3l-5.5 2.9 1-6.2L3 9.6l6.2-.9z"/>',
    "dot": '<circle cx="12" cy="12" r="4"/>',
    "ruler": '<path d="M3 17 17 3l4 4L7 21z"/><path d="m7.5 12.5 2 2"/><path d="m11 9 2 2"/><path d="m14.5 5.5 2 2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "spread": '<path d="M3 12h18"/><path d="m7 8-4 4 4 4"/><path d="m17 8 4 4-4 4"/>',
    "column": '<path d="M6 20v-8"/><path d="M12 20V6"/><path d="M18 20v-11"/><path d="M3 20h18"/>',
    "lollipop": '<path d="M4 6h10"/><path d="M4 12h14"/><path d="M4 18h7"/><circle cx="16" cy="6" r="2"/><circle cx="20" cy="12" r="2"/><circle cx="13" cy="18" r="2"/>',
    "line": '<path d="M3 20h18"/><path d="m4 15 5-5 4 3 7-7"/>',
    "file": '<path d="M14 3H6v18h12V7l-4-4z"/><path d="M14 3v4h4"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/>',
    "cap": '<path d="m2 9 10-5 10 5-10 5L2 9z"/><path d="M6 11v5c3 2.5 9 2.5 12 0v-5"/><path d="M22 9v6"/>',
}


def ic(name, color="currentColor", size=18):
    """Return a small line icon as inline SVG."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:-0.2em;">'
            f'{ICON_PATHS[name]}</svg>')


LOGO_SVG = """
<svg width="46" height="46" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#2563EB"/><stop offset="1" stop-color="#1E3A8A"/>
</linearGradient>
</defs>
<rect x="1" y="1" width="46" height="46" rx="13" fill="url(#lg)"/>
<rect x="14" y="13" width="4" height="9" rx="1.5" fill="#FACC15"/>
<rect x="21" y="9" width="4" height="13" rx="1.5" fill="#FACC15"/>
<rect x="28" y="15" width="4" height="7" rx="1.5" fill="#34D399"/>
<path d="M8 26 C14 23 20 23 24 27 C28 23 34 23 40 26 L40 38 C34 35 28 35 24 39 C20 35 14 35 8 38 Z" fill="#FFFFFF"/>
<path d="M24 27 L24 39" stroke="#1E3A8A" stroke-width="1.4"/>
</svg>
"""


def inject_css():
    html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
--ink: #0F1E4A;
--blue: #2563EB;
--navy: #1E3A8A;
--yellow: #FACC15;
--green: #10B981;
--coral: #F97066;
--muted: #5B6680;
--line: #DCE6F5;
}

/* Exercise-book paper: ruled blue lines */
[data-testid="stAppViewContainer"] {
background-color: #FBFCFE;
background-image: linear-gradient(var(--line) 1px, transparent 1px);
background-size: 100% 34px;
}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stDecoration"] {display: none;}
#MainMenu, footer {visibility: hidden;}

.block-container {
max-width: 1120px;
padding-top: 1.4rem !important;
padding-bottom: 6rem !important;
position: relative;
}
/* Red margin line of an exercise book */
.block-container::before {
content: ""; position: absolute; top: 0; bottom: 0; left: -14px;
width: 2px; background: rgba(249, 112, 102, 0.45);
}

html, body, p, li, label, .stMarkdown {
font-family: 'Inter', sans-serif; color: var(--ink);
}
h1, h2, h3, .jakarta {font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--ink);}

/* Top bar with logo on the right */
.topbar {
display: flex; align-items: center; justify-content: space-between;
margin-bottom: 1.2rem;
}
.pill {
display: inline-flex; align-items: center; gap: 0.45rem;
background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px;
padding: 0.4rem 0.9rem; font-size: 0.85rem; color: var(--muted);
box-shadow: 0 2px 8px rgba(15, 30, 74, 0.05);
}
.dot {width: 8px; height: 8px; border-radius: 50%; background: var(--green);
box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.15);}
.brand {display: flex; align-items: center; gap: 0.7rem; text-align: right;}
.brand-name {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 1.45rem; letter-spacing: 0.06em; color: var(--navy); line-height: 1;
}
.brand-sub {font-size: 0.7rem; color: var(--muted); letter-spacing: 0.04em; margin-top: 0.2rem;}

/* Section titles */
.eyebrow {
font-size: 0.78rem; font-weight: 600; letter-spacing: 0.14em;
text-transform: uppercase; color: var(--blue); margin-bottom: 0.35rem;
}
.section-title {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 1.9rem; color: var(--ink); margin: 0 0 0.4rem 0; line-height: 1.2;
}
.section-title .hl {
background: linear-gradient(transparent 62%, rgba(250, 204, 21, 0.55) 62%);
padding: 0 0.15rem;
}
.lead {font-size: 1.02rem; color: var(--muted); line-height: 1.65; max-width: 720px;}
.gap {height: 2.6rem;}

/* Stat cards with progress bars */
.stat {
background: #FFFFFF; border-radius: 18px; padding: 1.3rem 1.3rem 1.1rem;
border: 1px solid var(--line); box-shadow: 0 10px 30px rgba(15, 30, 74, 0.07);
height: 100%;
}
.stat-top {display: flex; justify-content: space-between; align-items: center;}
.stat-icon {
width: 40px; height: 40px; border-radius: 12px; display: flex;
align-items: center; justify-content: center; font-size: 1.25rem;
}
.stat-num {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 2.2rem; color: var(--ink); margin-top: 0.8rem; line-height: 1;
}
.stat-num small {font-size: 1rem; color: var(--muted); font-weight: 600;}
.stat-label {font-weight: 600; margin-top: 0.35rem; font-size: 0.95rem;}
.bar {height: 8px; background: #EEF2F9; border-radius: 99px; margin-top: 0.9rem; overflow: hidden;}
.bar > span {display: block; height: 100%; border-radius: 99px;}
.stat-note {font-size: 0.78rem; color: var(--muted); margin-top: 0.55rem;}

/* Feature cards */
.feature {
background: #FFFFFF; border-radius: 18px; padding: 1.5rem;
border: 1px solid var(--line); height: 100%;
transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature:hover {transform: translateY(-4px); box-shadow: 0 16px 36px rgba(37, 99, 235, 0.12);}
.feature-icon {
width: 52px; height: 52px; border-radius: 15px; display: flex;
align-items: center; justify-content: center; font-size: 1.5rem; margin-bottom: 0.9rem;
}
.feature-title {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.12rem;}
.feature-text {color: var(--muted); font-size: 0.92rem; line-height: 1.6; margin-top: 0.35rem;}

/* Steps timeline */
.steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; position: relative;}
.step {
background: #FFFFFF; border: 1px dashed #B9C9E6; border-radius: 16px;
padding: 1.1rem; position: relative;
}
.step-num {
width: 34px; height: 34px; border-radius: 50%; background: var(--navy); color: #FFFFFF;
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700;
display: flex; align-items: center; justify-content: center; margin-bottom: 0.7rem;
}
.step-title {font-weight: 600; font-size: 0.98rem;}
.step-text {color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; line-height: 1.5;}

/* Audience chips */
.chip {
display: inline-flex; align-items: center; gap: 0.4rem;
padding: 0.5rem 0.95rem; margin: 0.3rem 0.4rem 0.3rem 0;
background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px;
font-size: 0.9rem; font-weight: 500;
}

/* Call to action band */
.cta {
background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 70%, #3B82F6 100%);
border-radius: 24px; padding: 2rem 2.2rem; color: #FFFFFF;
position: relative; overflow: hidden;
}
.cta::after {
content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
border-radius: 50%; background: rgba(250, 204, 21, 0.16);
}
.cta h3, .cta p {position: relative; z-index: 1;}
.cta h3 {color: #FFFFFF !important; font-size: 1.6rem; margin: 0;}
.cta p {color: #DBE6FF !important; margin: 0.4rem 0 0 0;}
.cta p.cta-for {font-size: 0.82rem; color: #B8C8F0 !important; margin-top: 0.8rem;}

/* Buttons */
.stButton > button {
border-radius: 12px; font-weight: 600; padding: 0.65rem 1.3rem;
border: 1px solid var(--line); background: #FFFFFF; color: var(--ink);
transition: all 0.15s ease;
}
.stButton > button:hover {border-color: var(--blue); color: var(--blue); transform: translateY(-1px);}
.stButton > button[kind="primary"] {
background: var(--yellow); color: var(--ink); border: none;
box-shadow: 0 8px 20px rgba(250, 204, 21, 0.4); font-size: 1.05rem;
}
.stButton > button[kind="primary"]:hover {background: #FDE047; color: var(--ink);}
.stButton > button p {font-weight: 600;}
.stFormSubmitButton > button {border-radius: 12px; font-weight: 600; border: 1px solid var(--line); background: #FFFFFF; color: var(--ink);}
.stFormSubmitButton > button[kind="primaryFormSubmit"] {background: var(--yellow); color: var(--ink); border: none; box-shadow: 0 8px 20px rgba(250, 204, 21, 0.4);}
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {background: #FDE047; color: var(--ink);}
[data-testid="stChatInput"]:focus-within {border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;}
[data-testid="stChatInput"] > div {border-color: transparent !important;}
[data-testid="stTextInput"] input:focus {border-color: var(--blue) !important;}

.footer-note {font-size: 0.8rem !important; color: var(--muted) !important; text-align: center; margin-top: 3.2rem !important;}

/* Chat page */
.chat-hero {
background: #FFFFFF; border: 1px solid var(--line); border-radius: 20px;
padding: 1.4rem 1.6rem; box-shadow: 0 10px 30px rgba(15, 30, 74, 0.06);
}
.chat-hero h2 {margin: 0; font-size: 1.6rem;}
.chat-hero p {color: var(--muted) !important; margin: 0.3rem 0 0 0;}

[data-testid="stChatMessage"] {
background: #FFFFFF; border: 1px solid var(--line); border-radius: 20px;
padding: 1.1rem 1.2rem; box-shadow: 0 12px 32px rgba(15, 30, 74, 0.06);
margin-bottom: 0.4rem;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {font-size: 0.93rem; line-height: 1.6;}
/* User question: small blue bubble on the right */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
background: linear-gradient(120deg, #1E3A8A, #2563EB); border: none;
width: fit-content; max-width: 75%; margin-left: auto;
padding: 0.7rem 1.1rem; border-radius: 18px 18px 4px 18px; margin-top: 0.6rem;
box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p {color: #FFFFFF !important; margin: 0; line-height: 1.5;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] {margin-bottom: 0 !important;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div[aria-hidden="true"] {display: none;}
.ver-count {font-size: 0.8rem; color: var(--muted); font-weight: 600; padding: 0 0.1rem; line-height: 2.2;}
[class*="st-key-actions_"] {margin-top: -0.3rem; margin-bottom: 0.2rem;}
[class*="st-key-actions_"] button {font-size: 0.82rem !important; color: var(--muted) !important; padding: 0.15rem 0.5rem !important; min-height: 0 !important; border-radius: 8px !important;}
[class*="st-key-actions_"] button:hover {color: var(--blue) !important; background: #EEF4FF !important;}
.typing {display: inline-flex; gap: 5px; padding: 0.5rem 0.2rem;}
.typing span {width: 8px; height: 8px; border-radius: 50%; background: #93A4C3; animation: blink 1.2s infinite ease-in-out;}
.typing span:nth-child(2) {animation-delay: 0.2s;}
.typing span:nth-child(3) {animation-delay: 0.4s;}
@keyframes blink {0%, 80%, 100% {opacity: 0.25; transform: translateY(0);} 40% {opacity: 1; transform: translateY(-3px);}}
.suggest-label {font-size: 0.8rem !important; color: var(--muted) !important; margin: 0.6rem 0 0.1rem 0 !important;}
[class*="st-key-followups_"] button {border-radius: 999px !important; padding: 0.3rem 0.85rem !important; min-height: 0 !important; background: #FFFBEB !important; border: 1px solid #FDE68A !important; box-shadow: none !important; text-align: left !important;}
[class*="st-key-followups_"] button p {font-weight: 500 !important; font-size: 0.85rem !important;}
[class*="st-key-followups_"] button:hover {border-color: var(--blue) !important; color: var(--blue) !important; background: #EEF4FF !important; transform: none !important;}
.st-key-suggestions button {border-radius: 999px !important; padding: 0.3rem 0.85rem !important; min-height: 0 !important; font-size: 0.85rem !important; background: #F5F8FE !important; border: 1px solid var(--line) !important; box-shadow: none !important;}
.st-key-suggestions button p {font-weight: 500 !important; font-size: 0.85rem !important;}
.st-key-suggestions button:hover {border-color: var(--blue) !important; color: var(--blue) !important; background: #EEF4FF !important; transform: none !important;}
.st-key-history button {justify-content: flex-start !important; text-align: left !important; font-size: 0.85rem !important; padding: 0.35rem 0.6rem !important; min-height: 0 !important; border-radius: 10px !important; box-shadow: none !important;}
.st-key-history button p {font-weight: 500 !important; font-size: 0.85rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.st-key-history button[kind="tertiary"] {color: var(--ink) !important; border: none !important; background: transparent !important;}
.st-key-history {gap: 0.15rem !important;}
.st-key-history button > div {justify-content: flex-start !important; width: 100%; min-width: 0;}
.st-key-history button > div > span {justify-content: flex-start !important; min-width: 0; overflow: hidden;}
.st-key-history [data-testid="stMarkdownContainer"] {text-align: left !important; min-width: 0; overflow: hidden;}
.st-key-history button[kind="tertiary"]:hover {background: #F5F8FE !important; color: var(--blue) !important;}
.st-key-history button[kind="primary"] {background: #EEF4FF !important; color: var(--navy) !important; border: 1px solid #C9DAFB !important; font-size: 0.85rem !important;}
.understood {font-size: 0.78rem !important; color: var(--muted) !important; margin: 0 0 0.5rem 0 !important; font-style: italic;}
.map-key {font-size: 0.75rem !important; color: var(--muted) !important; margin: 0.3rem 0 0 0 !important;}
.map-key span {font-size: 0.95rem;}
.school-note {font-size: 0.75rem !important; color: var(--muted) !important; margin-top: 0.4rem !important;}
[data-testid="stChatInputSubmitButton"]:not(:disabled) {background: var(--blue) !important; color: #FFFFFF !important;}
.st-key-dictation {position: absolute; height: 0 !important; overflow: hidden; opacity: 0; pointer-events: none;}
.way {background: #FFFFFF; border: 1px solid var(--line); border-radius: 20px; padding: 1.4rem 1.5rem; height: 100%;
box-shadow: 0 10px 30px rgba(15, 30, 74, 0.06); border-top: 4px solid var(--blue);}
.way.way-parent {border-top-color: var(--yellow);}
.way-top {display: flex; gap: 0.9rem; align-items: center; margin-bottom: 0.8rem;}
.way-top .feature-icon {margin-bottom: 0;}
.way-who {font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);}
.way-title {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.25rem; color: var(--ink);}
.way-text {color: var(--muted); font-size: 0.93rem; line-height: 1.6; margin-bottom: 0.8rem;}
.way-ask {background: #F5F8FE; border: 1px solid var(--line); border-radius: 12px; padding: 0.45rem 0.8rem;
font-size: 0.86rem; color: var(--ink); margin-top: 0.45rem;}
.way-data {font-size: 0.75rem; color: var(--muted); margin-top: 0.9rem;}
.dir-row {display: flex; align-items: center; gap: 0.5rem; padding: 0.45rem 0; border-bottom: 1px solid #EEF2F9;}
.dir-row:last-of-type {border-bottom: none;}
.dir-name {flex: 1; font-size: 0.85rem; font-weight: 600; color: var(--ink); line-height: 1.3; min-width: 0;}
.dir-sub {display: block; font-weight: 400; font-size: 0.74rem; color: var(--muted);}
.dir-go {font-size: 0.78rem; font-weight: 600; color: var(--blue) !important; text-decoration: none !important;
white-space: nowrap; background: #EEF4FF; border-radius: 999px; padding: 0.25rem 0.6rem;}
.dir-go:hover {background: #DBEAFE;}
.dir-note {font-size: 0.72rem !important; color: var(--muted) !important; margin: 0.5rem 0 0 0 !important;}
.map-title {font-size: 0.85rem !important; color: var(--ink) !important; margin: 1rem 0 0.4rem 0 !important;}
[data-testid="stBaseLinkButton-primary"] {background: var(--blue) !important; border: none !important; color: #FFFFFF !important; border-radius: 12px !important;}
[data-testid="stBaseLinkButton-primary"]:hover {background: var(--navy) !important;}
.map-pick {background: #F5F8FE; border: 1px solid var(--line); border-radius: 14px; padding: 0.7rem 0.9rem; margin: 0.6rem 0 0.5rem 0;}
.map-pick-name {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: var(--ink);}
.map-pick-sub {font-size: 0.8rem; color: var(--muted);}
.heard {font-size: 0.75rem !important; color: #C9DAFB !important; margin-top: 0.2rem !important;}

/* Answer dashboard */
.dash-head {display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 0.8rem;}
.dash-eyebrow {font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--blue);}
.dash-headline {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--ink); margin-top: 0.25rem; line-height: 1.35;}
.dash-head .source-tag {margin-top: 0; white-space: nowrap;}
.chart-chip {display: inline-block; font-size: 0.78rem; font-weight: 600; color: var(--navy); background: #EEF4FF; border: 1px solid #C9DAFB; border-radius: 999px; padding: 0.25rem 0.75rem; margin-bottom: 0.8rem;}
.chart-chip span {color: var(--muted); font-weight: 500;}
.chg {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; border-radius: 8px; padding: 0.1rem 0.5rem; font-size: 0.85rem;}
.chg.up {background: #FEE2E2; color: #B42318;}
.chg.down {background: #D1FAE5; color: #047857;}
.lb-note {font-size: 0.76rem; color: var(--muted); margin: 0.15rem 0 0.2rem 0;}
.swatch {width: 12px; height: 12px; border-radius: 4px; display: inline-block;}
.dash-divider {height: 1px; background: var(--line); margin: 0.9rem 0 1rem 0;}
.kpi-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.7rem; margin-bottom: 0.9rem;}
.kpi {display: flex; gap: 0.65rem; align-items: center; background: #F8FAFF; border: 1px solid var(--line); border-radius: 14px; padding: 0.7rem 0.8rem;}
.kpi-icon {width: 36px; height: 36px; min-width: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1rem;}
.kpi-label {font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600;}
.kpi-value {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.25rem; color: var(--ink); line-height: 1.2;}
.kpi-sub {font-size: 0.76rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 130px;}
.lb {background: #F8FAFF; border: 1px solid var(--line); border-radius: 14px; padding: 0.9rem 1rem;}
.lb-title {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 600; margin-bottom: 0.6rem;}
.lb-row {display: flex; align-items: center; gap: 0.55rem; font-size: 0.88rem; margin-top: 0.55rem;}
.lb-rank {width: 22px; height: 22px; border-radius: 50%; background: var(--navy); color: #FFFFFF; font-size: 0.72rem; font-weight: 700; display: flex; align-items: center; justify-content: center;}
.lb-name {flex: 1; font-weight: 600; color: var(--ink);}
.lb-val {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: var(--navy);}
.lb-bar {height: 6px; background: #E6ECF7; border-radius: 99px; margin: 0.3rem 0 0 1.9rem; overflow: hidden;}
.lb-bar > span {display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #FACC15, #F97066);}
[data-testid="stChatMessage"] [data-testid="stExpander"] {border: 1px solid var(--line); border-radius: 12px; margin-top: 0.6rem;}
[data-testid="stChatMessage"] table {border-collapse: collapse; width: 100%;}
[data-testid="stChatMessage"] th {
background: #EEF4FF; text-align: left; padding: 0.45rem; border-bottom: 2px solid var(--blue);
}
[data-testid="stChatMessage"] td {border-bottom: 1px solid var(--line); padding: 0.45rem;}

[data-testid="stBottom"] > div {background: transparent;}
[data-testid="stChatInput"] {
border: 2px solid var(--blue) !important; border-radius: 16px; background: #FFFFFF;
box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
}

.source-tag {
display: inline-block; font-size: 0.75rem; font-weight: 600; color: var(--navy);
background: #FEF9C3; border-radius: 6px; padding: 0.15rem 0.55rem; margin-top: 0.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid var(--line);}
.side-std {
display: flex; align-items: center; gap: 0.6rem; background: #F5F8FE;
border-radius: 12px; padding: 0.6rem 0.75rem; margin: 0.45rem 0; font-size: 0.88rem;
}
.side-std b {font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.1rem; color: var(--blue);}

.ergit-error {
border: 1px solid #FBC4BE; background: #FEF1EF; color: #9F2D22;
border-radius: 14px; padding: 1rem 1.1rem;
}

@media (max-width: 760px) {
.steps {grid-template-columns: 1fr 1fr;}
.section-title {font-size: 1.5rem;}
.block-container::before {display: none;}
}

/* Phones (iPhone and Android) */
@media (max-width: 640px) {
.block-container {padding-left: 0.9rem !important; padding-right: 0.9rem !important; padding-top: 0.8rem !important;}
.brand-sub {display: none;}
.brand-name {font-size: 1.2rem;}
.pill {display: none;}          /* too long for a phone; the logo is enough */
.topbar {justify-content: flex-end; margin-bottom: 0.8rem;}
.steps {grid-template-columns: 1fr;}
.section-title {font-size: 1.35rem;}
.lead {font-size: 0.95rem;}
.stat-num {font-size: 1.8rem;}
.cta {padding: 1.4rem 1.3rem;}
.cta h3 {font-size: 1.3rem;}
[data-testid="stChatMessage"] {padding: 0.8rem 0.85rem; border-radius: 16px;}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {font-size: 0.9rem;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {max-width: 88%;}
.dash-head {flex-direction: column; gap: 0.4rem;}
.dash-headline {font-size: 1.02rem;}
.kpi-row {grid-template-columns: 1fr 1fr; gap: 0.5rem;}
.kpi {padding: 0.55rem 0.6rem; gap: 0.45rem; align-items: flex-start;}
.kpi-icon {width: 28px; height: 28px; min-width: 28px; border-radius: 8px;}
.kpi-icon svg {width: 15px; height: 15px;}
.kpi-label {font-size: 0.62rem;}
.kpi-value {font-size: 1.05rem;}
.kpi-sub {white-space: normal; max-width: none; font-size: 0.7rem;}
.lb {padding: 0.75rem 0.8rem;}
.chart-chip {font-size: 0.72rem;}
.st-key-suggestions button {font-size: 0.8rem !important; padding: 0.3rem 0.7rem !important;}
[class*="st-key-followups_"] button p {font-size: 0.8rem !important;}
[class*="st-key-followups_"] button, .st-key-suggestions button {height: auto !important; text-align: left !important;}
[class*="st-key-followups_"] button p, .st-key-suggestions button p {white-space: normal !important; text-align: left !important; line-height: 1.35 !important;}
/* Number cards: two per row on phones instead of one long column */
[data-testid="stHorizontalBlock"]:has(.stat) {flex-wrap: wrap !important; gap: 0.5rem !important;}
[data-testid="stHorizontalBlock"]:has(.stat) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 0.5rem) !important; min-width: calc(50% - 0.5rem) !important; width: calc(50% - 0.5rem) !important;}
.stat {padding: 0.85rem 0.85rem 0.75rem;}
.stat-icon {width: 32px; height: 32px; border-radius: 9px;}
.stat-num {font-size: 1.45rem; margin-top: 0.5rem;}
.stat-label {font-size: 0.82rem;}
.stat-note {font-size: 0.7rem;}
.way {padding: 1.1rem 1.1rem;}
.way-title {font-size: 1.1rem;}
.way-ask {font-size: 0.82rem;}
[class*="st-key-actions_"] button {font-size: 0.78rem !important;}
}
</style>
""")


def topbar():
    html(f"""
<div class="topbar">
<div class="pill"><span class="dot"></span> Lagos State · 20 LGAs · Public schools for parents</div>
<div class="brand">
<div>
<div class="brand-name">ERGIT</div>
<div class="brand-sub">Education Resource-Gap Intelligence Tool</div>
</div>
{LOGO_SVG}
</div>
</div>
""")


inject_css()


# --------------------------------------------------
# 3. Load environment variables
# --------------------------------------------------

load_dotenv(override=True)


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

# The keys are passed in, so the app rebuilds the models
# automatically whenever a key changes.
@st.cache_resource
def load_rag_components(api_key, embedding_api_key):
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://qwen-embed.publicaai.com/v1",
        api_key=embedding_api_key,
        check_embedding_ctx_length=False
    )

    vector_db = Chroma(
        collection_name="lerkb",
        embedding_function=embeddings,
        persist_directory=os.path.join(BASE_DIR, "chroma_store")
    )

    chat_model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=4096          # room for the full answer, so it is never cut off
    )

    return vector_db, chat_model


@st.cache_resource
def get_fast_model(api_key):
    """A small, very fast model for quick jobs (fixing voice questions,
    building dashboard data), so the user waits less."""
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=0,
        timeout=15
    )


def run_quick(prompt_template, values, chat_model):
    """Try the fast model first; fall back to the main model if it fails."""
    for model in (get_fast_model(api_key), chat_model):
        try:
            return (prompt_template | model | StrOutputParser()).invoke(values)
        except Exception as e:
            print(f"Quick model error: {type(e).__name__}: {e}")
    return ""


# --------------------------------------------------
# 5. Create the prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are ERGIT, a friendly assistant for Lagos State public school data.

Answer the user's question using only the supplied context.

The user's name: {name}

Tone:
- Be warm, friendly and professional, like a helpful teacher. Use plain
  English and explain terms simply. Do not sound stiff or robotic.
- You may use the user's name now and then, but not in every answer.
- Answer the question directly with the figures in the context. Do not
  ask the user to choose or clarify when you can already answer; if they
  say "yes" or "do that" to your earlier suggestion, do it right away.
- The context contains full rankings of all 20 LGAs ("Ranking of LGAs by
  ..."), state summaries and averages by school level: use them. The
  resource-gap ranking text lists all 20 LGAs with scores out of 100; use
  it when the user asks for the full overall ranking.
- If the user mentions a goal such as empowering, supporting or funding,
  answer with the data that serves it (which LGAs lack the most, by how
  much, and in which resource). Do not write general policy advice, and
  never invent thresholds, budgets or percentages that are not in the
  context.
- End with one short, friendly suggestion of what they could ask next.
- If the user only greets you, says thanks or chats, reply warmly in one
  or two sentences and suggest a question they could ask.

Understanding the question:
- The question may have spelling mistakes, or words misheard by voice
  typing (for example "world of times" may mean "over time", "bad agree"
  may mean "Badagry"). Work out the most likely meaning from the context
  and answer that, instead of saying the information is not available.
  If you had to guess, start with "I think you're asking about ...".
- For "top N", "least N", "bottom N" or "all LGAs" questions, use the
  ranking lists in the context ("Ranking of LGAs by ..., from most lacking
  to least lacking"). Top = the start of the list; least or bottom = the
  end of the list. Give exactly the number of LGAs asked for, with their
  figures, and put all of them in the dashboard block.
- When the question is about one LGA, give the figure for all public
  schools combined (Primary, JSS and SSS) first, then each school level
  if it is in the context, and say which year the figures are for.

Instructions:
- Be clear, helpful, and concise.
- Use tables when they make comparisons easier to understand.
- Preserve LGA names, figures, rankings, and standards accurately.
- Resource-gap scores are relative scores out of 100, not percentages
  of missing resources.
- Do not invent facts, statistics, rankings, or citations.
- If the context really does not contain the answer, say so kindly and
  suggest a related question that the data can answer.
- Distinguish between a resource-gap score and an official national
  education standard.
- Do not claim that a source supports information it does not contain.

Dashboard instructions:
- Keep the written answer short: 2 to 4 sentences, then a table only if
  it really helps. If the user asks for a summary, give a short summary.
- If your answer contains numbers that can be compared, add ONE
  dashboard block at the very end, in exactly this format:

<chart>
{{"type": "bar", "title": "Short chart title", "headline": "One-sentence key finding", "label_name": "LGA", "value_name": "What the number means", "standard": null, "data": [{{"label": "Badagry", "value": 62.6}}, {{"label": "Ojo", "value": 55.1}}]}}
</chart>

- Choose the "type" that best fits the question:
  * "bar": ranking many LGAs (7 or more items).
  * "lollipop": a short ranking of scores, such as a top five.
  * "column": comparing 2 to 6 items, such as school levels or a few LGAs.
  * "pie": shares of a total with 6 parts or fewer, such as learners by
    school level or male and female learners.
  * "line": change over years.
  * "compare": one LGA or school level against the UBE standards for
    teachers, classrooms and toilets. Give each item its own standard,
    for example {{"label": "Students per toilet", "value": 62.6, "standard": 30}}.
- If the user asks for a chart type (for example a pie chart or a line
  chart), use that type.
- "standard": the UBE standard the values are compared with (35 for
  students per teacher, 35 for students per classroom, 30 for students
  per toilet). For scores given as "times the standards", use 1. Use null
  for anything else (learner counts, percentages).
- Use only numbers that appear in the context. Values must be plain
  numbers, with no units or % signs.
- Put only ONE kind of number in a chart: never mix a resource-gap score
  with students per teacher, or learners with ratios.
- If the answer has no numbers to compare, do not add a dashboard block.
- Never mention the dashboard block in the written answer.

Conversation so far (use it for follow-up questions such as "the first
LGA you mentioned" or "what about toilets?"):
{history}

Context:
{context}

Question:
{question}

Answer:
""")


# --------------------------------------------------
# 6. Route questions to the appropriate source
# --------------------------------------------------

# Words a user may type -> the LGA name as written in the PDF
LGA_NAMES = {
    "agege": "Agege", "ajeromi": "Ajeromi", "ifelodun": "Ajeromi",
    "alimosho": "Alimosho", "amuwo": "Amuwo", "apapa": "Apapa",
    "badagry": "Badagry", "epe": "Epe", "eti-osa": "Eti-Osa",
    "eti osa": "Eti-Osa", "ibeju": "Ibeju", "lekki": "Ibeju",
    "ifako": "Ifako", "ikeja": "Ikeja", "ikorodu": "Ikorodu",
    "kosofe": "Kosofe", "lagos island": "Lagos Island",
    "lagos mainland": "Lagos Mainland", "mushin": "Mushin", "ojo": "Ojo",
    "oshodi": "Oshodi", "isolo": "Oshodi", "shomolu": "Shomolu",
    "somolu": "Shomolu", "surulere": "Surulere"
}


# Questions that ask for a list or ranking (top 10, least 5, all LGAs ...)
RANKING_PATTERN = re.compile(
    r"\b(top|bottom|least|lowest|highest|most|rank\w*|list|all|worst|best|"
    r"order|lacking|shortages?|greatest|largest|smallest|fewest|biggest|"
    r"which lgas?|\d+\s+lgas?)\b"
)

# Questions about the overall resource-gap score (all three resources together)
OVERALL_WORDS = [
    "overall", "resource gap", "resource-gap", "gap score", "biggest gap",
    "greatest gap", "most lacking", "least lacking", "lacking the most",
    "lack the most", "lacking most", "lacking resources", "lack resources",
    "need the most", "needs the most", "most in need", "empower", "support",
    "fund", "resources"
]
RESOURCE_WORDS = ["teacher", "classroom", "toilet", "learner", "student", "enrol"]

# Which documents are in the knowledge base. Filled in when the app starts,
# from the database itself, so moving or renaming the files does not break it.
SOURCE_FILTERS = {"pdf": None, "txt": None}


NAME_KEYS = ["source", "file_path", "filename", "file_name", "path", "title"]


def make_filter(key, values):
    values = sorted(values)
    if not key or not values:
        return None            # no label to filter on: search the whole knowledge base
    if len(values) == 1:
        return {key: values[0]}
    return {key: {"$in": values}}


@st.cache_data(show_spinner=False)
def read_sources(_vector_db, pieces):
    """Every document name stored in the knowledge base, per label key."""
    data = _vector_db._collection.get(include=["metadatas"])
    found = {}
    for meta in data["metadatas"]:
        for key in NAME_KEYS:
            value = (meta or {}).get(key)
            if isinstance(value, str) and value:
                found.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in found.items()}


def set_source_filters(vector_db, pieces):
    """Work out how the documents are labelled, whatever the notebook used.
    If they carry no label at all, ERGIT simply searches everything."""
    found = read_sources(vector_db, pieces)
    pdf_key, pdfs, txt_key, txts = None, [], None, []
    for key in NAME_KEYS:
        values = found.get(key, [])
        if not pdfs:
            pdfs = [v for v in values if v.lower().endswith(".pdf")]
            pdf_key = key if pdfs else None
        if not txts:
            txts = [v for v in values if "ranking" in v.lower()]
            txt_key = key if txts else None
    SOURCE_FILTERS["pdf"] = make_filter(pdf_key, pdfs)
    SOURCE_FILTERS["txt"] = make_filter(txt_key, txts)
    return found, pdfs, txts


def squash(text):
    """Lower-case and join broken lines, so PDF line breaks don't hide words."""
    return re.sub(r"\s+", " ", text).lower()


@st.cache_resource(show_spinner=False)
def load_all_pieces(_vector_db, pieces):
    """The knowledge base is small, so keep every piece in memory once.
    Exact-word search on this is fast and is not fooled by line breaks."""
    data = _vector_db._collection.get(include=["documents", "metadatas"])
    out = []
    for order, (text, meta) in enumerate(zip(data["documents"], data["metadatas"])):
        if text:
            out.append({"doc": Document(page_content=text, metadata=meta or {}),
                        "flat": squash(text), "order": order})
    return out


def search_containing(vector_db, query, must_contain, k=4, source_filter="pdf"):
    """Pieces that contain a phrase (ignoring line breaks and capitals),
    the most relevant first."""
    try:
        pieces = load_all_pieces(vector_db, vector_db._collection.count())
    except Exception:
        return []
    phrase = squash(must_contain)
    hits = [p for p in pieces if phrase in p["flat"]]
    if source_filter == "txt":
        hits = [p for p in hits if "ranking" in str(p["doc"].metadata).lower()] or hits

    def score(p):
        s = p["flat"].count(phrase)
        if f"summary for {phrase}" in p["flat"] or f"{phrase} lga" in p["flat"]:
            s += 10
        if "ranking of lgas" in p["flat"] or "resource-gap ranking" in p["flat"]:
            s += 5
        return (-s, p["order"])

    return [p["doc"] for p in sorted(hits, key=score)[:k]]


def retrieve_context(question, vector_db, general_retriever):
    question_lower = question.lower()
    docs, seen, sources = [], set(), []

    def add(new_docs, source_name):
        added = False
        for doc in new_docs:
            if doc.page_content not in seen:
                docs.append(doc)
                seen.add(doc.page_content)
                added = True
        if added and source_name not in sources:
            sources.append(source_name)

    is_ranking = bool(RANKING_PATTERN.search(question_lower))
    # "Which LGAs lack the most?" with no particular resource = the overall ranking
    is_overall = any(w in question_lower for w in OVERALL_WORDS) or (
        is_ranking and not any(w in question_lower for w in RESOURCE_WORDS))

    # 1. Overall resource-gap ranking (the ranking text file + PDF summary)
    if is_overall:
        ranking_docs = search_containing(
            vector_db, "LAGOS STATE EDUCATION RESOURCE-GAP RANKING",
            "RESOURCE-GAP RANKING", k=2, source_filter="txt"
        )
        ranking_docs.sort(key=lambda d: 0 if "LAGOS STATE EDUCATION RESOURCE-GAP RANKING"
                          in d.page_content.upper() else 1)
        add(ranking_docs, "Ranking TXT")
        add(search_containing(vector_db, "most and least lacking LGAs overall",
                              "lacking LGAs overall", k=2), "Education PDF")

    # 2. Full rankings of all 20 LGAs (teachers, classrooms, toilets, learners)
    if is_ranking or is_overall:
        add(search_containing(vector_db, "ranking of LGAs from most lacking to least lacking",
                              "Ranking of LGAs by", k=6), "Education PDF")
        add(search_containing(vector_db, "how many LGAs are lacking teachers classrooms toilets",
                              "Out of 20 LGAs", k=1), "Education PDF")

        # A long ranking list can be split over two pieces of the PDF.
        # The list items look like "(57.7 students per teacher);", so
        # this phrase finds every piece of the list.
        wanted = [w for w in ["teacher", "classroom", "toilet"] if w in question_lower]
        for word in wanted or ["teacher", "classroom", "toilet"]:
            add(search_containing(vector_db, f"ranking of LGAs by students per {word}",
                                  f"students per {word});", k=3), "Education PDF")

    # 3. A named LGA: the pieces that contain that LGA's name
    for typed, name in LGA_NAMES.items():
        if re.search(rf"\b{re.escape(typed)}\b", question_lower):
            add(search_containing(vector_db, f"{name} LGA students per teacher classroom toilet",
                                  name), "Education PDF")

    # 4. School levels
    if any(w in question_lower for w in ["level", "primary", "jss", "sss", "junior", "senior",
                                         "average", "lagos state", "whole state", "statewide",
                                         "all of lagos", "across lagos"]):
        add(search_containing(vector_db, "how many LGAs are lacking teachers classrooms toilets",
                              "Out of 20 LGAs", k=1), "Education PDF")
        add(search_containing(vector_db, "school level average students per teacher classroom toilet",
                              "across Lagos averaged"), "Education PDF")

    # 5. Normal meaning-based search, for everything else
    add(general_retriever.invoke(question), "Education PDF")

    source_used = " + ".join(sources) if sources else "Education PDF"

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return context, source_used


# --------------------------------------------------
# 6b. Live dashboard from the answer
# --------------------------------------------------

CHART_TYPES = {
    "bar": ("bars", "Ranking bar chart"),
    "column": ("column", "Column chart"),
    "lollipop": ("lollipop", "Dot ranking"),
    "pie": ("pie", "Donut chart"),
    "line": ("line", "Trend line"),
    "compare": ("target", "Actual vs UBE standard"),
}

# Words a user may type to ask for a chart type
REQUESTED_TYPES = [
    ("pie", "pie"), ("donut", "pie"), ("doughnut", "pie"),
    ("line", "line"), ("trend", "line"),
    ("column", "column"), ("vertical", "column"),
    ("lollipop", "lollipop"), ("dot", "lollipop"), ("dots", "lollipop"),
    ("bar", "bar"),
]

PALETTE = ["#2563EB", "#FACC15", "#10B981", "#F97066",
           "#8B5CF6", "#0EA5E9", "#F59E0B", "#1E3A8A"]


def to_number(value):
    try:
        number = float(value)
        return number if number else None
    except (TypeError, ValueError):
        return None


def split_chart(answer):
    """Separate the written answer from the dashboard data."""
    match = re.search(r"<chart>(.*?)</chart>", answer, re.S)
    if not match:
        return answer.strip(), None

    text = (answer[:match.start()] + answer[match.end():]).strip()
    return text, parse_chart_json(match.group(1))


def parse_chart_json(raw):
    """Read the dashboard data, even if the AI wrapped it in extra text."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    raw = re.sub(r",\s*([}\]])", r"\1", raw[start:end + 1])  # remove trailing commas

    try:
        chart = json.loads(raw)
        rows = []
        for d in chart.get("data", []):
            row = {"label": str(d["label"]), "value": float(d["value"])}
            item_standard = to_number(d.get("standard"))
            if item_standard:
                row["standard"] = item_standard
            rows.append(row)
        if not rows:
            return None
        chart["data"] = rows
        chart["standard"] = to_number(chart.get("standard"))
        chart["type"] = str(chart.get("type", "bar")).lower()
        if chart["type"] not in CHART_TYPES:
            chart["type"] = "bar"
        return chart
    except Exception:
        return None


# Used when the main answer comes back empty
simple_prompt = ChatPromptTemplate.from_template("""
You are ERGIT, a friendly assistant for Lagos State public school data.
The user's name: {name}
Using only the context, answer the question in 3 to 6 friendly, clear
sentences with the key figures and years. If the question is unclear, answer
the most likely meaning. Do not add any chart or code.

Conversation so far:
{history}

Context:
{context}

Question:
{question}
""")


# Used when the main answer has figures but no dashboard block
chart_prompt = ChatPromptTemplate.from_template("""
Turn the figures in this answer into dashboard data.
Reply with ONE JSON object only, no other words, in this format:

{{"type": "bar", "title": "Short chart title", "headline": "One-sentence key finding", "label_name": "LGA", "value_name": "What the number means", "standard": null, "data": [{{"label": "Badagry", "value": 62.6}}, {{"label": "Ojo", "value": 55.1}}]}}

Choose "type": "bar" (ranking 7 or more items), "lollipop" (short ranking),
"column" (2 to 6 items such as school levels), "pie" (shares of a total,
6 parts or fewer), "line" (change over years), or "compare" (one LGA or
school level against the UBE standards; give each item its own
"standard", e.g. {{"label": "Students per toilet", "value": 54.1, "standard": 30}}).
"standard" is 35 for students per teacher, 35 for students per classroom,
30 for students per toilet, or null. Use only numbers from the answer.
Put only ONE kind of number in the chart (never a score next to ratios).
If the answer gives one figure and its standard, use "compare" with one item.
If the answer has no figures that can be compared, reply: {{}}

Question: {question}

Answer: {answer}
""")


def make_chart(question, answer_text, chat_model):
    if len(re.findall(r"\d+(?:\.\d+)?", answer_text)) < 2:
        return None
    try:
        raw = run_quick(chart_prompt, {"question": question, "answer": answer_text}, chat_model)
        return parse_chart_json(raw)
    except Exception as e:
        print(f"Chart error: {type(e).__name__}: {e}")
        return None


def chart_is_grounded(chart, source_text):
    """Keep a chart only if its numbers really appear in the documents or answer."""
    found = set()
    for n in re.findall(r"\d[\d,]*\.?\d*", source_text):
        try:
            found.add(round(float(n.replace(",", "")), 1))
        except ValueError:
            pass
    values = [round(float(d["value"]), 1) for d in chart.get("data", [])]
    if not values:
        return False
    hits = sum(1 for v in values if v in found or round(v) in found)
    return hits / len(values) >= 0.6


def add_standard_line(chart):
    """Always show the UBE standard line when the values are compared with it."""
    # Never mix units in one chart (e.g. a score of 100 next to 62.6 students per
    # toilet). Keep the per-teacher / classroom / toilet bars and compare each
    # with its own UBE standard.
    def ratio_standard(label):
        t = str(label).lower()
        if not re.search(r"\bper\b|ratio", t):
            return None
        if "teacher" in t and "female" not in t:
            return 35.0
        if "classroom" in t:
            return 35.0
        if "toilet" in t:
            return 30.0
        return None
    data = chart.get("data", [])
    ratio_items = [d for d in data if ratio_standard(d.get("label")) and d.get("value", 0) > 5]
    if len(ratio_items) >= 2 or (ratio_items and len(ratio_items) < len(data)):
        for d in ratio_items:
            d["standard"] = ratio_standard(d["label"])
        chart["data"] = ratio_items
        chart["type"] = "compare"
        chart["value_name"] = "Students per resource"
        chart.pop("standard", None)
        return chart
    words = (str(chart.get("value_name", "")) + " " + str(chart.get("title", ""))).lower()
    # A score out of 100, a count of learners or a percentage has no UBE line.
    # The AI sometimes copies the example's "standard": 30, so remove it here.
    labels = " ".join(str(d.get("label", "")) for d in chart.get("data", [])).lower()
    is_ratio = re.search(r"\bper (teacher|classroom|toilet)|ratio|times", words + " " + labels)
    if not is_ratio or re.search(r"score|index|out of 100|0-100|percent|%", words):
        chart.pop("standard", None)
        for d in chart.get("data", []):
            d.pop("standard", None)
        return chart
    if chart.get("standard"):
        return chart
    values = [d["value"] for d in chart["data"]]
    if ("times" in words or "standard" in words or "gap score" in words) and values \
            and 0.2 <= min(values) and max(values) <= 5:
        chart["standard"] = 1.0                      # 1.0 = exactly at the standards
    elif "per teacher" in words or "per classroom" in words:
        chart["standard"] = 35.0
    elif "per toilet" in words:
        chart["standard"] = 30.0
    return chart


def choose_chart_type(question, chart):
    """Respect the user's choice of chart; otherwise keep the AI's choice,
    with a few safety checks."""
    chart = add_standard_line(chart)
    question_lower = question.lower()
    for word, kind in REQUESTED_TYPES:
        if len(chart["data"]) > 1 and re.search(rf"\b{word}\b", question_lower):
            chart["type"] = kind
            chart["requested"] = True
            return chart

    n = len(chart["data"])
    if n == 1:
        # One figure: only worth a chart when it can be compared with a standard
        if not (chart["data"][0].get("standard") or chart.get("standard")):
            return None
        chart["type"] = "compare"
    labels = " ".join(d["label"] for d in chart["data"])
    looks_like_years = len(re.findall(r"(19|20)\d\d", labels)) >= n

    if looks_like_years and chart["type"] not in ("line", "column"):
        chart["type"] = "line"
    if chart["type"] == "pie" and n > 6:
        chart["type"] = "bar"
    if chart["type"] == "compare":
        for d in chart["data"]:
            if "standard" not in d and chart.get("standard"):
                d["standard"] = chart["standard"]
        if not all("standard" in d for d in chart["data"]):
            chart["type"] = "column"
    return chart


def fmt(value):
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def tooltip_for(chart):
    return [alt.Tooltip("label:N", title=chart.get("label_name", "Item")),
            alt.Tooltip("value:Q", title=chart.get("value_name", "Value"), format=",.2f")]


def status_colour(df, standard, scale=None):
    # A school chart gives each bar its own status (from the school's label)
    by_value = df["value"].apply(
        lambda v: "Above standard" if v > standard else "Within standard"
    )
    df["status"] = df["status"].fillna(by_value) if "status" in df else by_value
    domain, colours = scale or (["Above standard", "Within standard"], ["#F97066", "#10B981"])
    return alt.Color(
        "status:N", title=None,
        scale=alt.Scale(domain=domain, range=colours),
        # school charts draw their own key above the chart (it wraps on small screens)
        legend=None if scale else alt.Legend(orient="top", labelLimit=0)
    )


def draw_chart(chart):
    df = pd.DataFrame(chart["data"])
    kind = chart["type"]
    standard = chart.get("standard")
    value_name = chart.get("value_name", "Value")
    tooltip = tooltip_for(chart)
    order = (df["label"].tolist() if chart.get("keep_order") else
             df.sort_values("value", ascending=bool(chart.get("ascending")))["label"].tolist())

    if kind == "pie":
        total = df["value"].sum()
        df["share"] = df["value"] / total * 100
        df["share_txt"] = df["share"].round().astype(int).astype(str) + "%"
        base = alt.Chart(df).encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color("label:N", title=None, sort=order,
                            scale=alt.Scale(range=PALETTE),
                            legend=alt.Legend(orient="bottom")),
            tooltip=tooltip + [alt.Tooltip("share:Q", title="Share %", format=".1f")]
        )
        donut = base.mark_arc(innerRadius=78, outerRadius=128,
                              stroke="#FFFFFF", strokeWidth=3, cornerRadius=4)
        labels = base.mark_text(radius=148, fontWeight=700, color="#0F1E4A").encode(
            text="share_txt:N"
        )
        centre = alt.Chart(pd.DataFrame({"t": [fmt(total)]})).mark_text(
            fontSize=24, fontWeight=800, color="#0F1E4A", dy=-8
        ).encode(text="t:N")
        centre_sub = alt.Chart(pd.DataFrame({"t": ["Total"]})).mark_text(
            fontSize=12, color="#5B6680", dy=16
        ).encode(text="t:N")
        chart_obj = (donut + labels + centre + centre_sub).properties(height=340)

    elif kind == "line":
        base = alt.Chart(df).encode(
            x=alt.X("label:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=value_name, scale=alt.Scale(zero=False)),
            tooltip=tooltip
        )
        area = base.mark_area(
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="rgba(37,99,235,0.35)", offset=0),
                       alt.GradientStop(color="rgba(37,99,235,0.02)", offset=1)],
                x1=1, x2=1, y1=0, y2=1),
            line={"color": "#2563EB", "strokeWidth": 3},
            interpolate="monotone"
        )
        points = base.mark_point(size=110, filled=True, color="#1E3A8A")
        labels = base.mark_text(dy=-14, fontWeight=700, color="#0F1E4A").encode(
            text=alt.Text("value:Q", format=",.1f"))
        layers = area + points + labels
        if standard:
            layers += alt.Chart(pd.DataFrame({"y": [standard]})).mark_rule(
                color="#F97066", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
        chart_obj = layers.properties(height=300)

    elif kind == "compare":
        long = pd.concat([
            pd.DataFrame({"label": df["label"], "series": "Actual", "value": df["value"]}),
            pd.DataFrame({"label": df["label"], "series": "UBE standard",
                          "value": df["standard"]}),
        ])
        bars = alt.Chart(long).mark_bar(cornerRadiusEnd=6, width={"band": 0.38}).encode(
            x=alt.X("label:N", title=None, sort=None,
                    axis=alt.Axis(labelAngle=0, labelLimit=180)),
            xOffset=alt.XOffset("series:N", sort=["Actual", "UBE standard"]),
            y=alt.Y("value:Q", title=value_name),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["Actual", "UBE standard"],
                                            range=["#F97066", "#CBD5E1"]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("label:N", title="Measure"),
                     alt.Tooltip("series:N", title=""),
                     alt.Tooltip("value:Q", title="Value", format=",.1f")]
        )
        labels = bars.mark_text(dy=-8, fontWeight=700, color="#0F1E4A").encode(
            text=alt.Text("value:Q", format=",.1f"), color=alt.value("#0F1E4A"))
        chart_obj = (bars + labels).properties(height=320)

    elif kind == "column":
        colour = status_colour(df, standard, chart.get("status_scale")) if standard else alt.Color(
            "label:N", legend=None, sort=order, scale=alt.Scale(range=PALETTE))
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=8, width={"band": 0.6}).encode(
            x=alt.X("label:N", title=None, sort=order,
                    axis=alt.Axis(labelAngle=0 if len(df) <= 6 else -35, labelLimit=140)),
            y=alt.Y("value:Q", title=value_name),
            color=colour, tooltip=tooltip
        )
        labels = bars.mark_text(dy=-9, fontWeight=700).encode(
            text=alt.Text("value:Q", format=",.1f"), color=alt.value("#0F1E4A"))
        layers = bars + labels
        if standard:
            layers += alt.Chart(pd.DataFrame({"y": [standard]})).mark_rule(
                color="#0F1E4A", strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
        chart_obj = layers.properties(height=320)

    elif kind == "lollipop":
        base = alt.Chart(df).encode(
            y=alt.Y("label:N", title=None, sort=order,
                    axis=alt.Axis(labelLimit=160, labelOverlap=False)),
            tooltip=tooltip
        )
        stems = base.mark_rule(color="#C7D2FE", strokeWidth=3).encode(
            x=alt.X("value:Q", title=value_name), x2=alt.value(0))
        heads = base.mark_circle(size=230, opacity=1).encode(
            x="value:Q",
            color=alt.Color("value:Q", legend=None,
                            scale=alt.Scale(range=["#FACC15", "#F97066"])))
        labels = base.mark_text(align="left", dx=12, fontWeight=700, color="#0F1E4A").encode(
            x="value:Q", text=alt.Text("value:Q", format=",.2f"))
        chart_obj = (stems + heads + labels).properties(height={"step": 34})

    else:  # horizontal ranking bar
        colour = status_colour(df, standard, chart.get("status_scale")) if standard else alt.Color(
            "value:Q", legend=None, scale=alt.Scale(range=["#93C5FD", "#1E3A8A"]))
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=7, height={"band": 0.7}).encode(
            x=alt.X("value:Q", title=value_name,
                    axis=alt.Axis(grid=True, gridColor="#EEF2F9")),
            y=alt.Y("label:N", title=None, sort=order,
                    axis=alt.Axis(labelLimit=160, labelOverlap=False)),
            color=colour, tooltip=tooltip
        )
        labels = bars.mark_text(align="left", dx=5, fontWeight=600).encode(
            text=alt.Text("value:Q", format=",.2f"), color=alt.value("#0F1E4A"))
        layers = bars + labels
        if standard:
            layers += alt.Chart(pd.DataFrame({"x": [standard]})).mark_rule(
                color="#0F1E4A", strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
        chart_obj = layers.properties(height={"step": 34})

    st.altair_chart(
        chart_obj.configure_view(strokeWidth=0).configure_axis(
            labelFont="Inter", titleFont="Inter", labelColor="#5B6680",
            titleColor="#5B6680", domain=False, tickColor="#DCE6F5",
            gridColor="#EEF2F9"
        ).configure_legend(labelFont="Inter", labelColor="#0F1E4A"),
        width="stretch"
    )


TILE_COLOURS = {"#FEE2E2": "#DC2626", "#D1FAE5": "#059669", "#DBEAFE": "#2563EB",
                "#FEF3C7": "#B45309", "#EDE9FE": "#7C3AED"}


def tile(label, value, sub, bg, icon):
    svg = ic(icon, TILE_COLOURS.get(bg, "#2563EB"), 18)
    return (f'<div class="kpi"><div class="kpi-icon" style="background:{bg};">{svg}</div>'
            f'<div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div></div>')


def kpi_tiles(chart):
    """Number cards that change with the type of question."""
    df = pd.DataFrame(chart["data"])
    kind = chart["type"]
    top = df.loc[df["value"].idxmax()]
    low = df.loc[df["value"].idxmin()]

    if kind == "line":
        first, last = df.iloc[0], df.iloc[-1]
        change = last["value"] - first["value"]
        pct = change / first["value"] * 100 if first["value"] else 0
        arrow = "up" if change > 0 else "down" if change < 0 else "flat"
        tiles = [
            tile("Start", fmt(first["value"]), first["label"], "#DBEAFE", "flag"),
            tile("Latest", fmt(last["value"]), last["label"], "#EDE9FE", "pin"),
            tile("Change", f"{change:+,.1f}", f"{pct:+.1f}% overall", "#FEE2E2" if change > 0 else "#D1FAE5", arrow),
            tile("Peak", fmt(top["value"]), top["label"], "#FEF3C7", "peak"),
        ]
    elif kind == "pie":
        total = df["value"].sum()
        tiles = [
            tile("Total", fmt(total), chart.get("value_name", ""), "#DBEAFE", "sum"),
            tile("Largest share", f"{top['value'] / total * 100:.0f}%", top["label"], "#FEF3C7", "star"),
            tile("Smallest share", f"{low['value'] / total * 100:.0f}%", low["label"], "#D1FAE5", "dot"),
            tile("Parts", str(len(df)), "in this chart", "#EDE9FE", "pie"),
        ]
    elif kind == "compare":
        df["std"] = df["standard"]
        df["gap"] = df["value"] - df["std"]
        worst = df.loc[df["gap"].idxmax()]
        above = int((df["gap"] > 0).sum())
        tiles = [
            tile("Above standard", f"{above}/{len(df)}", "measures", "#FEE2E2", "alert"),
            tile("Biggest gap", f"{worst['gap']:+,.1f}", escape(str(worst["label"])), "#FEF3C7", "ruler"),
            tile("Worst value", fmt(worst["value"]), f"Standard: {fmt(worst['std'])}", "#DBEAFE", "up"),
            tile("Measures", str(len(df)), "compared", "#EDE9FE", "target"),
        ]
    else:
        standard = chart.get("standard")
        tiles = [
            tile("Highest", fmt(top["value"]), escape(str(top["label"])), "#FEE2E2", "up"),
            tile("Lowest", fmt(low["value"]), escape(str(low["label"])), "#D1FAE5", "down"),
            tile("Average", fmt(df["value"].mean()), f"{len(df)} compared", "#DBEAFE", "bars"),
        ]
        if standard:
            above = int((df["value"] > standard).sum())
            tiles.append(tile("Above standard", f"{above}/{len(df)}",
                              f"Standard: {fmt(standard)}", "#FEF3C7", "alert"))
        else:
            spread = top["value"] - low["value"]
            tiles.append(tile("Spread", fmt(spread), "highest − lowest", "#EDE9FE", "spread"))

    html(f'<div class="kpi-row">{"".join(tiles)}</div>')


def side_panel(chart):
    """The small panel beside the chart changes with the chart type."""
    df = pd.DataFrame(chart["data"])
    kind = chart["type"]

    if kind == "line":
        rows = ""
        for prev, cur in zip(df.itertuples(), list(df.itertuples())[1:]):
            diff = cur.value - prev.value
            cls = "up" if diff > 0 else "down"
            rows += (f'<div class="lb-row"><span class="lb-name">{escape(str(prev.label))} → {escape(str(cur.label))}</span>'
                     f'<span class="chg {cls}">{diff:+,.1f}</span></div>')
        html(f'<div class="lb"><div class="lb-title">Year-to-year change</div>{rows}</div>')
        return

    if kind == "pie":
        total = df["value"].sum()
        df = df.sort_values("value", ascending=False)
        rows = "".join(
            f'<div class="lb-row"><span class="swatch" style="background:{PALETTE[i % len(PALETTE)]};"></span>'
            f'<span class="lb-name">{escape(str(r.label))}</span><span class="lb-val">{r.value / total * 100:.1f}%</span></div>'
            for i, r in enumerate(df.itertuples())
        )
        html(f'<div class="lb"><div class="lb-title">Share of total</div>{rows}</div>')
        return

    if kind == "compare":
        rows = ""
        for r in df.itertuples():
            std = getattr(r, "standard", None) or chart.get("standard") or 0
            gap = r.value - std
            cls = "up" if gap > 0 else "down"
            word = "above" if gap > 0 else "within"
            rows += (f'<div class="lb-row"><span class="lb-name">{escape(str(r.label))}</span>'
                     f'<span class="chg {cls}">{gap:+,.1f}</span></div>'
                     f'<div class="lb-note">{fmt(r.value)} vs standard {fmt(std)} · {word}</div>')
        html(f'<div class="lb"><div class="lb-title">Gap from the standard</div>{rows}</div>')
        return

    df = df.sort_values("value", ascending=False).head(5)
    top_value = df["value"].max() or 1
    rows = "".join(
        f'<div class="lb-row"><span class="lb-rank">{i}</span>'
        f'<span class="lb-name">{escape(str(r.label))}</span><span class="lb-val">{fmt(r.value)}</span></div>'
        f'<div class="lb-bar"><span style="width:{max(6, r.value / top_value * 100):.0f}%;"></span></div>'
        for i, r in enumerate(df.itertuples(), 1)
    )
    html(f'<div class="lb"><div class="lb-title">Highest {len(df)}</div>{rows}</div>')


def show_assistant_message(message, index):
    """Written answer first, then a live dashboard that fits the question."""
    chart = message.get("chart")
    source = message.get("source", "")

    # Memory at work: show how a follow-up was understood
    asked, understood = message.get("asked", ""), message.get("understood", "")
    def norm(t):
        return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    if understood and asked and norm(understood) != norm(quick_fix(asked)) and source != "ERGIT":
        html(f'<p class="understood">{ic("chat", "#5B6680", 13)} Understood as: {escape(understood)}</p>')

    if message.get("schools"):
        show_school_dashboard(message, index)
        return

    if not chart:
        st.markdown(message["content"])
        html(f'<span class="source-tag">{ic("file", "#1E3A8A", 13)} {source}</span>')
        return

    icon, chart_name = CHART_TYPES.get(chart["type"], CHART_TYPES["bar"])
    chosen = "as you asked" if chart.get("requested") else "chosen for this question"

    html(f'''
<div class="dash-head">
<div>
<div class="dash-eyebrow">Live dashboard · {escape(str(chart.get("title", "")))}</div>
<div class="dash-headline">{escape(str(chart.get("headline", "")))}</div>
</div>
<span class="source-tag">{ic("file", "#1E3A8A", 13)} {source}</span>
</div>
''')

    st.markdown(message["content"])

    html(f'''
<div class="dash-divider"></div>
<div class="chart-chip">{ic(icon, "#1E3A8A", 15)} {chart_name} <span>· {chosen}</span></div>
''')

    kpi_tiles(chart)

    left, right = st.columns([1.75, 1], gap="medium")
    with left:
        draw_chart(chart)
    with right:
        side_panel(chart)


# --------------------------------------------------
# 6c. Voice questions (speech to text with Groq Whisper)
# --------------------------------------------------

VOICE_HINT = (
    "Questions about Lagos State public schools. LGAs: Agege, "
    "Ajeromi/Ifelodun, Alimosho, Amuwo Odofin, Apapa, Badagry, Epe, "
    "Eti-Osa, Ibeju/Lekki, Ifako/Ijaiye, Ikeja, Ikorodu, Kosofe, "
    "Lagos Island, Lagos Mainland, Mushin, Ojo, Oshodi/Isolo, Shomolu, "
    "Surulere. Terms: UBE standard, JSS, SSS, student-teacher ratio, "
    "classroom, toilet, resource gap. Charts: line chart, donut chart, "
    "pie chart, bar chart, column chart, dot chart, dashboard."
)


def transcribe(audio_file):
    """Turn a recorded voice question into text."""
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    result = client.audio.transcriptions.create(
        file=("question.wav", audio_file.getvalue()),
        model="whisper-large-v3",      # the most accurate Whisper model
        language="en",
        prompt=VOICE_HINT,
        temperature=0,
    )
    return result.text.strip()


# --------------------------------------------------
# 7. Answer questions
# --------------------------------------------------

# Everyday chat: greetings, thanks, "how are you", "who are you"...
SMALL_TALK = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|good (morning|afternoon|evening|day)|"
    r"how are you|how('?s| is) (it going|everything|your day)|how far|"
    r"what'?s up|sup|thanks?( you)?|thank u|ok(ay)?|cool|nice|great|"
    r"bye|goodbye|see you|who are you|what are you|what can you do|"
    r"what do you do|help|nice to meet you)\b[\s!.,?😊👋🙂]*",
    re.I
)
DATA_WORDS = re.compile(
    r"\b(lga|teacher|classroom|toilet|school|student|learner|ratio|rank|"
    r"standard|ube|jss|sss|primary|enrol|gap|badagry|lagos|chart)\w*",
    re.I
)

chat_prompt = ChatPromptTemplate.from_template("""
You are ERGIT, a warm and friendly assistant for Lagos State public school
data (teachers, classrooms and toilets in the 20 Lagos LGAs, compared with
the UBE standards). Reply naturally to what the user said, like a friendly
person, in 1 to 3 short sentences. If it fits, mention one thing you can
help with, for example: "Which LGAs need teachers most?". Do not make up
any statistics. The user's name: {name}

Conversation so far:
{history}

User: {question}
""")


def is_small_talk(question):
    """True for everyday chat that does not need the documents."""
    return bool(SMALL_TALK.match(question)) and not DATA_WORDS.search(question)


def friendly_reply(question, chat_model, name=None, history="(none)"):
    text = run_quick(chat_prompt, {"question": question, "name": name or "not given",
                                   "history": history}, chat_model)
    return text.strip() or "Hi there! 😊 Ask me anything about Lagos public schools."


# --------------------------------------------------
# 7a. Public school comparison for parents (NMIS survey, 2012-2014)
#
# School questions are answered from a table with code (filter + sort),
# not from document search: exact numbers and full lists matter here.
# --------------------------------------------------

SCHOOL_FILE = os.path.join(BASE_DIR, "lagos_public_schools_clean.csv")
SCHOOL_NOTE = "School figures: NMIS school survey, 2012–2014 (public schools)."
UBE = {"teacher": ("Students_per_Teacher", 35, "students per teacher"),
       "classroom": ("Students_per_Classroom", 35, "students per classroom"),
       "toilet": ("Students_per_Toilet", 30, "students per toilet")}
LABEL_COLOURS = {
    "Within UBE benchmark": "#10B981",
    "Above UBE benchmark – check capacity": "#F59E0B",
    "High-pressure facilities – visit before enrolment": "#F97066",
    "Insufficient data – verify with school": "#94A3B8",
}
SHORT_LABELS = {"Within UBE benchmark": "Within benchmark",
                "Above UBE benchmark – check capacity": "Above benchmark",
                "High-pressure facilities – visit before enrolment": "High pressure"}


@st.cache_data(show_spinner=False)
def load_schools(path, modified):
    df = pd.read_csv(path)
    # Some names in the survey contain line breaks or double spaces: tidy every text column
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    df["_name"] = df["School_Name"].str.lower()
    df["_place"] = (df["Community"].fillna("") + " " + df["Ward"].fillna("")).str.lower()
    return df


def get_schools():
    if not os.path.exists(SCHOOL_FILE):
        return None
    return load_schools(SCHOOL_FILE, os.path.getmtime(SCHOOL_FILE))


school_filter_prompt = ChatPromptTemplate.from_template("""
Read a parent's question about public schools in Lagos and fill in a search.

Conversation so far:
{history}

Question: {question}

Lagos LGAs: Agege, Ajeromi/Ifelodun, Alimosho, Amuwo Odofin, Apapa, Badagry,
Epe, Eti-Osa, Ibeju/Lekki, Ifako/Ijaye, Ikeja, Ikorodu, Kosofe, Lagos Island,
Lagos Mainland, Mushin, Ojo, Oshodi/Isolo, Shomolu, Surulere.

Reply with ONE JSON object only:
{{"lga": "one LGA from the list, or null",
  "place": "a town, area or community named by the user that is not an LGA, or null",
  "level": "Primary" or "JSS" or "SSS" or "Secondary" (JSS or SSS) or null,
  "school_name": "the name of ONE particular school the user named (for
                  example 'Ikorodu Grammar School'), or null. A town, area
                  or LGA is NOT a school name.",
  "sort_by": "teacher" or "classroom" or "toilet" or "overall",
  "order": "best" (least crowded first) or "worst" (most crowded first;
           use this for schools that are lacking, in need, or to support),
  "top_n": a number from 1 to 20 (default 10)}}

"Smallest classes" means sort_by "classroom"; "enough teachers" means
"teacher"; "toilets" means "toilet"; otherwise "overall". Use the conversation
to fill in what the user means by "there", "that school" or "the first one".
""")


LGA_LIST = ["Agege", "Ajeromi/Ifelodun", "Alimosho", "Amuwo Odofin", "Apapa", "Badagry",
            "Epe", "Eti-Osa", "Ibeju/Lekki", "Ifako/Ijaye", "Ikeja", "Ikorodu", "Kosofe",
            "Lagos Island", "Lagos Mainland", "Mushin", "Ojo", "Oshodi/Isolo", "Shomolu",
            "Surulere"]
LGA_ALIASES = {"ajeromi": "Ajeromi/Ifelodun", "ifelodun": "Ajeromi/Ifelodun",
               "ajegunle": "Ajeromi/Ifelodun", "amuwo": "Amuwo Odofin", "festac": "Amuwo Odofin",
               "eti osa": "Eti-Osa", "ibeju": "Ibeju/Lekki", "lekki": "Ibeju/Lekki",
               "ifako": "Ifako/Ijaye", "ijaye": "Ifako/Ijaye", "ijaiye": "Ifako/Ijaye",
               "oshodi": "Oshodi/Isolo", "isolo": "Oshodi/Isolo", "somolu": "Shomolu",
               "mainland": "Lagos Mainland", "lagos island": "Lagos Island"}


def match_lga(text):
    """Turn any way of writing an LGA into its official name."""
    if not text:
        return None
    t = str(text).lower().strip()
    for lga in sorted(LGA_LIST, key=len, reverse=True):
        if re.search(rf"\b{re.escape(lga.lower())}\b", t):
            return lga
    for alias, lga in LGA_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return lga
    close = difflib.get_close_matches(t, [x.lower() for x in LGA_LIST], n=1, cutoff=0.75)
    return next((x for x in LGA_LIST if x.lower() == close[0]), None) if close else None


SCHOOL_WORDS = re.compile(r"\b(school|schools|college|grammar|academy|high|comprehensive|"
                          r"primary|secondary|jss|sss|model|memorial|nursery|community|"
                          r"c\.?p\.?s|l\.?a\.?|unity)\b")


def guess_school_filters(question):
    """Simple rules, used when the AI reply cannot be read."""
    q = question.lower()
    level = ("JSS" if re.search(r"\b(jss|junior)\b", q) else
             "SSS" if re.search(r"\b(sss|senior)\b", q) else
             "Secondary" if "secondary" in q else
             "Primary" if "primary" in q else None)
    sort_by = next((k for k in UBE if k in q), "overall")
    if re.search(r"class ?size|smallest class|crowded class", q):
        sort_by = "classroom"
    order = "worst" if re.search(r"\b(worst|most crowded|highest|over ?crowded|lacking|lack|need|needs|"
                                 r"under pressure|empower|support|help|fund|poorest)\b", q) else "best"
    n = re.search(r"\b(top|best|first|worst|bottom|least)\s+(\d{1,2})\b", q) or re.search(r"\b(\d{1,2})\s+(?:public\s+)?schools\b", q)
    top = int(n.group(n.lastindex)) if n else 10
    return {"lga": match_lga(q), "place": None, "level": level, "school_name": None,
            "sort_by": sort_by, "order": order, "top_n": max(1, min(20, top))}


def read_school_filters(raw, question):
    out = guess_school_filters(question)
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return out
    for key in out:
        value = data.get(key)
        if value not in (None, "", "null", "None"):
            out[key] = value
    # "Ikorodu" alone is an LGA, not a school. Keep a school name only if it
    # looks like one (has a word like school/college/grammar) or is not a place.
    name = str(out.get("school_name") or "").strip()
    if name and not SCHOOL_WORDS.search(name.lower()) and (
            match_lga(name) or name.lower() in str(question).lower().split()):
        if not data.get("lga") and match_lga(name):
            data["lga"] = match_lga(name)
        elif not data.get("place") and not match_lga(name):
            out["place"] = name
        out["school_name"] = None
    explicit_lga = match_lga(data.get("lga"))
    # A school's name often contains a place name ("Ifako Junior..."), so for a
    # named school only use an LGA the AI picked out explicitly.
    out["lga"] = explicit_lga if out.get("school_name") else (explicit_lga or match_lga(question))
    if out.get("place") and match_lga(out["place"]) and not out["lga"]:
        out["lga"], out["place"] = match_lga(out["place"]), None
    out["sort_by"] = out["sort_by"] if out["sort_by"] in (*UBE, "overall") else "overall"
    out["order"] = "worst" if str(out["order"]).lower() == "worst" else "best"
    try:
        out["top_n"] = max(1, min(20, int(out["top_n"])))
    except (TypeError, ValueError):
        out["top_n"] = 10
    if out["level"] not in ("Primary", "JSS", "SSS", "Secondary"):
        out["level"] = None
    return out


def km_between(lat1, lon1, lat2, lon2):
    """Straight-line distance in km (an estimate, not travel distance)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def search_schools(f):
    """Filter and sort the schools. Returns (rows, notes)."""
    df = get_schools()
    notes = {"filters": f, "place_found": None, "total_in_area": 0}
    if df is None:
        return None, notes
    rows = df[~df["Level"].isin(["Pre-primary", "Unknown", "Technical/Vocational"])]

    if f.get("school_name"):
        pool = rows[rows["LGA"] == f["lga"]] if f.get("lga") else rows
        names = difflib.get_close_matches(str(f["school_name"]).lower(), pool["_name"].tolist(),
                                          n=3, cutoff=0.55)
        if not names and f.get("lga"):
            pool = rows
            names = difflib.get_close_matches(str(f["school_name"]).lower(), pool["_name"].tolist(),
                                              n=3, cutoff=0.55)
        wanted = str(f["school_name"]).lower()
        if names and difflib.SequenceMatcher(None, wanted, names[0]).ratio() >= 0.9:
            names = names[:1]          # a clear match: show just that school
        hit = pool[pool["_name"].isin(names)].copy()
        hit["_rank"] = hit["_name"].map({n: i for i, n in enumerate(names)})
        hit = hit.sort_values("_rank").drop(columns="_rank")
        if hit.empty:
            word = str(f["school_name"]).lower().split()[0]
            hit = pool[pool["_name"].str.contains(re.escape(word), na=False)].head(3)
        notes["total_in_area"] = len(hit)
        return hit, notes

    if f.get("lga"):
        rows = rows[rows["LGA"] == f["lga"]]
    if f.get("place"):
        place = str(f["place"]).lower().strip()
        near = rows[rows["_place"].str.contains(re.escape(place), na=False) |
                    rows["_name"].str.contains(re.escape(place), na=False)]
        if not near.empty:
            lat, lon = near["Latitude"].median(), near["Longitude"].median()
            base = df[~df["Level"].isin(["Pre-primary", "Unknown", "Technical/Vocational"])]
            base = base.assign(Distance_km=km_between(lat, lon, base["Latitude"], base["Longitude"]).round(1))
            rows = base[base["Distance_km"] <= 5]
            notes["place_found"] = f["place"]
    if f.get("level") == "Secondary":
        rows = rows[rows["Offers_JSS"] | rows["Offers_SSS"]]
    elif f.get("level"):
        rows = rows[rows[f"Offers_{f['level']}"]]

    notes["total_in_area"] = len(rows)
    best = f["order"] == "best"
    col = "Pressure_Score" if f["sort_by"] == "overall" else UBE[f["sort_by"]][0]
    rows = rows.assign(_missing=rows[col].isna())
    if f["sort_by"] == "overall":
        # Within benchmark first (or high-pressure first), then by the average
        rank = {"Within UBE benchmark": 0, "Above UBE benchmark – check capacity": 1,
                "High-pressure facilities – visit before enrolment": 2}
        rows = rows.assign(_label=rows["Overall_Label"].map(rank).fillna(3))
        if not best:
            rows["_label"] = rows["_label"].where(rows["_label"] == 3, 2 - rows["_label"])
        rows = rows.sort_values(["_missing", "_label", col], ascending=[True, True, best])
        rows = rows.drop(columns="_label")
    else:
        rows = rows.sort_values(["_missing", col], ascending=[True, best])
    return rows.head(f["top_n"]), notes


school_answer_prompt = ChatPromptTemplate.from_template("""
You are ERGIT, a warm and friendly guide helping a parent compare Lagos
public schools. The user's name: {name}

Conversation so far:
{history}

Question: {question}

Search used: {search}
Schools found (in this order):
{schools}

Write ONLY a short, warm opening of 1 or 2 sentences (plain English, no
list, no table, no headings). Do NOT list the schools, do NOT repeat their
figures, do NOT add advice, and do NOT mention "the app", "the list below"
or these instructions.
In the opening:
- Say what was found, e.g. "Here are 10 public schools in Ikorodu, with the
  schools within the UBE benchmark first." Describe the order exactly as
  given in "Search used", once, and add nothing else about the sorting (do
  not say "the first few are the least crowded"). If there is a NOTE in
  "Search used", mention it.
- If the user asked about one school, say in one sentence how it compares
  with the standards, using its label exactly as given.
- Never call a school good or bad: the figures measure crowding and
  facilities, not teaching quality.
{goal_rule}
Use only the information above. Do not invent schools or numbers.
""")


def school_list_markdown(records):
    """The numbered school list, written by code so it is always complete and exact."""
    def v(x):
        return "n/a" if x is None or pd.isna(x) else f"{float(x):.1f}"
    short = {"Within UBE benchmark": "Within benchmark",
             "Above UBE benchmark – check capacity": "Above benchmark – check capacity",
             "High-pressure facilities – visit before enrolment": "High pressure – visit before enrolment",
             "Insufficient data – verify with school": "Insufficient data – verify with school"}
    lines = []
    for i, r in enumerate(records, 1):
        place = r.get("Community") or r.get("LGA") or ""
        dist = f", {r['Distance_km']} km away" if r.get("Distance_km") is not None else ""
        label = short.get(r.get("Overall_Label"), r.get("Overall_Label") or "")
        lines.append(
            f"{i}. **{r['School_Name']}** ({r.get('Level', '')}, {place}{dist}): "
            f"{v(r.get('Students_per_Teacher'))} per teacher · "
            f"{v(r.get('Students_per_Classroom'))} per classroom · "
            f"{v(r.get('Students_per_Toilet'))} per toilet — *{label}*")
    return "\n".join(lines)


def school_rows_text(rows):
    lines = []
    for i, r in enumerate(rows.itertuples(), 1):
        dist = f", {r.Distance_km} km away (straight line)" if hasattr(r, "Distance_km") else ""
        def v(x):
            return "n/a" if pd.isna(x) else f"{x:.1f}"
        lines.append(
            f"{i}. {r.School_Name} ({r.Level}, {r.Community or r.Ward}, {r.LGA}{dist}): "
            f"{v(r.Students_per_Teacher)} students per teacher, {v(r.Students_per_Classroom)} "
            f"per classroom, {v(r.Students_per_Toilet)} per toilet. Overall: {r.Overall_Label}. "
            f"Survey {int(r.Survey_Year) if not pd.isna(r.Survey_Year) else 'n/a'}."
        )
    return "\n".join(lines)


def school_records(rows):
    keep = ["School_Name", "LGA", "Community", "Level", "Students", "Teachers", "Classrooms",
            "Toilets", "Students_per_Teacher", "Students_per_Classroom", "Students_per_Toilet",
            "Overall_Label", "Pressure_Score", "Survey_Year", "Latitude", "Longitude"]
    if "Distance_km" in rows:
        keep.append("Distance_km")
    return json.loads(rows[keep].to_json(orient="records"))


def directions_url(row):
    """Google Maps directions from wherever the user is to the school.
    On a phone this opens the Maps app with the route."""
    lat, lon = getattr(row, "Latitude", None), getattr(row, "Longitude", None)
    if lat is None or pd.isna(lat) or pd.isna(lon):
        place = quote_plus(f"{row.School_Name}, {row.LGA}, Lagos")
        return f"https://www.google.com/maps/search/?api=1&query={place}"
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}&travelmode=driving"


def directions_list(schools):
    """A short list of the schools, each with a one-tap directions link."""
    rows = ""
    for i, r in enumerate(schools.head(8).itertuples(), 1):
        colour = LABEL_COLOURS.get(r.Overall_Label, "#94A3B8")
        rows += (f'<div class="dir-row"><span class="lb-rank">{i}</span>'
                 f'<span class="dir-name"><span style="color:{colour};">●</span> {escape(r.School_Name)}'
                 f'<span class="dir-sub">{escape(str(r.Community or r.LGA))}</span></span>'
                 f'<a class="dir-go" href="{directions_url(r)}" target="_blank" rel="noopener">'
                 f'{ic("pin", "#2563EB", 14)} Directions</a></div>')
    html(f'<div class="lb"><div class="lb-title">Get directions</div>{rows}'
         f'<p class="dir-note">Opens Google Maps with the route from where you are.</p></div>')


def hex_to_rgb(colour):
    colour = colour.lstrip("#")
    return [int(colour[i:i + 2], 16) for i in (0, 2, 4)] + [230]


def school_map(schools, index):
    """A large map: tap a school to see its details and get directions.
    Use the expand icon (top right of the map) for full screen."""
    points = schools.dropna(subset=["Latitude", "Longitude"]).copy()
    if points.empty:
        return
    points["colour"] = points["Overall_Label"].map(LABEL_COLOURS).fillna("#94A3B8").map(hex_to_rgb)
    points["label_short"] = points["Overall_Label"].str.split(" –").str[0]
    for c in ["Students_per_Teacher", "Students_per_Classroom", "Students_per_Toilet"]:
        points[c + "_txt"] = points[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
    points = points.reset_index(drop=True)

    spread = max(points["Latitude"].max() - points["Latitude"].min(),
                 points["Longitude"].max() - points["Longitude"].min())
    zoom = 14 if spread < 0.01 else 13 if spread < 0.04 else 12 if spread < 0.1 else 11 if spread < 0.25 else 10

    map_data = points[["School_Name", "Latitude", "Longitude", "colour", "label_short",
                       "Students_per_Teacher_txt", "Students_per_Classroom_txt",
                       "Students_per_Toilet_txt"]]          # no empty values: the map needs clean data
    layer = pdk.Layer(
        "ScatterplotLayer", data=map_data, id="schools",
        get_position=["Longitude", "Latitude"], get_fill_color="colour",
        get_line_color=[255, 255, 255], line_width_min_pixels=2, stroked=True,
        get_radius=90, radius_min_pixels=8, radius_max_pixels=18, pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=float(points["Latitude"].mean()),
                                         longitude=float(points["Longitude"].mean()),
                                         zoom=zoom),
        tooltip={"html": "<b>{School_Name}</b><br/>{label_short}<br/>"
                         "{Students_per_Classroom_txt} per classroom · "
                         "{Students_per_Teacher_txt} per teacher · {Students_per_Toilet_txt} per toilet",
                 "style": {"backgroundColor": "#0F1E4A", "color": "white",
                           "fontFamily": "Inter, sans-serif", "fontSize": "12px",
                           "borderRadius": "8px", "padding": "8px 10px"}},
    )
    html('<p class="map-title">' + ic("pin", "#1E3A8A", 15) +
         ' <b>Map of these schools.</b> Tap a dot to see the school and get directions; '
         'use the expand icon on the map for full screen.</p>')
    state = st.pydeck_chart(deck, height=420, on_select="rerun",
                            selection_mode="single-object", key=f"school_map_{index}")
    html('<p class="map-key">'
         + " &nbsp; ".join(f'<span style="white-space:nowrap;"><span style="color:{c};">●</span> '
                           f'{k.split(" –")[0]}</span>' for k, c in LABEL_COLOURS.items()) + "</p>")

    # The school the user tapped on the map
    picked = None
    try:
        objects = state.selection.get("objects", {}).get("schools", [])
        picked = objects[0] if objects else None
    except Exception:
        picked = None
    if picked:
        name = picked.get("School_Name", "")
        row = points[points["School_Name"] == name]
        if not row.empty:
            r = next(row.itertuples())
            html(f'<div class="map-pick"><div class="map-pick-name">{escape(name)}</div>'
                 f'<div class="map-pick-sub">{escape(str(r.Community or ""))}, {escape(r.LGA)} · '
                 f'{escape(r.Level)} · {escape(r.label_short)}</div>'
                 f'<div class="map-pick-sub">{r.Students_per_Classroom_txt} per classroom · '
                 f'{r.Students_per_Teacher_txt} per teacher · {r.Students_per_Toilet_txt} per toilet</div></div>')
            st.link_button("Get directions to this school", directions_url(r),
                           icon=":material/directions:", type="primary")


def show_school_dashboard(message, index=0):
    """Answer + school dashboard: number cards, a map, a chart and a table."""
    schools = pd.DataFrame(message["schools"])
    for c in ["Students", "Teachers", "Classrooms", "Toilets", "Students_per_Teacher",
              "Students_per_Classroom", "Students_per_Toilet", "Latitude", "Longitude",
              "Distance_km", "Survey_Year", "Pressure_Score"]:
        if c in schools:
            schools[c] = pd.to_numeric(schools[c], errors="coerce")
    meta = message.get("school_meta", {})
    f = meta.get("filters", {})
    area = meta.get("place_found") or f.get("lga") or "Lagos"
    level = f.get("level") or "public"
    title = f"{level} schools · {area}" if f.get("level") else f"Public schools · {area}"
    sort_col, standard, sort_words = UBE.get(f.get("sort_by"), ("Pressure_Score", 1, "pressure score"))

    html(f'''
<div class="dash-head">
<div>
<div class="dash-eyebrow">Live dashboard · {escape(title)}</div>
<div class="dash-headline">{len(schools)} school{"s" if len(schools) != 1 else ""} compared with the UBE standards</div>
</div>
<span class="source-tag">{ic("file", "#1E3A8A", 13)} NMIS 2012–2014</span>
</div>
''')
    st.markdown(message["content"])
    html('<div class="dash-divider"></div>')

    within = int((schools["Overall_Label"] == "Within UBE benchmark").sum())
    complete = schools[["Students_per_Teacher", "Students_per_Classroom", "Students_per_Toilet"]].notna().all(axis=1)
    med_class = schools["Students_per_Classroom"].median()
    tiles = [
        tile("Schools shown", str(len(schools)), f"of {meta.get('total_in_area', len(schools))} in the area", "#DBEAFE", "classroom"),
        tile("Within benchmark", f"{within}/{len(schools)}", "on all three measures", "#D1FAE5", "target"),
        tile("Typical class", f"{med_class:.1f}" if not pd.isna(med_class) else "n/a",
             "students per classroom (median)", "#FEF3C7", "teacher"),
        tile("Full data", f"{int(complete.sum())}/{len(schools)}", "schools with all figures", "#EDE9FE", "file"),
    ]
    html(f'<div class="kpi-row">{"".join(tiles)}</div>')

    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        if f.get("sort_by") not in UBE and "Pressure_Score" not in schools:
            schools["Pressure_Score"] = (schools["Students_per_Teacher"] / 35 +
                                         schools["Students_per_Classroom"] / 35 +
                                         schools["Students_per_Toilet"] / 30) / 3
        if f.get("sort_by") not in UBE:
            # Overall: average of the three ratios ÷ their standards; 1.0 = exactly at the standards
            sort_col, standard, sort_words = ("Pressure_Score", 1.0,
                                              "times the UBE standards (average)")
        if schools[sort_col].notna().sum() >= 1:
            chart = {"type": "bar", "title": sort_words, "label_name": "School",
                     "value_name": sort_words[0].upper() + sort_words[1:], "standard": standard,
                     "ascending": f.get("order") != "worst", "keep_order": True,
                     "data": [{"label": r.School_Name[:38], "value": float(getattr(r, sort_col))}
                              for r in schools.itertuples() if not pd.isna(getattr(r, sort_col))]}
            overall = f.get("sort_by") not in UBE
            if overall:
                # Colour by the school's label, so the chart agrees with the list:
                # one measure above its standard makes the school "Above".
                for d, r in zip(chart["data"], [r for r in schools.itertuples()
                                                if not pd.isna(getattr(r, sort_col))]):
                    d["status"] = SHORT_LABELS.get(str(r.Overall_Label), "No data")
                names = list(SHORT_LABELS.values()) + ["No data"]
                used = {d.get("status") for d in chart["data"]}
                pairs = [(n, c) for n, c in zip(names, LABEL_COLOURS.values()) if n in used]
                chart["status_scale"] = ([n for n, _ in pairs], [c for _, c in pairs])
                html('<p class="map-key" style="margin:0 0 0.3rem 0;">' + " &nbsp; ".join(
                    f'<span style="white-space:nowrap;"><span style="color:{c};">■</span> {n}</span>'
                    for n, c in pairs) + '</p>')
            draw_chart(chart)
            html('<p class="map-key">' + (
                "Bar length = the average of the three measures (dotted line = the standard). "
                "Bar colour = the school's label, as on the map: a school is only "
                "\"Within benchmark\" if all three measures are within their standards."
                if overall else
                "Dotted line = the UBE standard. Bars past it are above the standard.") + '</p>')
        else:
            counts = schools["Overall_Label"].value_counts()
            chart = {"type": "pie", "title": "Overall labels", "label_name": "Label",
                     "value_name": "Schools", "standard": None,
                     "data": [{"label": k.split(" –")[0], "value": float(v)} for k, v in counts.items()]}
            if len(chart["data"]) >= 2:
                draw_chart(chart)
            else:
                st.caption("All schools shown have the same overall label.")
    with right:
        directions_list(schools)

    school_map(schools, index)

    short = {"Within UBE benchmark": "Within benchmark",
             "Above UBE benchmark – check capacity": "Above – check capacity",
             "High-pressure facilities – visit before enrolment": "High pressure – visit first",
             "Insufficient data – verify with school": "Insufficient data"}
    schools["Overall_Label"] = schools["Overall_Label"].map(short).fillna(schools["Overall_Label"])
    table = schools.rename(columns={
        "School_Name": "School", "Students_per_Teacher": "Per teacher",
        "Students_per_Classroom": "Per classroom", "Students_per_Toilet": "Per toilet",
        "Overall_Label": "Label", "Distance_km": "Distance (km, est.)", "Survey_Year": "Survey"})
    table["Directions"] = [directions_url(r) for r in schools.itertuples()]
    cols = ["School", "Community", "Level", "Per teacher", "Per classroom", "Per toilet", "Label",
            "Directions"]
    if "Distance (km, est.)" in table:
        cols.insert(3, "Distance (km, est.)")
    st.dataframe(
        table[cols], hide_index=True, width="stretch",
        column_config={
            "School": st.column_config.TextColumn(width="medium"),
            "Per teacher": st.column_config.NumberColumn(format="%.1f"),
            "Per classroom": st.column_config.NumberColumn(format="%.1f"),
            "Per toilet": st.column_config.NumberColumn(format="%.1f"),
            "Label": st.column_config.TextColumn(width="medium"),
            "Directions": st.column_config.LinkColumn(display_text="Get directions"),
        })
    html(f'<p class="school-note">{SCHOOL_NOTE} UBE standards: 35 per teacher, 35 per classroom, '
         '30 per toilet. Labels describe crowding and facilities, not teaching quality. '
         'Visit schools before deciding.</p>')


# --------------------------------------------------
# 7b. The ERGIT agent (LangGraph)
#
#   message -> [understand] --chat---> [chat] -----------------------> answer
#                    |
#                    +--data---> [retrieve] --found--> [answer] -----> answer
#                    |               +--nothing found--> [chat] -----> answer
#                    |
#                    +--school-> [find schools] --found--> [school answer] -> answer
#                                    +--nothing found--> [chat] -----> answer
#
# The AI looks at the message (and the conversation so far) FIRST and decides:
# everyday chat is answered directly; only data questions search the documents.
# --------------------------------------------------

router_prompt = ChatPromptTemplate.from_template("""
You decide how ERGIT should handle a message. ERGIT is an assistant for
Lagos State public school data (LGAs, teachers, classrooms, toilets,
students, UBE standards, rankings, trends, charts).

Conversation so far:
{history}

Latest message: {question}

Reply with ONE JSON object only, no other words:
{{"route": "chat" or "data" or "school", "question": "the latest message as a complete question"}}

- "chat": greetings, thanks, small talk, how are you, questions about ERGIT
  itself, or anything not about school or education data.
- "school": questions about INDIVIDUAL schools: finding or choosing a school
  for a child, schools in an area or near a place, a school by name, or
  follow-ups about schools listed earlier ("the first school you mentioned").
- "data": questions about LGAs, school levels, Lagos as a whole, standards,
  rankings, trends or charts, INCLUDING follow-ups to earlier answers.
- "question": rewrite the latest message so it makes sense on its own.
  Replace words like "it", "they", "there", "that LGA", "the first one",
  "the second LGA you mentioned" with the exact names from the conversation.
  Keep any chart type the user asked for. If it already makes sense, copy it.
  Keep the user's intent: when they state a goal ("I want to empower /
  support / fund the LGAs or schools lacking the most"), the question is
  WHICH ones lack the most, e.g. "Which LGAs are lacking resources the
  most?". Never turn it into a question about how to do something.
""")


class ErgitState(TypedDict, total=False):
    question: str        # what the user typed or said
    history: str         # the conversation so far in this chat
    name: str
    route: str           # "chat" or "data"
    standalone: str      # the question rewritten to make sense on its own
    context: str
    source: str
    answer: str
    chart: Optional[dict]
    schools: Optional[list]   # school rows for the school dashboard
    school_meta: Optional[dict]


def parse_route(raw, question):
    """Read the router's JSON; fall back to simple rules if it is unclear."""
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        route = str(data.get("route", "")).lower()
        standalone = str(data.get("question") or question).strip()
        if route in ("chat", "data", "school"):
            return route, standalone or question
    except Exception:
        pass
    if is_small_talk(question):
        return "chat", question
    if re.search(r"\bschools?\b", question, re.I) and re.search(
            r"\b(my (child|son|daughter|kid|ward)|enrol|choose|pick|near|which|best|smallest|list|named)\b",
            question, re.I) and not re.search(r"\blgas?\b", question, re.I):
        return "school", question
    return "data", question


def build_agent(vector_db, chat_model, general_retriever):
    def understand(state):
        question = quick_fix(state["question"])
        if state.get("history", "(none)") == "(none)" and is_small_talk(question):
            return {"route": "chat", "standalone": question}   # obvious hello: skip a call
        raw = run_quick(router_prompt, {"history": state.get("history", "(none)"),
                                        "question": question}, chat_model)
        route, standalone = parse_route(raw, question)
        return {"route": route, "standalone": quick_fix(standalone)}

    def chat(state):
        note = ""
        if state.get("route") == "data":
            note = ("\n(Note for ERGIT: the documents had nothing on this. Say so "
                    "kindly and suggest a question about Lagos LGAs.)")
        if state.get("route") == "school":
            note = ("\n(Note for ERGIT: no public schools matched this search in the "
                    "2012-2014 school survey. Say so kindly and suggest naming an LGA, "
                    "for example: 'Public JSS schools in Kosofe with the smallest classes'.)")
            if (state.get("school_meta") or {}).get("need_area"):
                note = ("\n(Note for ERGIT: the user wants public schools but did not say "
                        "where. Ask kindly which LGA or area of Lagos they live in or are "
                        "moving to, and whether the child needs primary, JSS or SSS. Example "
                        "they can type: 'Public primary schools in Surulere'.)")
        reply = friendly_reply(state["question"] + note, chat_model,
                               state.get("name"), state.get("history", "(none)"))
        return {"answer": reply, "source": "ERGIT", "chart": None}

    def retrieve(state):
        context, source = retrieve_context(state["standalone"], vector_db, general_retriever)
        return {"context": context, "source": source}

    def answer(state):
        values = {"context": state["context"], "question": state["standalone"],
                  "name": state.get("name") or "not given",
                  "history": state.get("history", "(none)")}
        text, chart = split_chart((prompt | chat_model | StrOutputParser()).invoke(values))
        if not text.strip():
            # The AI sent no words: ask again, simply, without the dashboard part
            text, extra = split_chart((simple_prompt | chat_model | StrOutputParser()).invoke(values))
            chart = chart or extra
        if not text.strip():
            text = (chart or {}).get("headline") or (
                "Sorry, I couldn't put that answer together. Could you ask it "
                "in a shorter way? For example: *How did the student-teacher "
                "ratio change from 2016/17 to 2019/20?*"
            )
        if not chart:
            chart = make_chart(state["standalone"], text, chat_model)
        if chart and not chart_is_grounded(chart, state["context"] + "\n" + text):
            chart = None        # numbers we cannot find in the documents: no chart
        if chart:
            chart = choose_chart_type(state["standalone"], chart)
        return {"answer": text, "chart": chart}

    def find_schools(state):
        raw = run_quick(school_filter_prompt, {"history": state.get("history", "(none)"),
                                               "question": state["standalone"]}, chat_model)
        filters = read_school_filters(raw, state["standalone"])
        asked = (state["question"] + " " + state["standalone"]).lower()
        if not (filters.get("lga") or filters.get("place") or filters.get("school_name")) \
                and not re.search(r"\b(all of lagos|across lagos|anywhere in lagos|whole (of )?lagos|"
                                     r"all lgas|any lga|statewide|whole state)\b", asked):
            # A parent who has not said where: ask, instead of guessing all of Lagos
            return {"schools": [], "school_meta": {"filters": filters, "need_area": True}}
        rows, notes = search_schools(filters)
        if rows is None or rows.empty:
            return {"schools": [], "school_meta": notes}
        return {"schools": school_records(rows), "school_meta": notes,
                "context": school_rows_text(rows)}

    def school_answer(state):
        meta = state["school_meta"]
        f = meta["filters"]
        search = ", ".join(f"{k}: {v}" for k, v in f.items() if v)
        if f.get("sort_by") == "overall":
            search += (". Order: schools within the benchmark first, then the rest; "
                       "least crowded first within each group" if f.get("order") == "best"
                       else ". Order: high-pressure schools first, then the rest; "
                       "most crowded first within each group")
        else:
            search += (". Order: least crowded first" if f.get("order") == "best"
                       else ". Order: most crowded first")
        if f.get("place") and not meta.get("place_found"):
            where = f.get("lga") or "all of Lagos"
            search += (f". NOTE: the place '{f['place']}' is not in the survey data, so "
                       f"these schools are from {where}. Tell the user this.")
        search += f". Schools in this area: {meta.get('total_in_area')}"
        wants_support = re.search(r"\b(empower\w*|support\w*|fund\w*|donat\w*|ngo|sponsor\w*)\b",
                                  (state.get("question", "") + " " + state["standalone"]).lower())
        goal_rule = ("- The user wants to support, fund or empower schools: say these schools "
                     "showed the greatest pressure in the 2012-2014 survey, and suggest confirming "
                     "their current needs with the Lagos SUBEB office or the LGEA before acting."
                     if wants_support else
                     "- The user is a parent choosing a school. Do NOT talk about funding, "
                     "supporting or empowering schools.")
        opening = (school_answer_prompt | chat_model | StrOutputParser()).invoke({
            "name": state.get("name") or "not given", "history": state.get("history", "(none)"),
            "question": state["standalone"], "search": search, "schools": state["context"],
            "goal_rule": goal_rule}).strip()
        records = state["schools"]
        # Drop any sentence where the AI talks about the app or its instructions
        sentences = re.split(r"(?<=[.!?])\s+", opening)
        opening = " ".join(x for x in sentences if not re.search(
            r"\b(the app|your words|below|following list|list of schools|instructions?)\b", x, re.I)).strip()
        if not opening:
            opening = f"Here {'is' if len(records) == 1 else 'are'} the school{'s' if len(records) != 1 else ''} I found."
        parts = [opening, school_list_markdown(records)]
        thin = len(records) < 3 and not f.get("school_name")
        if thin:
            parts.append("The survey has only a few schools for this search, so the data here is thin.")
        if wants_support:
            parts.append("These figures come from the 2012–2014 NMIS school survey and may have "
                         "changed. Before acting, confirm each school's current needs with the "
                         "Lagos SUBEB office or the Local Government Education Authority (LGEA).")
        else:
            parts.append("These figures come from the 2012–2014 NMIS school survey and may have "
                         "changed. Before deciding, visit your top 3 to 5 choices and ask about "
                         "class sizes, teacher numbers and travel time.")
        text = "\n\n".join(parts)
        return {"answer": text, "source": "NMIS 2012–2014", "chart": None}

    graph = StateGraph(ErgitState)
    graph.add_node("understand", understand)
    graph.add_node("chat", chat)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("find_schools", find_schools)
    graph.add_node("school_answer", school_answer)
    graph.add_edge(START, "understand")
    graph.add_conditional_edges("understand", lambda s: s["route"],
                                {"chat": "chat", "data": "retrieve", "school": "find_schools"})
    graph.add_conditional_edges("find_schools",
                                lambda s: "school_answer" if s.get("schools") else "chat",
                                {"school_answer": "school_answer", "chat": "chat"})
    graph.add_conditional_edges("retrieve",
                                lambda s: "answer" if s.get("context", "").strip() else "chat",
                                {"answer": "answer", "chat": "chat"})
    graph.add_edge("chat", END)
    graph.add_edge("answer", END)
    graph.add_edge("school_answer", END)
    return graph.compile()


def build_history(turns, upto, max_turns=6):
    """The earlier questions and answers in THIS chat (its memory)."""
    lines = []
    for turn in turns[max(0, upto - max_turns):upto]:
        version = turn["versions"][turn["current"]]
        result = version.get("answer") or {}
        if "content" not in result:
            continue
        reply = result["content"]
        reply = reply if len(reply) <= 700 else reply[:700] + "…"
        chart = result.get("chart")
        if chart:
            items = ", ".join(f'{d["label"]} ({fmt(d["value"])})' for d in chart["data"][:10])
            reply += f"\n[Dashboard, in order: {items}]"
        if result.get("schools"):
            names = ", ".join(f'{i}. {r["School_Name"]} ({r["LGA"]})'
                              for i, r in enumerate(result["schools"][:10], 1))
            reply += f"\n[Schools listed, in order: {names}]"
        lines.append(f"User: {version.get('understood') or version['question']}\nERGIT: {reply}")
    return "\n\n".join(lines) if lines else "(none)"


def answer_question(question, vector_db, chat_model, general_retriever, name=None,
                    history="(none)"):
    agent = build_agent(vector_db, chat_model, general_retriever)
    result = agent.invoke({"question": question, "history": history, "name": name or ""})
    return (result.get("answer", ""), result.get("source", "ERGIT"),
            result.get("chart"), result.get("standalone", question),
            result.get("schools") or None, result.get("school_meta"))


# Fix voice questions that were misheard
voice_fix_prompt = ChatPromptTemplate.from_template("""
A user asked a question by voice about Lagos State public school data
(LGAs, teachers, classrooms, toilets, UBE standards, school levels
Primary/JSS/SSS, enrolment, years 2016/17 to 2019/20 and 2024/25).
Speech-to-text may have misheard some words.

Lagos LGAs: Agege, Ajeromi/Ifelodun, Alimosho, Amuwo Odofin, Apapa,
Badagry, Epe, Eti-Osa, Ibeju/Lekki, Ifako/Ijaiye, Ikeja, Ikorodu, Kosofe,
Lagos Island, Lagos Mainland, Mushin, Ojo, Oshodi/Isolo, Shomolu, Surulere.

Some words may sound alike but be wrong. Any word that does not make sense
for this app was probably misheard: replace it with the closest-sounding
word that does make sense. Examples:
- "lime chart", "lined chart", "lion chart" -> "line chart"
- "Donald chart", "donor chart", "doughnut chat" -> "donut chart"
- "do not chart", "dog chart" -> "dot chart"
- "pie chat", "pi chart" -> "pie chart"; "bar chat" -> "bar chart"
- "bad agree", "bad agri" -> "Badagry"; "alley mosho" -> "Alimosho"
- "world of times", "over the times" -> "over time"
- "student teacher ratio point" -> "student-teacher ratio"
- "toy let", "toilette" -> "toilet"; "J.S.S." -> "JSS"

Then rewrite it as the clear question the user most likely meant.
Keep it short, about as long as what was heard, and do not add topics
the user did not say. Keep every chart type they asked for.
Reply with the question only.

Heard: {heard}
""")


VOICE_FIXES = [
    (r"\b(lime|lined|lion|lyme)\s+chart", "line chart"),
    (r"\b(donald|donor|doughnut|donuts?|donat)\s+cha?rt", "donut chart"),
    (r"\b(do not|dog|dots?)\s+chart", "dot chart"),
    (r"\b(pie|pi|pye)\s+chat\b", "pie chart"),
    (r"\bpi\s+chart", "pie chart"),
    (r"\b(bar|ba)\s+chat\b", "bar chart"),
    (r"\bbad\s*agr(ee|i|y)\b", "Badagry"),
    (r"\bworld of times?\b", "over time"),
]


def quick_fix(text):
    for pattern, replacement in VOICE_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def understand_voice(heard, chat_model):
    heard = quick_fix(heard) if heard else heard
    try:
        fixed = run_quick(voice_fix_prompt, {"heard": heard}, chat_model)
        fixed = fixed.strip().strip('"').strip()
        if len(fixed) > 3 * len(heard) + 40:   # it rambled: keep what was heard
            fixed = heard
        return fixed or heard
    except Exception:
        return heard


# --------------------------------------------------
# 8. 3D hero (floating books, pencils, caps and data bars)
# --------------------------------------------------

HERO_3D = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body {margin: 0; height: 100%; overflow: hidden; font-family: 'Inter', sans-serif;}
.wrap {
position: relative; height: 100%; border-radius: 28px; overflow: hidden;
background:
radial-gradient(circle at 78% 30%, rgba(250,204,21,0.28), transparent 40%),
radial-gradient(circle at 92% 85%, rgba(16,185,129,0.22), transparent 38%),
linear-gradient(135deg, #0F1E4A 0%, #1E3A8A 55%, #2563EB 100%);
}
.grid {
position: absolute; inset: 0;
background-image:
linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
background-size: 40px 40px;
}
canvas {position: absolute; top: 0; bottom: 0; right: 0;}   /* left is set by the script */
.text {
position: absolute; left: 6%; top: 50%; transform: translateY(-50%);
max-width: 520px; color: #FFFFFF; z-index: 2;
}
.badge {
display: inline-block; background: rgba(255,255,255,0.12);
border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
padding: 6px 14px; font-size: 13px; letter-spacing: 0.04em; color: #E0E9FF;
backdrop-filter: blur(6px);
}
h1 {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 46px; line-height: 1.08; margin: 18px 0 14px 0; letter-spacing: -0.01em;
}
h1 span {color: #FACC15;}
p {font-size: 17px; line-height: 1.6; color: #D6E2FF; margin: 0;}
.row {display: flex; gap: 22px; margin-top: 26px;}
.mini b {display: block; font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; font-size: 26px; color: #FFFFFF;}
.mini span {font-size: 12.5px; color: #B8C8F0;}
@media (max-width: 760px) {
h1 {font-size: 32px;} .text {left: 7%; right: 7%;}
}
</style>
</head>
<body>
<div class="wrap">
<div class="grid"></div>
<canvas id="scene"></canvas>
<div class="text">
<div class="badge">Education Resource-Gap Intelligence Tool</div>
<h1>Every child deserves a <span>teacher</span>, a classroom and a clean toilet.</h1>
<p>ERGIT shows which Lagos LGAs are furthest from the national school standards, and helps parents compare public schools before they enrol a child.</p>
<div class="row">
<div class="mini"><b>20</b><span>LGAs ranked</span></div>
<div class="mini"><b>1,689</b><span>public schools</span></div>
<div class="mini"><b>3</b><span>UBE standards</span></div>
</div>
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
if (!window.THREE) return;
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera.position.set(0, 0.6, 12);

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(5, 8, 6); scene.add(sun);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.45);
fill.position.set(-4, -2, 6); scene.add(fill);
const warm = new THREE.PointLight(0xfacc15, 0.8, 30);
warm.position.set(6, 2, 5); scene.add(warm);

const group = new THREE.Group();
scene.add(group);
const mat = c => new THREE.MeshStandardMaterial({color: c, roughness: 0.45, metalness: 0.08});
const pages = mat(0xFFFDF5);
const floaters = [];

function book(color, w, h, d) {
  const g = new THREE.Group();
  const cover = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(color));
  const pg = new THREE.Mesh(new THREE.BoxGeometry(w * 0.94, h * 0.78, d * 1.01), pages);
  pg.position.x = 0.03;
  g.add(cover); g.add(pg);
  return g;
}

const shiny = (c, m, r) => new THREE.MeshStandardMaterial({color: c, metalness: m, roughness: r});

// A real, sharpened hexagonal pencil: painted body, scalloped paint edge,
// bare wood cone, shiny graphite point, ridged metal ferrule and an eraser.
function pencil(color) {
  const g = new THREE.Group();
  const R = 0.095, L = 2.3;                                  // slim, like a real pencil
  const paint = new THREE.MeshStandardMaterial({color: color, roughness: 0.35, metalness: 0.05, flatShading: true});
  const body = new THREE.Mesh(new THREE.CylinderGeometry(R, R, L, 6), paint);
  // the painted corners run a little way down the cone (the wavy edge you see on real pencils)
  const scallop = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 0.8, 0.07, 6), paint);
  scallop.position.y = -L / 2 - 0.035;
  // sharpened wood: a smooth cone of pale cedar
  const woodLen = 0.42, leadR = 0.03;
  const wood = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.8, leadR, woodLen, 32),
                              new THREE.MeshStandardMaterial({color: 0xE9C29A, roughness: 0.85}));
  wood.position.y = -L / 2 - 0.07 - woodLen / 2;
  // graphite point, continuing the same slope, slightly shiny
  const lead = new THREE.Mesh(new THREE.ConeGeometry(leadR, 0.14, 32), shiny(0x2B2F36, 0.6, 0.3));
  lead.rotation.x = Math.PI;
  lead.position.y = -L / 2 - 0.07 - woodLen - 0.07;
  // ridged metal ferrule
  const ferrule = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.04, R * 1.04, 0.22, 32), shiny(0xC9CED8, 0.9, 0.25));
  ferrule.position.y = L / 2 + 0.11;
  for (let k = 0; k < 3; k++) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(R * 1.05, 0.008, 6, 32), shiny(0xA7AEBB, 0.9, 0.3));
    ring.rotation.x = Math.PI / 2; ring.position.y = L / 2 + 0.04 + k * 0.07;
    g.add(ring);
  }
  // pink eraser with a rounded end
  const eraser = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 0.16, 32),
                                new THREE.MeshStandardMaterial({color: 0xF4A3A8, roughness: 0.9}));
  eraser.position.y = L / 2 + 0.3;
  const eraserTop = new THREE.Mesh(new THREE.SphereGeometry(R, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2),
                                   new THREE.MeshStandardMaterial({color: 0xF4A3A8, roughness: 0.9}));
  eraserTop.scale.y = 0.35; eraserTop.position.y = L / 2 + 0.38;
  g.add(body, scallop, wood, lead, ferrule, eraser, eraserTop);
  return g;
}

// A ballpoint pen: glossy barrel, rubber grip, metal cone and ball, clip on the cap end.
function pen(color) {
  const g = new THREE.Group();
  const R = 0.085;
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 1.9, 32), shiny(color, 0.25, 0.2));
  const grip = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.06, R * 0.95, 0.5, 32),
                              new THREE.MeshStandardMaterial({color: 0x1F2430, roughness: 0.95}));
  grip.position.y = -1.2;
  const cone = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.95, 0.018, 0.26, 32), shiny(0xD5DAE3, 0.95, 0.2));
  cone.position.y = -1.58;
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 16), shiny(0xB8BEC9, 1, 0.15));
  ball.position.y = -1.72;
  const top = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.08, R * 1.08, 0.5, 32), shiny(color, 0.25, 0.2));
  top.position.y = 1.0;
  const button = new THREE.Mesh(new THREE.SphereGeometry(R * 1.05, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2),
                                shiny(0xD5DAE3, 0.95, 0.2));
  button.scale.y = 0.6; button.position.y = 1.25;
  const band = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.1, R * 1.1, 0.05, 32), shiny(0xD5DAE3, 0.95, 0.2));
  band.position.y = 0.73;
  const clip = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.75, 0.035), shiny(0xD5DAE3, 0.95, 0.2));
  clip.position.set(0, 0.85, R * 1.35);
  const clipEnd = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), shiny(0xD5DAE3, 0.95, 0.2));
  clipEnd.position.set(0, 0.48, R * 1.25);
  g.add(barrel, grip, cone, ball, top, button, band, clip, clipEnd);
  return g;
}

// A wooden ruler with tick marks
function ruler() {
  const g = new THREE.Group();
  const board = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.04, 0.42),
                               new THREE.MeshStandardMaterial({color: 0xE2A93B, roughness: 0.6}));
  g.add(board);
  const tickMat = new THREE.MeshStandardMaterial({color: 0x3A2F1C});
  for (let k = 0; k <= 24; k++) {
    const long = k % 4 === 0;
    const t = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.01, long ? 0.16 : 0.08), tickMat);
    t.position.set(-1.2 + k * 0.1, 0.026, -0.21 + (long ? 0.08 : 0.04));
    g.add(t);
  }
  return g;
}

function cap() {
  const g = new THREE.Group();
  const board = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.08, 1.7), mat(0x111827));
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.62, 0.45, 24), mat(0x111827));
  base.position.y = -0.27;
  const button = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.05, 12), mat(0xFACC15));
  button.position.y = 0.06;
  const cord = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.85), mat(0xFACC15));
  cord.position.set(0.42, 0.06, 0.42); cord.rotation.y = Math.PI / 4;
  const tassel = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.08, 0.5, 8), mat(0xFACC15));
  tassel.position.set(0.82, -0.2, 0.82);
  g.add(board, base, button, cord, tassel);
  return g;
}

// Stack of books (right side)
const stack = new THREE.Group();
const s1 = book(0x2563EB, 2.6, 0.42, 1.8); s1.position.y = 0;
const s2 = book(0xFACC15, 2.3, 0.38, 1.6); s2.position.y = 0.42; s2.rotation.y = 0.25;
const s3 = book(0x10B981, 2.0, 0.34, 1.45); s3.position.y = 0.8; s3.rotation.y = -0.2;
stack.add(s1, s2, s3);
stack.position.set(4.2, -1.9, 0); stack.rotation.set(0.25, -0.5, 0);
group.add(stack);

// Rising data bars standing on the stack
const bars = [];
const barColors = [0x93C5FD, 0x60A5FA, 0x3B82F6, 0xFACC15];
for (let i = 0; i < 4; i++) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(0.32, 1, 0.32), mat(barColors[i]));
  b.position.set(-0.6 + i * 0.42, 1.0, 0);
  bars.push(b); stack.add(b);
}

// Floating objects
const cap1 = cap(); cap1.position.set(6.6, 2.4, -1); group.add(cap1);
floaters.push({o: cap1, base: 2.4, speed: 0.9, spin: 0.35});

const p1 = pencil(0xFACC15); p1.position.set(3.0, 1.9, 1); p1.rotation.set(0.3, 0, -0.75); group.add(p1);
floaters.push({o: p1, base: 1.9, speed: 1.2, spin: 0.0});

const p2 = pencil(0x2563EB); p2.position.set(8.1, -0.5, 0.5); p2.rotation.set(-0.4, 0, 0.6); group.add(p2);
floaters.push({o: p2, base: -0.5, speed: 1.0, spin: 0.0});

const fb = book(0xF97066, 1.4, 0.26, 1.0); fb.position.set(3.0, -0.3, 2); fb.rotation.set(0.6, 0.4, 0.3); group.add(fb);
floaters.push({o: fb, base: -0.3, speed: 0.8, spin: 0.4});

const fb2 = book(0x8B5CF6, 1.2, 0.22, 0.9); fb2.position.set(5.6, 0.9, 2.2); fb2.rotation.set(-0.3, 0.9, -0.2); group.add(fb2);
floaters.push({o: fb2, base: 0.9, speed: 1.1, spin: -0.3});

const pn = pen(0xDC2626); pn.scale.setScalar(1.25); pn.position.set(3.4, -1.85, 1.8); pn.rotation.set(0.35, 0.2, 1.1); group.add(pn);
floaters.push({o: pn, base: -1.85, speed: 0.95, spin: 0.0});

const rl = ruler(); rl.position.set(6.4, -2.0, 0.8); rl.rotation.set(1.0, 0.2, 0.35); group.add(rl);
floaters.push({o: rl, base: -2.0, speed: 0.7, spin: 0.0});

const fb3 = book(0x10B981, 1.0, 0.2, 0.75); fb3.position.set(4.7, 3.0, -1.6); fb3.rotation.set(0.5, -0.6, 0.2); group.add(fb3);
floaters.push({o: fb3, base: 3.0, speed: 1.05, spin: 0.25});

// Small glowing particles (ideas)
const pts = new THREE.BufferGeometry();
const n = 140, arr = new Float32Array(n * 3);
for (let i = 0; i < n; i++) {
  arr[i * 3] = (Math.random() - 0.2) * 16;
  arr[i * 3 + 1] = (Math.random() - 0.5) * 8;
  arr[i * 3 + 2] = (Math.random() - 0.5) * 6;
}
pts.setAttribute('position', new THREE.BufferAttribute(arr, 3));
const dots = new THREE.Points(pts, new THREE.PointsMaterial({color: 0xFFFFFF, size: 0.05, transparent: true, opacity: 0.6}));
scene.add(dots);

let mx = 0, my = 0;
window.addEventListener('mousemove', e => {
  mx = (e.clientX / window.innerWidth - 0.5);
  my = (e.clientY / window.innerHeight - 0.5);
});

// The 3D objects live only in the empty space to the RIGHT of the text,
// so they never cover the words. On a phone there is no room: hide them.
const textBox = document.querySelector('.text');
function resize() {
  const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
  const textRight = textBox.getBoundingClientRect().right
                    - canvas.parentElement.getBoundingClientRect().left;
  const free = w - textRight - 24;
  if (w < 760 || free < 220) { canvas.style.display = 'none'; return; }
  canvas.style.display = 'block';
  canvas.style.left = (w - free) + 'px';
  canvas.style.width = free + 'px';
  renderer.setSize(free, h, false);
  camera.aspect = free / h;
  // centre on the objects and step back far enough to fit them all
  camera.position.set(5.6, 0.4, Math.max(12, 4.6 / (Math.tan(20 * Math.PI / 180) * camera.aspect)));
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

const clock = new THREE.Clock();
function animate() {
  const t = clock.getElapsedTime();
  floaters.forEach((f, i) => {
    f.o.position.y = f.base + Math.sin(t * f.speed + i) * 0.25;
    f.o.rotation.y += f.spin * 0.01;
  });
  bars.forEach((b, i) => {
    const h = 0.9 + i * 0.45 + Math.sin(t * 1.5 + i) * 0.18;
    b.scale.y = h; b.position.y = 0.95 + h / 2 - 0.5;
  });
  stack.rotation.y = -0.5 + Math.sin(t * 0.3) * 0.12;
  dots.rotation.y = t * 0.02;
  group.rotation.y += ((mx * 0.25) - group.rotation.y) * 0.05;
  group.rotation.x += ((my * 0.12) - group.rotation.x) * 0.05;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------
# 9. Sidebar
# --------------------------------------------------

# --------------------------------------------------
# 9b. Chat history (kept for this visit)
# --------------------------------------------------

def chat_title(chat):
    turns = chat["turns"]
    if not turns:
        return "New chat"
    first = turns[0]["versions"][0]["question"]
    return first if len(first) <= 34 else first[:32].rstrip() + "…"


def remember_current_chat():
    """Add the open conversation to the history list (once)."""
    if st.session_state.turns and st.session_state.current_chat_id is None:
        st.session_state.next_chat_id += 1
        chat_id = st.session_state.next_chat_id
        # The history shares the same list, so new questions are saved automatically
        st.session_state.chats.append({"id": chat_id, "turns": st.session_state.turns})
        st.session_state.current_chat_id = chat_id


def start_new_chat():
    remember_current_chat()
    st.session_state.turns = []
    st.session_state.current_chat_id = None
    st.session_state.editing = None
    st.session_state.voice_draft = None
    st.session_state.greeted = False
    st.session_state.page = "chat"


def open_chat(chat_id):
    remember_current_chat()
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            st.session_state.turns = chat["turns"]
            st.session_state.current_chat_id = chat_id
    st.session_state.editing = None
    st.session_state.greeted = True
    st.session_state.page = "chat"


def render_sidebar():
    with st.sidebar:
        html(f"""
<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
{LOGO_SVG}
<div>
<div class="brand-name" style="font-size:1.25rem;">ERGIT</div>
<div class="brand-sub">Lagos State, Nigeria</div>
</div>
</div>
<p style="font-size:0.88rem; color:#5B6680; line-height:1.55;">
A data tool that ranks Lagos LGAs by how urgently their public schools
need teachers, classrooms and toilets, and helps parents compare
public schools in an area.
</p>
<div class="eyebrow" style="margin-top:1.2rem;">UBE minimum standards</div>
<div class="side-std">{ic("teacher", "#2563EB")} <span><b>35</b> students per teacher</span></div>
<div class="side-std">{ic("classroom", "#2563EB")} <span><b>35</b> students per classroom</span></div>
<div class="side-std">{ic("toilet", "#2563EB")} <span><b>30</b> students per toilet</span></div>
<p style="font-size:0.8rem; color:#5B6680; margin-top:1.2rem; line-height:1.5;">
Answers come only from the project's documents. If the data does not
cover a question, ERGIT will say so.
</p>
""")

        st.write("")
        if st.button("New chat", icon=":material/add_comment:", key="new_chat",
                     width="stretch"):
            start_new_chat()
            st.rerun()

        # Previous chats from this visit, newest first
        if st.session_state.chats:
            html('<div class="eyebrow" style="margin-top:1.2rem;">Recent chats</div>')
            with st.container(key="history"):
                for chat in reversed(st.session_state.chats):
                    is_current = (chat["id"] == st.session_state.current_chat_id
                                  and st.session_state.page == "chat")
                    st.button(
                        chat_title(chat),
                        key=f"chat_{chat['id']}",
                        icon=":material/chat_bubble:" if is_current else ":material/chat_bubble_outline:",
                        type="primary" if is_current else "tertiary",
                        width="stretch",
                        on_click=open_chat, args=(chat["id"],)
                    )


# --------------------------------------------------
# 10. Page 1: Home (what ERGIT does)
# --------------------------------------------------

def render_home():
    topbar()

    # Number of schools, read from the school file so it always matches it
    school_file = get_schools()
    n_schools = len(school_file) if school_file is not None else 1689
    hero = HERO_3D.replace("<b>1,689</b>", f"<b>{n_schools:,}</b>")

    if hasattr(st, "iframe"):
        st.iframe(hero, height=470)
    else:
        components.html(hero, height=470)

    html('<div class="gap"></div>')

    # Two ways to use ERGIT
    html("""
<div class="eyebrow">Two ways to use ERGIT</div>
<div class="section-title">For the people who <span class="hl">plan</span>, and the families who <span class="hl">choose</span></div>
<p class="lead">One assistant, two kinds of question. Ask about an area to see where Lagos needs
support, or ask about schools to find a place for your child.</p>
<div style="height:1rem;"></div>
""")
    w1, w2 = st.columns(2)
    with w1:
        html(f"""
<div class="way">
<div class="way-top"><div class="feature-icon" style="background:#DBEAFE;">{ic("bars", "#2563EB", 24)}</div>
<div><div class="way-who">For government, NGOs and donors</div>
<div class="way-title">Compare the 20 LGAs</div></div></div>
<div class="way-text">See which LGAs are furthest from the UBE standards for teachers,
classrooms and toilets, how levels compare, and how things changed over time.</div>
<div class="way-ask">“Which LGAs need teachers most?”</div>
<div class="way-ask">“Compare Alimosho and Ikeja on toilets”</div>
<div class="way-ask">“Teacher ratio trend 2016/17 to 2019/20”</div>
<div class="way-data">{ic("file", "#5B6680", 13)} DNEMIS School Census 2024/25 · Lagos Bureau of Statistics</div>
</div>
""")
    with w2:
        html(f"""
<div class="way way-parent">
<div class="way-top"><div class="feature-icon" style="background:#FEF3C7;">{ic("cap", "#B45309", 24)}</div>
<div><div class="way-who">For parents and families</div>
<div class="way-title">Find a public school</div></div></div>
<div class="way-text">Moving to a new area? Compare the public schools there on class size,
teachers and toilets, see them on a map, and get a shortlist to visit.</div>
<div class="way-ask">“Public JSS schools in Kosofe with the smallest classes”</div>
<div class="way-ask">“Primary schools near Ikotun”</div>
<div class="way-ask">“How does Mende Senior High compare with the standards?”</div>
<div class="way-data">{ic("file", "#5B6680", 13)} NMIS school survey 2012–2014 · {n_schools:,} public schools</div>
</div>
""")

    html('<div class="gap"></div>')

    # Key findings
    html("""
<div class="eyebrow">Key findings · 2024/25</div>
<div class="section-title">Facilities are the <span class="hl">biggest gap</span> in Lagos public schools</div>
<p class="lead">Each LGA was compared with the Universal Basic Education (UBE) standards.
Here is how many of the 20 LGAs fall short.</p>
<div style="height:1rem;"></div>
""")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "toilet", "#FEF3C7", "#B45309", "17", "/20", "LGAs lack toilets", 85, "#F59E0B", "Standard: 30 per toilet"),
        (c2, "classroom", "#FEE2E2", "#DC2626", "13", "/20", "LGAs lack classrooms", 65, "#F97066", "Standard: 35 per classroom"),
        (c3, "teacher", "#DBEAFE", "#2563EB", "8", "/20", "LGAs lack teachers", 40, "#2563EB", "Standard: 35 per teacher"),
        (c4, "layers", "#D1FAE5", "#059669", "JSS", "", "Most pressured level", 100, "#10B981", "70.6 students per classroom"),
    ]
    for col, icon, bg, fg, num, small, label, width, bar, note in cards:
        with col:
            html(f"""
<div class="stat">
<div class="stat-top"><div class="stat-icon" style="background:{bg};">{ic(icon, fg, 20)}</div></div>
<div class="stat-num">{num}<small>{small}</small></div>
<div class="stat-label">{label}</div>
<div class="bar"><span style="width:{width}%; background:{bar};"></span></div>
<div class="stat-note">{note}</div>
</div>
""")

    html("""
<p style="font-size:0.8rem; color:#5B6680; margin-top:0.8rem;">
Public Primary, JSS and SSS schools combined for each LGA. Sources: DNEMIS Annual School Census 2024/25; Lagos Bureau of Statistics.
</p>
<div style="height:1.6rem;"></div>
""")

    # School findings (worked out from the school file, so they always match it)
    schools = get_schools()
    if schools is not None:
        total = len(schools)
        full = int(schools[["Students_per_Teacher", "Students_per_Classroom",
                            "Students_per_Toilet"]].notna().all(axis=1).sum())
        within = int((schools["Overall_Label"] == "Within UBE benchmark").sum())
        high = int((schools["Overall_Label"] == "High-pressure facilities – visit before enrolment").sum())
        html("""
<div class="eyebrow">Public schools · 2012–2014 survey</div>
<p class="lead" style="margin-bottom:1rem;">Each public school was compared with the same UBE standards,
so parents can see how crowded a school was before they visit.</p>
""")
        s1, s2, s3, s4 = st.columns(4)
        school_cards = [
            (s1, "classroom", "#DBEAFE", "#2563EB", f"{total:,}", "", "Public schools", 100, "#2563EB", "Across all 20 LGAs"),
            (s2, "file", "#EDE9FE", "#7C3AED", f"{full:,}", "", "With full figures", round(full / total * 100), "#8B5CF6", "Teachers, classrooms and toilets"),
            (s3, "target", "#D1FAE5", "#059669", f"{within:,}", "", "Within all 3 standards", round(within / max(full, 1) * 100), "#10B981", f"{within / max(full, 1):.0%} of schools with full figures"),
            (s4, "alert", "#FEE2E2", "#DC2626", f"{high:,}", "", "High-pressure schools", round(high / max(full, 1) * 100), "#F97066", f"{high / max(full, 1):.0%} of schools with full figures"),
        ]
        for col, icon, bg, fg, num, small, label, width, bar, note in school_cards:
            with col:
                html(f"""
<div class="stat">
<div class="stat-top"><div class="stat-icon" style="background:{bg};">{ic(icon, fg, 20)}</div></div>
<div class="stat-num">{num}<small>{small}</small></div>
<div class="stat-label">{label}</div>
<div class="bar"><span style="width:{width}%; background:{bar};"></span></div>
<div class="stat-note">{note}</div>
</div>
""")
        html("""
<p style="font-size:0.8rem; color:#5B6680; margin-top:0.8rem;">
Source: NMIS school survey 2012–2014 (Federal Government). School figures are older than the LGA figures and may have changed, so ERGIT always shows the survey year.
</p>
""")

    html('<div class="gap"></div>')

    # What ERGIT does
    html("""
<div class="eyebrow">What ERGIT does</div>
<div class="section-title">From scattered reports to <span class="hl">clear answers</span></div>
<p class="lead">Nigeria's education data sits in many separate reports. ERGIT brings it together,
so decision-makers can see where support should go first, and parents can choose a school with facts.</p>
<div style="height:1rem;"></div>
""")

    f1, f2, f3 = st.columns(3)
    with f1:
        html(f"""
<div class="feature">
<div class="feature-icon" style="background:#DBEAFE;">{ic("bars", "#2563EB", 24)}</div>
<div class="feature-title">Ranks every LGA</div>
<div class="feature-text">Compares all 20 Lagos LGAs with the national standards for teachers, classrooms and toilets, and ranks them by need.</div>
</div>
""")
    with f2:
        html(f"""
<div class="feature">
<div class="feature-icon" style="background:#FEF3C7;">{ic("cap", "#B45309", 24)}</div>
<div class="feature-title">Compares public schools</div>
<div class="feature-text">Lists the public schools in an area or near a place, compares them with the standards, labels each one fairly and shows them on a map.</div>
</div>
""")
    with f3:
        html(f"""
<div class="feature">
<div class="feature-icon" style="background:#D1FAE5;">{ic("chat", "#059669", 24)}</div>
<div class="feature-title">Answers with live dashboards</div>
<div class="feature-text">Type or speak a question. ERGIT replies in plain English, shows its source, and draws the chart that fits the question best.</div>
</div>
""")

    html('<div class="gap"></div>')

    # How it works
    html("""
<div class="eyebrow">How it works</div>
<div class="section-title">Four steps, <span class="hl">one clear picture</span></div>
<div style="height:1rem;"></div>
<div class="steps">
<div class="step"><div class="step-num">1</div><div class="step-title">Collect</div>
<div class="step-text">LGA data from DNEMIS and the Lagos Bureau of Statistics; school data from the NMIS survey.</div></div>
<div class="step"><div class="step-num">2</div><div class="step-title">Clean &amp; compare</div>
<div class="step-text">Checked, cleaned and measured against the UBE standards.</div></div>
<div class="step"><div class="step-num">3</div><div class="step-title">Rank &amp; label</div>
<div class="step-text">LGAs ranked by need; each school labelled against the standards.</div></div>
<div class="step"><div class="step-num">4</div><div class="step-title">Explain &amp; chart</div>
<div class="step-text">The AI decides: LGA question or school search, then answers with a live dashboard.</div></div>
</div>
<div class="gap"></div>
""")

    # Call to action
    left, right = st.columns([3, 1.3])
    with left:
        html("""
<div class="cta">
<h3>Ready to explore the data?</h3>
<p>Ask which LGAs need teachers, classrooms or toilets most, or find a public school for your child.</p>
<p class="cta-for">Useful for Lagos SUBEB, the Ministry of Basic &amp; Secondary Education, UBEC, NGOs, donors and researchers, and for parents choosing a public school.</p>
</div>
""")
    with right:
        html('<div style="height:3.4rem;"></div>')
        if st.button("Start asking ERGIT  →", type="primary", width="stretch"):
            st.session_state.page = "chat"
            st.rerun()

    html("""
<p class="footer-note">
ERGIT is a student data science capstone project. Figures come from official public sources.
</p>
""")


# --------------------------------------------------
# 11. Page 2: Chat with ERGIT
# --------------------------------------------------

SAMPLE_QUESTIONS = [
    (":material/leaderboard:", "Top 5 LGAs with the biggest resource gaps"),
    (":material/person:", "Which LGAs need teachers most?"),
    (":material/water_drop:", "How does Badagry compare with the toilet standard?"),
    (":material/show_chart:", "Teacher ratio trend 2016/17 to 2019/20"),
    (":material/school:", "Public JSS schools in Kosofe with the smallest classes"),
    (":material/location_on:", "Primary schools near Ikotun"),
]


def render_chat():
    topbar()

    if st.button("Back to home", icon=":material/arrow_back:"):
        st.session_state.page = "home"
        st.rerun()

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
        vector_db, chat_model = load_rag_components(api_key, embedding_api_key)
    except Exception as e:
        html(f'<div class="ergit-error"><b>Could not initialize the RAG system:</b> {e}</div>')
        st.stop()

    # The knowledge base must sit next to this file, in a folder called chroma_store
    try:
        pieces = vector_db._collection.count()
    except Exception:
        pieces = 0
    if pieces == 0:
        html(f"""
<div class="ergit-error">
<b>The knowledge base was not found.</b><br>
ERGIT looked for the folder <code>chroma_store</code> here:<br>
<code>{escape(os.path.join(BASE_DIR, "chroma_store"))}</code><br><br>
Put the <code>chroma_store</code> folder (with everything inside it) in the
<b>same folder as this app file</b> on GitHub, then reboot the app.
</div>
""")
        st.stop()

    # Find the document names actually stored (they change if files are moved)
    set_source_filters(vector_db, pieces)

    search_kwargs = {"k": 8}
    if SOURCE_FILTERS["pdf"]:
        search_kwargs["filter"] = SOURCE_FILTERS["pdf"]
    general_retriever = vector_db.as_retriever(search_kwargs=search_kwargs)

    # ERGIT says hello first, like a chatbot (typed out the first time only)
    with st.chat_message("assistant", avatar=":material/school:"):
        if not st.session_state.greeted:
            bubble = st.empty()
            bubble.markdown(
                '<div class="typing"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True
            )
            time.sleep(1.3)          # "ERGIT is typing..."
            bubble.write_stream(type_out(GREETING))
            st.session_state.greeted = True
        else:
            st.markdown(GREETING)

        # Suggested questions: small chips under the greeting, gone after
        # the first question
        if not st.session_state.turns:
            html('<p class="suggest-label">You could ask:</p>')
            with st.container(horizontal=True, gap="small", key="suggestions"):
                for i, (icon, sample) in enumerate(SAMPLE_QUESTIONS):
                    if st.button(sample, icon=icon, key=f"sample_{i}"):
                        st.session_state.pending_question = sample
                        st.rerun()

    # Each turn = one question with all its versions (retries and edits)
    for t_index, turn in enumerate(st.session_state.turns):
        render_turn(t_index, turn, vector_db, chat_model, general_retriever)

    # One input bar: type, or tap the mic and watch your words appear as you speak.
    # Nothing is sent until you press the send arrow, so you can fix any word first.
    voice_question = None
    voice_heard = None
    submission = st.chat_input(
        "Ask about an LGA or find a school...",
        key="ergit_chat"
    )
    live_dictation()

    typed_question = None
    if submission:
        text = submission if isinstance(submission, str) else (submission.text or "")
        typed_question = text.strip() or None

    question = (
        typed_question
        or voice_question
        or st.session_state.pop("pending_question", None)
    )

    if question:
        by_voice = bool(voice_question) and not typed_question
        version = new_version(question, by_voice, voice_heard if by_voice else None)
        version["needs_fix"] = False   # the user has already checked the words
        st.session_state.turns.append({"versions": [version], "current": 0})
        remember_current_chat()
        st.rerun()  # the new turn is answered in place on the next run


# --------------------------------------------------
# 6d. Live voice typing in the chat box
#
# Uses the browser's own speech recognition (Chrome, Edge, Android, iPhone
# Safari). As the user speaks, the words appear in the typing box; they can
# fix anything, then press send. Nothing is sent until they do.
# --------------------------------------------------

LIVE_DICTATION = """
<script>
(function () {
  const P = window.parent, doc = P.document;
  const SR = P.SpeechRecognition || P.webkitSpeechRecognition;
  const owner = "o" + Math.random().toString(36).slice(2);

  const MIC = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/></svg>';
  const STOP = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

  if (!doc.getElementById("ergit-mic-style")) {
    const css = doc.createElement("style");
    css.id = "ergit-mic-style";
    css.textContent = `
      #ergit-mic {border:none; background:transparent; color:#5B6680; width:34px; height:34px;
        border-radius:10px; display:flex; align-items:center; justify-content:center;
        cursor:pointer; margin-right:6px; flex-shrink:0; align-self:flex-end; margin-bottom:4px;}
      #ergit-mic:hover {background:#EEF4FF; color:#2563EB;}
      #ergit-mic.on {background:#F97066; color:#fff; animation:ergitPulse 1.3s infinite;}
      @keyframes ergitPulse {0%{box-shadow:0 0 0 0 rgba(249,112,102,.55);}
        70%{box-shadow:0 0 0 10px rgba(249,112,102,0);} 100%{box-shadow:0 0 0 0 rgba(249,112,102,0);}}
      #ergit-mic-hint {position:absolute; left:16px; top:-24px; font:600 12px Inter,sans-serif;
        color:#F97066; background:#fff; padding:2px 8px; border-radius:8px;
        box-shadow:0 2px 8px rgba(15,30,74,.12);}`;
    doc.head.appendChild(css);
  }

  let rec = null, listening = false, base = "";

  // A recording left from an earlier page load: stop it and clear its note
  if (P.__ergitRec) { try { P.__ergitRec.abort(); } catch (e) {} P.__ergitRec = null; }
  const oldHint = doc.getElementById("ergit-mic-hint"); if (oldHint) oldHint.remove();

  function box() { return doc.querySelector('[data-testid="stChatInput"]'); }
  function area() { const b = box(); return b ? b.querySelector("textarea") : null; }

  function setText(v) {
    const ta = area(); if (!ta) return;
    const setter = Object.getOwnPropertyDescriptor(P.HTMLTextAreaElement.prototype, "value").set;
    setter.call(ta, v);
    ta.dispatchEvent(new P.Event("input", { bubbles: true }));
  }

  function hint(show) {
    const b = box(); if (!b) return;
    let h = doc.getElementById("ergit-mic-hint");
    if (show && !h) {
      h = doc.createElement("div"); h.id = "ergit-mic-hint";
      h.textContent = "Listening… tap the mic again when you finish";
      b.style.position = "relative"; b.appendChild(h);
    }
    if (!show && h) h.remove();
  }

  function stop() { if (rec) { try { rec.stop(); } catch (e) {} } }

  function start(lang) {
    rec = new SR();
    rec.lang = lang;
    rec.interimResults = true;     // show words while the user is still speaking
    rec.continuous = true;         // keep listening through pauses, until the user taps again
    const ta = area();
    base = ta && ta.value.trim() ? ta.value.trim() + " " : "";
    rec.onresult = function (e) {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      setText(base + text.replace(/\\s+/g, " ").trimStart());
    };
    rec.onerror = function (e) {
      if (e.error === "language-not-supported" && lang !== "en-US") { listening = false; start("en-US"); return; }
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        hint(true);
        const h = doc.getElementById("ergit-mic-hint");
        if (h) h.textContent = "Microphone blocked: allow it in your browser settings";
        setTimeout(function () { hint(false); }, 4000);
      }
    };
    rec.onend = function () {
      listening = false;
      const btn = doc.getElementById("ergit-mic");
      if (btn) { btn.classList.remove("on"); btn.innerHTML = MIC; btn.title = "Speak your question"; }
      hint(false);
      const t = area(); if (t) t.focus();
    };
    P.__ergitRec = rec;
    rec.start();
    listening = true;
    const btn = doc.getElementById("ergit-mic");
    if (btn) { btn.classList.add("on"); btn.innerHTML = STOP; btn.title = "Stop"; }
    hint(true);
  }

  function setup() {
    const b = box(); if (!b || !SR) return;
    let btn = doc.getElementById("ergit-mic");
    if (btn && btn.dataset.owner === owner) return;
    if (btn) btn.remove();                       // a button left from an earlier page load
    const send = b.querySelector('[data-testid="stChatInputSubmitButton"]');
    btn = doc.createElement("button");
    btn.id = "ergit-mic"; btn.type = "button"; btn.dataset.owner = owner;
    btn.title = "Speak your question"; btn.setAttribute("aria-label", "Speak your question");
    btn.innerHTML = MIC;
    btn.addEventListener("click", function (ev) {
      ev.preventDefault();
      if (listening) stop(); else start("en-NG");
    });
    if (send && send.parentElement) send.parentElement.insertBefore(btn, send);
    else b.appendChild(btn);

    // Sending the question stops the listening, so no words land in the empty box
    const ta = area();
    if (ta) ta.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey && listening) stop();
    });
    if (send) send.addEventListener("click", function () { if (listening) stop(); });
  }

  setup();
  setInterval(setup, 800);
})();
</script>
"""


def live_dictation():
    """Add the live voice-typing mic to the chat box."""
    with st.container(key="dictation"):
        if hasattr(st, "iframe"):
            st.iframe(LIVE_DICTATION, height=1)
        else:
            components.html(LIVE_DICTATION, height=1)


# --------------------------------------------------
# 11b. Retry, edit and versions for each question
# --------------------------------------------------

GREETING = ("Hi, welcome to **ERGIT** 👋 How can I help you today? "
            "Ask me about Lagos LGAs, or find a public school for your child.")


def type_out(text):
    """Show the greeting word by word, like someone typing."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.09)


def new_version(question, voice=False, heard=None):
    return {"question": question, "voice": voice, "heard": heard, "answer": None}


def retry_turn(t_index):
    turn = st.session_state.turns[t_index]
    old = turn["versions"][turn["current"]]
    turn["versions"].append(new_version(old["question"], old["voice"], old["heard"]))
    turn["current"] = len(turn["versions"]) - 1


def start_edit(t_index):
    st.session_state.editing = t_index


def move_version(t_index, step):
    turn = st.session_state.turns[t_index]
    turn["current"] = max(0, min(len(turn["versions"]) - 1, turn["current"] + step))


def render_turn(t_index, turn, vector_db, chat_model, general_retriever):
    version = turn["versions"][turn["current"]]
    total = len(turn["versions"])

    # ---- the question (or the edit box) ----
    if st.session_state.editing == t_index:
        with st.chat_message("user", avatar="🧑🏾"):
            with st.form(f"edit_form_{t_index}", border=False):
                new_text = st.text_area(
                    "Edit your question", value=version["question"],
                    label_visibility="collapsed", height=90
                )
                b1, b2, _ = st.columns([1, 1, 3])
                with b1:
                    send = st.form_submit_button("Send", icon=":material/send:", type="primary", width="stretch")
                with b2:
                    cancel = st.form_submit_button("Cancel", width="stretch")
        if send and new_text.strip():
            turn["versions"].append(new_version(new_text.strip()))
            turn["current"] = len(turn["versions"]) - 1
            st.session_state.editing = None
            st.rerun()
        if cancel:
            st.session_state.editing = None
            st.rerun()
    else:
        with st.chat_message("user", avatar="🧑🏾"):
            if version["voice"]:
                html(f'<p>{ic("mic", "#FFFFFF", 16)} {escape(version["question"])}</p>')
            else:
                st.markdown(version["question"])
            heard = version.get("heard")
            if heard and heard.strip().lower() != version["question"].strip().lower():
                html(f'<p class="heard">I heard: “{escape(heard)}”</p>')

        # ---- small action row under the question ----
        busy = version["answer"] is None
        with st.container(horizontal=True, horizontal_alignment="right",
                          gap="small", key=f"actions_{t_index}"):
            if total > 1:
                st.button("", icon=":material/chevron_left:", key=f"prev_{t_index}", type="tertiary",
                          help="Previous version", disabled=turn["current"] == 0 or busy,
                          on_click=move_version, args=(t_index, -1))
                html(f'<span class="ver-count">{turn["current"] + 1} / {total}</span>')
                st.button("", icon=":material/chevron_right:", key=f"next_{t_index}", type="tertiary",
                          help="Next version", disabled=turn["current"] == total - 1 or busy,
                          on_click=move_version, args=(t_index, 1))
            st.button("Retry", icon=":material/refresh:", key=f"retry_{t_index}", type="tertiary",
                      help="Ask this question again", disabled=busy,
                      on_click=retry_turn, args=(t_index,))
            st.button("Edit", icon=":material/edit_note:", key=f"edit_{t_index}", type="tertiary",
                      help="Change or add to this question", disabled=busy,
                      on_click=start_edit, args=(t_index,))

    # ---- the answer (worked out here, in place, the first time) ----
    with st.chat_message("assistant", avatar=":material/school:"):
        if version["answer"] is None:
            if version.get("needs_fix"):
                with st.spinner("Making sure I understood you..."):
                    version["question"] = understand_voice(version["heard"], chat_model)
                    version["needs_fix"] = False
            with st.spinner("Searching the education documents..."):
                try:
                    history = build_history(st.session_state.turns, t_index)
                    answer, source_used, chart, understood, schools, meta = answer_question(
                        version["question"], vector_db, chat_model,
                        general_retriever, st.session_state.user_name, history
                    )
                    version["understood"] = understood
                    version["answer"] = {"content": answer, "source": source_used,
                                         "chart": chart, "understood": understood,
                                         "asked": version["question"],
                                         "schools": schools, "school_meta": meta}
                    version["answer"]["followups"] = make_followups(version["answer"])
                except Exception as e:
                    print(f"RAG error: {type(e).__name__}: {e}")
                    version["answer"] = {"error": f"{type(e).__name__}: {e}"}
            st.rerun()

        if "error" in version["answer"]:
            err = version["answer"]["error"].lower()
            if "429" in err or "rate" in err:
                st.warning("ERGIT is very busy right now. Please wait a minute, then tap Retry.")
            elif "401" in err or "api key" in err or "auth" in err:
                st.error("ERGIT could not sign in to the AI service. The app owner needs to check the API key.")
            elif "connection" in err or "timeout" in err or "timed out" in err:
                st.warning("The connection dropped while answering. Check your internet, then tap Retry.")
            else:
                st.error("Something went wrong while answering. Tap Retry to try again.")
            with st.expander("Details"):
                st.code(version["answer"]["error"])
        else:
            show_assistant_message(version["answer"], t_index)
            if t_index == len(st.session_state.turns) - 1 and st.session_state.editing is None:
                show_followups(version["answer"], t_index)


# --------------------------------------------------
# 11c. Two suggested follow-up questions after every answer
#
# They link the two sides of ERGIT: an LGA answer suggests looking at the
# schools in that LGA, and a school answer suggests looking at its LGA.
# --------------------------------------------------

LEVEL_WORDS = {"Primary": "primary", "JSS": "JSS", "SSS": "SSS", "Secondary": "secondary"}
DEFAULT_FOLLOWUPS = ["Which LGAs need teachers most?",
                     "Public primary schools near Ikotun"]


def make_followups(answer):
    """Two short questions the user can tap next."""
    asked = (answer.get("understood") or answer.get("asked") or "").lower()
    ideas = []

    schools = answer.get("schools") or []
    chart = answer.get("chart") or {}
    labels = [d["label"] for d in chart.get("data", [])]
    lgas = [match_lga(x) for x in labels if match_lga(x)]
    if not lgas:
        named = next((lga for lga in LGA_LIST
                      if re.search(rf"\b{re.escape(lga.lower())}\b", asked)), None)
        lgas = [named] if named else []

    if schools:
        f = (answer.get("school_meta") or {}).get("filters", {})
        lga = f.get("lga") or schools[0]["LGA"]
        level = LEVEL_WORDS.get(f.get("level"), "")
        if len(schools) > 1:
            ideas.append(f"How does {schools[0]['School_Name']} compare with the UBE standards?")
        measure = "toilet" if f.get("sort_by") != "toilet" else "teacher"
        ideas.append(f"Which public {level + ' ' if level else ''}schools in {lga} "
                     f"have the fewest students per {measure}?")
        ideas.append(f"How does {lga} LGA compare with the UBE standards?")
    elif lgas:
        top = lgas[0]
        ideas.append(f"Public schools in {top} with the smallest classes")
        ideas.append(f"How does {top} compare with the UBE standards?")
        if len(lgas) > 1:
            ideas.append(f"Compare {lgas[0]} and {lgas[1]} on teachers, classrooms and toilets")
    elif chart:
        ideas += ["Which LGAs have the most students per classroom?",
                  "Public JSS schools in Alimosho with the smallest classes"]
    ideas += DEFAULT_FOLLOWUPS

    out = []
    for q in ideas:
        if q.lower() != asked and q not in out:
            out.append(q)
        if len(out) == 2:
            break
    return out


def show_followups(answer, t_index):
    questions = answer.get("followups") or make_followups(answer)
    if not questions:
        return
    html('<p class="suggest-label">You could also ask:</p>')
    with st.container(horizontal=True, gap="small", key=f"followups_{t_index}"):
        for i, q in enumerate(questions[:2]):
            icon = ":material/school:" if "school" in q.lower() else ":material/leaderboard:"
            if st.button(q, icon=icon, key=f"follow_{t_index}_{i}"):
                st.session_state.pending_question = q
                st.rerun()


# --------------------------------------------------
# 12. Run the app
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "turns" not in st.session_state:
    st.session_state.turns = []

if "editing" not in st.session_state:
    st.session_state.editing = None

if "voice_draft" not in st.session_state:
    st.session_state.voice_draft = None

if "chats" not in st.session_state:
    st.session_state.chats = []
    st.session_state.current_chat_id = None
    st.session_state.next_chat_id = 0

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "greeted" not in st.session_state:
    st.session_state.greeted = False

render_sidebar()

if st.session_state.page == "home":
    render_home()
else:
    render_chat()