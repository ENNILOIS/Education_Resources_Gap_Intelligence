# Education Resource-Gap Intelligence Tool

A data-driven tool that ranks the 20 Local Government Areas (LGAs) of Lagos State by how urgently their public schools need more **teachers, classrooms and toilets**, forecasts teacher shortages with machine learning, and answers plain-English questions through a Retrieval-Augmented Generation (RAG) assistant.

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Features](#features)
- [Who It Is For](#who-it-is-for)
- [Data](#data)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Limitations and Ethics](#limitations-and-ethics)
- [Future Work](#future-work)
- [Data Sources and References](#data-sources-and-references)
- [Team](#team)

## Overview

Nigeria's education data is scattered across many portals and reports, so it is hard for decision-makers to see where schools need support most. This project combines two official sources into one 360-row dataset and compares every Lagos LGA with the Universal Basic Education (UBE) minimum standards:

| Resource | UBE standard |
| --- | --- |
| Teachers | 35 students per teacher |
| Classrooms | 35 students per classroom |
| Toilets | 30 students per toilet |

The tool is designed to give the Lagos State Universal Basic Education Board (SUBEB), the Ministry of Education, NGOs and donors an evidence base for deciding where to post teachers and build classrooms and toilets.

## Key Findings

- **Classrooms:** all 20 LGAs exceed the standard (most lacking: Ojo, Ikorodu, Alimosho).
- **Toilets:** 19 of 20 LGAs exceed the standard (most lacking: Badagry, Ojo, Alimosho). Only Mushin is within it.
- **Teachers:** 9 of 20 LGAs exceed the standard (most lacking: Ajeromi/Ifelodun, Badagry, Amuwo Odofin).
- **Most lacking overall:** Badagry, Ojo, Ajeromi/Ifelodun, Ibeju/Lekki and Alimosho.
- **Junior secondary (JSS) schools** are under the most pressure, with classrooms holding about double the standard.
- **The teacher shortage worsened every year from 2016/17 to 2019/20.** In public primary schools the average rose from 38.4 to 53.0 students per teacher.

## Features

- **LGA ranking** against the UBE standards for each resource and overall.
- **Most and least lacking lists**, for example the five LGAs most in need of teachers.
- **Teacher-shortage forecast** using Linear Regression and Random Forest, with an estimate of teachers needed per LGA.
- **Expected-versus-actual flag** for LGAs getting worse faster than the model predicts.
- **RAG assistant** that answers questions using only the project's data and names its source.
- **Streamlit dashboard** (Overview, Rankings, LGA profile, Forecast, Ask the assistant) so users need no coding skills.

## Who It Is For

| End user | How they use the tool |
| --- | --- |
| Lagos State Universal Basic Education Board (SUBEB) | Decide where to post new teachers and build classrooms and toilets |
| Lagos State Ministry of Basic and Secondary Education | Use trends and forecasts for annual planning and budgets |
| Universal Basic Education Commission (UBEC) | Check LGAs against UBE standards when allocating funds |
| NGOs and development partners | Target projects and write proposals for under-served LGAs |
| Corporate donors and CSR teams | Choose where to fund classroom or sanitation projects |
| Researchers and journalists | Use sourced figures for analysis and reporting |

## Data

The combined dataset has **360 rows**. Each row is one LGA, school level, ownership type and year (panel data).

| Source | Years | What it records | Rows |
| --- | --- | --- | --- |
| Lagos Bureau of Statistics, *Spotlight on Lagos Statistics 2021* | 2016/17 to 2019/20 | Public Primary, JSS and SSS enrolment by gender, and student-teacher ratio | 240 |
| DNEMIS Annual School Census | 2024/25 | Learners and teacher, classroom and toilet ratios, by public school level and private total | 120 |

Most missing values are structural (the two sources record different information), so gaps were left as `NaN` rather than filled. Instead the data was split into task-specific datasets:

| Dataset | Rows | Used for |
| --- | --- | --- |
| Full | 360 | EDA and the RAG knowledge base |
| Machine learning | 180 | Training and testing the forecast models (2017/18 to 2019/20, public schools) |
| Ranking | 60 | Ranking LGAs (2024/25, public Primary, JSS, SSS) |

The full column list is in `Lagos_Dataset_Data_Dictionary.xlsx`.

## Methodology

1. **Data collection and cleaning:** combine both sources, standardise column names and types, check LGA totals against published totals, and calculate estimated teachers, classrooms and toilets and the gap against the teacher standard.
2. **Exploratory data analysis:** compare every LGA and school level with the UBE standards and track the 2016/17 to 2019/20 trend.
3. **Machine learning:** forecast next year's student-teacher ratio as a regression problem. Linear Regression is the baseline, Random Forest is the main model, and both are compared with a naive "same as last year" check using MAE, RMSE and R². The data is split by year, not randomly, to avoid leakage.
4. **RAG assistant:** convert the cleaned data into plain-English sentences, embed them, store them in a vector database, retrieve the most relevant chunks for each question, and have a language model answer only from those chunks. The assistant never calculates numbers itself.
5. **Dashboard:** present rankings, forecasts and the assistant in a Streamlit app.

## Repository Structure

```
Education_Resources_Gap_Intelligence/
├── Capstone_DS/                            # Machine learning work
├── data/                                   # Prepared datasets
├── chroma_store/                           # Vector store used by the RAG assistant and dashboard
├── Data_Cleaning.ipynb                     # Data cleaning and preparation
├── EDA.ipynb                               # Exploratory data analysis
├── RAG.ipynb                               # RAG assistant
├── edu_app.py                              # Streamlit dashboard
├── requirements.txt                        # Python dependencies for the dashboard
├── Lagos_LGA_Education_Dataset.csv         # Combined dataset
├── Lagos_Dataset_Data_Dictionary.xlsx      # Column definitions
├── Lagos_Education_RAG_Knowledge_Base.pdf  # RAG knowledge base
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git
- API keys for the chat model and the embedding model (see below)

### Installation

```bash
# Clone the repository
git clone https://github.com/ENNILOIS/Education_Resources_Gap_Intelligence.git
cd Education_Resources_Gap_Intelligence

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# To run the dashboard
pip install -r requirements.txt

# To also run the notebooks (cleaning, EDA, machine learning, RAG)
pip install pandas numpy matplotlib seaborn openpyxl scikit-learn ipykernel langchain-community pymupdf langgraph ddgs
```

### API keys

Create a file named `.env` in the project root with your own keys:

```
OPENAI_API_KEY=your_chat_model_key
EMBEDDING_API_KEY=your_embedding_model_key
```

> **Never commit your `.env` file.** It is listed in `.gitignore`. If a key is ever pushed to GitHub, regenerate it immediately.

### Running the notebooks

Open the project in VS Code or Jupyter, select the `venv` kernel, and run the notebooks in this order:

1. `Data_Cleaning.ipynb`
2. `EDA.ipynb`
3. Machine learning notebook in `Capstone_DS/`
4. `RAG.ipynb`

### Running the dashboard

With your virtual environment active and your `.env` file in place:

```bash
streamlit run edu_app.py
```

## Tech Stack

| Area | Tools |
| --- | --- |
| Programming | Python, Jupyter Notebook |
| Data handling | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn (Linear Regression, Random Forest Regressor) |
| RAG | LangChain, ChromaDB, PyMuPDF; chat model `openai/gpt-oss-20b` via Groq; embedding model `Qwen/Qwen3-Embedding-0.6B` |
| Dashboard | Streamlit |

## Limitations and Ethics

**Limitations**

- No LGA-level exam results are published, so the tool cannot test whether bigger resource gaps lead to worse results.
- The two sources cover different periods (2016-2020 and 2024/25) and come from different bodies, so comparisons across them need care.
- Classroom and toilet data exists for one year only, so only the teacher ratio can be forecast.
- With only 20 LGAs, a model can learn only so much.
- LGA averages hide differences between individual schools.
- About 85% of Lagos schools reported to the 2024/25 census.

**Ethics**

All data is public and aggregated by LGA, so no student, teacher or school can be identified. A low ranking describes a resource gap, not the quality of teachers or students, and should be used to direct support, not to blame. The assistant answers only from verified data, and final decisions remain with responsible officials.

## Future Work

- Extend the tool to other states (DNEMIS uses the same format for every state).
- Update it each year with new census data.
- Add exam results if they are published by LGA.
- Add a map view of Lagos LGAs.

## Data Sources and References

- Federal Ministry of Education (2026). *Nigeria Education Statistics 2024: Annual School Census.* DNEMIS. https://emis.education.gov.ng/portal/
- Federal Ministry of Education (2026). *Lagos State Education Statistics 2024: Annual School Census.* DNEMIS. https://emis.education.gov.ng/portal/asc/state-H2ZhSMudlMI
- Lagos Bureau of Statistics (2022). *Spotlight on Lagos Statistics, 2021 Edition.* https://lagosmepb.org/wp-content/uploads/Hotline_Stat.pdf
- Wada, O. Z., Oloruntoba, E. O., Adejumo, M. and Aluko, O. O. (2020). *Classification of Sanitation Services and Students' Sanitation Practices among Schools in Lagos, Nigeria.*
- UNICEF and Federal Ministry of Budget and Economic Planning (2024). *Situation Analysis of Children and Adolescents in Nigeria.*

## Team

| Name | Role | Contribution | GitHub |
| --- | --- | --- | --- |
| Cynthia | Group Leader, Data Analyst | Data collection, cleaning and preparation; exploratory data analysis (EDA) | [@ENNILOIS](https://github.com/ENNILOIS) |
| Hadiya | Machine Learning Engineer, Dashboard Developer | Forecast models and Streamlit dashboard | [@Haddiee](https://github.com/Haddiee) |
| Mercy | RAG Developer, Documentation | RAG assistant and project README | [@Mercyyme](https://github.com/Mercyyme) |

For the full write-up, see the project report, *Education Resource-Gap Intelligence Tool: Project Report*.
