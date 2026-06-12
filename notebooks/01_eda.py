# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown] id="F9UtaL2xhxmS"
# # Project EDA: Expert Discovery System and Collaboration Network Analysis with HORUS data

# %% [markdown] id="bUPmv3NcjLhe"
# **General Objective**: To develop an intelligent model that identifies latent areas of expertise and recommends thesis advisors based on semantic affinity, overcoming traditional administrative classifications.
#
# **Main Mining Technique**: Topic detection (Topic Modeling) with the use of embeddings (Language Models - LLMs) for vector representation of descriptions (e.g., SBERT or ada-002) combined with Hierarchical Clustering (e.g., HDBSCAN).
#
# **Granularity level**: The base level of granularity is academic productions. For now, data has been collected from a single university campus (sede Bogotá) and its faculties (without a more specific breakdown by departments, mainly for ease of extraction, as we have not yet been given access to the official databases). However, I this would still be acceptable to a certain extent, as the focus is on professors and their output rather than on any divisions that may exist, allowing the data to reveal the actual structure of research  (thematic affinity rather than by faculty or department). Also, we will only consider productions of active professors.

# %% [markdown]
# # 1. Setup and imports

# %% id="b72fd66a"
import pandas as pd
import numpy as np
import re
import unicodedata
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import nltk

from wordcloud import WordCloud
from pathlib import Path
from itertools import combinations
from collections import Counter
from IPython.display import display
from nltk.corpus import stopwords as nltk_stopwords
from collections import Counter

# %% [markdown] id="bfaf216a"
# ### 1.1. Environment Configuration (Mounting Drive)

# %%
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    base_path = Path("/content/drive/Shareddrives/Minería/proyecto_horus/data/")
else:
    base_path = Path("../data/")

# %% [markdown] id="15a33eed"
# ### 1.2. Faculty-Level Data Loading

# %% [markdown] id="bde526d8"
# Load faculty-specific datasets and assign the faculty label explicitly.

# %% id="bde526d8"
products_engineering = pd.read_excel(base_path / "raw/productos_ingenieria.xlsx").assign(faculty='Ingeniería')
products_arts = pd.read_excel(base_path / "raw/productos_artes.xlsx").assign(faculty='Artes')
products_agricultural_sciences = pd.read_excel(base_path / "raw/productos_ciencias_agrarias.xlsx").assign(faculty='Ciencias Agrarias')
products_economic_sciences = pd.read_excel(base_path / "raw/productos_ciencias_economicas.xlsx").assign(faculty='Ciencias Económicas')
products_human_sciences = pd.read_excel(base_path / "raw/productos_ciencias_humanas.xlsx").assign(faculty='Ciencias Humanas')
products_sciences = pd.read_excel(base_path / "raw/productos_ciencias.xlsx").assign(faculty='Ciencias')
products_law = pd.read_excel(base_path / "raw/productos_derecho.xlsx").assign(faculty='Derecho')
products_nursing = pd.read_excel(base_path / "raw/productos_enfermeria.xlsx").assign(faculty='Enfermería')
products_medicine = pd.read_excel(base_path / "raw/productos_medicina.xlsx").assign(faculty='Medicina')
products_dentistry = pd.read_excel(base_path / "raw/productos_odontologia.xlsx").assign(faculty='Odontología')
products_veterinary = pd.read_excel(base_path / "raw/productos_veterinaria.xlsx").assign(faculty='Veterinaria')

products_bogota = pd.concat([
    products_engineering,
    products_arts,
    products_agricultural_sciences,
    products_economic_sciences,
    products_human_sciences,
    products_sciences,
    products_law,
    products_nursing,
    products_medicine,
    products_dentistry,
    products_veterinary
], ignore_index=True)

column_mapping = {
    'Título original': 'original_title',
    'Descripción original': 'original_description',
    'Revista / Conferencia': 'journal_conference',
    'Coautores': 'coauthors',
    'Doi': 'doi',
    'ISBN': 'isbn',
    'ISSN': 'issn',
    'Citaciones': 'citations',
    'Idioma': 'language',
    'Fecha': 'date',
    'Tipo': 'type',
    'Fuente': 'source',
    'Enlace': 'link',
    'faculty': 'faculty'
}
products_bogota.rename(columns=column_mapping, inplace=True)

research_professors = pd.read_csv(base_path / 'external/docentes_investigadores.csv')
column_mapping_2 = {
    'Nombre': 'name',
    'Vinculación': 'affiliation',
    'Cantidad de productos': 'product_count'
}
research_professors.rename(columns=column_mapping_2, inplace=True)
active_research_professors = research_professors[research_professors['affiliation'] == 'Activo'].reset_index(drop=True)


# %% [markdown] id="55202302"
# ### 1.3. Useful functions

# %% id="f817ec88"
# Process coauthors
def count_coauthors(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return len([x for x in str(text).split(',') if x.strip()])

# Extract all individual coauthors
def extract_coauthors(text):
    if pd.isna(text):
        return []
    return [name.strip() for name in str(text).split(',') if name.strip()]

# Normalize names for comparison
def normalize_name(name):
    if pd.isna(name):
        return ""
    # Convert to lowercase and remove accents
    name = str(name).lower()
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                   if unicodedata.category(c) != 'Mn')
    # Remove periods and commas
    name = re.sub(r'[.,]', '', name)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name

# Group names by similarity (surname)
def extract_surname(name_norm):
    parts = name_norm.split()
    if len(parts) > 0:
        return parts[-1]  # Last element (likely the surname)
    return name_norm

# Convention strings to treat as missing
NULL_CONVENTIONS = ['no reportado', 'no disponible', 'sin información', 'n/a', 'na', 'none', '-', 'nd']
def is_null_by_convention(val):
    if pd.isna(val):
        return True
    return str(val).strip().lower() in NULL_CONVENTIONS

# Simple heuristic: infer language from title characters if missing
def infer_language_heuristic(title):
    """Very rough heuristic — flag for manual review or langdetect library."""
    if pd.isna(title) or str(title).strip() == '':
        return 'Desconocido'
    title = str(title).lower()
    # Spanish stopwords signals
    if any(w in title for w in [' de ', ' del ', ' la ', ' el ', ' en ', ' y ', ' con ', ' para ']):
        return 'Español (inferido)'
    elif any(w in title for w in [' the ', ' of ', ' and ', ' for ', ' in ', ' with ']):
        return 'Inglés (inferido)'
    return 'Indeterminado'

# Normalize names for matching (surnames differ from full names)
def normalize_for_match(name):
    name = str(name).lower().strip()
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                   if unicodedata.category(c) != 'Mn')
    name = re.sub(r'[.,]', '', name)
    return ' '.join(name.split())

# Check which coauthors are active professors
def is_active_professor(name, professors_set):
    return normalize_for_match(name) in professors_set

# Filter: keep only productions where at least one coauthor is an active professor
def has_active_professor(coauthors_str, professors_set):
    names = extract_coauthors(coauthors_str)
    return any(normalize_for_match(n) in professors_set for n in names)

noisy_expressions = [
    r'\btexto tomado\b',
    r'\buniversidad nacional\b',
    r'\bnacional colombia\b',
    r'\bet al\b'
]

stopwords_auto_df = set()

def normalize_text(text, remove_accents=True, lowercase=True):
    """
    Normalize text: lowercase and remove accents.
    
    Args:
        text: Text to normalize
        remove_accents: If True, remove accents
        lowercase: If True, convert to lowercase
    
    Returns:
        Normalized text
    """
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text = str(text)
    
    if lowercase:
        text = text.lower()
    
    if remove_accents:
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    return text

def clean_text(text, min_token_length=2, remove_numbers=True):
    """
    Clean text by removing punctuation, short tokens and numbers.
    
    Args:
        text: Text to clean
        min_token_length: Minimum token length to keep
        remove_numbers: If True, remove tokens that are only numbers
    
    Returns:
        Cleaned text
    """
    if pd.isna(text) or str(text).strip() == '':
        return ""
    
    text = str(text)
    
    # Remove punctuation (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Split into tokens
    tokens = text.split()
    
    # Filter tokens
    filtered_tokens = []
    for token in tokens:
        # Remove very short tokens
        if len(token) < min_token_length:
            continue
        
        # Remove tokens that are only numbers
        if remove_numbers and token.isdigit():
            continue
        
        filtered_tokens.append(token)
    
    return ' '.join(filtered_tokens)

def clean_thematic_noise(text):

    if not text or str(text).strip() == "":
        return ""

    t = str(text)

    for pattern in noisy_expressions:
        t = re.sub(pattern, " ", t, flags=re.IGNORECASE)

    tokens = t.split()

    tokens = [
        tok for tok in tokens
        if tok not in stopwords_auto_df
    ]

    return " ".join(tokens)



# %% [markdown]
# ### 1.4. Stopwords Definition and Processing Functions

# %%
# -----------------------------
# SPANISH STOPWORDS
# -----------------------------
STOPWORDS_ES = {
    'el','la','los','las','lo',
    'un','una','unos','unas','uno',
    'de','del','al',
    'que','y','o','u','ni',
    'a','ante','bajo','con','contra','desde','durante','en','entre',
    'hacia','hasta','para','por','segun','sin','sobre','tras',

    'ser','estar','haber','tener','hacer','poder','decir','ir','ver','dar',
    'saber','querer','llegar','pasar','deber','poner','parecer',
    'quedar','creer','hablar','llevar','dejar','seguir','encontrar',

    'este','esta','estos','estas',
    'ese','esa','esos','esas',
    'aquel','aquella','aquellos','aquellas',
    'esto','eso','aquello',

    'yo','tu','tú','el','ella','nosotros','nosotras','ellos','ellas',
    'me','te','se','nos','les','le','lo','la','los','las',

    'mi','mis','tu','tus','su','sus','nuestro','nuestra','nuestros','nuestras',

    'si','no','sí',
    'muy','mas','más','menos','tan','tanto','tanta','tantos','tantas',

    'tambien','también','ya','aun','aún',
    'asi','así',

    'cuando','donde','dónde','como','cómo','porque','porqué','pues',

    'uno','dos','tres',

    'algo','alguien','alguno','algunos','alguna','algunas',
    'nada','nadie','todo','todos','todas',

    'mismo','misma','mismos','mismas',

    'tiempo','dia','día','ano','año'
}


# -----------------------------
# ENGLISH STOPWORDS
# -----------------------------
STOPWORDS_EN = {
    'the','a','an','and','or','but',
    'of','to','in','on','for','with','at','by','from',
    'into','onto','about','over','under','between','through',

    'is','are','was','were','be','been','being',
    'have','has','had','having',
    'do','does','did',

    'this','that','these','those',
    'there','here',

    'i','you','he','she','it','we','they',
    'me','him','her','us','them',

    'my','your','his','her','its','our','their',

    'what','which','who','whom','whose',
    'when','where','why','how',

    'can','could','should','would','may','might','must',

    'not','no','nor','only','just','also','even',

    'very','more','most','less',

    'one','two','three',

    'new','old','same','other','another'
}


# -----------------------------
# ACADEMIC / SCIENTIFIC NOISE
# -----------------------------
STOPWORDS_CONTEXT = {

    # institutions / geography
    'colombia','colombian',
    'universidad','university',
    'nacional','national',
    'bogota','bogotá',

    # research words
    'study','studies','research','paper','article',
    'work','works','project',

    # results words
    'result','results','finding','findings','found',

    # methodological words
    'method','methods','methodology',
    'approach','approaches',
    'analysis','analyses',
    'model','models',
    'data','dataset',
    'variable','variables',
    'process','processes',
    'system','systems',

    # reporting language
    'based','using','use','used',
    'show','shows','shown',
    'present','presents','presented',
    'propose','proposed',
    'evaluate','evaluated',

    # generic academic words
    'case','cases',
    'group','groups',
    'type','types',
    'level','levels',
    'value','values',
    'factor','factors',

    'high','low','large','small',

    'different','various','several',

    'general','overall',

    'during','within','across',

    # spanish academic filler
    'resultado','resultados',
    'analisis','análisis',
    'estudio','estudios', 'traves',
    'investigacion','investigación',
    'metodo','método', 'such', 'mayor', 'presente',
    'metodos','métodos', 'años', 'diseño',
    'modelo','modelos', 'anos', 'diseno',
    'datos', 'uso', 'partir', 'objetivo', 
    'proceso','procesos',
    'sistema','sistemas',

    'forma','nivel','valor',

    'caso','casos',
    'grupo','grupos',

    'diferente','diferentes',

    'generalmente',

    'durante',

    'ademas','además',

    'cual','cuales',

    'son','sus','esta','este','esto',
    'han','ha','fue','tiene'
}

STOPWORDS_ALL = STOPWORDS_ES | STOPWORDS_EN | STOPWORDS_CONTEXT

noisy_expressions = [
    r'\btexto tomado\b',
    r'\buniversidad nacional\b',
    r'\bnacional colombia\b',
    r'\bet al\b'
]

try:
    _ = nltk_stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS_NLTK_EN = set(nltk_stopwords.words('english'))
STOPWORDS_NLTK_ES = set(nltk_stopwords.words('spanish'))
print(f"NLTK stopwords loaded: EN={len(STOPWORDS_NLTK_EN)}, ES={len(STOPWORDS_NLTK_ES)}")


def _normalize_stopword_set(words):
    normalized = set()
    for w in words:
        if pd.isna(w):
            continue
        w = str(w).strip().lower()
        if not w:
            continue
        w = ''.join(
            c for c in unicodedata.normalize('NFD', w)
            if unicodedata.category(c) != 'Mn'
        )
        normalized.add(w)
    return normalized

# Normalize stopword sets
STOPWORDS_ES = _normalize_stopword_set(STOPWORDS_ES)
STOPWORDS_EN = _normalize_stopword_set(STOPWORDS_EN)
STOPWORDS_NLTK_ES = _normalize_stopword_set(STOPWORDS_NLTK_ES)
STOPWORDS_NLTK_EN = _normalize_stopword_set(STOPWORDS_NLTK_EN)
STOPWORDS_CONTEXT = _normalize_stopword_set(STOPWORDS_CONTEXT)

# Reinforce contextual noise detected in EDA
STOPWORDS_CONTEXT_EXTRA = {
    'proyecto', 'trabajo', 'uso', 'mayor', 'parte', 'cada', 'puede'
}
STOPWORDS_CONTEXT |= _normalize_stopword_set(STOPWORDS_CONTEXT_EXTRA)

# Unified list: remove stopwords from both languages + context
STOPWORDS_COMBINED = STOPWORDS_ES | STOPWORDS_EN | STOPWORDS_NLTK_ES | STOPWORDS_NLTK_EN
STOPWORDS_ALL = STOPWORDS_COMBINED | STOPWORDS_CONTEXT

def remove_stopwords(text, stopwords=None):
    if pd.isna(text) or str(text).strip() == '':
        return ""

    if stopwords is None:
        stopwords = STOPWORDS_ALL

    tokens = str(text).split()
    filtered_tokens = [tok for tok in tokens if tok.lower() not in stopwords]

    return ' '.join(filtered_tokens)

def preprocess_text_full(text, 
                         lowercase=True,
                         remove_accents=True, 
                         min_token_length=2,
                         remove_numbers=True,
                         remove_stopwords_flag=True):
    """
    Full text preprocessing pipeline.

    Steps:
    1. Normalization (lowercase, accents)
    2. Cleaning (punctuation, short tokens, numbers)
    3. Stopword removal using STOPWORDS_ALL (ES + EN + context)
    4. Thematic noise cleaning
    """
    if pd.isna(text) or str(text).strip() == '':
        return ""

    text = normalize_text(text, remove_accents=remove_accents, lowercase=lowercase)
    text = clean_text(text, min_token_length=min_token_length, remove_numbers=remove_numbers)

    if remove_stopwords_flag:
        text = remove_stopwords(text)

    text = clean_thematic_noise(text)
    return text.strip()

print(
    f"Total stopwords: base={len(STOPWORDS_COMBINED)}, ",
    f"context={len(STOPWORDS_CONTEXT)}, total={len(STOPWORDS_ALL)}"
)

# %% [markdown] id="2ba728da"
# ### 1.5. Filter to active professors only

# %%
# Build a set of normalized names from the CSV
active_professors_names = set(
    active_research_professors['name']
    .dropna()
    .apply(normalize_for_match)
)

active_mask = products_bogota['coauthors'].apply(
    lambda x: has_active_professor(x, active_professors_names)
)

products_bogota = products_bogota[active_mask].copy()
print(f"Products with at least one active professor: {len(products_bogota)}")
print(f"Products excluded: {len(active_mask) - active_mask.sum()}")

# %% [markdown] id="41ecb912"
# # 2. EDA
#
# This Exploratory Data Analysis (EDA) aims to understand the structure, quality, and behavioral patterns of academic production data across faculties. It begins with data validation, examining dimensions, missing values, duplicates, and consistency in key fields such as dates and text columns. It then explores categorical distributions (faculty, type, source, language) to identify dominant patterns and concentration effects. The analysis also investigates authorship dynamics, including coauthor counts, productivity distribution (long-tail behavior), collaboration frequency, and potential name disambiguation issues.

# %% [markdown] colab={"base_uri": "https://localhost:8080/"} id="01ab3457" outputId="119d642d-4bc0-4769-cf18-f267db00a4ea"
# ## 2.1. Dimensional overview

# %% colab={"base_uri": "https://localhost:8080/", "height": 625} id="87d0f864" outputId="4e100a34-d8e6-4494-c423-0447f90d7936"
print("Dimensions (rows, columns):", products_bogota.shape)
products_bogota.sample(5, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="9df37e2b" outputId="bf539bdc-d6e0-42e3-8ecf-fd6ad2b03ab1"
products_bogota.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="32a07b96" outputId="282df977-c5a1-4856-f9cb-ef51577b2fb7"
active_research_professors.sample(5, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="7d08053f" outputId="5ed93f89-0afc-49ab-e323-61e867d76fca"
active_research_professors.info()

# %% [markdown] id="1V0_qhm9u5AV"
# ## 2.2. Missing Values & Uniqueness Analysis

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="73f142dc" outputId="95fdb5c7-109b-4f62-84f7-f8e00bf0d047"
columns_summary = pd.DataFrame({
    "dtype": products_bogota.dtypes.astype(str),
    "null": products_bogota.isna().sum(),
    "null_%": (products_bogota.isna().mean() * 100).round(2),
    "unique": products_bogota.nunique(dropna=True)
}).sort_values(["null_%", "unique"], ascending=[False, True])

columns_summary

# %% [markdown] id="zQ3GtsGDvty9"
# ## 2.3. Duplicate Detection

# %% colab={"base_uri": "https://localhost:8080/", "height": 523} id="fb7b0d68" outputId="5017a1c0-f76e-421a-dcc8-5c6955f3053f"
duplicates = products_bogota.duplicated().sum()
print("Duplicates:", duplicates)
print("Duplicates percentage:", round((duplicates / len(products_bogota)) * 100, 2), "%")

products_bogota.describe(include="all").transpose()

# %% [markdown] id="Dns8Sx_JwBbU"
# ## 2.4. Date Validation and temporal analysis

# %% colab={"base_uri": "https://localhost:8080/", "height": 527} id="1d755541" outputId="a35aeb9a-239f-4c0f-bda0-e55dc417cda9"
date_dt = pd.to_datetime(products_bogota["date"], errors="coerce", dayfirst=True)

date_summary = pd.DataFrame({
    "valid_dates": [date_dt.notna().sum()],
    "invalid_dates": [date_dt.isna().sum()],
    "date_min": [date_dt.min()],
    "date_max": [date_dt.max()]
})

display(date_summary)

publications_per_year = date_dt.dt.year.value_counts().sort_index()
publications_per_year.to_frame("records")

# %% [markdown]
# ### 2.4.1. Temporal Analysis

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="524a2508" outputId="2fbd92a7-64f6-4d99-9c88-4e900e8f3641"
# Publications Temporal Analysis
print("="*60)
print("PUBLICATIONS TEMPORAL ANALYSIS")
print("="*60)

# Convert fecha to datetime if not done
if 'date_dt' not in locals():
    date_dt = pd.to_datetime(products_bogota["date"], errors="coerce", dayfirst=True)

# Add temporary time columns to the DataFrame
products_bogota['year'] = date_dt.dt.year
products_bogota['month'] = date_dt.dt.month
products_bogota['quarter'] = date_dt.dt.quarter

# Filter rows with valid dates
temporal_data = products_bogota[products_bogota['year'].notna()].copy()

print(f"Records with valid date: {len(temporal_data)} ({len(temporal_data)/len(products_bogota)*100:.1f}%)")
print(f"Time range: {temporal_data['year'].min():.0f} - {temporal_data['year'].max():.0f}")

# 1. Production per year
print("\n--- Annual Production ---")
annual_prod = temporal_data['year'].value_counts().sort_index()
display(annual_prod.tail(10).to_frame("publications"))

# 2. Statistics by faculty and year (Top 5 faculties)
print("\n--- Production by Faculty (last 5 years) ---")
top_faculties = temporal_data['faculty'].value_counts().head(5).index
last_5_years = temporal_data['year'].max() - 4

faculty_year = temporal_data[
    (temporal_data['faculty'].isin(top_faculties)) &
    (temporal_data['year'] >= last_5_years)
].groupby(['faculty', 'year']).size().unstack(fill_value=0)

display(faculty_year)

# 3. Growth trend
print("\n--- Trend Analysis ---")
yearly_growth = annual_prod.pct_change() * 100
print(f"Average annual growth: {yearly_growth.mean():.2f}%")
print(f"Year with highest growth: {yearly_growth.idxmax():.0f} ({yearly_growth.max():.1f}%)")
print(f"Year with highest production: {annual_prod.idxmax():.0f} ({annual_prod.max()} publications)")

# 4. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Full time series
axes[0, 0].plot(annual_prod.index, annual_prod.values, marker='o', linewidth=2)
axes[0, 0].set_title('Temporal Evolution of Publications', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Publications')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Last 10 years (bars)
last_10 = annual_prod.tail(10)
axes[0, 1].bar(last_10.index.astype(str), last_10.values, color='steelblue', edgecolor='black')
axes[0, 1].set_title('Publications - Last 10 Years', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Number of Publications')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Trend by faculty (Top 5, last 5 years)
faculty_year.T.plot(ax=axes[1, 0], marker='o', linewidth=2)
axes[1, 0].set_title('Trend by Faculty (Top 5 - Last 5 years)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Number of Publications')
axes[1, 0].legend(title='Faculty', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Quarterly distribution (last 3 years)
last_3_years = temporal_data['year'].max() - 2
quarter_data = temporal_data[temporal_data['year'] >= last_3_years]
trim_pivot = quarter_data.groupby(['year', 'quarter']).size().unstack(fill_value=0)
trim_pivot.plot(kind='bar', ax=axes[1, 1], width=0.8, edgecolor='black')
axes[1, 1].set_title('Quarterly Distribution (Last 3 years)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Number of Publications')
axes[1, 1].legend(title='Quarter', labels=['Q1', 'Q2', 'Q3', 'Q4'])
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 5. Seasonality analysis (publications by month)
print("\n--- Seasonality by Month ---")
monthly_prod = temporal_data['month'].value_counts().sort_index()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(1, 13), [monthly_prod.get(i, 0) for i in range(1, 13)],
       color='teal', edgecolor='black', alpha=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_title('Publications Distribution by Month (all years)', fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Publications')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 6. Publication rate by faculty (last 5 years)
print("\n--- Growth Rate by Faculty (last 5 years) ---")
faculty_growth = temporal_data[
    temporal_data['year'] >= last_5_years
].groupby('faculty')['year'].agg(['min', 'max', 'count'])

display(faculty_growth.sort_values('count', ascending=False).head(10))

# Clean up temporary columns
products_bogota.drop(columns=['year', 'month', 'quarter'], inplace=True)

# %% [markdown]
# ### 2.4.2. Correlation: Missing Descriptions vs. Publication Date
#
# There is a systematic artifact where older publications are much more likely to lack a description because digital metadata standards weren't established until the 2000s. We analyze this by looking at description availability per decade to justify retaining older publications despite semantic sparsity.

# %%
print("="*60)
print("CORRELATION: DESCRIPTIONS BY DECADE")
print("="*60)

# 1. Create a Decade column based on valid dates
df_temp = products_bogota.copy()
df_temp['Decade'] = date_dt.dt.year // 10 * 10
df_temp = df_temp.dropna(subset=['Decade'])

# 2. Flag records that have a usable description
df_temp['has_desc'] = df_temp['original_description'].notna() & (df_temp['original_description'].str.strip() != '') & ~df_temp['original_description'].apply(is_null_by_convention)

# 3. Aggregate
decade_correlation = df_temp.groupby('Decade').agg(
    total_records=('original_title', 'count'),
    with_description=('has_desc', 'sum')
)
decade_correlation['pct_with_description'] = (decade_correlation['with_description'] / decade_correlation['total_records']) * 100

display(decade_correlation.round(1))

# 4. Plot
plt.figure(figsize=(10, 5))
sns.barplot(
    data=decade_correlation.reset_index(), 
    x='Decade', 
    y='pct_with_description', 
    color='steelblue'
)
plt.title('Percentage of Records with Valid Descriptions by Decade')
plt.ylabel('% with Description')
plt.xlabel('Decade')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# The lack of description is strongly monotonic with age. We will not use publication date as a strict filter to avoid biasing against historical expertise.

# %% [markdown] id="nLLydnv3xK4o"
# ## 2.5. Metadata Quality and Text Field Analysis ('original_title', 'original_description')

# %% colab={"base_uri": "https://localhost:8080/", "height": 420} id="a0abe821" outputId="b98775ac-6124-4808-ce29-8da9933d8fda"
# Text columns analysis (original_title, original_description)
print("="*60)
print("TEXT COLUMNS ANALYSIS")
print("="*60)

# Titles
products_bogota['title_length'] = products_bogota['original_title'].fillna('').astype(str).str.len()
display(pd.DataFrame({
    'column': ['original_title'],
    'total': [len(products_bogota)],
    'non_null': [products_bogota['original_title'].notna().sum()],
    'empty': [(products_bogota['original_title'].fillna('').astype(str).str.strip() == '').sum()],
    'avg_length': [products_bogota['title_length'].mean()],
    'median_length': [products_bogota['title_length'].median()],
    'min_length': [products_bogota['title_length'].min()],
    'max_length': [products_bogota['title_length'].max()]
}))

# Description/Abstracts
products_bogota['desc_length'] = products_bogota['original_description'].fillna('').astype(str).str.len()
desc_non_null = products_bogota['original_description'].notna()
short_descs = (products_bogota['desc_length'] > 0) & (products_bogota['desc_length'] < 50)

display(pd.DataFrame({
    'column': ['original_description'],
    'total': [len(products_bogota)],
    'non_null': [desc_non_null.sum()],
    'empty': [(products_bogota['desc_length'] == 0).sum()],
    'very_short (<50 chars)': [short_descs.sum()],
    'avg_length': [products_bogota.loc[desc_non_null, 'desc_length'].mean()],
    'median_length': [products_bogota.loc[desc_non_null, 'desc_length'].median()],
    'min_length': [products_bogota.loc[desc_non_null, 'desc_length'].min()],
    'max_length': [products_bogota.loc[desc_non_null, 'desc_length'].max()]
}))

# Clean temporary columns
products_bogota.drop(columns=['title_length', 'desc_length'], inplace=True)

# Show examples of very short abstracts
if short_descs.sum() > 0:
    print("\nExamples of very short descriptions:")
    display(products_bogota.loc[short_descs, ['original_title', 'original_description']].head(5))

# %% colab={"base_uri": "https://localhost:8080/", "height": 450} id="0fb9769a" outputId="0fd31d96-9585-4890-b0cf-478b3df86236"
print("="*60)
print("DATA QUALITY ANALYSIS - MISSING VALUES & CONVENTIONS")
print("="*60)

# Check each column for conventions
cols_to_check = ['original_title', 'original_description', 'language',
                 'source', 'type', 'journal_conference', 'doi', 'isbn', 'issn']

quality_summary = []
for col in cols_to_check:
    if col not in products_bogota.columns:
        continue
    real_nans = products_bogota[col].isna().sum()
    convention_nans = products_bogota[col].apply(is_null_by_convention).sum() - real_nans
    empties = (products_bogota[col].fillna('').astype(str).str.strip() == '').sum() - real_nans
    quality_summary.append({
        'Column': col,
        'Real NaNs': real_nans,
        'Empty Strings': empties,
        'Convention NaNs': convention_nans,
        'Total Effectively Null': real_nans + empties + convention_nans,
        '% Effective Null': round((real_nans + empties + convention_nans) / len(products_bogota) * 100, 2),
        'Valid Uniques': products_bogota[col].dropna().nunique()
    })

quality_summary_df = pd.DataFrame(quality_summary)
display(quality_summary_df)

# Rows with empty title
no_title_rows = products_bogota[
    products_bogota['original_title'].fillna('').astype(str).str.strip() == ''
]
print(f"\nRows without title: {len(no_title_rows)}")
display(no_title_rows[['type', 'source', 'faculty', 'original_description']].head(10))

# %% [markdown] id="4fPpHMj9xQjW"
# ## 2.6. Linguistic Metadata Analysis ('language')

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1c5c72bf" outputId="dc5686a6-2995-497f-e95b-d7d32d84d655"
print("="*60)
print("DETAILED LANGUAGE ANALYSIS")
print("="*60)

print("\nFull language distribution:")
language_counts = products_bogota['language'].value_counts(dropna=False)
display(language_counts.to_frame("frequency").assign(
    percentage=lambda df: (df['frequency'] / len(products_bogota) * 100).round(2)
))

has_description_mask = products_bogota['original_description'].notna()
missing_language_mask = products_bogota['language'].isna()

print(f"\nRecords without language specified: {missing_language_mask.sum()}")
print(f"Of these, have description: {(missing_language_mask & has_description_mask).sum()}")

# Language vs Faculty heatmap
print("\nLanguage by faculty (top 8 languages):")
top_languages = products_bogota['language'].value_counts().head(8).index
language_faculty = products_bogota[
    products_bogota['language'].isin(top_languages)
].groupby(['faculty', 'language']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(language_faculty, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Language Distribution by Faculty', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Language vs type
print("\nLanguage by production type (top 8 languages):")
language_type = products_bogota[
    products_bogota['language'].isin(top_languages)
].groupby(['type', 'language']).size().unstack(fill_value=0)
display(language_type)

missing_language_convention_mask = products_bogota['language'].apply(is_null_by_convention)
print(f"\nRecords with missing language (by convention): {missing_language_convention_mask.sum()}")
inferred_languages = products_bogota.loc[missing_language_convention_mask, 'original_title'].apply(infer_language_heuristic)
print("Heuristic language inference for records without language:")
display(inferred_languages.value_counts().to_frame("count"))

# %%
products_bogota[products_bogota['language'].isna()][['original_title', 'original_description', 'faculty', 'type', 'date']]

# %% [markdown] id="_mx3VnRfxVMQ"
# ## 2.7. Categorical Structure Analysis ('type', 'source', 'faculty', 'journal_conference')

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="e3e9e287" outputId="0d11533c-b7db-42e6-f2f2-3ae72a693bc8"
# Categorical columns analysis
print("="*60)
print("CATEGORICAL COLUMNS ANALYSIS")
print("="*60)

categorical_columns = ['type', 'source', 'faculty', 'journal_conference']

for col in categorical_columns:
    print(f"\n--- {col} ---")
    counts = products_bogota[col].value_counts(dropna=False)
    print(f"Unique categories: {products_bogota[col].nunique(dropna=False)}")
    print("Top 30:")
    display(counts.head(30).to_frame("frequency"))

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.head(15).plot(kind='barh', ax=ax)
    ax.set_title(f'Top 15 - {col}')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c269a120" outputId="772155d5-7cee-4d56-9f83-f290ee7a2c7a"
print("="*60)
print("RELATION: SOURCE x FACULTY x TYPE")
print("="*60)

# source vs faculty
top_sources = products_bogota['source'].value_counts().head(10).index
source_faculty = products_bogota[
    products_bogota['source'].isin(top_sources)
].groupby(['source', 'faculty']).size().unstack(fill_value=0)

print("Top sources by faculty:")
display(source_faculty)

# source vs type
source_type = products_bogota[
    products_bogota['source'].isin(top_sources)
].groupby(['source', 'type']).size().unstack(fill_value=0)

print("\nTop sources by production type:")
display(source_type)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(source_faculty, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0])
axes[0].set_title('source x faculty', fontsize=12, fontweight='bold')
sns.heatmap(source_type, annot=True, fmt='d', cmap='YlGn', ax=axes[1])
axes[1].set_title('source x type', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown] id="uwAuhySXxZa3"
# ## 2.8. Authorship Structure Analysis ('coauthors')

# %% id="ffd0fad5" outputId="7638e912-bdba-4c9b-ca66-3cb4817613b5"
# Coauthors analysis
print("="*60)
print("COAUTHORS ANALYSIS")
print("="*60)

products_bogota['num_coauthors'] = products_bogota['coauthors'].apply(count_coauthors)
products_bogota.loc[products_bogota['coauthors'].isna(), 'num_coauthors'] = 0

display(products_bogota['num_coauthors'].describe())

all_coauthors = []
for coauthors in products_bogota['coauthors']:
    all_coauthors.extend(extract_coauthors(coauthors))

coauthors_counter = Counter(all_coauthors)
print(f"\nTotal unique coauthor names: {len(coauthors_counter)}")

fig, ax = plt.subplots(figsize=(14, 5))
products_bogota['num_coauthors'].plot(kind='box', vert=False, ax=ax)
ax.set_title('Boxplot - Number of coauthors')
ax.set_ylabel('Number of coauthors')
plt.tight_layout()
plt.show()

products_bogota['num_coauthors'].plot(kind='box', vert=False, figsize=(10, 3), showfliers=False)
plt.tight_layout()
plt.show()

# %% id="01c5e766" outputId="e8325e17-b25b-4f71-83f6-3be75b068393"
print("=" * 60)
print("INSPECTION OF ROWS WITH MANY COAUTHORS")
print("=" * 60)

# Discretization with custom bins
bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 5000]
labels = ['1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '50+']

products_bogota['author_groups'] = pd.cut(products_bogota['num_coauthors'], bins=bins, labels=labels)

# Plot
ax = products_bogota['author_groups'].value_counts()[labels].plot(
    kind='bar',
    color='steelblue',
    edgecolor='black',
    figsize=(15, 10)
)

ax.set_title("Distribution of Coauthors per Academic Production")
ax.set_xlabel("Number of Coauthors")
ax.set_ylabel("Number of Productions")
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.locator_params(axis='y', nbins=30)
plt.xticks(rotation=0)
plt.show()

products_bogota.drop(columns=['author_groups'], inplace=True)

outlier_threshold = 50  # adjust as needed
outliers_coauthors = products_bogota[products_bogota['num_coauthors'] > outlier_threshold]

print(f"Rows with more than {outlier_threshold} coauthors: {len(outliers_coauthors)}")
display(outliers_coauthors.sort_values('num_coauthors', ascending=False).head())

# %%
print("="*60)
print("ADDITIONAL COAUTHORS ANALYSIS BY TYPE")
print("="*60)

# --- A. Correlation between num_coauthors and type ---
print("coauthors by production type:")
coauthors_by_type = products_bogota.groupby('type')['num_coauthors'].agg(
    ['mean', 'median', 'max', 'count']
).sort_values('mean', ascending=False)
display(coauthors_by_type)

fig, ax = plt.subplots(figsize=(15, 8))
products_bogota.boxplot(column='num_coauthors', by='type', ax=ax,
                          showfliers=False, rot=90)
ax.set_title('Distribution of coauthors by production type')
ax.set_xlabel('')
plt.suptitle('')
plt.tight_layout()
plt.show()

# %% [markdown] id="d45ba504"
# ## 2.9. Author frequency

# %% id="567965b8" outputId="ff841750-6157-4b87-e05f-28a75b1a86d0"
# Distribution by Principal Author (ACTIVE PROFESSORS ONLY)
print("="*60)
print("DISTRIBUTION ANALYSIS BY ACTIVE AUTHOR")
print("="*60)

print("Analyzing distribution of active professors from the 'coauthors' column...")

# Set of normalized active professor names for fast lookup
if 'active_professors_names' not in locals():
    active_professors_names = set(
        active_research_professors['name']
        .dropna()
        .apply(normalize_for_match)
    )

# Filter only coauthors who are active professors
active_coauthors = []
for coauthors in products_bogota['coauthors']:
    names = extract_coauthors(coauthors)
    actives = [n for n in names if normalize_for_match(n) in active_professors_names]
    active_coauthors.extend(actives)

author_counts = pd.Series(active_coauthors).value_counts()

print(f"Total unique active authors: {len(author_counts)}")
print(f"Total mentions of active authors: {len(active_coauthors)}")

# Coverage
papers_with_actives = products_bogota['coauthors'].apply(
    lambda x: has_active_professor(x, active_professors_names)
).sum()
print(f"Productions with at least 1 active professor: {papers_with_actives}/{len(products_bogota)} ({papers_with_actives/len(products_bogota)*100:.1f}%)")

print(f"\nStatistics of productions per active author:")
display(author_counts.describe())

print(f"\nTop 20 most frequent active authors:")
display(author_counts.head(20).to_frame("productions"))

# Long-tail analysis for active authors
authors_with_1 = (author_counts == 1).sum()
authors_with_2_5 = ((author_counts >= 2) & (author_counts <= 5)).sum()
authors_with_6_10 = ((author_counts >= 6) & (author_counts <= 10)).sum()
authors_with_more_10 = (author_counts > 10).sum()

print("\n--- Long Tail Distribution (Active Authors) ---")
print(f"Authors with 1 production: {authors_with_1} ({authors_with_1/len(author_counts)*100:.1f}%)")
print(f"Authors with 2-5 productions: {authors_with_2_5} ({authors_with_2_5/len(author_counts)*100:.1f}%)")
print(f"Authors with 6-10 productions: {authors_with_6_10} ({authors_with_6_10/len(author_counts)*100:.1f}%)")
print(f"Authors with >10 productions: {authors_with_more_10} ({authors_with_more_10/len(author_counts)*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(author_counts, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
axes[0].set_title('Distribution of productions per active author')
axes[0].set_xlabel('Number of productions')
axes[0].set_ylabel('Number of authors')
axes[0].set_yscale('log')
axes[0].grid(alpha=0.3)

# Top 20 active authors
author_counts.head(20).sort_values().plot(kind='barh', ax=axes[1], color='darkgreen')
axes[1].set_title('Top 20 most frequent active authors')
axes[1].set_xlabel('Number of productions')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Additional analysis: active coauthorship by faculty
print("\n" + "="*60)
print("ACTIVE COAUTHORS BY FACULTY")
print("="*60)

# Build table of active professors by faculty
active_by_faculty = []

for faculty in products_bogota['faculty'].dropna().unique():
    subset = products_bogota[products_bogota['faculty'] == faculty]
    
    faculty_authors = []
    for coauthors in subset['coauthors']:
        names = extract_coauthors(coauthors)
        actives = [n for n in names if normalize_for_match(n) in active_professors_names]
        faculty_authors.extend(actives)
    
    if len(faculty_authors) > 0:
        counts = pd.Series(faculty_authors).value_counts()
        active_by_faculty.append({
            'faculty': faculty,
            'Productions': len(subset),
            'Unique active professors': len(counts),
            'Total active mentions': len(faculty_authors),
            'Avg mentions per professor': counts.mean()
        })

active_faculty_df = pd.DataFrame(active_by_faculty).sort_values('Total active mentions', ascending=False)
display(active_faculty_df)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: unique active professors by faculty
axes[0].barh(active_faculty_df['faculty'], active_faculty_df['Unique active professors'], color='teal')
axes[0].set_xlabel('Number of unique active professors')
axes[0].set_title('Unique active professors by faculty', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Chart 2: total active mentions by faculty
axes[1].barh(active_faculty_df['faculty'], active_faculty_df['Total active mentions'], color='coral')
axes[1].set_xlabel('Total mentions of active professors')
axes[1].set_title('Active professor mentions by faculty', fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown] id="09652e82"
# ## 2.10. Topic modeling feasibility

# %% id="6a906c5f" outputId="af628e62-a278-47f5-b8c3-35ac801bd453"
# --- Density of text fields: useful for modeling feasibility ---
print("\nFeasibility for Topic Modeling:")
has_desc = products_bogota['original_description'].apply(
    lambda x: not is_null_by_convention(x) and len(str(x).strip()) >= 50
)
has_title = products_bogota['original_title'].apply(
    lambda x: not is_null_by_convention(x) and len(str(x).strip()) > 3
)

print(f"Records with usable description (≥50 chars): {has_desc.sum()} ({has_desc.mean()*100:.1f}%)")
print(f"Records with usable title: {has_title.sum()} ({has_title.mean()*100:.1f}%)")
print(f"Records with both: {(has_desc & has_title).sum()} ({(has_desc & has_title).mean()*100:.1f}%)")
print(f"Title only available: {(~has_desc & has_title).sum()}")

# %% [markdown]
# ## 2.11. Text Preprocessing Analysis
#
# This section demonstrates the text preprocessing functions defined earlier and shows their effect on actual data. 

# %%
print("="*60)
print("TEXT PREPROCESSING EXAMPLE")
print("="*60)

# Select one example with a valid description
example = products_bogota[
    products_bogota['original_description'].notna() &
    (products_bogota['original_description'].str.len() > 100)
].sample(1, random_state=42).iloc[0]

# Title
title_orig = example['original_title']
title_clean = preprocess_text_full(title_orig)

print(f"\n📌 TITLE:")
print(f"Original: {title_orig}")
print(f"Processed: {title_clean}")
print(f"Tokens: {len(str(title_orig).split())} → {len(title_clean.split())}")

# Description
desc_orig = str(example['original_description'])[:200]
desc_clean = preprocess_text_full(example['original_description'])

print(f"\n📝 DESCRIPTION (first 200 chars):")
print(f"Original: {desc_orig}...")
print(f"Processed: {desc_clean[:200]}...")
print(f"Total tokens: {len(str(example['original_description']).split())} → {len(desc_clean.split())}")

if example['language']:
    print(f"language: {example['language']}")
if example['faculty']:
    print(f"faculty: {example['faculty']}")

# %%
print("\n" + "="*70)
print("TOKENS COMPARISON: RAW vs PREPROCESSED")
print("="*70)

# Apply preprocessing to titles and descriptions
products_with_text = products_bogota[
    products_bogota['original_title'].notna()
].copy()

products_with_text['title_tokens_original'] = products_with_text['original_title'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
products_with_text['title_processed'] = products_with_text['original_title'].apply(preprocess_text_full)
products_with_text['title_tokens_clean'] = products_with_text['title_processed'].apply(
    lambda x: len(str(x).split()) if x else 0
)

products_with_desc = products_bogota[
    products_bogota['original_description'].notna()
].copy()
products_with_desc['desc_tokens_original'] = products_with_desc['original_description'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
products_with_desc['desc_processed'] = products_with_desc['original_description'].apply(preprocess_text_full)
products_with_desc['desc_tokens_clean'] = products_with_desc['desc_processed'].apply(
    lambda x: len(str(x).split()) if x else 0
)

# ===== TITLES =====
orig_title_total = int(products_with_text['title_tokens_original'].sum())
proc_title_total = int(products_with_text['title_tokens_clean'].sum())
red_title_abs = orig_title_total - proc_title_total
red_title_pct = (red_title_abs / orig_title_total * 100) if orig_title_total > 0 else 0.0

print("\n📘 TITLES")
print(f"Records analyzed: {len(products_with_text):,}")
print(f"TOTAL TOKENS RAW: {orig_title_total:,}")
print(f"TOTAL TOKENS PREPROCESSED:   {proc_title_total:,}")
print(f"ABSOLUTE DIFFERENCE:            {red_title_abs:,} tokens")
print(f"PERCENT REDUCTION:           {red_title_pct:.2f}%")
print(f"Average per record:          {products_with_text['title_tokens_original'].mean():.2f} → {products_with_text['title_tokens_clean'].mean():.2f}")

# ===== DESCRIPTIONS =====
orig_desc_total = int(products_with_desc['desc_tokens_original'].sum())
proc_desc_total = int(products_with_desc['desc_tokens_clean'].sum())
red_desc_abs = orig_desc_total - proc_desc_total
red_desc_pct = (red_desc_abs / orig_desc_total * 100) if orig_desc_total > 0 else 0.0

print("\n📝 DESCRIPTIONS")
print(f"Records analyzed: {len(products_with_desc):,}")
print(f"TOTAL TOKENS RAW: {orig_desc_total:,}")
print(f"TOTAL TOKENS PREPROCESSED:   {proc_desc_total:,}")
print(f"ABSOLUTE DIFFERENCE:            {red_desc_abs:,} tokens")
print(f"PERCENT REDUCTION:           {red_desc_pct:.2f}%")
print(f"Average per record:          {products_with_desc['desc_tokens_original'].mean():.2f} → {products_with_desc['desc_tokens_clean'].mean():.2f}")

# Compact summary table
tokens_summary = pd.DataFrame([
    {
        'Field': 'Title',
        'Records': len(products_with_text),
        'Total raw': orig_title_total,
        'Total preprocessed': proc_title_total,
        'Difference': red_title_abs,
        '% reduction': round(red_title_pct, 2)
    },
    {
        'Field': 'Description',
        'Records': len(products_with_desc),
        'Total raw': orig_desc_total,
        'Total preprocessed': proc_desc_total,
        'Difference': red_desc_abs,
        '% reduction': round(red_desc_pct, 2)
    }
])

print("\n TABULAR SUMMARY")
display(tokens_summary)

# Edge cases
print("\n⚠️ PROBLEMATIC CASES AFTER PREPROCESSING")
titles_empty_post = int((products_with_text['title_tokens_clean'] == 0).sum())
print(f"Titles that became empty: {titles_empty_post:,}")

descs_empty_post = int((products_with_desc['desc_tokens_clean'] == 0).sum())
print(f"Descriptions that became empty: {descs_empty_post:,}")

# Clean temporary columns
products_with_text.drop(columns=['title_tokens_original', 'title_processed', 'title_tokens_clean'], inplace=True)
products_with_desc.drop(columns=['desc_tokens_original', 'desc_processed', 'desc_tokens_clean'], inplace=True)

# %%
# Visualization of token reduction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sample_titles_base = products_bogota[products_bogota['original_title'].notna()].copy()
sample_titles = sample_titles_base.sample(
    min(500, len(sample_titles_base)),
    random_state=42
)

sample_desc_base = products_bogota[products_bogota['original_description'].notna()].copy()
sample_desc = sample_desc_base.sample(
    min(300, len(sample_desc_base)),
    random_state=42
)

titles_orig_tokens = sample_titles['original_title'].apply(lambda x: len(str(x).split()))
titles_clean_tokens = sample_titles['original_title'].apply(preprocess_text_full).apply(lambda x: len(str(x).split()))
delta_titles = titles_orig_tokens - titles_clean_tokens

desc_orig_tokens = sample_desc['original_description'].apply(lambda x: len(str(x).split()))
desc_clean_tokens = sample_desc['original_description'].apply(preprocess_text_full).apply(lambda x: len(str(x).split()))
delta_desc = desc_orig_tokens - desc_clean_tokens

# =========================
# 1) AVERAGE BARS BEFORE vs AFTER
# =========================
labels = ['Titles', 'Descriptions']
avg_orig = [titles_orig_tokens.mean(), desc_orig_tokens.mean()]
avg_clean = [titles_clean_tokens.mean(), desc_clean_tokens.mean()]
x = np.arange(len(labels))
width = 0.35

axes[0].bar(x - width/2, avg_orig, width, label='Original', color='steelblue')
axes[0].bar(x + width/2, avg_clean, width, label='Preprocessed', color='orange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel('Average tokens per record')
axes[0].set_title('Average tokens: Before vs After')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(avg_orig):
    axes[0].text(i - width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
for i, v in enumerate(avg_clean):
    axes[0].text(i + width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)

# =========================
# 2) % OF RECORDS THAT REDUCE
# =========================
pct_docs_reduced_titles = (delta_titles > 0).mean() * 100
pct_docs_reduced_desc = (delta_desc > 0).mean() * 100

axes[1].bar(
    ['Titles', 'Descriptions'],
    [pct_docs_reduced_titles, pct_docs_reduced_desc],
    color=['teal', 'darkorange'],
    edgecolor='black'
)
axes[1].set_ylim(0, 105)
axes[1].set_ylabel('% records with reduction')
axes[1].set_title('Coverage of reduction by text type')
axes[1].grid(axis='y', alpha=0.3)

axes[1].text(0, pct_docs_reduced_titles + 1, f'{pct_docs_reduced_titles:.1f}%', ha='center')
axes[1].text(1, pct_docs_reduced_desc + 1, f'{pct_docs_reduced_desc:.1f}%', ha='center')

plt.tight_layout()
plt.show()

# =========================
# NUMERIC SUMMARY
# =========================
pct_reduction_titles = (1 - titles_clean_tokens.mean() / titles_orig_tokens.mean()) * 100
pct_reduction_desc = (1 - desc_clean_tokens.mean() / desc_orig_tokens.mean()) * 100

print(f"\nVALIDATION ORIGINAL vs PREPROCESSED DIFFERENCE")
print(f"Titles - average reduction: {pct_reduction_titles:.1f}%")
print(f"Titles - % records with reduction: {pct_docs_reduced_titles:.1f}%")
print(f"Descriptions - average reduction: {pct_reduction_desc:.1f}%")
print(f"Descriptions - % records with reduction: {pct_docs_reduced_desc:.1f}%")

summary_difference = pd.DataFrame([
    {
        'Field': 'Titles',
        'Sample': len(sample_titles),
        'Avg original tokens': round(titles_orig_tokens.mean(), 2),
        'Avg preprocessed tokens': round(titles_clean_tokens.mean(), 2),
        'Avg reduction (%)': round(pct_reduction_titles, 2),
        '% records reduced': round(pct_docs_reduced_titles, 2)
    },
    {
        'Field': 'Descriptions',
        'Sample': len(sample_desc),
        'Avg original tokens': round(desc_orig_tokens.mean(), 2),
        'Avg preprocessed tokens': round(desc_clean_tokens.mean(), 2),
        'Avg reduction (%)': round(pct_reduction_desc, 2),
        '% records reduced': round(pct_docs_reduced_desc, 2)
    }
])

display(summary_difference)

# %% [markdown]
# ## 2.12. Deep Text Analysis (Word Clouds & Token Statistics)
#
# This section provides a deeper analysis of the textual content, focusing on the most frequent words, word clouds by language and faculty, and text quality metrics.

# %% [markdown]
# ### 2.12.1. Word Clouds

# %%
print("="*60)
print("Word cloud: with and without generic terms")
print("="*60)

# =========================
# 1) Build base text
# =========================
base_text = products_bogota.apply(
    lambda row: f"{str(row['original_title']) if pd.notna(row['original_title']) else ''} "
                f"{str(row['original_description']) if pd.notna(row['original_description']) else ''}",
    axis=1
)

base_text = base_text.apply(preprocess_text_full)
base_text = base_text[base_text.str.len() > 10].copy()

# =========================
# 2) Automatic document-frequency filtering
# =========================
df_threshold = 0.12

doc_freq = Counter()

for txt in base_text:
    tokens = set(txt.split())
    doc_freq.update(tokens)

n_docs = len(base_text)

auto_df_stopwords = {
    tok for tok, df in doc_freq.items()
    if (df / n_docs) >= df_threshold and len(tok) >= 2
}

# =========================
# 3) Informative text (apply thematic + automatic stopwords)
# =========================
def clean_all(text):
    return remove_stopwords(text, stopwords=STOPWORDS_ALL | auto_df_stopwords)

informative_text = base_text.apply(clean_all)
informative_text = informative_text[informative_text.str.len() > 10].copy()

print(f"Valid records (original cloud): {len(base_text)}")
print(f"Valid records (informative cloud): {len(informative_text)}")
print(f"Terms removed by automatic DF filter (DF >= {df_threshold:.0%}): {len(auto_df_stopwords)}")

# =========================
# 5) Generate clouds
# =========================
global_text_original = " ".join(base_text)
global_text_informative = " ".join(informative_text)

wordcloud_original = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='viridis',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10,
    collocations=False
).generate(global_text_original)

wordcloud_informative = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='magma',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10,
    collocations=False
).generate(global_text_informative)

# =========================
# 6) Visualization
# =========================
fig, axes = plt.subplots(1, 2, figsize=(20,7))

axes[0].imshow(wordcloud_original, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Original cloud', fontsize=14, fontweight='bold')

axes[1].imshow(wordcloud_informative, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Informative cloud (after filtering)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# %%
# ==========================================================
# NUBES DE PALABRAS POR FACULTAD (TOP 3) 
# ==========================================================

print("="*60)
print("WORD CLOUDS BY FACULTY")
print("="*60)

# ----------------------------------------------------------
# 1. Build combined text
# ----------------------------------------------------------
products_bogota['full_text'] = (
    products_bogota['original_title'].fillna('').astype(str) + " " +
    products_bogota['original_description'].fillna('').astype(str)
)

# ----------------------------------------------------------
# 2. Apply preprocessing pipeline
# ----------------------------------------------------------
products_bogota['processed_text'] = products_bogota['full_text'].apply(
    lambda t: preprocess_text_full(
        t,
        lowercase=True,
        remove_accents=True,
        min_token_length=3,
        remove_numbers=True,
        remove_stopwords_flag=True
    )
)

# ----------------------------------------------------------
# 3. Filter valid texts
# ----------------------------------------------------------
valid_texts = products_bogota[
    products_bogota['processed_text'].notna() &
    (products_bogota['processed_text'].str.len() > 10)
].copy()

print(f"\nPublications with valid text: {len(valid_texts)}")

# ----------------------------------------------------------
# 4. Select the 3 faculties with the most publications
# ----------------------------------------------------------
top_faculties = valid_texts['faculty'].value_counts().head(3).index

texts_by_faculty = {}

for faculty in top_faculties:

    subset = valid_texts[valid_texts['faculty'] == faculty]

    texts = subset['processed_text']

    filtered_texts = texts.apply(
        lambda t: remove_stopwords(t, stopwords=STOPWORDS_ALL | auto_df_stopwords)
    )

    filtered_texts = filtered_texts[filtered_texts.str.len() > 10]

    if len(filtered_texts) > 0:
        texts_by_faculty[faculty] = " ".join(filtered_texts)

    print(f"{faculty}: {len(filtered_texts)} publications")

# ----------------------------------------------------------
# 5. Generate word clouds
# ----------------------------------------------------------
num_faculties = len(texts_by_faculty)

fig, axes = plt.subplots(1, num_faculties, figsize=(6 * num_faculties, 5))

if num_faculties == 1:
    axes = [axes]

colormaps = ['Greens', 'Blues', 'Oranges']

for idx, (faculty, text) in enumerate(texts_by_faculty.items()):

    wordcloud = WordCloud(
        width=800,
        height=500,
        background_color='white',
        colormap=colormaps[idx % len(colormaps)],
        max_words=60,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)

    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].axis('off')

    total_pub = len(valid_texts[valid_texts["faculty"] == faculty])

    axes[idx].set_title(
        f"{faculty}\n({total_pub} publications)",
        fontsize=12,
        fontweight='bold'
    )

plt.tight_layout()
plt.show()

# %%
print("="*60)
print("MOST FREQUENT WORDS ANALYSIS")
print("="*60)

# ----------------------------------------------------------
# 1. Create cleaned text column if it doesn't exist
# ----------------------------------------------------------
if 'clean_text' not in products_bogota.columns:

    products_bogota['combined_text'] = (
        products_bogota['original_title'].fillna('').astype(str) + " " +
        products_bogota['original_description'].fillna('').astype(str)
    )

    products_bogota['clean_text'] = products_bogota['combined_text'].apply(
        lambda t: preprocess_text_full(
            t,
            lowercase=True,
            remove_accents=True,
            min_token_length=3,
            remove_numbers=True,
            remove_stopwords_flag=True
        )
    )


# ----------------------------------------------------------
# 2. Filter valid texts (minimum quality)
# ----------------------------------------------------------
valid_texts = products_bogota[
    products_bogota['clean_text'].notna() &
    (products_bogota['clean_text'].str.len() > 10)
].copy()

print(f"\nDocuments analyzed: {len(valid_texts)}")


# ----------------------------------------------------------
# 3. Apply final cleaning (additional stopwords)
# ----------------------------------------------------------
analysis_texts = valid_texts['clean_text'].apply(
    lambda t: remove_stopwords(t, stopwords=STOPWORDS_ALL | auto_df_stopwords)
)


# ----------------------------------------------------------
# 4. Extract tokens
# ----------------------------------------------------------
all_tokens = []

for txt in analysis_texts:
    if txt and len(txt) > 0:
        all_tokens.extend(txt.split())

print(f"\nTotal tokens processed: {len(all_tokens):,}")
print(f"Unique tokens: {len(set(all_tokens)):,}")


# ----------------------------------------------------------
# 5. Word frequency
# ----------------------------------------------------------
token_counter = Counter(all_tokens)

top_30 = token_counter.most_common(30)

print("\nTOP 30 MOST FREQUENT WORDS")

top_30_df = pd.DataFrame(top_30, columns=['Word', 'Frequency'])
top_30_df['%'] = (top_30_df['Frequency'] / len(all_tokens) * 100).round(3)

display(top_30_df)


# ----------------------------------------------------------
# 6. Visualization: Top words
# ----------------------------------------------------------
top_20_df = top_30_df.head(20)

plt.figure(figsize=(10,6))

plt.barh(
    top_20_df['Word'][::-1],
    top_20_df['Frequency'][::-1],
    edgecolor='black'
)

plt.xlabel("Frequency")
plt.title("Top 20 Most Frequent Words", fontweight="bold")
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.12.2. Text Quality Analysis

# %%
if 'products_bogota' not in globals():
    try:
        products_bogota = df.copy()
        print("⚠️ 'products_bogota' did not exist. Created from 'df'.")
    except:
        raise ValueError("DataFrame 'products_bogota' or 'df' not found. Load your data first.")

# ----------------------------------------------------------
# Verify required columns
# ----------------------------------------------------------
required_columns = ['original_title', 'original_description']
for col in required_columns:
    if col not in products_bogota.columns:
        raise ValueError(f"The column '{col}' does not exist in the dataframe.")

print("TEXT QUALITY ANALYSIS")
print("="*60)

# ----------------------------------------------------------
# 1. Compute title tokens
# ----------------------------------------------------------
products_bogota['title_tokens'] = (
    products_bogota['original_title']
    .fillna('')
    .astype(str)
    .apply(lambda x: len(preprocess_text_full(x).split()) if x.strip() else 0)
)

# ----------------------------------------------------------
# 2. Compute description tokens
# ----------------------------------------------------------
products_bogota['desc_tokens'] = (
    products_bogota['original_description']
    .fillna('')
    .astype(str)
    .apply(lambda x: len(preprocess_text_full(x).split()) if x.strip() else 0)
)

# ----------------------------------------------------------
# 3. Quality metrics
# ----------------------------------------------------------
total_records = len(products_bogota)

title_empty = (products_bogota['title_tokens'] == 0).sum()
title_very_short = (
    (products_bogota['title_tokens'] > 0) &
    (products_bogota['title_tokens'] < 3)
).sum()

desc_empty = (products_bogota['desc_tokens'] == 0).sum()
desc_very_short = (
    (products_bogota['desc_tokens'] > 0) &
    (products_bogota['desc_tokens'] < 10)
).sum()

print(f"\n📊 QUALITY METRICS")

print(f"\nTITLES")
print(f"No useful content (0 tokens): {title_empty} ({title_empty/total_records*100:.1f}%)")
print(f"Very short (1-2 tokens): {title_very_short} ({title_very_short/total_records*100:.1f}%)")

print(f"\nDESCRIPTIONS")
print(f"No useful content (0 tokens): {desc_empty} ({desc_empty/total_records*100:.1f}%)")
print(f"Very short (1-9 tokens): {desc_very_short} ({desc_very_short/total_records*100:.1f}%)")

# ----------------------------------------------------------
# 4. Records with good quality text
# ----------------------------------------------------------
good_quality = (
    (products_bogota['title_tokens'] >= 3) &
    (products_bogota['desc_tokens'] >= 10)
).sum()

print(f"\nRecords with complete good-quality text")
print(f"{good_quality} / {total_records} ({good_quality/total_records*100:.1f}%)")

# ----------------------------------------------------------
# 5. Visualizations
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Title length distribution
axes[0].hist(
    products_bogota['title_tokens'],
    bins=30,
    edgecolor='black',
    alpha=0.7
)

axes[0].axvline(x=3, linestyle='--', label='Quality threshold (3 tokens)')
axes[0].set_xlabel('Title tokens (post-preprocessing)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Title Length Distribution', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim(left=0)

# Description length distribution
axes[1].hist(
    products_bogota['desc_tokens'],
    bins=50,
    edgecolor='black',
    alpha=0.7
)

axes[1].axvline(x=10, linestyle='--', label='Quality threshold (10 tokens)')
axes[1].set_xlabel('Description tokens (post-preprocessing)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Description Length Distribution', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim(left=0)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 6. Problematic examples
# ----------------------------------------------------------
print("\n⚠️ EXAMPLES WITH VERY SHORT OR EMPTY DESCRIPTIONS")

problematic_cases = products_bogota[
    products_bogota['desc_tokens'] < 10
][['original_title', 'original_description']].head(10)

display(problematic_cases)

# ----------------------------------------------------------
# 7. Clean temporary columns
# ----------------------------------------------------------
products_bogota.drop(columns=['title_tokens', 'desc_tokens'], inplace=True)

# %% [markdown]
# ## 2.13. Cross Analysis: Language × Text × Active Coauthors

# %%
print("="*60)
print("CROSS ANALYSIS: LANGUAGE × TEXT × ACTIVE COAUTHORS")
print("="*60)

# Create working copy
cross_df = products_bogota.copy()

# Text length (tokens after preprocessing)
cross_df['title_tokens_clean'] = cross_df['original_title'].fillna('').astype(str).apply(
    lambda x: len(preprocess_text_full(x).split()) if x.strip() else 0
)
cross_df['desc_tokens_clean'] = cross_df['original_description'].fillna('').astype(str).apply(
    lambda x: len(preprocess_text_full(x).split()) if x.strip() else 0
)

# Number of active coauthors per publication
def count_actives_in_paper(coauthors_str, professors_set):
    names = extract_coauthors(coauthors_str)
    return sum(1 for n in names if normalize_for_match(n) in professors_set)

cross_df['num_actives'] = cross_df['coauthors'].apply(
    lambda x: count_actives_in_paper(x, active_professors_names)
)

# Keep top languages
top_languages = cross_df['language'].value_counts().head(5).index
cross_top = cross_df[cross_df['language'].isin(top_languages)].copy()

print("\n📊 SUMMARY BY LANGUAGE:")
language_summary = cross_top.groupby('language').agg({
    'title_tokens_clean': ['mean', 'median'],
    'desc_tokens_clean': ['mean', 'median'],
    'num_actives': ['mean', 'median', 'max'],
    'language': 'count'
}).round(2)

language_summary.columns = ['title_mean', 'title_median', 'desc_mean', 'desc_median',
                            'actives_mean', 'actives_median', 'actives_max', 'n_publications']
language_summary = language_summary.sort_values('n_publications', ascending=False)
display(language_summary)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Title length by language
sns.boxplot(data=cross_top, x='language', y='title_tokens_clean', ax=axes[0, 0], showfliers=False)
axes[0, 0].set_title('Title Tokens by Language', fontweight='bold')
axes[0, 0].set_xlabel('language')
axes[0, 0].set_ylabel('Title tokens')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Description length by language
sns.boxplot(data=cross_top, x='language', y='desc_tokens_clean', ax=axes[0, 1], showfliers=False)
axes[0, 1].set_title('Description Tokens by Language', fontweight='bold')
axes[0, 1].set_xlabel('language')
axes[0, 1].set_ylabel('Description tokens')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Average active coauthors per language
actives_by_language = cross_top.groupby('language')['num_actives'].mean().sort_values(ascending=False)
axes[1, 0].bar(range(len(actives_by_language)), actives_by_language.values, color='forestgreen')
axes[1, 0].set_xticks(range(len(actives_by_language)))
axes[1, 0].set_xticklabels(actives_by_language.index, rotation=45, ha='right')
axes[1, 0].set_title('Average Active Coauthors by Language', fontweight='bold')
axes[1, 0].set_ylabel('Average active coauthors')
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Scatter: description length vs active coauthors (color = title tokens)
sample_cross = cross_top.sample(min(1000, len(cross_top)), random_state=42)
scatter = axes[1, 1].scatter(sample_cross['desc_tokens_clean'], sample_cross['num_actives'],
                             c=sample_cross['title_tokens_clean'], cmap='viridis', alpha=0.6, s=20)
axes[1, 1].set_title('Description Tokens vs Active Coauthors\n(color = title tokens)', fontweight='bold')
axes[1, 1].set_xlabel('Description tokens')
axes[1, 1].set_ylabel('Number of active coauthors')
axes[1, 1].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Title tokens')

plt.tight_layout()
plt.show()

# Key findings
print("\n🔍 KEY FINDINGS:")
lang_longest_desc = language_summary['desc_mean'].idxmax()
print(f"1. Language with longest descriptions (avg): {lang_longest_desc} ({language_summary.loc[lang_longest_desc, 'desc_mean']:.1f} tokens)")

lang_most_collab = language_summary['actives_mean'].idxmax()
print(f"2. Language with most active coauthors per publication: {lang_most_collab} ({language_summary.loc[lang_most_collab, 'actives_mean']:.2f})")

# Clean temporary objects
del cross_df, cross_top

# %% [markdown]
# ## 2.14. Secondary Analysis linked to core variables (Faculty × Missing Text)

# %%
print("="*60)
print("ANALYSIS: FACULTY × MISSING DESCRIPTION")
print("="*60)

# Define what counts as a missing / not useful description
def missing_description(desc):
    if pd.isna(desc):
        return True
    desc_str = str(desc).strip()
    if desc_str == '' or is_null_by_convention(desc_str):
        return True
    # Consider very short descriptions as not useful
    desc_clean = preprocess_text_full(desc_str)
    return len(desc_clean.split()) < 10

products_bogota['desc_missing_or_short'] = products_bogota['original_description'].apply(missing_description)

# Summary by faculty
faculty_desc_summary = products_bogota.groupby('faculty').agg(
    total_publications=('faculty', 'count'),
    desc_missing_or_short=('desc_missing_or_short', 'sum')
).reset_index()

faculty_desc_summary['pct_missing'] = (
    faculty_desc_summary['desc_missing_or_short'] / faculty_desc_summary['total_publications'] * 100
).round(2)

faculty_desc_summary = faculty_desc_summary.sort_values('pct_missing', ascending=False)

print("\n📊 Percentage of missing or not-useful descriptions by faculty:")
display(faculty_desc_summary)

# Hypothesis test for faculty 'Artes'
if 'Artes' in faculty_desc_summary['faculty'].values:
    artes_pct = faculty_desc_summary[faculty_desc_summary['faculty'] == 'Artes']['pct_missing'].iloc[0]
    overall_mean = faculty_desc_summary['pct_missing'].mean()
    
    print(f"\n🎨 Hypothesis for faculty 'Artes':")
    print(f"   % missing in Artes: {artes_pct:.2f}%")
    print(f"   Overall average %: {overall_mean:.2f}%")
    
    if artes_pct > overall_mean:
        print(f"   ✅ Artes is ABOVE the average (+{artes_pct - overall_mean:.2f} pp)")
    else:
        print(f"   ℹ️ Artes is BELOW the average ({artes_pct - overall_mean:.2f} pp)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: % missing by faculty
colors = ['crimson' if f == 'Artes' else 'steelblue' for f in faculty_desc_summary['faculty']]
axes[0].barh(faculty_desc_summary['faculty'], faculty_desc_summary['pct_missing'], color=colors, alpha=0.8)
axes[0].axvline(x=faculty_desc_summary['pct_missing'].mean(), color='black', linestyle='--', label='Overall mean')
axes[0].set_xlabel('% missing / not-useful descriptions')
axes[0].set_title('Description quality by faculty', fontweight='bold')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Plot 2: Absolute counts
df_plot = faculty_desc_summary.sort_values('desc_missing_or_short', ascending=False)
axes[1].bar(df_plot['faculty'], df_plot['desc_missing_or_short'], color='orange', edgecolor='black')
axes[1].set_xlabel('faculty')
axes[1].set_ylabel('Number of missing / not-useful descriptions')
axes[1].set_title('Absolute count of problematic descriptions', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Relation with language
print("\n🌐 Faculty × language (only records with useful descriptions):")
usable_products = products_bogota[~products_bogota['desc_missing_or_short']]
faculty_language_usable = pd.crosstab(
    usable_products['faculty'],
    usable_products['language'],
    normalize='index'
).round(3) * 100

display(faculty_language_usable)

# Clean temporary column
products_bogota.drop(columns=['desc_missing_or_short'], inplace=True)

# %% [markdown] id="Lmv_NylrwW91"
# ## 2.15. Executive Summary (Text, Active Coauthors, Language Focus)

# %% id="d7d025df" outputId="905ad028-58fc-4c5a-e28d-35172b9b8ac8"
# Final EDA summary (text, active coauthors, language focus)
print("EXECUTIVE SUMMARY OF EDA - MAIN FOCUS")

# Base text metrics
valid_titles = products_bogota['original_title'].notna().sum()
valid_descriptions = products_bogota['original_description'].notna().sum()

title_tokens = products_bogota['original_title'].fillna('').astype(str).apply(
    lambda x: len(preprocess_text_full(x).split()) if x.strip() else 0
)
desc_tokens = products_bogota['original_description'].fillna('').astype(str).apply(
    lambda x: len(preprocess_text_full(x).split()) if x.strip() else 0
)

# Active coauthors metrics
papers_with_actives = products_bogota['coauthors'].apply(
    lambda x: has_active_professor(x, active_professors_names)
).sum()

# Main language distribution
language_dist = products_bogota['language'].value_counts(dropna=False)
top_5_languages = language_dist.head(5)

summary_text = f"""
DATASET:
  - {len(products_bogota)} academic production records

FOCUS 1: TEXT VARIABLES
  - Records with non-null title: {valid_titles} ({valid_titles/len(products_bogota)*100:.1f}%)
  - Records with non-null description: {valid_descriptions} ({valid_descriptions/len(products_bogota)*100:.1f}%)
  - Avg title length (post-preprocessing): {title_tokens.mean():.2f} tokens
  - Avg description length (post-preprocessing): {desc_tokens.mean():.2f} tokens
  - Very short titles (<3 tokens): {(title_tokens < 3).sum()} ({(title_tokens < 3).sum()/len(products_bogota)*100:.1f}%)
  - Not-useful descriptions (<10 tokens): {(desc_tokens < 10).sum()} ({(desc_tokens < 10).sum()/len(products_bogota)*100:.1f}%)

FOCUS 2: ACTIVE COAUTHORS
  - Records with at least 1 active faculty: {papers_with_actives} ({papers_with_actives/len(products_bogota)*100:.1f}%)
  - Avg coauthors per production (overall): {products_bogota['num_coauthors'].mean():.2f}
  - Note: non-faculty collaborators were filtered for main-frequency analysis

FOCUS 3: LANGUAGE
  - Detected languages (including NaN): {products_bogota['language'].nunique(dropna=False)}
  - Records without language specified: {products_bogota['language'].isna().sum()} ({products_bogota['language'].isna().sum()/len(products_bogota)*100:.1f}%)
  - Top 5 languages by frequency:
{top_5_languages.to_string()}

RELATED ANALYSES (SECONDARY, CONNECTED TO FOCUS)
  - faculty × missing / not-useful description: assessed to identify capture biases
  - language × text length × active coauthors: evaluated in cross-analysis

OPERATIONAL CONCLUSION FOR NEXT PHASE (PREPROCESSING NOTEBOOK)
  1. Prioritize records with useful descriptions for topic modeling.
  2. Keep titles as fallback when descriptions are missing.
  3. Use active-faculty filter for primary authorship metrics.
  4. Handle records without language explicitly (heuristic or auto-detection).
"""

print(summary_text)
