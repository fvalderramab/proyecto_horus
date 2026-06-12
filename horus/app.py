import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import plotly.express as px

from horus.config import (
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR
)
from horus.recommender import HorusRecommender
from horus.graph_utils import (
    get_professor_ego_network,
    normalize_name
)

# ------------------------------------------------------------
# 1. STREAMLIT PAGE SETUP & WOW AESTHETICS
# ------------------------------------------------------------
st.set_page_config(
    page_title="HORUS - Academic Recommender",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode Glassmorphism Style injection
st.markdown("""
<style>
    /* Main app layout modifications */
    .stApp {
        background-color: #0d0e12;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #151821;
        padding: 8px 16px;
        border-radius: 12px;
        border: 1px solid #272c38;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        color: #6366f1 !important;
        background-color: #1e2230 !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    /* Cards for recommendations */
    .advisor-card {
        background: rgba(21, 24, 33, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .advisor-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Metric badges */
    .badge-active {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-retired {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-score {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        float: right;
    }
    
    /* Highlight titles */
    h1 {
        background: linear-gradient(135deg, #a5b4fc, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 8px !important;
    }
    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    
    /* Input box styling */
    .stTextArea textarea {
        background-color: #151821 !important;
        color: #e2e8f0 !important;
        border: 1px solid #272c38 !important;
        border-radius: 12px !important;
    }
    
    /* Accordions */
    .st-d5 {
        background-color: #151821 !important;
        border: 1px solid #272c38 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. DATA LOADER HELPERS
# ------------------------------------------------------------
@st.cache_resource
def get_recommender():
    rec = HorusRecommender()
    rec.lazy_load()
    return rec

@st.cache_resource
def load_llm_labels() -> dict[str, dict]:
    """
    Loads the LLM topic labels generated during the final model evaluation (Notebook 11).
    """
    labels_path = OUTPUTS_DIR / "11_llm_evals_Graph_Leiden_MinComm.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

try:
    recommender = get_recommender()
    llm_labels = load_llm_labels()
except Exception as e:
    st.error(f"Error loading system resources: {e}")
    st.info("Ensure you have run the notebooks and generated all processed files in `data/processed/`.")
    st.stop()

# ------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ------------------------------------------------------------
# Use an emoji as a logo
st.sidebar.markdown("## 🦉 HORUS PILOT SYSTEM")
st.sidebar.markdown("This portal matches thesis proposals with academic advisors using **PRISM semantic embeddings** and **topological co-authorship graphs**.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Recommendation Weights")

# Sliders for weights with tooltips explaining their influence
w_sem = st.sidebar.slider(
    "Semantic Relevance", 0.0, 1.0, 0.5, 0.05,
    help="Determines the weight of the semantic similarity (Jina v5 + LoRA/PRISM) between your proposal and the professor's past papers. High values favor strict topic alignment."
)
w_pr = st.sidebar.slider(
    "Network Authority", 0.0, 1.0, 0.3, 0.05,
    help="Determines the weight of the professor's PageRank centrality in the global co-authorship network. High values favor highly collaborative and influential researchers."
)
w_dens = st.sidebar.slider(
    "Co-occurrence Density", 0.0, 1.0, 0.2, 0.05,
    help="Determines the weight of the volume/count of matching publications. High values favor professors with a large quantity of papers in the relevant topic."
)

# Normalise weights on display
total_w = w_sem + w_pr + w_dens
if total_w > 0:
    w_sem_norm = w_sem / total_w
    w_pr_norm = w_pr / total_w
    w_dens_norm = w_dens / total_w
else:
    w_sem_norm, w_pr_norm, w_dens_norm = 0.34, 0.33, 0.33
    
st.sidebar.caption(f"Normalized weights: Semantic {w_sem_norm:.2f} | PageRank {w_pr_norm:.2f} | Density {w_dens_norm:.2f}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Search Configuration")
top_k = st.sidebar.slider(
    "Retrieved Publications (top_k)", 10, 100, 50, 5,
    help="Number of nearest publications retrieved from the FAISS vector space to perform the aggregation. Larger values broaden the search, smaller values narrow focus to precise matches."
)
active_only = st.sidebar.checkbox("Show ACTIVE professors only", value=True)

# ------------------------------------------------------------
# 4. PORTAL TABS
# ------------------------------------------------------------
tab_rec, tab_leiden, tab_inter = st.tabs([
    "🤝 Advisor Recommender",
    "🕸️ Leiden Community Explorer",
    "🧬 Interdisciplinary Diagnosis"
])

# ============================================================
# TAB 1: ADVISOR RECOMMENDER
# ============================================================
with tab_rec:
    st.subheader("Match Your Thesis Proposal")
    st.markdown("Enter your thesis details below. Horus will match it semantic-by-semantic to find professors with matching expertise and high network authority.")
    
    # User Input
    title_input = st.text_input("Title of the thesis", placeholder="e.g., Optimización multiobjetivo para el tráfico urbano en Bogotá")
    abstract_input = st.text_area("Abstract / Proposal description", placeholder="Write a short summary (1-2 paragraphs) of your goals, methods, and expected results...", height=150)
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        search_clicked = st.button("🚀 Find Advisors", use_container_width=True)
        
    if search_clicked or "last_recs" in st.session_state:
        # Save recommendations to session state to prevent losing them on network selector interaction
        if search_clicked:
            if not title_input.strip() and not abstract_input.strip():
                st.warning("Please enter a title or description for your proposal.")
                st.stop()
            combined_query = f"{title_input}. {abstract_input}"
            with st.spinner("Generating PRISM semantic embeddings & computing hybrid scores..."):
                st.session_state.last_recs = recommender.recommend_advisors(
                    proposal_text=combined_query,
                    top_k_docs=top_k,
                    weight_semantic=w_sem_norm,
                    weight_pagerank=w_pr_norm,
                    weight_density=w_dens_norm,
                    active_only=active_only
                )
                st.session_state.search_performed = True

        recs = st.session_state.last_recs
        
        if not recs:
            st.error("No official advisors found matching the criteria.")
        else:
            if search_clicked:
                st.success(f"Matched {len(recs)} potential advisors. Showing the top matches.")
                
            # Split in two columns: Left for cards, Right for Network visualization
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                st.subheader("💡 Recommended Advisors")
                
                # Display top 5 matches
                for i, r in enumerate(recs[:5]):
                    badge_html = f'<span class="badge-active">Activo</span>' if r['status'] == 'Activo' else f'<span class="badge-retired">Retirado</span>'
                    
                    st.markdown(f"""
                    <div class="advisor-card">
                        <span class="badge-score">Match: {r['hybrid_score']*100:.1f}%</span>
                        <h3 style="margin: 0 0 6px 0;">{i+1}. {r['name']}</h3>
                        <div style="margin-bottom: 16px;">
                            {badge_html}
                            <span style="color: #94a3b8; font-size: 14px; margin-left: 10px;">Total Products: <b>{r['total_products']}</b></span>
                            <span style="color: #94a3b8; font-size: 14px; margin-left: 10px;">Matched Papers: <b>{r['papers_matched_count']}</b></span>
                        </div>
                        <div style="font-size: 14px; color: #cbd5e1; margin-bottom: 8px;">
                            <b>Score Breakdown:</b>
                        </div>
                        <div style="font-size: 13px; color: #94a3b8; margin-bottom: 12px; display: flex; gap: 20px;">
                            <span>🧬 Semantic Similarity: <b>{r['sub_scores']['semantic']*100:.1f}%</b></span>
                            <span>👑 Network Authority: <b>{r['sub_scores']['pagerank']*100:.1f}%</b></span>
                            <span>📚 Paper Density: <b>{r['sub_scores']['density']*100:.1f}%</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Accordion showing matched papers
                    with st.expander(f"View matching publications of {r['name']}"):
                        for paper in r['matched_papers']:
                            st.markdown(f"""
                            - **{paper['title']}**
                              *Similarity: {paper['similarity']:.3f} | Faculty: {paper['faculty'].capitalize()} | Year: {paper['date'][:4]}*
                            """)
                            
            with col_right:
                st.subheader("🕸️ Advisor's Local Network")
                
                # Interactive Selector for the Graph to display (Punto 4)
                advisor_options = {r['name']: r for r in recs[:5]}
                selected_advisor_name = st.selectbox(
                    "Select recommended advisor to view their local network:",
                    options=list(advisor_options.keys())
                )
                
                selected_advisor = advisor_options[selected_advisor_name]
                st.markdown(f"Visualizing the local co-authorship ego-network for **{selected_advisor['name']}** (depth=1).")
                
                # Compute ego-network
                ego_G = get_professor_ego_network(recommender.G, selected_advisor['username'], depth=1)
                
                if ego_G.number_of_nodes() > 0:
                    norm_advisor = normalize_name(selected_advisor['username'])
                    # Filter neighbors to top 15 collaborators by weight to prevent browser freeze (Punto 4)
                    ego_nodes = list(ego_G.neighbors(norm_advisor))
                    weighted_neighbors = []
                    for n in ego_nodes:
                        weight = ego_G.get_edge_data(norm_advisor, n).get('weight', 1)
                        weighted_neighbors.append((n, weight))
                    weighted_neighbors = sorted(weighted_neighbors, key=lambda x: x[1], reverse=True)
                    
                    top_neighbors = [n for n, w in weighted_neighbors[:15]]
                    nodes_to_keep = [norm_advisor] + top_neighbors
                    ego_G = ego_G.subgraph(nodes_to_keep).copy()
                    # Build PyVis
                    net = Network(height="450px", width="100%", bgcolor="#0d0e12", font_color="white")
                    net.repulsion(node_distance=130, central_gravity=0.33, spring_length=100, spring_strength=0.10, damping=0.95)
                    
                    # Add nodes with custom design
                    norm_advisor = normalize_name(selected_advisor['username'])
                    for node in ego_G.nodes:
                        node_data = ego_G.nodes[node]
                        is_center = (node == norm_advisor)
                        
                        label = node_data.get('original_name', node.title())
                        status = node_data.get('status', 'Coauthor')
                        
                        # Designing nodes
                        if is_center:
                            color = "#fbbf24"  # Gold
                            size = 25
                            title = f"Center: {label} (Selected Advisor)"
                        elif status == "Activo":
                            color = "#6366f1"  # Indigo
                            size = 18
                            title = f"Professor: {label} (Active)"
                        elif status == "Retirado":
                            color = "#ef4444"  # Red
                            size = 15
                            title = f"Professor: {label} (Retired)"
                        else:
                            color = "#10b981"  # Emerald
                            size = 12
                            title = f"Co-author: {label} (External/Student)"
                            
                        net.add_node(
                            node, 
                            label=label, 
                            size=size, 
                            color=color, 
                            title=title
                        )
                        
                    # Add edges
                    for u, v, data in ego_G.edges(data=True):
                        weight = data.get('weight', 1)
                        net.add_edge(u, v, value=weight, color="#4b5563", title=f"{weight} co-publications")
                        
                    # Render HTML inside an IFrame to prevent Streamlit deprecation warnings
                    net_html_path = "horus_temp_ego.html"
                    net.write_html(net_html_path)
                    with open(net_html_path, "r", encoding="utf-8") as f:
                        net_html = f.read()
                    os.remove(net_html_path)
                    
                    # Use iframe standard base64 embedding to satisfy modern Streamlit requirements
                    b64_ego_html = base64.b64encode(net_html.encode('utf-8')).decode('utf-8')
                    components.iframe(src=f"data:text/html;base64,{b64_ego_html}", height=480)
                else:
                    st.info("This professor has no co-authors in the filtered co-authorship database.")

# ============================================================
# TAB 2: LEIDEN COMMUNITY EXPLORER
# ============================================================
with tab_leiden:
    st.subheader("Global Topic Map (Leiden Communities)")
    st.markdown("This interactive graph shows the semantic topics discovered in the university publications. Nodes represent **Leiden communities** (topics) and edges are the aggregate similarity links between them.")
    
    # Load and render Macro Graph HTML (Punto 1: Deszoom & IFrame warnings fix)
    macro_graph_path = OUTPUTS_DIR / "10_topic_graph_min_communities.html"
    if macro_graph_path.exists():
        with open(macro_graph_path, "r", encoding="utf-8") as f:
            macro_html = f.read()
        
        # Programmatically inject vis.js camera zoom scale (moveTo scale: 0.15 for deszoom view)
        # Replacing default network.fit() call inside PyVis script with a deszoomed camera fit
        macro_html = macro_html.replace(
            "network.fit()", 
            "network.fit(); network.moveTo({scale: 0.18});"
        )
        
        # Use st.components.v1.iframe to avoid the st.components.v1.html deprecation warning
        b64_macro_html = base64.b64encode(macro_html.encode('utf-8')).decode('utf-8')
        components.iframe(src=f"data:text/html;base64,{b64_macro_html}", height=600)
    else:
        st.info("Topic Graph visualization file missing. Make sure `outputs/10_topic_graph_min_communities.html` is generated.")

    st.markdown("---")
    st.subheader("🔍 Query Leiden Community Statistics")
    
    # Select community (Punto 1: Displaying LLM label in selectbox)
    valid_comms = sorted(recommender.df_meta[recommender.df_meta['community_id'] != -1]['community_id'].unique())
    
    def format_community_option(comm_id):
        str_id = str(comm_id)
        if str_id in llm_labels:
            return f"{comm_id} - {llm_labels[str_id]['label']}"
        return f"{comm_id} - Topic {comm_id}"
        
    selected_comm = st.selectbox(
        "Select Community ID to inspect", 
        options=valid_comms,
        format_func=format_community_option
    )
    
    if selected_comm is not None:
        comm_docs = recommender.df_meta[recommender.df_meta['community_id'] == selected_comm]
        
        # Display Cohesion Score and Reasoning from LLM (Punto 3)
        str_selected = str(selected_comm)
        if str_selected in llm_labels:
            cohesion = llm_labels[str_selected].get("cohesion_score", "N/A")
            reasoning = llm_labels[str_selected].get("reasoning", "No reasoning provided.")
            
            # Badge styling based on score
            if cohesion == "High":
                cohesion_html = '<span style="background-color: rgba(16, 185, 129, 0.2); color: #10b981; padding: 4px 10px; border-radius: 6px; font-size: 14px; font-weight: bold; border: 1px solid rgba(16, 185, 129, 0.4);">High</span>'
            elif cohesion == "Medium":
                cohesion_html = '<span style="background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 4px 10px; border-radius: 6px; font-size: 14px; font-weight: bold; border: 1px solid rgba(245, 158, 11, 0.4);">Medium</span>'
            else:
                cohesion_html = '<span style="background-color: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 4px 10px; border-radius: 6px; font-size: 14px; font-weight: bold; border: 1px solid rgba(239, 68, 68, 0.4);">Low</span>'
                
            st.markdown(f"**Cohesión Temática (LLM Eval):** {cohesion_html}", unsafe_allow_html=True)
            st.info(f"**Análisis de Cohesión (DeepSeek):** {reasoning}")
            st.markdown("---")
            
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Display Label as Title
            str_selected = str(selected_comm)
            label_title = llm_labels[str_selected]["label"] if str_selected in llm_labels else f"Topic {selected_comm}"
            st.markdown(f"#### 📊 Distribution: {label_title}")
            st.metric("Total Publications", len(comm_docs))
            
            # Faculty Distribution
            fac_counts = comm_docs['faculty'].value_counts()
            st.bar_chart(fac_counts)
            
        with col2:
            st.markdown(f"#### 🔬 Sample Titles in Community {selected_comm}")
            samples = comm_docs['original_title'].dropna().sample(n=min(8, len(comm_docs)), random_state=42)
            for s in samples:
                st.markdown(f"- {s}")
                
            # Top Authors
            st.markdown("#### 👑 Top active researchers in this community")
            all_coauthors = []
            for coauths in comm_docs['active_coauthors'].dropna():
                all_coauthors.extend(coauths.split(','))
            
            import collections
            counts = collections.Counter([normalize_name(c) for c in all_coauthors if c.strip()])
            top_profs = []
            for u_name, count in counts.most_common(5):
                if u_name in recommender.registry:
                    top_profs.append({
                        "Researcher": recommender.registry[u_name]["original_name"],
                        "Publications in Topic": count,
                        "Status": recommender.registry[u_name]["status"]
                    })
            if top_profs:
                st.table(pd.DataFrame(top_profs))
            else:
                st.caption("No registered official professors found in this community's sample.")

# ============================================================
# TAB 3: INTERDISCIPLINARY DIAGNOSIS
# ============================================================
with tab_inter:
    st.subheader("Interdisciplinary Overlap Diagnosis")
    st.markdown("""
    Administrative divisions in universities (e.g., faculties and departments) are often too rigid, dividing areas of knowledge that are inherently linked. 
    **HORUS** addresses this by modeling latent topics organically from semantic text representations, revealing the underlying multidisciplinary landscape.
    """)
    
    # Heatmap of Faculty vs. Leiden Community Overlap (Punto 3: Plotly Heatmap for ALL communities)
    st.subheader("🧬 Faculty to Leiden Community Overlap Heatmap")
    st.markdown("This interactive chart maps the percentage distribution of publications from each administrative faculty across *all* Leiden communities. Hover, zoom, and pan to explore the overlaps.")
    
    # Calculate cross-tabulation of all faculties vs all communities
    df_valid = recommender.df_meta[recommender.df_meta['community_id'] != -1].copy()
    
    # Get sorted lists
    all_comms = sorted(df_valid['community_id'].unique())
    all_faculties = sorted(df_valid['faculty'].unique())
    
    # Build crosstab (% distribution of each faculty)
    crosstab = pd.crosstab(
        df_valid['faculty'], 
        df_valid['community_id'],
        normalize='index' 
    ) * 100
    
    # Reindex to ensure consistency
    crosstab = crosstab.reindex(index=all_faculties, columns=all_comms, fill_value=0.0)
    
    # Format columns in the index to display: "ID: Label" (Punto 3)
    formatted_cols = []
    for col in crosstab.columns:
        str_col = str(col)
        if str_col in llm_labels:
            formatted_cols.append(f"{col}: {llm_labels[str_col]['label']}")
        else:
            formatted_cols.append(f"{col}: Topic {col}")
            
    crosstab.columns = formatted_cols
    
    # Draw Plotly Heatmap
    # Using Plotly solves the black-text contrast issue on dark mode natively and enables horizontal scrolling/zooming
    fig = px.imshow(
        crosstab,
        labels=dict(x="Leiden Community (Topic)", y="Administrative Faculty", color="% of Faculty Papers"),
        x=crosstab.columns,
        y=crosstab.index,
        color_continuous_scale="dense",
        aspect="auto"
    )
    
    # Update styling to match HSL Dark theme
    fig.update_layout(
        paper_bgcolor='#0d0e12',
        plot_bgcolor='#0d0e12',
        font=dict(color='#e2e8f0', family='Inter'),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=8, color='#94a3b8'),
            title=dict(font=dict(size=12, color='#e2e8f0')),
            gridcolor='rgba(255,255,255,0.05)',
            fixedrange=False # Allow zooming/panning on X-axis (communities)
        ),
        yaxis=dict(
            tickfont=dict(size=10, color='#94a3b8'),
            title=dict(font=dict(size=12, color='#e2e8f0')),
            gridcolor='rgba(255,255,255,0.05)',
            fixedrange=True # Lock Y-axis (faculties) to prevent vertical zoom issues (Punto 2)
        ),
        margin=dict(l=50, r=50, t=50, b=150),
        coloraxis_colorbar=dict(
            title="% of Papers",
            title_font=dict(color='#e2e8f0'),
            tickfont=dict(color='#94a3b8')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **🔍 Key Diagnosis Insights:**
    - **Horizontal Spreading**: Drag or zoom into specific Leiden communities (columns). If a single community contains significant percentages from multiple distinct faculties, it marks a **real-world multidisciplinary intersection** (e.g., Engineering, Medicine, and Sciences collaborating in Bio-materials).
    - **Vertical Segments**: Faculty rows spread wide across communities prove that administrative boundaries host highly diverse scientific lines of work, which are better categorized by Leiden communities than administrative labels.
    """)
