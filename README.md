# Horus: Comparative Analysis of Classical vs. State-of-the-Art Topic Modeling on Academic Production

Horus is an advanced data mining and semantic analysis platform designed to map, group, and analyze academic production data from the Universidad Nacional de Colombia (Bogotá campus). 

Traditional university structures classify research using rigid administrative divisions (faculties and departments). However, actual scientific research is inherently fluid, cross-disciplinary, and collaborative. Horus overcomes these administrative silos by identifying latent research topics and mapping collaborative networks directly from the texts (titles, abstracts, and metadata) of academic publications.

The primary objective of this project is to propose, implement, and validate a **State-of-the-Art (SOTA) topic modeling pipeline** using multilingual transformer embeddings and topology-based community detection, and benchmark it against **classical NLP baselines** (TF-IDF, LSA, K-Means, and LDA).

---

## 📊 Academic Requirements Mapping (Competencias de Minería de Datos)

To satisfy the requirements of the academic curriculum, this repository implements a complete data mining pipeline covering the five key stages of data analysis:

| Stage (Etapa) | Description (Descripción) | Implementations (Técnicas Aplicadas) | Notebooks |
| :--- | :--- | :--- | :--- |
| **1. Descriptive Analysis**<br>*(Análisis Descriptivo)* | Exploratory analysis of raw academic datasets across faculties. | Missing value analysis, temporal distribution analysis, faculty productivity, co-author distribution, and language representation checks. | [01_eda.ipynb](file:///notebooks/01_eda.ipynb) |
| **2. Preprocessing**<br>*(Procesamiento)* | Cleaning and formatting text for embedding and symbolic models. | Unicode normalization, mojibake resolution, dual preprocessing strategy (minimal cleaning for transformer-based pipelines vs. spaCy lemmatization, stopword removal, and n-gram extraction for classical models). | [02_preprocessing.ipynb](file:///notebooks/02_preprocessing.ipynb) |
| **3. Association**<br>*(Asociación)* | Mining co-authorship transactional patterns and groups. | Transactional formatting of author groups, **Apriori Algorithm** for frequent itemset mining, co-authorship association rules, and co-authorship graph construction. | [03_association.ipynb](file:///notebooks/03_association.ipynb) |
| **4. Clustering / Topic Modeling**<br>*(Agrupación)* | Unsupervised discovery of latent academic topics. | **Classical NLP:** TF-IDF, Latent Semantic Analysis (LSA), K-Means, Agglomerative Hierarchical Clustering, and classical LDA.<br>**Modern NLP:** Jina Embeddings v5, UMAP + HDBSCAN, FASTopic neural modeling, LLM-XTM (VAE + LLM refinement), and PRISM (LoRA fine-tuning) + Leiden community detection. | [04_classical_clustering.ipynb](file:///notebooks/04_classical_clustering.ipynb)<br>[05_modern_clustering.ipynb](file:///notebooks/05_modern_clustering.ipynb)<br>[07_fastopic_modeling.ipynb](file:///notebooks/07_fastopic_modeling.ipynb)<br>[08_llm_xtm_modeling.ipynb](file:///notebooks/08_llm_xtm_modeling.ipynb)<br>[09_prism_finetuning.ipynb](file:///notebooks/09_prism_finetuning.ipynb)<br>[10_graph_topic_modeling.ipynb](file:///notebooks/10_graph_topic_modeling.ipynb) |
| **5. Classification**<br>*(Clasificación)* | Predictive modeling of administrative departments. | **Logistic Regression, Random Forest, and LightGBM** classifiers trained to predict administrative faculties from semantic embeddings. Evaluates classification overlap as a measure of interdisciplinary boundary crossing. | [12_classification.ipynb](file:///notebooks/12_classification.ipynb) |

---

## 🛠️ Repository Architecture & Notebook Pipeline

The project is structured as a sequential 12-notebook pipeline:

1. **`01_eda.ipynb` (Exploratory Data Analysis):** Investigates raw Excel datasets for 11 university faculties, examining duplicate publications, null distributions, date features, and text lengths.
2. **`02_preprocessing.ipynb` (Dual Preprocessing):** Implements a clean Unicode mapping and prepares two subsets: a rich text dataset preserving grammatical context for neural embeddings, and a normalized, lemmatized, stopword-free dataset for classical bag-of-words models.
3. **`03_association.ipynb` (Association Rules & Networks):** Discovers transactional authorship patterns, identifying frequent collaborating teams and generating co-authorship network graphs.
4. **`04_classical_clustering.ipynb` (Classical NLP Baselines):** Creates sparse TF-IDF matrices, applies LSA, and benchmarks K-Means (optimizing $K$ via Elbow Method), Agglomerative Clustering, and Latent Dirichlet Allocation (LDA) with pyLDAvis visualization.
5. **`05_modern_clustering.ipynb` (Modern Embeddings & Clustering):** Extracts dense multilingual representations using `jina-embeddings-v5-text-small` (with the `clustering` task adapter) and runs Modern K-Means and UMAP + HDBSCAN.
6. **`06_comparative_evaluation.ipynb` (Baseline Comparison):** Performs an initial comparative evaluation of the basic clustering methods using geometric internal metrics.
7. **`07_fastopic_modeling.ipynb` (Neural Topic Modeling):** Implements **FASTopic**, a fast, dual-relation neural topic model. Grids-searches parameters for topic coherence (NPMI) and diversity (PUV) before training on the entire dataset.
8. **`08_llm_xtm_modeling.ipynb` (Cross-Lingual LLM Refinement):** Trains a Variational Autoencoder (VAE) cross-lingual topic model (LLM-XTM) and integrates the DeepSeek API to align and refine multi-language topic representative terms.
9. **`09_prism_finetuning.ipynb` (PRISM Fine-Tuning):** Implements the **Precision-Informed Semantic Modeling (PRISM)** framework. Samples intra- and inter-faculty document pairs, labels semantic similarity via a DeepSeek teacher model, and fine-tunes Jina v5's adapter using LoRA and CoSENT loss.
10. **`10_graph_topic_modeling.ipynb` (Leiden Graph Community Detection):** Constructs a cosine-similarity k-NN graph in the PRISM-adapted vector space using FAISS, followed by Leiden community detection to extract natural topological topics.
11. **`11_final_model_evaluation.ipynb` (Universal Benchmarking & Verdict):** Consolidates assignments across all 9 modeling approaches. Computes universal geometry metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and semantic metrics (NPMI, Diversity) on the same embedding space. Conducts a qualitative review of topic cohesion using DeepSeek to produce a final meta-evaluation.
12. **`12_classification.ipynb` (Faculty Classification & Overlap):** Evaluates classification accuracy when predicting administrative faculties from Base vs. PRISM embeddings, proving the success of PRISM's space adaptation and measuring interdisciplinary boundaries.

---

## 📈 Key Results & Insights

* **Classical vs. Modern Geometry:** Classical TF-IDF methods fail to group publications cleanly in geometric vector space (Silhouette scores near $-0.01$). Standard Jina v5 embeddings improve geometry, while PRISM fine-tuning adapts the space to group research domains with high precision.
* **Semantic Coherence:** Modern K-Means on Jina v5 embeddings (NPMI = $0.51$) and FASTopic (NPMI = $0.46$) show significantly higher semantic coherence than Classical LDA (NPMI = $0.31$).
* **PRISM Adaptation:** The classification experiment in `12_classification.ipynb` shows that Base Embeddings achieve ~31% accuracy when predicting the rigid administrative faculty (since they dilute the faculty tokens), while **PRISM Adapted Embeddings reach ~94% accuracy**, showcasing the adapter's ability to warp the vector space to capture institutional groupings while preserving natural interdisciplinary bridges.
* **Interdisciplinary Boundaries:** In classification matrices, persistent confusion between certain faculties (e.g., *Science* vs. *Medicine*, or *Engineering* vs. *Agrarian Sciences*) exposes actual interdisciplinary research overlap, confirming that administrative groupings do not represent strict boundaries for scientific knowledge.

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fvalderramab/proyecto_horus.git
   cd proyecto_horus
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements-prep.txt
   ```

4. **Environment Variables:**
   For notebooks utilizing LLM labeling and refinement (08, 09, 11), set your DeepSeek API key:
   ```bash
   export DEEPSEEK_API_KEY="your-api-key-here"
   ```

5. **Run the notebooks:**
   Start the Jupyter Lab server and execute notebooks sequentially:
   ```bash
   jupyter lab
   ```

*Note: Due to the size of the embeddings matrix and neural model training, notebooks 05, 07, 08, 09, and 10 are optimized for execution on GPU-accelerated environments (such as Google Colab with T4 GPU).*