import os
import json
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm as async_tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import openai
from openai import AsyncOpenAI

class CentroidTopicLabeler:
    """
    A reusable class for automatically labeling clustered topic embeddings
    using the DeepSeek LLM. Extracted from the topic modeling pipeline.
    """
    def __init__(
        self, 
        api_key=None, 
        top_k=10, 
        semaphore_limit=15, 
        checkpoint_dir="data/processed", 
        checkpoint_filename="llm_labels_checkpoint.json"
    ):
        self.top_k = top_k
        self.semaphore_limit = semaphore_limit
        
        self.checkpoint_path = Path(checkpoint_dir) / checkpoint_filename
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                try:
                    from google.colab import userdata
                    api_key = userdata.get('DEEPSEEK_API_KEY')
                except ImportError:
                    pass
                    
        self.api_key = api_key
        if not self.api_key:
            print("WARNING: DEEPSEEK_API_KEY not found. API calls will fail.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def _safe_save_checkpoint(self, data):
        temp_path = str(self.checkpoint_path) + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_path, self.checkpoint_path)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _query_llm_json(self, comm_id, docs, semaphore):
        """Sends prompt to DeepSeek API strictly expecting JSON output."""
        if not self.client:
            return comm_id, "No API Key"
            
        system_prompt = (
            "You are an expert academic taxonomist at Universidad Nacional de Colombia. "
            "Synthesize a single, concise academic research area name that encompasses the provided documents. "
            "Output MUST be strictly in Spanish. Max 5 words. No conversational filler. "
            "You MUST output a valid JSON object with a single key 'label'."
        )
        
        docs_text = "\n".join([f"- {doc}" for doc in docs])
        user_prompt = f"Documents:\n{docs_text}"
        
        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=64
                )
            except openai.RateLimitError as e:
                print(f"Rate limit exceeded for cluster {comm_id}: {e}")
                raise
            except openai.APIConnectionError as e:
                print(f"Network error for cluster {comm_id}: {e}")
                raise
            except openai.APIError as e:
                print(f"API error for cluster {comm_id}: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error for cluster {comm_id}: {e}")
                raise
            
            message = response.choices[0].message
            if getattr(message, 'refusal', None):
                print(f"Model refused the request for cluster {comm_id}: {message.refusal}")
                return int(comm_id), "Error: Request Refused"
                
            content = message.content
            if not content:
                return int(comm_id), "Error: No Content"
                
            content = content.strip()
            try:
                parsed = json.loads(content)
                label = parsed.get("label", "Error en Formato")
            except json.JSONDecodeError:
                label = "Error en JSON"
                
            return int(comm_id), str(label).strip()

    async def generate_labels(self, df, embeddings, cluster_col='cluster_id', text_col='embeddings_text', noise_label_id=-1):
        """
        Generates and returns LLM labels for each valid cluster.
        
        Args:
            df (pd.DataFrame): The metadata dataframe.
            embeddings (np.ndarray): The embeddings matching the dataframe rows.
            cluster_col (str): Column name containing the opaque cluster IDs.
            text_col (str): Column name containing the documents text.
            noise_label_id (int): ID representing noise/outliers to be skipped.
            
        Returns:
            dict: Mapping of {cluster_id (int): "Topic Label"}.
        """
        if len(df) != embeddings.shape[0]:
            raise ValueError("Mismatch between metadata rows and embeddings size!")

        valid_clusters = df[df[cluster_col] != noise_label_id][cluster_col].unique()
        print(f"Found {len(valid_clusters)} valid clusters (excluding noise).")

        cluster_top_docs = {}
        for cluster_id in valid_clusters:
            idx = df[df[cluster_col] == cluster_id].index
            comm_embeddings = embeddings[idx]
            
            # L2 normalized centroid
            centroid = np.mean(comm_embeddings, axis=0)
            centroid_norm = centroid / np.linalg.norm(centroid)
            
            sims = cosine_similarity(comm_embeddings, centroid_norm.reshape(1, -1)).flatten()
            
            k_actual = min(self.top_k, len(sims))
            top_k_local_idx = np.argsort(sims)[::-1][:k_actual]
            top_k_global_idx = idx[top_k_local_idx]
            
            top_texts = df.loc[top_k_global_idx, text_col].tolist()
            cluster_top_docs[cluster_id] = top_texts

        print("Finished extracting core representative documents for all clusters.")

        if self.checkpoint_path.exists():
            print(f"Resuming from checkpoint: {self.checkpoint_path}")
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                # keys will be strings in JSON
                str_dict = json.load(f)
                checkpoint_dict = {int(k): v for k, v in str_dict.items()}
        else:
            checkpoint_dict = {}

        semaphore = asyncio.Semaphore(self.semaphore_limit)
        tasks = []
        for cluster_id, docs in cluster_top_docs.items():
            if int(cluster_id) not in checkpoint_dict:
                tasks.append(self._query_llm_json(cluster_id, docs, semaphore))
                
        if not tasks:
            print("All clusters are already labeled.")
            return checkpoint_dict
            
        print(f"Processing {len(tasks)} missing clusters...")
        
        for f in async_tqdm.as_completed(tasks):
            try:
                cluster_id, label = await f
                checkpoint_dict[cluster_id] = label
                self._safe_save_checkpoint(checkpoint_dict)
            except Exception as e:
                print(f"Failed to process a cluster after max retries: {e}")
                
        return checkpoint_dict

class TopTermsTopicLabeler:
    """
    A class for automatically labeling and evaluating cohesion of topics
    based on their top terms using the DeepSeek LLM.
    """
    def __init__(
        self, 
        api_key=None, 
        semaphore_limit=15, 
        checkpoint_dir="data/processed", 
        checkpoint_filename="llm_top_terms_checkpoint.json"
    ):
        self.semaphore_limit = semaphore_limit
        
        self.checkpoint_path = Path(checkpoint_dir) / checkpoint_filename
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                try:
                    from google.colab import userdata
                    api_key = userdata.get('DEEPSEEK_API_KEY')
                except ImportError:
                    pass
                    
        self.api_key = api_key
        if not self.api_key:
            print("WARNING: DEEPSEEK_API_KEY not found. API calls will fail.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def _safe_save_checkpoint(self, data):
        temp_path = str(self.checkpoint_path) + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_path, self.checkpoint_path)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _query_llm_json(self, topic_id, top_terms, semaphore):
        if not self.client:
            return topic_id, {"label": "No API Key", "cohesion_score": "Low", "reasoning": ""}
            
        system_prompt = (
            "You are an expert academic taxonomist. You will be provided with the most representative words "
            "(top terms) of a research cluster from a university. Your task is to:\n"
            "1. Assign a 'label' (maximum 5 words, in English) that summarizes the research area.\n"
            "2. Evaluate the cohesion of the cluster ('cohesion_score': 'High', 'Medium', or 'Low'). A 'High' score "
            "indicates that the words represent a clear and consistent research niche. 'Low' indicates "
            "that the words are a diffuse mix of unrelated concepts.\n"
            "3. Provide a brief 'reasoning' (maximum 2 sentences) explaining your cohesion evaluation.\n"
            "You MUST respond STRICTLY in JSON format with the keys: 'label', 'cohesion_score', 'reasoning'."
        )
        
        user_prompt = f"Top Terms: {top_terms}"
        
        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=150
                )
            except Exception as e:
                print(f"Error for topic {topic_id}: {e}")
                raise
            
            message = response.choices[0].message
            if getattr(message, 'refusal', None):
                print(f"Model refused the request for topic {topic_id}: {message.refusal}")
                return str(topic_id), {"label": "Error: Request Refused", "cohesion_score": "Low", "reasoning": ""}
                
            content = message.content
            if not content:
                return str(topic_id), {"label": "Error: No Content", "cohesion_score": "Low", "reasoning": ""}
                
            content = content.strip()
            try:
                parsed = json.loads(content)
                label = parsed.get("label", "Error")
                score = parsed.get("cohesion_score", "Low")
                reasoning = parsed.get("reasoning", "")
                result = {"label": label, "cohesion_score": score, "reasoning": reasoning}
            except json.JSONDecodeError:
                result = {"label": "JSON Error", "cohesion_score": "Low", "reasoning": ""}
                
            return str(topic_id), result

    async def generate_labels(self, df, id_col='cluster', terms_col='top_terms', noise_id=-1):
        """
        Generates and returns LLM evaluation for each valid cluster.
        
        Args:
            df (pd.DataFrame): The metadata dataframe containing top terms.
            id_col (str): Column name containing the cluster IDs.
            terms_col (str): Column name containing the comma-separated top terms.
            noise_id (int or str): ID representing noise/outliers to be skipped.
            
        Returns:
            dict: Mapping of {cluster_id (str): {"label": ..., "cohesion_score": ..., "reasoning": ...}}.
        """
        valid_df = df[df[id_col] != noise_id]
        print(f"Found {len(valid_df)} valid clusters (excluding noise).")

        cluster_terms = {str(row[id_col]): row[terms_col] for _, row in valid_df.iterrows()}

        if self.checkpoint_path.exists():
            print(f"Resuming from checkpoint: {self.checkpoint_path}")
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)
        else:
            checkpoint_dict = {}

        semaphore = asyncio.Semaphore(self.semaphore_limit)
        tasks = []
        for cluster_id, terms in cluster_terms.items():
            if str(cluster_id) not in checkpoint_dict:
                tasks.append(self._query_llm_json(cluster_id, terms, semaphore))
                
        if not tasks:
            print("All clusters are already evaluated.")
            return checkpoint_dict
            
        print(f"Processing {len(tasks)} missing clusters...")
        
        for f in async_tqdm.as_completed(tasks):
            try:
                cluster_id, result = await f
                checkpoint_dict[cluster_id] = result
                self._safe_save_checkpoint(checkpoint_dict)
            except Exception as e:
                print(f"Failed to process a cluster after max retries: {e}")
                
        return checkpoint_dict
