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

class DocumentTopicLabeler:
    """
    A reusable class for automatically labeling clustered topics and evaluating cohesion
    using the DeepSeek LLM based on representative documents.
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
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _query_llm_json(self, comm_id, docs, semaphore):
        """Sends prompt to DeepSeek API strictly expecting JSON output."""
        if not self.client:
            return comm_id, "No API Key"
            
        system_prompt = (
            "You are an expert academic taxonomist. You will be provided with the most representative documents "
            "of a research cluster from a university. Your task is to:\n"
            "1. Assign a 'label' (maximum 5 words, in Spanish) that summarizes the research area.\n"
            "2. Evaluate the cohesion of the cluster ('cohesion_score': 'High', 'Medium', or 'Low'). A 'High' score "
            "indicates that the documents represent a clear and consistent research niche. 'Low' indicates "
            "that the documents are a diffuse mix of unrelated concepts.\n"
            "3. Provide a brief 'reasoning' (maximum 2 sentences) explaining your cohesion evaluation.\n"
            "You MUST respond STRICTLY in JSON format with the keys: 'label', 'cohesion_score', 'reasoning'."
        )
        
        docs_text = "\n".join([f"- {doc}" for doc in docs])
        user_prompt = f"Documents:\n{docs_text}"
        
        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
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
                raise ValueError(f"Model refused the request: {message.refusal}")
                
            content = message.content
            if not content:
                raise ValueError("API returned empty content")
                
            content = content.strip()
            # Strip markdown blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                parsed = json.loads(content)
                label = parsed.get("label", "Error")
                score = parsed.get("cohesion_score", "Low")
                reasoning = parsed.get("reasoning", "")
                result = {"label": label, "cohesion_score": score, "reasoning": reasoning}
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON Parsing Error: {e} | Content: {content[:100]}")
                
            return int(comm_id), result

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
            dict: Mapping of {cluster_id (int): {"label": ..., "cohesion_score": ..., "reasoning": ...}}.
        """
        if len(df) != embeddings.shape[0]:
            raise ValueError("Mismatch between metadata rows and embeddings size!")

        # Reset index to ensure alignment with numpy embeddings matrix
        df = df.reset_index(drop=True)

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
                str_dict = json.load(f)
                checkpoint_dict = {}
                for k, v in str_dict.items():
                    # If it's an error, we ignore it so it gets re-evaluated
                    if isinstance(v, dict) and "Error" in v.get("label", ""):
                        continue
                    if isinstance(v, str) and "Error" in v:
                        continue
                    checkpoint_dict[int(k)] = v
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
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
            except Exception as e:
                print(f"Error for topic {topic_id}: {e}")
                raise
            
            message = response.choices[0].message
            if getattr(message, 'refusal', None):
                raise ValueError(f"Model refused the request: {message.refusal}")
                
            content = message.content
            if not content:
                raise ValueError("API returned empty content")
                
            content = content.strip()
            # Strip markdown blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                parsed = json.loads(content)
                label = parsed.get("label", "Error")
                score = parsed.get("cohesion_score", "Low")
                reasoning = parsed.get("reasoning", "")
                result = {"label": label, "cohesion_score": score, "reasoning": reasoning}
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON Parsing Error: {e} | Content: {content[:100]}")
                
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

class TopicMetaEvaluator:
    """
    A class for summarizing and comparing topic modeling methods using DeepSeek.
    """
    def __init__(self, api_key=None):
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
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com", max_retries=5)

    def evaluate_method(self, method_name, method_evaluations, metrics=None):
        if not self.client:
            return "No API Key"
            
        system_prompt = (
            "You are an expert academic taxonomist. Analyze the topic evaluations (labels and cohesion scores) "
            "for a specific topic modeling method. You will also be provided with quantitative metrics (Silhouette, NPMI, etc.) "
            "if available. Write a concise, 1-paragraph summary of its strengths, "
            "weaknesses, and overall quality in grouping academic texts. Do not use conversational filler."
        )
        
        # Format evaluations to avoid context limit (just label and cohesion)
        compact_evals = []
        for cid, eval_data in method_evaluations.items():
            if isinstance(eval_data, dict):
                compact_evals.append(f"- Topic {cid}: {eval_data.get('label', '')} [{eval_data.get('cohesion_score', '')}]")
            else:
                compact_evals.append(f"- Topic {cid}: {eval_data}")
                
        eval_text = "\n".join(compact_evals)
        
        metrics_text = ""
        if metrics:
            metrics_text = "Quantitative Metrics:\n" + "\n".join([f"- {k}: {v}" for k, v in metrics.items()]) + "\n\n"
            
        user_prompt = f"Method: {method_name}\n\n{metrics_text}Topic Evaluations:\n{eval_text}\n\nProvide the summary."
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error evaluating method {method_name}: {e}")
            return f"Error evaluating method: {e}"

    def generate_final_verdict(self, methods_summaries):
        if not self.client:
            return "No API Key"
            
        system_prompt = (
            "You are the head of an academic data mining team. You have received summaries of how different "
            "topic modeling methods performed on the university's research dataset. Your task is to write a "
            "final, concise comparative conclusion. Identify the best method(s) for the project (focusing on "
            "providing accurate, highly cohesive and useful topics for a recommendation system) and briefly "
            "explain why it outperforms the others. Use markdown formatting. Do not exceed 3 paragraphs."
        )
        
        summaries_text = "\n\n".join([f"### {name}\n{summary}" for name, summary in methods_summaries.items()])
        user_prompt = f"Here are the summaries of each method:\n{summaries_text}\n\nPlease provide the final comparative verdict."
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating final verdict: {e}")
            return f"Error generating final verdict: {e}"
