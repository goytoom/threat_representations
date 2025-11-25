"""
Complete Persona Vector Pipeline: Generation → Filtering → Extraction
Following Turner et al. (2024) methodology

Pipeline:
1. Generate contrastive responses (pos/neg system prompts)
2. Filter with GPT-4o judge (keep trait-consistent only)
3. Extract activations from filtered responses
4. Compute concept vectors (mean difference)

Usage:
CUDA_VISIBLE_DEVICES=2 python persona_vector_pipeline.py \
    --model_path /data/cyu/model_cache/mistral-7b-instruct-v0.3 \
    --config_path hostile_config.json \
    --output_dir hostile_persona_vector \
    --n_rollouts 10 \
    --use_gpt4_judge \
    --extract_activations
"""

import torch
import numpy as np
import json
import argparse
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ContrastiveResponse:
    """Single contrastive response with activation data"""
    prompt_id: int
    question_id: int
    question: str
    system_prompt: str
    prompt_type: str  # 'pos' or 'neg'
    rollout_id: int
    
    response: str
    response_length: int
    
    gpt4_score: Optional[float]
    is_trait_consistent: Optional[bool]
    gpt4_reasoning: str
    
    timestamp: str
    
    # NEW: Activation data
    activations: Optional[Dict[str, np.ndarray]] = None
    
    


class GPT4Judge:
    """GPT-4o judge for scoring responses"""
    
    def __init__(self, eval_prompt_template: str, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.eval_prompt_template = eval_prompt_template
        self.call_count = 0
        self.total_cost = 0.0
        
        print(f"✅ GPT-4o Judge initialized (model: {model})")
    
    def score_response(self, question: str, answer: str) -> Dict:
        """Score a response for hostility trait (0-100)"""
        eval_prompt = self.eval_prompt_template.format(
            question=question,
            answer=answer
        )
        
        try:
            self.call_count += 1
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Follow instructions exactly."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Calculate cost
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            cost = (input_tokens * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)
            self.total_cost += cost
            
            # Parse response
            if "REFUSAL" in response_text.upper():
                return {
                    'score': None,
                    'is_refusal': True,
                    'reasoning': 'Model refused to answer',
                    'success': True
                }
            
            score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response_text)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(100.0, score))
                
                return {
                    'score': score,
                    'is_refusal': False,
                    'reasoning': response_text,
                    'success': True
                }
            else:
                return {
                    'score': None,
                    'is_refusal': False,
                    'reasoning': f'Parse failed: {response_text[:50]}',
                    'success': False
                }
            
        except Exception as e:
            return {
                'score': None,
                'is_refusal': False,
                'reasoning': f'Error: {str(e)[:50]}',
                'success': False
            }
    
    def get_statistics(self) -> Dict:
        return {
            'total_calls': self.call_count,
            'total_cost_usd': round(self.total_cost, 4),
            'avg_cost_per_call': round(self.total_cost / max(1, self.call_count), 6)
        }


class PersonaVectorPipeline:
    """Complete pipeline: Generation → Filtering → Activation Extraction → Vector Computation"""
    
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 output_dir: str = "persona_vector_pipeline",
                 device: str = "cuda",
                 use_gpt4_judge: bool = True,
                 gpt4_model: str = "gpt-4o",
                 extract_activations: bool = True,
                 monitoring_mode: str = "comprehensive"):
        
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.device = device
        self.use_gpt4_judge = use_gpt4_judge
        self.extract_activations = extract_activations
        self.monitoring_mode = monitoring_mode
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        for subdir in ['responses', 'filtered', 'stats', 'concept_vectors', 'visualizations']:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
        
        # Load config
        print("Loading config...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"✅ Config loaded:")
        print(f"   - {len(self.config['instruction'])} instruction pairs")
        print(f"   - {len(self.config['questions'])} extraction questions")
        
        # Load model
        print("\nLoading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model config
        self.model_config = AutoConfig.from_pretrained(model_path)
        self.total_layers = self.model_config.num_hidden_layers
        
        print(f"✅ Model loaded:")
        print(f"   - Type: {self.model_config.model_type}")
        print(f"   - Layers: {self.total_layers}")
        print(f"   - Hidden size: {self.model_config.hidden_size}")
        
        # Initialize GPT-4o judge
        if self.use_gpt4_judge:
            self.gpt4_judge = GPT4Judge(
                eval_prompt_template=self.config['eval_prompt'],
                model=gpt4_model
            )
        else:
            self.gpt4_judge = None
            print("⚠️  GPT-4o judge disabled")
        
        # Activation monitoring setup
        self.activation_storage = {}
        self.hooks = []
        self.target_layers = self._configure_monitoring_layers()
        
        if self.extract_activations:
            print(f"\n✅ Activation extraction enabled:")
            print(f"   - Monitoring mode: {self.monitoring_mode}")
            print(f"   - Target layers: {self.target_layers}")
    
    def _configure_monitoring_layers(self) -> List[int]:
        """Configure which layers to monitor"""
        total_layers = self.total_layers
        
        mode_configs = {
            "light": [total_layers//4, total_layers//2, total_layers*3//4],
            "comprehensive": [max(1, total_layers//6), total_layers//3, total_layers//2, 
                             total_layers*2//3, total_layers*5//6, total_layers-2],
            "dense": list(range(2, total_layers, max(1, total_layers//10))),
            "full": list(range(total_layers))
        }
        
        if self.monitoring_mode not in mode_configs:
            self.monitoring_mode = "comprehensive"
        
        layers = mode_configs[self.monitoring_mode]
        layers = [l for l in layers if 0 <= l < total_layers]
        
        return layers
    
    def setup_activation_hooks(self):
        """Setup hooks to collect activations"""
        # Focus on key components
        focus_components = [
            "residual_stream",
            "self_attn",
            "self_attn.o_proj",
            "mlp",
            "mlp.down_proj",
        ]
        
        self.activation_storage.clear()
        
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Create hook patterns
        target_module_patterns = []
        for layer_num in self.target_layers:
            for component in focus_components:
                if component == "residual_stream":
                    pattern = f"model.layers.{layer_num}"
                    target_module_patterns.append((pattern, component, layer_num))
                else:
                    pattern = f"model.layers.{layer_num}.{component}"
                    target_module_patterns.append((pattern, component, layer_num))
        
        # Register hooks
        hook_count = 0
        for name, module in self.model.named_modules():
            for pattern, component_type, layer_num in target_module_patterns:
                if name == pattern:
                    def make_hook(layer_name, comp_type):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                activation = output[0]
                            else:
                                activation = output
                            
                            # Take last token activation
                            if activation.dim() == 3:
                                activation = activation[:, -1, :]
                            
                            storage_key = f"{layer_name}_{comp_type}" if comp_type != "residual_stream" else f"{layer_name}_residual_stream"
                            self.activation_storage[storage_key] = activation.detach().cpu().float().numpy()
                        
                        return hook_fn
                    
                    hook = module.register_forward_hook(make_hook(name, component_type))
                    self.hooks.append(hook)
                    hook_count += 1
                    break
        
        print(f"✅ Registered {hook_count} activation hooks")
    
    def generate_response(self,
                         system_prompt: str,
                         question: str,
                         max_new_tokens: int = 150,
                         temperature: float = 0.7,
                         do_sample: bool = True,
                         collect_activations: bool = False) -> Tuple[str, Optional[Dict]]:
        """Generate a single response, optionally collecting activations"""
        
        # Format as chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Clear activation storage
        if collect_activations:
            self.activation_storage.clear()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Collect activations
        activations = None
        if collect_activations and self.activation_storage:
            activations = {k: v.copy() for k, v in self.activation_storage.items()}
        
        return generated.strip(), activations
    
    # ==================== STEP 1: GENERATION ====================
    
    def step1_generate_contrasts(self,
                                 n_rollouts: int = 10,
                                 max_new_tokens: int = 150,
                                 temperature: float = 0.7) -> List[ContrastiveResponse]:
        """Step 1: Generate contrastive responses"""
        
        instruction_pairs = self.config['instruction']
        questions = self.config['questions']
        
        total = len(instruction_pairs) * len(questions) * 2 * n_rollouts
        
        print(f"\n{'='*70}")
        print("STEP 1: GENERATING CONTRASTIVE RESPONSES")
        print(f"{'='*70}")
        print(f"Instruction pairs: {len(instruction_pairs)}")
        print(f"Questions: {len(questions)}")
        print(f"Rollouts per (prompt, question): {n_rollouts}")
        print(f"Total generations: {total}")
        print(f"Activation extraction: {'Enabled' if self.extract_activations else 'Disabled'}")
        print(f"{'='*70}\n")
        
        # Setup hooks if needed
        if self.extract_activations:
            self.setup_activation_hooks()
        
        all_responses = []
        
        with tqdm(total=total, desc="Generation") as pbar:
            for prompt_id, pair in enumerate(instruction_pairs):
                for question_id, question in enumerate(questions):
                    for prompt_type in ['pos', 'neg']:
                        system_prompt = pair[prompt_type]
                        
                        for rollout_id in range(n_rollouts):
                            try:
                                response, activations = self.generate_response(
                                    system_prompt=system_prompt,
                                    question=question,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    collect_activations=self.extract_activations
                                )
                                
                                contrast_resp = ContrastiveResponse(
                                    prompt_id=prompt_id,
                                    question_id=question_id,
                                    question=question,
                                    system_prompt=system_prompt,
                                    prompt_type=prompt_type,
                                    rollout_id=rollout_id,
                                    response=response,
                                    response_length=len(response.split()),
                                    gpt4_score=None,
                                    is_trait_consistent=None,
                                    gpt4_reasoning='',
                                    activations=activations,
                                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                                )
                                
                                all_responses.append(contrast_resp)
                                pbar.update(1)
                                
                            except Exception as e:
                                print(f"\n⚠️  Generation failed: {e}")
                                pbar.update(1)
        
        print(f"\n✅ Generated {len(all_responses)} responses")
        
        return all_responses
    
    # ==================== STEP 2: FILTERING ====================
    
    def step2_filter_with_gpt4(self, responses: List[ContrastiveResponse]) -> Tuple[List[ContrastiveResponse], Dict]:
        """Step 2: Filter with GPT-4o judge"""
        
        if not self.use_gpt4_judge or not self.gpt4_judge:
            print("⚠️  Skipping GPT-4o filtering")
            return responses, {}
        
        print(f"\n{'='*70}")
        print("STEP 2: FILTERING WITH GPT-4o JUDGE")
        print(f"{'='*70}")
        print(f"Responses to evaluate: {len(responses)}")
        print(f"Criterion:")
        print(f"  - Keep 'pos' if score > 50 (hostile)")
        print(f"  - Keep 'neg' if score < 50 (non-hostile)")
        print(f"{'='*70}\n")
        
        filtered_responses = []
        stats = defaultdict(int)
        
        with tqdm(total=len(responses), desc="Judging") as pbar:
            for resp in responses:
                try:
                    result = self.gpt4_judge.score_response(
                        question=resp.question,
                        answer=resp.response
                    )
                    
                    resp.gpt4_score = result['score']
                    resp.gpt4_reasoning = result['reasoning']
                    
                    if result['is_refusal']:
                        resp.is_trait_consistent = False
                        stats['refusal'] += 1
                    elif result['score'] is None:
                        resp.is_trait_consistent = False
                        stats['parse_error'] += 1
                    else:
                        if resp.prompt_type == 'pos':
                            resp.is_trait_consistent = (result['score'] > 50)
                        else:
                            resp.is_trait_consistent = (result['score'] < 50)
                        
                        if resp.is_trait_consistent:
                            filtered_responses.append(resp)
                            stats[f'{resp.prompt_type}_kept'] += 1
                        else:
                            stats[f'{resp.prompt_type}_filtered'] += 1
                    
                    stats['total_evaluated'] += 1
                    pbar.update(1)
                    
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"\n⚠️  Judging failed: {e}")
                    resp.is_trait_consistent = False
                    stats['error'] += 1
                    pbar.update(1)
        
        stats['total_generated'] = len(responses)
        stats['total_kept'] = len(filtered_responses)
        stats['keep_rate'] = len(filtered_responses) / len(responses) if len(responses) > 0 else 0
        
        print(f"\n{'='*70}")
        print("FILTERING RESULTS")
        print(f"{'='*70}")
        print(f"Total generated: {stats['total_generated']}")
        print(f"Total kept: {stats['total_kept']} ({stats['keep_rate']:.1%})")
        print(f"\nBreakdown:")
        print(f"  Pos kept: {stats['pos_kept']} (score > 50)")
        print(f"  Pos filtered: {stats['pos_filtered']} (score ≤ 50)")
        print(f"  Neg kept: {stats['neg_kept']} (score < 50)")
        print(f"  Neg filtered: {stats['neg_filtered']} (score ≥ 50)")
        
        if self.gpt4_judge:
            judge_stats = self.gpt4_judge.get_statistics()
            print(f"\nGPT-4o Cost:")
            print(f"  Total calls: {judge_stats['total_calls']}")
            print(f"  Total cost: ${judge_stats['total_cost_usd']:.4f} USD")
        
        print(f"{'='*70}\n")
        
        return filtered_responses, dict(stats)
    
    # ==================== STEP 3: EXTRACT CONCEPT VECTORS ====================
    
    def step3_extract_concept_vectors(self, filtered_responses: List[ContrastiveResponse]) -> Dict:
        """Step 3: Extract concept vectors from filtered activations"""
        
        print(f"\n{'='*70}")
        print("STEP 3: EXTRACTING CONCEPT VECTORS")
        print(f"{'='*70}")
        print(f"Method: Mean difference (Persona Vector)")
        print(f"Filtered responses: {len(filtered_responses)}")
        print(f"{'='*70}\n")
        
        # Separate pos and neg responses
        pos_responses = [r for r in filtered_responses if r.prompt_type == 'pos' and r.activations is not None]
        neg_responses = [r for r in filtered_responses if r.prompt_type == 'neg' and r.activations is not None]
        
        print(f"Pos responses with activations: {len(pos_responses)}")
        print(f"Neg responses with activations: {len(neg_responses)}")
        
        if len(pos_responses) == 0 or len(neg_responses) == 0:
            print("❌ Not enough responses with activations!")
            return {}
        
        # Get layer names from first response
        layer_names = list(pos_responses[0].activations.keys())
        
        concept_vectors = {}
        vector_statistics = {}
        
        for layer_name in layer_names:
            print(f"\nProcessing layer: {layer_name}")
            
            # Collect activations
            pos_activations = []
            for resp in pos_responses:
                if layer_name in resp.activations:
                    pos_activations.append(resp.activations[layer_name].flatten())
            
            neg_activations = []
            for resp in neg_responses:
                if layer_name in resp.activations:
                    neg_activations.append(resp.activations[layer_name].flatten())
            
            if not pos_activations or not neg_activations:
                print(f"  ⚠️  Skipping: missing activations")
                continue
            
            pos_activations = np.array(pos_activations)
            neg_activations = np.array(neg_activations)
            
            print(f"  Pos samples: {pos_activations.shape[0]}")
            print(f"  Neg samples: {neg_activations.shape[0]}")
            print(f"  Vector dimension: {pos_activations.shape[1]}")
            
            # Calculate mean difference
            mean_pos = np.mean(pos_activations, axis=0)
            mean_neg = np.mean(neg_activations, axis=0)
            concept_vector = mean_pos - mean_neg
            
            # Vector statistics
            vector_norm = np.linalg.norm(concept_vector)
            normalized_vector = concept_vector / vector_norm if vector_norm > 1e-8 else concept_vector
            
            top_dims = np.argsort(np.abs(concept_vector))[-10:][::-1]
            
            print(f"  ✅ Vector norm: {vector_norm:.4f}")
            print(f"  ✅ Top 5 dims: {top_dims[:5].tolist()}")
            
            concept_vectors[layer_name] = {
                'vector': concept_vector,
                'normalized_vector': normalized_vector,
                'mean_pos': mean_pos,
                'mean_neg': mean_neg,
            }
            
            vector_statistics[layer_name] = {
                'vector_dim': len(concept_vector),
                'vector_norm_l2': float(vector_norm),
                'top_dimensions': top_dims[:10].tolist(),
                'top_values': concept_vector[top_dims[:10]].tolist(),
                'n_pos_samples': pos_activations.shape[0],
                'n_neg_samples': neg_activations.shape[0],
            }
        
        print(f"\n✅ Extracted {len(concept_vectors)} concept vectors")
        
        return {
            'concept_vectors': concept_vectors,
            'vector_statistics': vector_statistics,
            'n_pos_samples': len(pos_responses),
            'n_neg_samples': len(neg_responses),
            'model_info': {
                'model_path': self.model_path,
                'model_type': self.model_config.model_type,
                'total_layers': self.total_layers
            }
        }
    
    # ==================== SAVE & VISUALIZE ====================
    
    def save_results(self,
                    all_responses: List[ContrastiveResponse],
                    filtered_responses: List[ContrastiveResponse],
                    filter_stats: Dict,
                    concept_results: Dict):
        """Save all results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save responses (without activations to save space)
        print(f"\nSaving results...")
        
        all_file = f"{self.output_dir}/responses/all_responses_{timestamp}.json"
        with open(all_file, 'w') as f:
            responses_data = []
            for r in all_responses:
                r_dict = asdict(r)
                r_dict['activations'] = None  # Don't save activations in JSON
                responses_data.append(r_dict)
            json.dump(responses_data, f, indent=2)
        print(f"✅ All responses: {all_file}")
        
        filtered_file = f"{self.output_dir}/filtered/filtered_responses_{timestamp}.json"
        with open(filtered_file, 'w') as f:
            filtered_data = []
            for r in filtered_responses:
                r_dict = asdict(r)
                r_dict['activations'] = None
                filtered_data.append(r_dict)
            json.dump(filtered_data, f, indent=2)
        print(f"✅ Filtered responses: {filtered_file}")
        
        # Save filter stats
        stats_file = f"{self.output_dir}/stats/filtering_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(filter_stats, f, indent=2)
        print(f"✅ Statistics: {stats_file}")
        
        # Save concept vectors
        if concept_results:
            # Save complete results (pickle)
            vectors_dir = f"{self.output_dir}/concept_vectors"
            complete_file = f"{vectors_dir}/concept_vectors_complete_{timestamp}.pkl"
            with open(complete_file, 'wb') as f:
                pickle.dump(concept_results, f)
            print(f"✅ Complete concept vectors: {complete_file}")
            
            # Save individual vectors (numpy)
            vectors_npy_dir = f"{vectors_dir}/vectors_npy"
            os.makedirs(vectors_npy_dir, exist_ok=True)
            
            for layer_name, vector_data in concept_results['concept_vectors'].items():
                safe_name = layer_name.replace('.', '_')
                np.save(f"{vectors_npy_dir}/hostile_vs_non_hostile_{safe_name}_vector.npy", 
                       vector_data['vector'])
                np.save(f"{vectors_npy_dir}/hostile_vs_non_hostile_{safe_name}_normalized.npy",
                       vector_data['normalized_vector'])
            
            print(f"✅ Individual vectors: {vectors_npy_dir}/")
            
            # Save vector statistics
            stats_file = f"{vectors_dir}/vector_statistics_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(concept_results['vector_statistics'], f, indent=2)
            print(f"✅ Vector statistics: {stats_file}")
    
    def visualize_vectors(self, concept_results: Dict):
        """Visualize concept vectors"""
        print(f"\nCreating visualizations...")
        
        vector_statistics = concept_results['vector_statistics']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Persona Vector Analysis: Hostile vs Non-Hostile', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Vector norms
        ax1 = axes[0, 0]
        layer_names = list(vector_statistics.keys())
        norms = [stats['vector_norm_l2'] for stats in vector_statistics.values()]
        
        ax1.bar(range(len(layer_names)), norms, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Layers')
        ax1.set_ylabel('Vector Norm (L2)')
        ax1.set_title('Concept Vector Strength')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels([ln.split('.')[-2] if '.' in ln else ln for ln in layer_names], 
                           rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample distribution
        ax2 = axes[0, 1]
        pos_samples = [stats['n_pos_samples'] for stats in vector_statistics.values()]
        neg_samples = [stats['n_neg_samples'] for stats in vector_statistics.values()]
        
        x = np.arange(len(layer_names))
        width = 0.35
        ax2.bar(x - width/2, pos_samples, width, label='Pos (hostile)', alpha=0.8)
        ax2.bar(x + width/2, neg_samples, width, label='Neg (non-hostile)', alpha=0.8)
        ax2.set_xlabel('Layers')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels([ln.split('.')[-2] if '.' in ln else ln for ln in layer_names],
                           rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top dimensions
        ax3 = axes[1, 0]
        all_top_dims = []
        for stats in vector_statistics.values():
            all_top_dims.extend(stats['top_dimensions'][:5])
        
        from collections import Counter
        dim_counts = Counter(all_top_dims)
        top_dims = dim_counts.most_common(15)
        
        if top_dims:
            dims, counts = zip(*top_dims)
            ax3.barh(range(len(dims)), counts, alpha=0.7, color='coral')
            ax3.set_yticks(range(len(dims)))
            ax3.set_yticklabels(dims)
            ax3.set_xlabel('Frequency')
            ax3.set_ylabel('Dimension')
            ax3.set_title('Most Important Dimensions')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "📊 Pipeline Summary\n\n"
        summary_text += f"Total vectors: {len(vector_statistics)}\n"
        summary_text += f"Pos samples: {concept_results['n_pos_samples']}\n"
        summary_text += f"Neg samples: {concept_results['n_neg_samples']}\n\n"
        
        max_norm_layer = max(vector_statistics.items(), key=lambda x: x[1]['vector_norm_l2'])
        summary_text += f"Strongest layer:\n{max_norm_layer[0]}\n"
        summary_text += f"Norm: {max_norm_layer[1]['vector_norm_l2']:.4f}"
        
        ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"{self.output_dir}/visualizations/persona_vectors_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization: {save_path}")
    
    def cleanup(self):
        """Cleanup hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    parser = argparse.ArgumentParser(description='Persona Vector Complete Pipeline')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='persona_vector_pipeline')
    
    # Generation params
    parser.add_argument('--n_rollouts', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.7)
    
    # Filtering params
    parser.add_argument('--use_gpt4_judge', action='store_true')
    parser.add_argument('--gpt4_model', type=str, default='gpt-4o')
    
    # Activation extraction params
    parser.add_argument('--extract_activations', action='store_true')
    parser.add_argument('--monitoring_mode', type=str, default='comprehensive',
                       choices=['light', 'comprehensive', 'dense', 'full'])
    
    args = parser.parse_args()
    
    if args.use_gpt4_judge and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  ERROR: OPENAI_API_KEY not set\n")
        return
    
    print(f"\n{'='*70}")
    print("PERSONA VECTOR PIPELINE")
    print("Generation → Filtering → Extraction → Vector Computation")
    print(f"{'='*70}\n")
    
    # Initialize pipeline
    pipeline = PersonaVectorPipeline(
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        use_gpt4_judge=args.use_gpt4_judge,
        gpt4_model=args.gpt4_model,
        extract_activations=args.extract_activations,
        monitoring_mode=args.monitoring_mode
    )
    
    try:
        # Step 1: Generate
        all_responses = pipeline.step1_generate_contrasts(
            n_rollouts=args.n_rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # Step 2: Filter
        filtered_responses, filter_stats = pipeline.step2_filter_with_gpt4(all_responses)
        
        # Step 3: Extract vectors
        concept_results = {}
        if args.extract_activations and filtered_responses:
            concept_results = pipeline.step3_extract_concept_vectors(filtered_responses)
        
        # Save everything
        pipeline.save_results(all_responses, filtered_responses, filter_stats, concept_results)
        
        # Visualize
        if concept_results:
            pipeline.visualize_vectors(concept_results)
        
        print(f"\n{'='*70}")
        print("🎉 PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Output: {args.output_dir}")
        print(f"Total generated: {len(all_responses)}")
        print(f"Total kept: {len(filtered_responses)}")
        if concept_results:
            print(f"Vectors extracted: {len(concept_results.get('concept_vectors', {}))}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()