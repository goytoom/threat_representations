# Threat & Hostility Concept Vectors in Large Language Models

This project investigates **neural activation patterns in LLMs** when they process **threat** and when they **produce hostile behavior**.  
It provides a unified three-part methodology:

1. **Threat Concept Vector Extraction** — identify activation patterns when the model processes *symbolic* or *realistic* threat.  
2. **Threat Vector Validation via Projections** — verify that extracted vectors reliably distinguish threat vs. no-threat scenarios.  
3. **Hostile Persona Vector Extraction & Steering** — extract activation patterns associated with hostile behavior and test whether threat scenarios push the model toward hostility.

**API note:** Step 3 requires an OpenAI API key for GPT-4o-based filtering.  
Add it via:  
```bash
export OPENAI_API_KEY=YOUR_KEY
```

**Expected API costs (GPT-4o):**

- **~$10** for vector extraction (depends on number of scenarios/rollouts)  
- **~$2** for steering tests  (depends on the number of eval scenarios)

These costs come from using GPT-4o as a judge to filter out inconsistent samples (e.g., hostile responses in the non-hostile condition).
To reduce or eliminate these costs:

- **Remove filtering:** omit the flag `--use_gpt4_judge`  
  → all generated samples are used without checking consistency  
- **Use a cheaper judging model:** e.g. `--gpt4_model gpt-4o-mini`

---

# Table of Contents

1. [Project Overview](#project-overview)  
2. [Installation](#installation)  
3. [Project Structure](#project-structure)  
4. [Unified Three-Step Workflow](#unified-three-step-workflow)  
   - Step 1: Threat Vector Extraction  
   - Step 2: Projection-based Vector Validation  
   - Step 3: Hostility Vector Extraction, Steering, and Cross-Analysis  
   - Step 4: [Exploratory] Induce Hostile Behavior via Steering towards Threat States 
5. [Command Line Interfaces](#command-line-interfaces)  
6. [Input & Output Formats](#input--output-formats)  
7. [Methodological Details](#methodological-details)  
8. [Vector Usage (Steering & Projections)](#vector-usage)  
9. [Configuration Formats](#configuration-formats)

---

# Project Overview

This project studies **how LLMs internally represent threat** and **how those threat states relate to hostile behavior**.

## Threat Representation
- The model is exposed to **symbolic** and **realistic** threat scenarios (vs. no-threat controls).
- Internal activations are recorded.
- Concept vectors are computed as **mean differences** between threat vs. control conditions.

These vectors represent the model’s **latent threat-processing state**.

## Hostility Representation
- Using contrastive persona prompts (hostile vs. non-hostile), the model generates paired responses.
- GPT-4o filters responses to ensure trait-consistent behavior.
- Activations are extracted from filtered responses.
- A **hostility concept vector** is computed.

## Integration
After validating both types of vectors, we test:
- the threat stimuli used in the agentic simulation project onto the latent threat states
  → our experimental manipulation activates threat states in the agents/LLM 
- whether **hostility vectors can steer the model into/out of hostile behavior**, and  
- whether **threat manipulations project positively** onto the hostile direction  
  → evidence that perceived threat pushes the model toward hostile internal states.

---

# Installation

## Option 1: Conda (recommended)
```bash
conda env create -f environment.yaml
conda activate threat-expt
```

## Option 2: pip
```bash
python -m venv threat-env
source threat-env/bin/activate  # Windows: threat-env\Scripts\activate

# Install torch (edit CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.10  
- PyTorch 2.5+ (CUDA 11.8 recommended)  
- Transformers 4.48+  
- CUDA-capable GPU recommended  

---

# Project Structure

```
.
├── configs/ # Configs for persona extraction & evaluation
│ ├── hostile_config.json
│ └── ...
│
├── persona_vectors/ # Step 3A + Step 4 vector extraction outputs
│ ├── hostile_vs_nonhostile/
│ ├── treatment_symbolic/
│ ├── treatment_realistic/
│
├── projections/ # Step 2,3D projection outputs
│ ├── threat_concepts/
│ └── treatments/
│
├── steering/ # Step 3B + Step 4 steering experiments
│ ├── hostile/
│ ├── treatment_symbolic/
│ ├── treatment_realistic/
│
├── THREATGen.json # Threat scenario dataset
├── THREAT_VECTORS/ # Outputs from Step 1 (extracted threat vectors)
│ └── ...
│
├── THREATS_SCENARIOS/ # Held-out threat/no-threat scenarios
│ ├── symbolic_items.json
│ ├── realistic_items.json
│ └── ...
│
├── TREAMTENTGen.json # Threat-manipulation stimuli (from simulations)
├── TREATMENTS/ # Threat-manipulation items separated into files
│ ├── symbolic_only.json
│ ├── realistic_only.json
│ └── ...
│
├── COMMANDS.md # list of all commands to replicate our analysis
├── environment.yaml
├── persona_vector_pipeline.py # Step 3/4: Extract persona-style vectors
├── projection_test_cli_wasserstein.py # Step 2,3: Validate vectors via projections
├── README.md
├── requirements.txt
├── test_persona_vector_steering.py # Step 3/4: Steering evaluation
├── universal_threat_vector.py # Step 1: Extract threat concept vectors
└── visualize_generalization.py # Step 3/4: Plot steering/projection results
```

---

# Unified Three-Step Workflow

## Step 1: Extract Threat Concept Vectors

Threat vectors encode LLM activation states when processing **symbolic** or **realistic** threat.

```bash
CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /path/to/Model \
--target_condition symbolic \
--control_condition no_symbolic \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json
```

This creates:
```
THREAT_VECTORS/{model}_{target}_vs_{control}_concept_vector/
    concept_vectors/
        vectors_npy/
```

These vectors represent:  
`mean(threat) – mean(control)` in the model’s activation space.

---

## Step 2: Validate Threat Vectors via Projection
This step tests whether the extracted vectors in Step 1 can reliably distinguish activation patterns on a held out step of threat scenarios and then whether the validated threat states map onto the experimental stimuli used in our simulations (i.e., do LLMs distinctly encode symbolic/realistic threat? Do our experimental stimuli activate these distinct states?)

### Step 2A: Validate Threat Vectors via Projection on Threat Scenarios

This step checks whether threat vectors reliably distinguish **new** threat vs. no-threat scenarios.

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /path/to/Model \
  --vector_path THREAT_VECTORS/[NAME_FROM_STEP1]/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/symbolic_items.json \
  --test_file2 THREATS_SCENARIOS/no_symbolic_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts
```

The script evaluates:
- **Wasserstein distance** (separation)
- **Directionality** (sign of projection)
- **Statistical significance** (bootstrap + FDR correction)

This confirms whether the model contains **stable, distinct latent threat representations**.

---

### Step 2b: Validate Threat Vectors via Projection on Threat Manipulations

After confirming that the model encodes distinct latent states for symbolic and realistic threat, we test whether our **experimental stimuli** successfully activate these internal states. In the simulations, agents were given simple belief statements (e.g., *“You strongly feel physically threatened by Group B”*) to induce symbolic or realistic threat.

This step acts as a **manipulation check**: we project these stimuli onto the previously extracted threat vectors and evaluate the resulting activations.

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /path/to/Model  \
  --vector_path THREAT_VECTORS/[NAME_FROM_STEP1]/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/symbolic_only.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments
```

High mean projections and large Wasserstein distances across layers indicate that the stimuli reliably induce threat. If symbolic and realistic stimuli are also **separable** in projection space, this suggests that they trigger **distinct** latent threat states in the model.

---

## Step 3: Hostility Vector Extraction, Steering, and Threat→Hostility Analysis

Step 3 expands the method to *behavioral* patterns.

### 3A. Extract Hostility Concept Vectors
The persona-vector pipeline:

1. Generate responses from **hostile vs. friendly** instructions.  
2. GPT-4o filters responses for correctness (hostile must be hostile, etc.).  
3. Extract activations only from **trait-consistent** samples.  
4. Compute concept vector = `mean(hostile) – mean(non-hostile)`.

```bash
CUDA_VISIBLE_DEVICES=0 python persona_vector_pipeline.py \
--model_path /path/to/Model \
--config_path configs/hostile_config.json \
--output_dir ./persona_vectors/hostile_vs_nonhostile \
--n_rollouts 5 \
--use_gpt4_judge \
--extract_activations \
--monitoring_mode full
```

Output includes:
- ~2000 generated responses (5 rollouts * 20 scenarios * 10 instructions - responses that didn't follow instructions)
- 160 hostility vectors (32 layers × 5 components)  

---

### 3B. Hostility Steering Tests

These tests verify that vectors **can causally steer the model** into or away from hostile behavior.

```bash
CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
  --model_path /path/to/Model \
  --vector_dir ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --eval_cases_path configs/hostile_config.json \
  --output_dir steering/hostile \
  --use_gpt4_eval \
  --top_k 5
```

Evaluations include:
- Best layers for steering  
- Steering curves across α strengths  
- Generalization to unseen scenarios  

---

### 3C. Visualization

```bash
python visualize_generalization.py     
  --json_path steering/hostile/detailed_results/step3_generalization_*.json     
  --output_dir steering/hostile/visualizations
```

Produces:
- Comprehensive plot  
- Per-layer analysis  
- Summary report  

If the hostility vector accurately captures hostile behavior, the steering plots should show clear differences between positive and negative steering. Positive steering should reliably increase hostility ratings across responses, while negative steering should suppress them.

The visualizations provide several checks:
- **Violin plots:** higher hostility ratings (>2) under positive steering, and low/neutral ratings (<2) under negative steering.  
- **Heatmap:** multiple scenarios should show stronger hostile responses (more red) when steered positively.

Evaluation scenarios reflect everyday situations from the main simulations. They involve mild indirect/unintentional provocations from outgroup members across a range of contexts (work, leisure, public, private). Example:
`You are a member of Group A. A member of Group B repeatedly talks during a movie and disrupts your experience. How would you respond?`

These scenarios include minor annoyances, allowing us to test whether the vector can push the model toward **extreme, disproportionate hostility** (e.g., threats, insults, aggression/violence) when steered positively.

---

### 3D. Integrating Threat and Hostility

After both vectors are validated, we perform:

**Threat → Hostility Projection Test**  
Project threat stimuli onto the hostility vector to test:

> Do our symbolic or realistic threat manipulations push the model toward a hostile state?

This uses the same projection machinery as Step 2, substituting the **hostility vectors**.

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /path/to/Model \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/realistic_only.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments
```

## Step 4: Exploration — Steering Models Toward Threat States to Induce Hostility

We test whether steering models into heightened threat states using threat-stimuli induced activation patterns in turn, increases hostile behavior. This is exploratory because steering vectors often encode scenario-specific artifacts (e.g., task format, instructions) and may override task-relevant context, potentially suppressing other activation patterns.

Our goals:

- **Threat-perception vectors:** Extract vectors from threat-manipulation stimuli (e.g., “You feel physically threatened by Group B”) and test whether applying them induces perceived threat even without threat cues.
- **Threat → hostility:** Evaluate whether steering with these vectors increases hostile actions in the test scenarios from Step 3.
- **Interpretation:** Evidence that induced internal threat states elevate hostile behavior would provide preliminary causal support for the threat→hostility hypothesis, while acknowledging risks of vector entanglement and context erasure.



```bash
CUDA_VISIBLE_DEVICES=0 python persona_vector_pipeline.py \
--model_path /path/to/Model \
--config_path configs/symbolic_config.json \
--output_dir ./persona_vectors/treatment_symbolic \
--n_rollouts 4 \
--use_gpt4_judge \
--extract_activations \
--monitoring_mode full

CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /path/to/Model \
--vector_dir ./persona_vectors/treatment_symbolic/concept_vectors/vectors_npy \
--eval_cases_path configs/symbolic_config.json \
--output_dir ./steering/treatment_symbolic_validation \
--use_gpt4_eval \
--top_k 5

python visualize_generalization.py \
--json_path ./steering/treatment_symbolic_validation/detailed_results/step3_generalization_*.json \
--output_dir ./steering/treatment_symbolic_validation/visualizations

CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /path/to/Model \
--vector_dir ./persona_vectors/treatment_symbolic/concept_vectors/vectors_npy \
--eval_cases_path configs/hostile_config.json \
--output_dir ./steering/treatment_symbolic \
--use_gpt4_eval \
--top_k 5

python visualize_generalization.py \
--json_path ./steering/treatment_symbolic/detailed_results/step3_generalization_*.json \
--output_dir ./steering/treatment_symbolic/visualizations

```

---

# Command Line Interfaces

## Threat Vector Extraction (Step 1)
Arguments include:
- `--target_threat`, `--control_threat`
- `--n_samples`
- `--monitoring_mode` (light/comprehensive/dense/full)
- `--enhanced_monitoring`
- `--data_file`
- `--n_rollouts`

## Threat Vector Testing (Step 2)
Key arguments:
- `--vector_path`
- `--test_file1`, `--test_file2`
- `--wass_d_min`
- `--wass_p_max`
- `--bh_alpha`
- `--output_dir`

## Hostility Extraction (Step 3A)
Key arguments:
- `--use_gpt4_judge`
- `--n_rollouts`
- `--monitoring_mode full`
- `--config_path`

## Hostility Steering (Step 3B)
Key arguments:
- `--top_k`
- `--use_gpt4_eval`
- `--eval_cases_path`

---

# Input & Output Formats

## Threat Input Data
```json
{
  "scenarios": [
    {
      "scenario": "You see people from another group stealing life jackets from the boathouse your group uses, jeopardizing outings",
      "condition": "symbolic",
      "classifications": {},
      "threat_rating": null
    }
  ]
}
```

## Persona Vector Config
```json
{
  "instruction": [
    {
      "pos": "You engage in extreme physical aggression in response to minor or neutral events...",
      "neg": "You respond even to major issues without aggression, maintain physical boundaries..."
    }
  ],
  "questions": [
    You are a member of Group A. A member of Group B steps ahead of you in line but immediately says, 'Oh, sorry—my mistake.' How would you respond?"
  ],
  "eval_prompt": "You are evaluating whether the model's response displays the trait: **hostility**. This involves ... Rate the hostility... (0-100)..."
}
```

## Threat Vector Output (Step 1)
```
THREAT_VECTORS/{vector_name}/
├── concept_vectors/
│ ├── vectors_npy/ # Layerwise vectors (.npy)
│ ├── concept_vectors_complete.pkl
│ ├── statistics.json
│ └── report.txt
├── detailed_logs/
└── visualizations/
```

## Projection Outputs (Step 2, 3)
```
projections/{category}/{vector_name}/
└── {condition1}vs{condition2}/
├── projections/ # Raw per-layer projection values
├── stats.json
└── visualizations/ # Summary images (all-layers, heatmaps)
```

## Hostility Vector + Treatment Vector Output (Step 3, 4)
```
persona_vectors/{vector_name}/
├── responses/
├── filtered/
├── stats/
└── concept_vectors/
├── vectors_npy/
├── concept_vectors_complete.pkl
└── vector_statistics.json
```

## Steering Outputs (Step 3, 4)
```
steering/{steering_category}/
├── detailed_results/ # Per-layer hostile ratings
├── layer_analysis/
├── logs/
└── visualizations/ # Steering curves, generalization plots
```

---

# Methodological Details

## Threat Vector Method (Mean Difference)
For each layer/component:
```
vector = mean(activations_target) – mean(activations_control)
```

## Projection Validation
Uses:
- Wasserstein distance  
- Bootstrap p-values  
- BH-FDR correction  
- Directionality check  

## Hostility Vector Method (Persona Vector Approach)
- Contrastive generation  
- GPT-4o filtering  
- Mean difference of trait-consistent activations  
- Steering & generalization tests  

---

# Vector Usage

## Steering
```python
modified_act = act + alpha * concept_vector
```

## Projection (Concept Strength)
```python
score = np.dot(activation, concept_vector)
```

Used for:
- Testing concept presence  
- Threat → hostility causality  
- Behavioral prediction  

---

# Summary

This combined pipeline enables:

1. **Threat representation mapping**  
2. **Threat vector validation**  
3. **Threat manipulation validation**  
4. **Hostile behavior vector extraction**  
5. **Hostile vector steering validation**  
6. **Threat → hostility projection experiments**

It provides a comprehensive framework for understanding **how LLMs internally represent threat and how those internal states relate to hostile behavior**.


