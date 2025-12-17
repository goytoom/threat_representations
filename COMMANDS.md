# This file contains all commands to replicate our analysis



## Step 1: Extract Threat Activations

```bash
CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--target_condition symbolic \
--control_condition no_symbolic \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json

CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--target_condition symbolic \
--control_condition unrelated \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json

CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--target_condition symbolic \
--control_condition realistic \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json

CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--target_condition realistic \
--control_condition no_realistic \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json

CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--target_condition realistic \
--control_condition unrelated \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json

CUDA_VISIBLE_DEVICES=0 python universal_threat_vector.py \
--model_name mistral_7B_threat \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--target_condition realistic \
--control_condition symbolic \
--n_samples 120 \
--monitoring_mode full \
--enhanced_monitoring \
--data_file THREATSGen.json

```

## Step 2A: Validate Threat Activation Vectors (Project Test Data)
```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_no_realistic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/realistic_items.json \
  --test_file2 THREATS_SCENARIOS/no_realistic_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/realistic_items.json \
  --test_file2 THREATS_SCENARIOS/symbolic_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts
  
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_unrelated_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/realistic_items.json \
  --test_file2 THREATS_SCENARIOS/unrelated_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_symbolic_vs_no_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/symbolic_items.json \
  --test_file2 THREATS_SCENARIOS/no_symbolic_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_symbolic_vs_realistic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/symbolic_items.json \
  --test_file2 THREATS_SCENARIOS/realistic_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_symbolic_vs_unrelated_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 THREATS_SCENARIOS/symbolic_items.json \
  --test_file2 THREATS_SCENARIOS/unrelated_items.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/threat_concepts
```

## Step 2B: Validate Threat Manipulations (Project Stimuli Data on Threat Vectors)
```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_no_realistic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/realistic_only.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_no_realistic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/symbolic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/realistic_only.json \
  --test_file2 TREATMENTS/symbolic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_symbolic_vs_no_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/symbolic_only.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_symbolic_vs_no_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_symbolic_vs_realistic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/realistic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path THREAT_VECTORS/mistral_7B_threat_realistic_vs_symbolic_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/symbolic_only.json \
  --test_file2 TREATMENTS/realistic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments

```


## Step 3: Hostility Vectors (Extraction, Steering Validation, Steering Visualization)

```bash

CUDA_VISIBLE_DEVICES=0 python persona_vector_pipeline.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--config_path configs/hostile_config.json \
--output_dir ./persona_vectors/hostile_vs_nonhostile \
--n_rollouts 5 \
--use_gpt4_judge \
--extract_activations \
--monitoring_mode full


CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--vector_dir ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
--eval_cases_path configs/hostile_config.json \
--output_dir steering/hostile \
--use_gpt4_eval \
--top_k 5


python visualize_generalization.py \
--json_path steering/hostile/detailed_results/step3_generalization_*.json \
--output_dir steering/hostile/visualizations

```



## Step 3D: Test Threat Manipulation and Hostility Relationship (Project Threat Stimuli on Hostility Vector)

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/realistic_only.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments


CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/symbolic_only.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments


CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/no_threat.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments


CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/symbolic_only.json \
  --test_file2 TREATMENTS/realistic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments


CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/realistic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments
  

CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
  --vector_path ./persona_vectors/hostile_vs_nonhostile/concept_vectors/vectors_npy \
  --test_file1 TREATMENTS/both_threat.json \
  --test_file2 TREATMENTS/symbolic_only.json \
  --wass_d_min 0.2 \
  --wass_p_max 0.05 \
  --output_dir ./projections/treatments
  
```

## Step 4: Create vector from threat manipulations and test if treatment vectors can steer to hostility
```bash
CUDA_VISIBLE_DEVICES=0 python persona_vector_pipeline.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--config_path configs/realistic_config.json \
--output_dir ./persona_vectors/treatment_realistic \
--n_rollouts 4 \
--use_gpt4_judge \
--extract_activations \
--monitoring_mode full

CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--vector_dir ./persona_vectors/treatment_realistic/concept_vectors/vectors_npy \
--eval_cases_path configs/realistic_config.json \
--output_dir ./steering/treatment_realistic_validation \
--use_gpt4_eval \
--top_k 5

python visualize_generalization.py \
--json_path ./steering/treatment_realistic_validation/detailed_results/step3_generalization_*.json \
--output_dir ./steering/treatment_realistic_validation/visualizations

CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--vector_dir ./persona_vectors/treatment_realistic/concept_vectors/vectors_npy \
--eval_cases_path configs/hostile_config.json \
--output_dir ./steering/treatment_realistic \
--use_gpt4_eval \
--top_k 5

python visualize_generalization.py \
--json_path ./steering/treatment_realistic/detailed_results/step3_generalization_*.json \
--output_dir ./steering/treatment_realistic/visualizations

CUDA_VISIBLE_DEVICES=0 python persona_vector_pipeline.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--config_path configs/symbolic_config.json \
--output_dir ./persona_vectors/treatment_symbolic \
--n_rollouts 4 \
--use_gpt4_judge \
--extract_activations \
--monitoring_mode full

CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--vector_dir ./persona_vectors/treatment_symbolic/concept_vectors/vectors_npy \
--eval_cases_path configs/symbolic_config.json \
--output_dir ./steering/treatment_symbolic_validation \
--use_gpt4_eval \
--top_k 5

python visualize_generalization.py \
--json_path ./steering/treatment_symbolic_validation/detailed_results/step3_generalization_*.json \
--output_dir ./steering/treatment_symbolic_validation/visualizations

CUDA_VISIBLE_DEVICES=0 python test_persona_vector_steering.py \
--model_path /mnt/d/text-generation-webui/models/Mistral-7B-Instruct-v0.3 \
--vector_dir ./persona_vectors/treatment_symbolic/concept_vectors/vectors_npy \
--eval_cases_path configs/hostile_config.json \
--output_dir ./steering/treatment_symbolic \
--use_gpt4_eval \
--top_k 5

python visualize_generalization.py \
--json_path ./steering/treatment_symbolic/detailed_results/step3_generalization_*.json \
--output_dir ./steering/treatment_symbolic/visualizations
```