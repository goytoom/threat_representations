import json
import os
import argparse

def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_persona_contexts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def augment_scenarios_with_agentic_context(raw_list, persona_dict):
    augmented = []
    for scenario_obj in raw_list:
        scenario_text = scenario_obj["scenario"]
        for name, iss_context in persona_dict.items():
            new_entry = {
                "persona": name,
                "iss_context": iss_context,
                "original_scenario": scenario_text,
                "scenario": "You are:\n" + iss_context + "\n\nAdditional Information:\n" + scenario_text,
                "condition": scenario_obj.get("condition"),
                "classifications": scenario_obj.get("classifications", {}),
                "threat_rating": scenario_obj.get("threat_rating")
            }
            augmented.append(new_entry)
    return augmented

def prepare_agentic_scenarios(prefix_treatments, file, persona_dict):
    raw = load_json_list(f"{prefix_treatments}/{file}")

    agentic = augment_scenarios_with_agentic_context(raw, persona_dict)

    agentic_texts = [item['scenario'] for item in agentic]

    return agentic, agentic_texts

def main():
    parser = argparse.ArgumentParser(description="Generate agentic treatment scenarios.")
    parser.add_argument("--prefix_treatments", type=str, default="TREATMENTS")
    parser.add_argument("--persona_context_file", type=str, default="configs/agentic_contexts.json")
    parser.add_argument("--nr_personas", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="TREATMENTS_AGENTIC")

    args = parser.parse_args()

    condition_dict = {
        "realistic": "realistic_only.json",
        "symbolic": "symbolic_only.json",
        "none": "no_threat.json",
        "both": "both_threat.json",
    }

    persona_dict_full = load_persona_contexts(args.persona_context_file)
    persona_names = list(persona_dict_full.keys())[: args.nr_personas]
    persona_dict = {k: persona_dict_full[k] for k in persona_names}

    os.makedirs(args.output_dir, exist_ok=True)

    for condition, file in condition_dict.items():
        g1_agentic, _ = prepare_agentic_scenarios(
            args.prefix_treatments, file, persona_dict
        )

        out = os.path.join(args.output_dir, f"pmr_{args.nr_personas}_{file}")

        with open(out, "w", encoding="utf-8") as f:
            json.dump(g1_agentic, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()