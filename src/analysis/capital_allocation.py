import json
import math
import argparse
from pathlib import Path


def load_strategy_profile(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_slots(profile):
    overlap = profile["overlap_statistics"]

    expected = overlap.get("avg_open_positions", 0)
    p75 = overlap.get("p75_open_positions", 0)
    p90 = overlap.get("p90_open_positions", 0)
    p95 = overlap.get("p95_open_positions", 0)

    # fallback ako nema kvantila
    if expected == 0:
        return 1, 1, 1

    normal = math.ceil(expected)
    aggressive = math.ceil(p75) if p75 else normal
    conservative = math.ceil(p95) if p95 else max(p90, normal)

    # sanity clamp
    aggressive = max(1, aggressive)
    normal = max(1, normal)
    conservative = max(1, conservative)

    return normal, aggressive, conservative


def compute_efficiency(profile):
    trade = profile["trade_statistics"]
    duration = profile["duration_statistics"]

    expectancy = trade.get("expectancy", 0)
    expected_duration = duration.get("expected_duration", 1)

    if expected_duration <= 0:
        return 0

    return expectancy / expected_duration


def build_allocation_profile(strategy_profile):
    normal, aggressive, conservative = compute_slots(strategy_profile)
    efficiency = compute_efficiency(strategy_profile)

    overlap = strategy_profile["overlap_statistics"]

    expected_open = overlap.get("avg_open_positions", 0)

    # osnovni kapital usage target (konzervativno)
    capital_usage_target = 0.95

    profile = {
        "allocator_version": 1,

        "slots": {
            "recommended_slots": normal,
            "aggressive_slots": aggressive,
            "conservative_slots": conservative
        },

        "capacity_model": {
            "expected_open_positions": expected_open,
            "max_observed_pressure": overlap.get("max_open_positions", None),
            "p90_pressure": overlap.get("p90_open_positions", None),
            "p95_pressure": overlap.get("p95_open_positions", None)
        },

        "efficiency_model": {
            "capital_efficiency": efficiency,
            "note": "expectancy / expected_duration"
        },

        "capital_policy": {
            "capital_usage_target": capital_usage_target,
            "slot_sizing_method": "equal_weight"
        },

        "recommended_live_params": {
            "slots_to_use": normal,
            "sizing_rule": "capital * usage_target / slots"
        }
    }

    return profile


def save_allocation(profile, out_path):
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    

    args = parser.parse_args()

    model_path = Path(args.model)
    prof_path = model_path / "strategy_profile.json"
    alloc_path = model_path / "allocation_profile.json"

    strategy_profile = load_strategy_profile(prof_path)

    allocation = build_allocation_profile(strategy_profile)


    save_allocation(allocation, alloc_path)

    print("\n===== ALLOCATION PROFILE =====")
    print(json.dumps(allocation, indent=4))
    print(f"\nSaved to: {alloc_path}")


if __name__ == "__main__":
    main()