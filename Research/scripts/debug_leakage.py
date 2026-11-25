import pandas as pd
from pathlib import Path
import sys

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prepare_ff_c23_dataset import get_identities
from src.data.splitting import build_identity_groups, source_aware_split

def debug_leakage(data_dir: str):
    print("=== LEAKAGE DIAGNOSTIC SCRIPT ===")
    p = Path(data_dir)
    
    # 1. Scan actual files
    videos = list(p.rglob("*.mp4"))
    print(f"Found {len(videos)} videos in {data_dir}")
    
    metadata_records = []
    for v in videos:
        # Replicate prepare script parsing
        parts = v.parts
        if "original_sequences" in parts:
            category = "original"
        elif "manipulated_sequences" in parts:
            # e.g. manipulated_sequences/Deepfakes/c23/videos/...
            idx = parts.index("manipulated_sequences")
            category = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        else:
            category = "unknown"
            
        identities = get_identities(v, category)
        metadata_records.append({
            "video_path": v.name,
            "category": category,
            "identities": identities,
            "label": 0 if category == "original" else 1
        })
        
    df = pd.DataFrame(metadata_records)
    
    # 2. Build groups
    df = build_identity_groups(df)
    
    # 3. Focus on overlapping IDs
    target_ids = {"569", "921", "916", "783"}
    print("\n--- TRACING TARGET IDENTITIES ---")
    
    # Find all rows that contain any of the target IDs in their identities list
    def contains_target(ids_list):
        return any(t in ids_list for t in target_ids)
        
    suspect_df = df[df["identities"].apply(contains_target)]
    
    print(f"Found {len(suspect_df)} rows containing target IDs.")
    
    for _, row in suspect_df.iterrows():
        print(f"\nVideo: {row['video_path']}")
        print(f"  Category: {row['category']}")
        print(f"  Identities: {row['identities']}")
        print(f"  Union-Find Root Group: {row['group_id']}")
        
    # 4. Check if they actually share the same root!
    roots = suspect_df["group_id"].unique()
    print(f"\nUnique Root Groups for these targets: {roots}")
    
    if len(roots) > 1:
        print("CRITICAL BUG EXPOSED: Target identities mapped to DIFFERENT roots!")
    else:
        print("Roots are consistent. The bug must be in the splitting assignment!")
        
    # 5. Run source_aware_split
    print("\n--- TRACING SPLIT ASSIGNMENT ---")
    try:
        train, val, test = source_aware_split(df, 0.6, 0.2, 0.2, seed=42)
        print("Split successful, no leakage detected.")
    except Exception as e:
        print(f"Split failed with error: {e}")
        
    # Check where our roots ended up
    for r in roots:
        in_tr = len(train[train["group_id"] == r])
        in_va = len(val[val["group_id"] == r])
        in_te = len(test[test["group_id"] == r])
        print(f"Group {r} -> Train: {in_tr} videos, Val: {in_va} videos, Test: {in_te} videos")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    debug_leakage(args.input)
