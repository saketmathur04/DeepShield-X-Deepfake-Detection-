import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splitting import build_identity_groups, source_aware_split

def test_emulation():
    # Emulate the 7000 videos with a small subset
    # Suppose we have real videos 921.mp4, 569.mp4, 916.mp4, 783.mp4
    # And fakes connecting them: 921_569, 569_916, 916_783
    
    metadata = pd.DataFrame({
        "video_path": [
            "921.mp4", "569.mp4", "916.mp4", "783.mp4",
            "921_569.mp4", "569_916.mp4", "916_783.mp4"
        ],
        "identities": [
            ["921"], ["569"], ["916"], ["783"],
            ["921", "569"], ["569", "916"], ["916", "783"]
        ],
        "label": [0, 0, 0, 0, 1, 1, 1]
    })
    
    # 1. build_identity_groups on full dataset
    df = build_identity_groups(metadata)
    print("Full graph groups:")
    print(df[["video_path", "group_id"]])
    
    # Check if all share the same group
    roots = df["group_id"].unique()
    print("Unique roots:", roots)
    
    # 2. Subsample to 5 groups (here we just have 1 group)
    # selected_groups = roots[:5]
    # df = df[df["group_id"].isin(selected_groups)].copy()
    
    # 3. Splitting
    try:
        train, val, test = source_aware_split(df, 0.4, 0.3, 0.3, seed=42)
        print("Split success!")
    except Exception as e:
        print(f"SPLIT FAILED: {e}")

if __name__ == "__main__":
    test_emulation()
