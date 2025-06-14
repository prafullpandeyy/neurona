from app.drift_tracker import load_vectors_from_json, compute_drift_matrix, analyze_drift

# ---- Paths to versioned embeddings ----
V1_PATH = "data/embeddings/sample_v1.json"
V2_PATH = "data/embeddings/sample_v2.json"
THRESHOLD = 0.85
TOP_N = 5

def main():
    print("📊 Drift Analysis Between Two Versions\n")

    vec1, meta1 = load_vectors_from_json(V1_PATH)
    vec2, meta2 = load_vectors_from_json(V2_PATH)

    print(f"✅ Loaded V1: {vec1.shape}, V2: {vec2.shape}")

    sim_matrix = compute_drift_matrix(vec1, vec2)
    drift = analyze_drift(sim_matrix, meta1, meta2, threshold=THRESHOLD)

    print(f"\n📉 Top {TOP_N} Highly Drifted Chunks:\n")
    for i, d in enumerate(drift[:TOP_N]):
        print(f"[{i+1}] Drift Score: {d['drift_score']}")
        print(f"    Page {d['page_number']} → {d['matched_page']} | Match: {d['matched_score']}")
        print(f"    Text: {d['chunk_text']}...\n")

if __name__ == "__main__":
    main()
