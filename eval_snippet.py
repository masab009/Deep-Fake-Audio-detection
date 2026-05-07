import torch
import torch.nn as nn
from main import get_config
from data_preprocessing import ASVspoofDataset, EpisodicSampler
from feature_engineering import DualStreamFeatureExtractor
from model import DeepfakeDetector

def main():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = 'outputs/checkpoints/best_model.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: {checkpoint_path} not found.")
        return

    feature_extractor = DualStreamFeatureExtractor()
    detector = DeepfakeDetector(feature_extractor).to(device)
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.eval()
    
    # Construct eval dataset
    # Note: ASVspoofDataset likely takes config or other params based on data_preprocessing.py
    eval_dataset = ASVspoofDataset(mode='eval')
    print(f"len(eval_dataset): {len(eval_dataset)}")
    
    # Sample one few-shot episode for attack A10 with k=5
    # Assuming EpisodicSampler takes (dataset, n_way, k_shot, n_query, iterations)
    # Finding A10 indices
    a10_indices = [i for i, meta in enumerate(eval_dataset.metadata) if meta.get('attack_type') == 'A10']
    
    if not a10_indices:
        print("Warning: No A10 samples found. Using all eval samples.")
        subset_indices = list(range(len(eval_dataset)))
    else:
        subset_indices = a10_indices

    # Simplified sampler behavior for the snippet
    sampler = EpisodicSampler(eval_dataset, n_way=2, k_shot=5, n_query=10)
    
    # Get one episode
    # EpisodicSampler.__iter__ usually yields indices for one episode
    indices = next(iter(sampler))
    
    # Load support/query audio tensors
    # EpisodicSampler usually returns [support_indices, query_indices] or similar
    # If it returns a flat list, we split it based on n_way, k_shot, n_query
    n_way = 2
    k_shot = 5
    n_query = 10
    
    support_indices = indices[:n_way * k_shot]
    query_indices = indices[n_way * k_shot:]
    
    support_x = torch.stack([eval_dataset[i][0] for i in support_indices]).to(device)
    query_x = torch.stack([eval_dataset[i][0] for i in query_indices]).to(device)
    
    print(f"support_x shape: {support_x.shape}")
    print(f"query_x shape: {query_x.shape}")
    
    with torch.no_grad():
        # Prototypes
        support_features = detector.feature_extractor(support_x) 
        # Reshape to (n_way, k_shot, feat_dim)
        feat_dim = support_features.shape[-1]
        z_proto = support_features.view(n_way, k_shot, feat_dim).mean(1)
        
        query_features = detector.feature_extractor(query_x)
        
        # Compute scores (negative distance to prototypes)
        # query_features: (n_query_total, feat_dim), z_proto: (n_way, feat_dim)
        dists = torch.cdist(query_features, z_proto) # (n_query_total, n_way)
        scores = -dists
        
        # spoof scores (assume class 1 is spoof)
        spoof_scores = scores[:, 1]
        
        print(f"Scores min: {spoof_scores.min().item():.4f}")
        print(f"Scores max: {spoof_scores.max().item():.4f}")
        print(f"Scores mean: {spoof_scores.mean().item():.4f}")

if __name__ == "__main__":
    main()
