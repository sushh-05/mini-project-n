"""
Training module for learning saree-blouse matching patterns.
Stores positive examples and improves matching accuracy over time.
"""
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class MatchTrainer:
    """
    Learns from user-confirmed saree-blouse matches to improve recommendations.
    """
    
    def __init__(self, training_file="outputs/training_pairs.json"):
        self.training_file = Path(training_file)
        self.training_pairs = []
        self.load_training_data()
    
    def load_training_data(self):
        """Load existing training data."""
        if self.training_file.exists():
            with open(self.training_file, 'r') as f:
                data = json.load(f)
                self.training_pairs = data.get('pairs', [])
            print(f"Loaded {len(self.training_pairs)} training pairs")
        else:
            self.training_pairs = []
    
    def save_training_data(self):
        """Save training data to file."""
        self.training_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'pairs': self.training_pairs,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.training_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.training_pairs)} training pairs")
    
    def add_training_pair(self, saree_id, blouse_id, saree_embedding, blouse_embedding, 
                         saree_meta=None, blouse_meta=None, score=1.0):
        """
        Add a confirmed saree-blouse match to training data.
        
        Args:
            saree_id: Saree item ID
            blouse_id: Blouse item ID  
            saree_embedding: Saree CLIP embedding (numpy array)
            blouse_embedding: Blouse CLIP embedding (numpy array)
            saree_meta: Saree metadata dict (color, fabric, etc.)
            blouse_meta: Blouse metadata dict
            score: User feedback score (0-1, default 1.0 for positive match)
        """
        pair = {
            'saree_id': saree_id,
            'blouse_id': blouse_id,
            'saree_embedding': saree_embedding.tolist() if isinstance(saree_embedding, np.ndarray) else saree_embedding,
            'blouse_embedding': blouse_embedding.tolist() if isinstance(blouse_embedding, np.ndarray) else blouse_embedding,
            'saree_meta': saree_meta or {},
            'blouse_meta': blouse_meta or {},
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        self.training_pairs.append(pair)
        self.save_training_data()
    
    def get_learned_boost(self, query_embedding, candidate_embedding, 
                         query_meta=None, candidate_meta=None):
        """
        Calculate a boost score based on similarity to known good matches.
        
        Returns:
            boost: Additional score (0-1) based on training data
        """
        if not self.training_pairs:
            return 0.0
        
        query_emb = np.array(query_embedding).reshape(1, -1)
        cand_emb = np.array(candidate_embedding).reshape(1, -1)
        
        boosts = []
        
        for pair in self.training_pairs:
            # Compare embeddings with training examples
            saree_sim = cosine_similarity(query_emb, 
                                         np.array(pair['saree_embedding']).reshape(1, -1))[0][0]
            blouse_sim = cosine_similarity(cand_emb,
                                          np.array(pair['blouse_embedding']).reshape(1, -1))[0][0]
            
            # If query saree is similar to a training saree AND 
            # candidate blouse is similar to its paired blouse, boost score
            if saree_sim > 0.85 and blouse_sim > 0.85:
                combined_sim = (saree_sim + blouse_sim) / 2
                boosts.append(combined_sim * pair['score'])
        
        if boosts:
            # Return weighted average of top matches
            boosts.sort(reverse=True)
            return np.mean(boosts[:3])  # Average of top 3
        
        return 0.0
    
    def get_training_stats(self):
        """Get statistics about training data."""
        if not self.training_pairs:
            return {
                'total_pairs': 0,
                'avg_score': 0.0,
                'color_pairs': {}
            }
        
        color_pairs = {}
        for pair in self.training_pairs:
            saree_color = pair['saree_meta'].get('color', 'unknown')
            blouse_color = pair['blouse_meta'].get('color', 'unknown')
            key = f"{saree_color}+{blouse_color}"
            color_pairs[key] = color_pairs.get(key, 0) + 1
        
        return {
            'total_pairs': len(self.training_pairs),
            'avg_score': np.mean([p['score'] for p in self.training_pairs]),
            'color_pairs': color_pairs,
            'most_common': max(color_pairs.items(), key=lambda x: x[1]) if color_pairs else None
        }
    
    def suggest_color_compatibility(self, saree_color):
        """
        Suggest compatible blouse colors based on training data.
        
        Args:
            saree_color: Color of the saree
            
        Returns:
            List of (blouse_color, confidence) tuples
        """
        color_matches = {}
        
        for pair in self.training_pairs:
            if pair['saree_meta'].get('color', '').lower() == saree_color.lower():
                blouse_color = pair['blouse_meta'].get('color', 'unknown')
                score = pair['score']
                
                if blouse_color in color_matches:
                    color_matches[blouse_color].append(score)
                else:
                    color_matches[blouse_color] = [score]
        
        # Average scores and sort
        suggestions = [(color, np.mean(scores)) 
                      for color, scores in color_matches.items()]
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions


def create_training_from_catalog(catalog_df, embeddings):
    """
    Create initial training data from catalog by matching similar items.
    Useful for bootstrap training.
    """
    trainer = MatchTrainer()
    
    sarees = catalog_df[catalog_df['item_type'].str.lower() == 'saree']
    blouses = catalog_df[catalog_df['item_type'].str.lower() == 'blouse']
    
    for _, saree in sarees.iterrows():
        saree_emb = embeddings[int(saree['embedding_index'])]
        saree_color = saree['color'].lower()
        
        # Find matching blouse by color/style rules
        for _, blouse in blouses.iterrows():
            blouse_emb = embeddings[int(blouse['embedding_index'])]
            blouse_color = blouse['color'].lower()
            
            # Auto-match based on color rules
            is_match = False
            
            # Exact color match
            if saree_color == blouse_color:
                is_match = True
                score = 0.95
            
            # Complementary pairs
            elif (saree_color, blouse_color) in [
                ('black', 'gold'), ('black', 'silver'),
                ('green', 'gold'), ('purple', 'gold'),
                ('silver', 'silver')  # Same color
            ]:
                is_match = True
                score = 0.90
            
            if is_match:
                trainer.add_training_pair(
                    saree_id=saree['item_id'],
                    blouse_id=blouse['item_id'],
                    saree_embedding=saree_emb,
                    blouse_embedding=blouse_emb,
                    saree_meta={'color': saree_color, 'fabric': saree['fabric']},
                    blouse_meta={'color': blouse_color, 'fabric': blouse['fabric']},
                    score=score
                )
    
    return trainer


if __name__ == "__main__":
    import pandas as pd
    
    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "outputs" / "catalog_with_status.csv"
    embeddings_path = project_root / "outputs" / "embeddings.npy"
    
    if not catalog_path.exists() or not embeddings_path.exists():
        print("Please run extract_embeddings.py first!")
        exit(1)
    
    catalog_df = pd.read_csv(catalog_path)
    embeddings = np.load(embeddings_path)
    
    print("Creating training data from catalog...")
    trainer = create_training_from_catalog(catalog_df, embeddings)
    
    stats = trainer.get_training_stats()
    print(f"\nTraining Statistics:")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Average score: {stats['avg_score']:.2f}")
    print(f"\nColor combinations:")
    for combo, count in stats['color_pairs'].items():
        print(f"  {combo}: {count}")
