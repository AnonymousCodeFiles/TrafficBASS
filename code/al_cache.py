import os
import pickle
import hashlib
from datetime import datetime

def get_cache_key(config, unlabeled_indices, strategy_name):
    key_info = {
        'data_path': config.data.data_path,
        'model_path': config.model.model_path,
        'strategy': strategy_name,
        'initial_labeled_samples': config.al.initial_labeled_samples,
        'max_length': config.data.max_length,
        'num_labels': len(set(unlabeled_indices))
    }
    
    key_str = str(sorted(key_info.items()))
    return hashlib.md5(key_str.encode()).hexdigest()

def save_strategy_state(strategy, unlabeled_indices, config, value_ranking=None):
    strategy_name = config.al.strategy
    index_dir = os.path.join(config.model.output_dir, 'index')
    os.makedirs(index_dir, exist_ok=True)
    
    cache_key = get_cache_key(config, unlabeled_indices, strategy_name)
    
    state = {
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy_name,
        'unlabeled_indices': unlabeled_indices,
        'remaining_indices': strategy._remaining_indices,
        'value_ranking': value_ranking if value_ranking is not None else []
    }
    
    cache_path = os.path.join(index_dir, f"{strategy_name}_{cache_key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Strategy state saved to {cache_path}")
    
    meta_path = os.path.join(index_dir, f"{strategy_name}_{cache_key}.meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Timestamp: {state['timestamp']}\n")
        f.write(f"Data path: {config.data.data_path}\n")
        f.write(f"Model path: {config.model.model_path}\n")
        f.write(f"Unlabeled samples: {len(unlabeled_indices)}\n")
        f.write(f"Remaining indices: {len(strategy._remaining_indices)}\n")
    
    return cache_path

def load_strategy_state(config, unlabeled_indices, strategy_name):
    index_dir = os.path.join(config.model.output_dir, 'index')
    if not os.path.exists(index_dir):
        return None
    
    cache_key = get_cache_key(config, unlabeled_indices, strategy_name)
    cache_path = os.path.join(index_dir, f"{strategy_name}_{cache_key}.pkl")
    
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'rb') as f:
            state = pickle.load(f)
        
        print(f"Loaded strategy state from {cache_path}")
        print(f"Cache created at: {state['timestamp']}")
        return state
    except Exception as e:
        print(f"Error loading cached strategy state: {str(e)}")
        return None

