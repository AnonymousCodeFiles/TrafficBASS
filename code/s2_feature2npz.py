import json
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from collections import Counter
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import argparse
from tqdm import tqdm

def process_original_data(data, label_mapping):
    
    fields_to_remove = ['protocol', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
    for item in data:
        for field in fields_to_remove:
            item['cic'].pop(field, None)
        
        ip_lengths = item.get('ip_lengths', [])
        for i in range(len(ip_lengths)):
            if abs(ip_lengths[i]) > 1600:
                ip_lengths[i] = 1600 if ip_lengths[i] > 0 else -1600
        item['ip_lengths'] = ip_lengths

        label = item.get('Label', None)
        if label in label_mapping:
            item['Label'] = label_mapping[label]
        else:
            print(f"Warning: Label '{label}' not found in mapping")
        
    return data

def load_json_files(file_paths, label_mapping):
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print("Sucessful Read in Json: ", file_path, "Number of samples: ", len(data))
        
        data = process_original_data(data, label_mapping)
        merged_data.extend(data)
        
    return merged_data

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, max_len=256, decay_factor=0.5):
        super(EnhancedTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.max_len = max_len
        self.decay_factor = decay_factor

    def forward(self, src):
        batch_size, seq_len, _ = src.shape
        
        t = torch.arange(1, seq_len + 1, dtype=torch.float32, device=src.device)
        weights = torch.exp(-self.decay_factor * (seq_len - t) / seq_len)
        weights = weights.view(1, -1, 1).expand(batch_size, -1, 1)
        weighted_src = src * weights
        embedded = self.embedding(weighted_src)
        pos_encoded = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(pos_encoded)
        return output.mean(dim=1)

def feature_selection():
    selected_features = ['init_win_bytes_forward', 'init_win_bytes_backward', 'packet_length_mean', 'avg_segment_size', 
                         'packet_length_max', 'total_length', 'packet_length_min', 'flow_packets_s', 'subflow_bytes', 
                         'packet_length_std', 'packet_length_variance', 'flow_duration', 'iat_total', 'iat_mean', 
                         'header_length', 'iat_min', 'iat_max', 'flow_bytes_s', 'iat_std', 'ack_flag_count', 'total_packets', 
                         'psh_flag_count', 'syn_flag_count', 'subflow_packets', 'idle_max']
    
    return selected_features

def process_ip_lengths(ip_lengths, max_length=256):
    if len(ip_lengths) > max_length:
        return ip_lengths[:max_length]
    else:
        return ip_lengths + [0] * (max_length - len(ip_lengths))

def encode_sequence_batch(model, sequences):
    sequences_tensor = torch.FloatTensor(sequences).unsqueeze(-1)
    with torch.no_grad():
        encoded = model(sequences_tensor)
    return encoded.numpy()

def enhance_statistical_features(features, labels):

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(features, labels)
    
    importance_scores = rf_model.feature_importances_
    
    normalized_scores = importance_scores / np.sum(importance_scores)
    
    importance_matrix = np.diag(normalized_scores)
    
    # Apply importance weighting: Xs' = Xs · diag(S)
    enhanced_features = np.dot(features, importance_matrix)
    
    print("Feature importance applied to statistical features")
    print(f"Top 5 important features: {np.argsort(normalized_scores)[-5:]}")
    
    return enhanced_features

def main(file_paths, output_file, batch_size):
    
    label_mapping = {}
    
    merged_data = load_json_files(file_paths, label_mapping)
    print("Successful merged data!")
    labels = [record.get('Label', None) for record in merged_data]
    print(merged_data[0])

    distribution = Counter(labels)
    for label, count in distribution.items():
        print(f"Label: {label}, Count: {count}")

    print("Performing feature selection...")
    selected_features = feature_selection()

    print("Encoding sequence features with time decay weights enhancement...")
    model = EnhancedTransformerEncoder(input_dim=1, d_model=64, nhead=4, num_layers=2, decay_factor=0.5)
    model.eval()

    feature_cic = []
    feature_lengths = []
    labels = []

    for i in tqdm(range(0, len(merged_data), batch_size)):
        batch = merged_data[i:i+batch_size]

        batch_cic = [
            [sample['cic'][feature] for feature in selected_features]
            for sample in batch
        ]
        
        feature_cic.extend(batch_cic)

        batch_lengths = [
            process_ip_lengths(sample['ip_lengths'])
            for sample in batch
        ]
        encoded_lengths = encode_sequence_batch(model, batch_lengths)
        feature_lengths.extend(encoded_lengths)

        batch_labels = [sample['Label'] for sample in batch]
        labels.extend(batch_labels)

    print("Enhancing statistical features using feature importance...")
    feature_cic = np.array(feature_cic)
    feature_lengths = np.array(feature_lengths)
    labels_array = np.array(labels)
    
    enhanced_cic_features = enhance_statistical_features(feature_cic, labels_array)
    
    concatenated_features = np.hstack((enhanced_cic_features, feature_lengths))

    print(f"Saving data to {output_file}...")
    np.savez(output_file, features=concatenated_features, labels=labels_array)
    
    print("Done!")
    print(f"Feature shape: {concatenated_features.shape}")
    print(f"Label shape: {labels_array.shape}")
    print(f"Unique labels: {np.unique(labels_array)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data files")
    parser.add_argument('files', nargs='+', help='Input JSON files')
    parser.add_argument('-o', '--output', default='processed_data.npz', help='Output file name')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Batch size for processing')
    args = parser.parse_args()

    main(args.files, args.output, args.batch_size)
    
    # python script_name.py file1.json file2.json file3.json -o output_file.npz -b 500