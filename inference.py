import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')
from model import SalesPredictor

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying with weights_only=True...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Get model configuration
    config = checkpoint.get('config', {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'input_seq_len': 28,
        'output_seq_len': 7
    })
    
    # Determine number of features from saved model state
    first_layer_weight = checkpoint['model_state_dict']['fc_input.weight']
    num_features = first_layer_weight.shape[1]
    
    # Create model
    model = SalesPredictor(num_features, **config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def prepare_input_sequence(test_df, feature_cols):
    """Prepare last 28 days of data for each restaurant_menu"""
    sequences = {}
    
    grouped = test_df.groupby('restaurant_menu')
    for name, group in grouped:
        group = group.sort_values('date').reset_index(drop=True)
        
        # Take last 28 days
        if len(group) >= 28:
            last_28_days = group.tail(28)[feature_cols].values
            sequences[name] = torch.FloatTensor(last_28_days).unsqueeze(0)  # Add batch dimension
        else:
            print(f"Warning: {name} has less than 28 days of data ({len(group)} days)")
            # Pad with zeros if less than 28 days
            padding_size = 28 - len(group)
            data = group[feature_cols].values
            padded_data = np.vstack([np.zeros((padding_size, len(feature_cols))), data])
            sequences[name] = torch.FloatTensor(padded_data).unsqueeze(0)
    
    return sequences

def denormalize_predictions(predictions, menu_scalers, restaurant_menu_list):
    """Denormalize predictions back to original scale"""
    denormalized = {}
    
    for i, restaurant_menu in enumerate(restaurant_menu_list):
        pred = predictions[i]
        
        if restaurant_menu in menu_scalers and menu_scalers[restaurant_menu] is not None:
            scaler = menu_scalers[restaurant_menu]
            # Reshape for inverse transform
            pred_reshaped = pred.reshape(-1, 1)
            denorm = scaler.inverse_transform(pred_reshaped).flatten()
            # Round to integers and clip negative values
            denorm = np.maximum(0, np.round(denorm)).astype(int)
        else:
            # If no scaler available, assume already in correct scale
            denorm = np.maximum(0, np.round(pred * 100)).astype(int)  # Rough estimate
        
        denormalized[restaurant_menu] = denorm
    
    return denormalized

def create_submission(predictions_dict, test_name, all_menus):
    """Create submission dataframe in the required format"""
    
    # Initialize submission dataframe
    submission_rows = []
    
    # Create 7 rows for 7 days prediction
    for day in range(1, 8):
        row = {
            '영업일자': f'{test_name}+{day}일'
        }
        
        # Fill predictions for each menu
        for menu in all_menus:
            if menu in predictions_dict:
                row[menu] = predictions_dict[menu][day-1]
            else:
                row[menu] = 0
        
        submission_rows.append(row)
    
    return pd.DataFrame(submission_rows)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scalers
    print("Loading scalers and mappings...")
    try:
        # Load scalers from pickle file
        with open('data_preprocessed/scalers.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
            menu_scalers = scaler_data['menu_scalers']
    except FileNotFoundError:
        print("Warning: scalers.pkl not found. Trying to load from embedding_models.pt...")
        try:
            embedding_data = torch.load('data_preprocessed/embedding_models.pt', map_location='cpu', weights_only=False)
            menu_scalers = embedding_data.get('menu_scalers', {})
        except Exception as e:
            print(f"Warning: Could not load menu scalers: {e}")
            print("Proceeding without proper denormalization...")
            menu_scalers = {}
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, device)
    
    # Get all unique menus from sample submission
    sample_submission = pd.read_csv('data/sample_submission.csv')
    all_menus = [col for col in sample_submission.columns if col != '영업일자']
    
    # Process each test file
    test_files = sorted(glob.glob('data_preprocessed/TEST_*_preprocessed.csv'))
    all_submissions = []
    
    for test_file in test_files:
        test_name = os.path.basename(test_file).replace('_preprocessed.csv', '')
        print(f"\nProcessing {test_name}...")
        
        # Load test data
        test_df = pd.read_csv(test_file)
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        # Get feature columns (excluding date and restaurant_menu)
        feature_cols = [col for col in test_df.columns if col not in ['date', 'restaurant_menu']]
        
        # Prepare input sequences
        input_sequences = prepare_input_sequence(test_df, feature_cols)
        
        # Make predictions
        predictions = {}
        with torch.no_grad():
            for restaurant_menu, input_seq in tqdm(input_sequences.items(), desc="Predicting"):
                input_seq = input_seq.to(device)
                
                # Model prediction
                output = model(input_seq, target=None, teacher_forcing_ratio=0)
                output = output.squeeze().cpu().numpy()
                
                predictions[restaurant_menu] = output
        
        # Denormalize predictions
        restaurant_menu_list = list(predictions.keys())
        predictions_array = np.array(list(predictions.values()))
        denormalized = denormalize_predictions(predictions_array, menu_scalers, restaurant_menu_list)
        
        # Create submission for this test file
        submission = create_submission(denormalized, test_name, all_menus)
        all_submissions.append(submission)
    
    # Combine all submissions
    final_submission = pd.concat(all_submissions, ignore_index=True)
    
    # Save submission
    output_path = f'submission_{args.output_suffix}.csv'
    final_submission.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {final_submission.shape}")
    
    # Display sample predictions
    print("\nSample predictions (first 5 menus, first test file):")
    print(final_submission[['영업일자'] + all_menus[:5]].head(7))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    
    parser.add_argument('--model_path', type=str, default='final_model_seed42.pt',
                      help='Path to trained model checkpoint')
    parser.add_argument('--output_suffix', type=str, default='final',
                      help='Suffix for output submission file')
    
    args = parser.parse_args()
    
    main(args)