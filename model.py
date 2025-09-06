import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RestaurantEmbedding(nn.Module):
    def __init__(self, num_restaurants, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(num_restaurants, embedding_dim)
        nn.init.uniform_(self.embedding.weight, 0, 1)
    
    def forward(self, x):
        return self.embedding(x)

class MenuEmbedding(nn.Module):
    def __init__(self, num_menus, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(num_menus, embedding_dim)
        nn.init.uniform_(self.embedding.weight, 0, 1)
    
    def forward(self, x):
        return self.embedding(x)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, 1, hidden_dim]
        encoder_outputs: [batch_size, seq_len, hidden_dim]
        """
        seq_len = encoder_outputs.size(1)
        
        hidden = hidden.repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        
        attention_scores = self.v(energy).squeeze(2)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        outputs, (hidden, cell) = self.lstm(x)
        
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)
        
        hidden = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
        hidden = self.fc(hidden)
        
        cell = torch.cat([cell[-2,:,:], cell[-1,:,:]], dim=1)
        cell = self.fc(cell)
        
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_features, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.num_layers = num_layers
        
        self.attention = Attention(hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim + num_features,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2 + num_features, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        """
        input: [batch_size, 1, num_features]
        hidden: [num_layers, batch_size, hidden_dim]
        cell: [num_layers, batch_size, hidden_dim]
        encoder_outputs: [batch_size, seq_len, hidden_dim]
        """
        context, attention_weights = self.attention(hidden[-1:].transpose(0, 1), encoder_outputs)
        
        lstm_input = torch.cat([input, context], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat([output, context, input], dim=2))
        
        return prediction, hidden, cell, attention_weights

class SalesPredictor(nn.Module):
    def __init__(self, 
                 num_features,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.2,
                 input_seq_len=28,
                 output_seq_len=7):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        self.encoder = Encoder(num_features, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(1, hidden_dim, num_features, num_layers, dropout)
        
        self.fc_input = nn.Linear(num_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
        
    def forward(self, x, target=None, teacher_forcing_ratio=0.5, future_features=None):
        """
        x: [batch_size, input_seq_len, num_features]
        target: [batch_size, output_seq_len] (only for training)
        future_features: [batch_size, output_seq_len, num_features] (should be provided for both training and inference)
        """
        batch_size = x.size(0)
        
        x = self.layer_norm(self.fc_input(x))
        
        encoder_outputs, hidden, cell = self.encoder(x)
        
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        outputs = []
        
        # If future_features not provided, use last day as fallback (backward compatibility)
        if future_features is None:
            last_day_features = x[:, -1:, :]
            future_features = last_day_features.repeat(1, self.output_seq_len, 1)
        
        for t in range(self.output_seq_len):
            # Get future features for day t
            input_features = future_features[:, t:t+1, :].clone()
            
            # Apply teacher forcing only to sales_count (first feature)
            if t == 0:
                # First prediction uses last day's actual sales
                input_features[:, :, 0] = x[:, -1, 0:1]
            else:
                # For subsequent days, use teacher forcing or previous prediction
                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    # Use actual sales from target
                    input_features[:, :, 0] = target[:, t-1:t]
                else:
                    # Use previous prediction
                    input_features[:, :, 0] = outputs[-1].squeeze(2)
            
            prediction, hidden, cell, _ = self.decoder(input_features, hidden, cell, encoder_outputs)
            
            outputs.append(prediction)
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs.squeeze(2)

class SalesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_seq_len=28, output_seq_len=7, stride=1):
        """
        data: preprocessed dataframe
        input_seq_len: number of days to use as input
        output_seq_len: number of days to predict
        stride: sliding window stride
        """
        self.data = data
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.stride = stride
        
        self.feature_cols = [col for col in data.columns 
                           if col not in ['date', 'restaurant_menu']]
        
        self.samples = self._create_sequences()
    
    def _create_sequences(self):
        samples = []
        
        grouped = self.data.groupby('restaurant_menu')
        
        for name, group in grouped:
            group = group.sort_values('date').reset_index(drop=True)
            
            if len(group) < self.input_seq_len + self.output_seq_len:
                continue
            
            for i in range(0, len(group) - self.input_seq_len - self.output_seq_len + 1, self.stride):
                input_seq = group.iloc[i:i + self.input_seq_len][self.feature_cols].values
                output_seq = group.iloc[i + self.input_seq_len:i + self.input_seq_len + self.output_seq_len]['sales_count_norm'].values
                # Add future features for the output period
                future_features = group.iloc[i + self.input_seq_len:i + self.input_seq_len + self.output_seq_len][self.feature_cols].values
                
                samples.append((input_seq, output_seq, future_features))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, output_seq, future_features = self.samples[idx]
        return torch.FloatTensor(input_seq), torch.FloatTensor(output_seq), torch.FloatTensor(future_features)

def create_model(num_features, config=None):
    """
    Create and return the sales prediction model
    
    Args:
        num_features: number of input features
        config: optional configuration dictionary
    """
    default_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'input_seq_len': 28,
        'output_seq_len': 7
    }
    
    if config:
        default_config.update(config)
    
    model = SalesPredictor(
        num_features=num_features,
        hidden_dim=default_config['hidden_dim'],
        num_layers=default_config['num_layers'],
        dropout=default_config['dropout'],
        input_seq_len=default_config['input_seq_len'],
        output_seq_len=default_config['output_seq_len']
    )
    
    return model