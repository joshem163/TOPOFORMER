
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
# Define the Transformer model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps,dropout):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers,
                                          num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim * num_timesteps)

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        transformer_output = self.bn(transformer_output)
        transformer_output = self.dropout(transformer_output)
        predictions = self.fc(transformer_output)
        return predictions


class DualTransformerClassifier(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim, n_heads, n_layers, num_timesteps,dropout):
        super(DualTransformerClassifier, self).__init__()
        # Transformer for first set of features
        self.transformer1 = TransformerClassifier(input_dim1, hidden_dim, hidden_dim, n_heads, n_layers, num_timesteps,dropout)
        # Transformer for second set of features
        self.transformer2 = TransformerClassifier(input_dim2, hidden_dim, hidden_dim, n_heads, n_layers, num_timesteps,dropout)
        # Learnable weight for linear combination
        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Initialize with 0.5
        self.batch_norm_combined = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X1, X2):
        out1 = self.transformer1(X1)
        out2 = self.transformer2(X2)
        # Learnable linear combination
        combined = self.alpha * out1 + (1 - self.alpha) * out2
        combined = self.batch_norm_combined(combined)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output



def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    # Reset the alpha parameter explicitly
    if hasattr(model, 'alpha'):
        nn.init.constant_(model.alpha, 0.5)  # Reset alpha to its initial value of 0.5


class DualTransformerWithMLPClassifier(nn.Module):
    def __init__(
            self, input_dim1, input_dim2, mlp_input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps_1,
            num_timesteps_2,dropout
    ):
        super(DualTransformerWithMLPClassifier, self).__init__()
        # Transformer for first set of features
        self.transformer1 = TransformerClassifier(input_dim1, hidden_dim, hidden_dim, n_heads, n_layers,
                                                  num_timesteps_1,dropout)

        # Transformer for second set of features
        self.transformer2 = TransformerClassifier(input_dim2, hidden_dim, hidden_dim, n_heads, n_layers,
                                                  num_timesteps_2,dropout)

        # MLP for third feature set
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim + 200),
            nn.BatchNorm1d(hidden_dim + 200),  # Add BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim + 200, hidden_dim),
            nn.BatchNorm1d(hidden_dim)  # Add BatchNorm
        )

        # Learnable weights for linear combination
        self.alpha = nn.Parameter(torch.tensor(0.33, requires_grad=True))  # Initialize alpha with 0.33
        self.beta = nn.Parameter(torch.tensor(0.33, requires_grad=True))  # Initialize beta with 0.33

        # BatchNorm for combined output
        self.batch_norm_combined = nn.BatchNorm1d(hidden_dim)  # Add BatchNorm for combined features

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X1, X2, X3):
        out1 = self.transformer1(X1)  # Output from Transformer 1
        out2 = self.transformer2(X2)  # Output from Transformer 2
        out3 = self.mlp(X3)  # Output from MLP

        # Learnable linear combination
        combined = self.alpha * out1 + self.beta * out2 + (1 - self.alpha - self.beta) * out3

        # Apply BatchNorm to combined features
        combined = self.batch_norm_combined(combined)

        # Classification output
        output = self.fc(combined)
        return output


def reset_weights_mpp(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    # Reset the alpha and beta parameters explicitly
    if hasattr(model, 'alpha'):
        nn.init.constant_(model.alpha, 0.33)  # Reset alpha to its initial value of 0.5
    if hasattr(model, 'beta'):
        nn.init.constant_(model.beta, 0.33)  # Reset beta to its initial value of 0.5
    if hasattr(model, 'gamma'):
        nn.init.constant_(model.gamma, 0.33)  # Reset beta to its initial value of 0.5
