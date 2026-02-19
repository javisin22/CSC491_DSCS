import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig

# --------------------------------------------------
# Define the Fine-Tuned CodeRankEmbed Model with LoRA
# --------------------------------------------------
class FineTunedCodeRankEmbed(nn.Module):
    def __init__(self, model_name, projection_dim=256, pooling_strategy="mean"):
        """
        model_name: Hugging Face model name ("nomic-ai/CodeRankEmbed")
        projection_dim: Dimension of the semantic embedding output
        pooling_strategy: "mean" for mean pooling, "first" to use the first token's embedding.
        """
        super().__init__()
        self.pooling_strategy = pooling_strategy

        # Load the pre-trained CodeRankEmbed model
        self.base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        
        # Configure LoRA: we target key modules in the transformer. 
        # Here we target (for example) the query and key projections.
        lora_config = LoraConfig(
            r=8,                    # Low rank dimension for the update matrices
            lora_alpha=32,          # Scaling factor for the LoRA updates
            target_modules=["encoder.layer.0.attention.self.query", "encoder.layer.0.attention.self.key"],
            lora_dropout=0.1,       # Regularization via dropout
            bias="none",            # Don't train bias parameters
            task_type="FEATURE_EXTRACTION"  # Task-specific setting
        )
        # Apply LoRA via the PEFT package to the base model
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Add a projection head that maps the model's hidden state to the desired embedding dimension
        self.projection = nn.Linear(self.base_model.config.hidden_size, projection_dim)
    
    def forward(self, input_ids, attention_mask):
        """
        Pass input code tokens through the model and return a normalized embedding.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # Pooling strategy: choose mean pooling over tokens or use first token representation.
        if self.pooling_strategy == "mean":
            pooled = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        else:
            pooled = hidden_states[:, 0, :]  # (batch_size, hidden_size)
        
        # Project the pooled representation to a fixed-dimension embedding
        embedding = self.projection(pooled)  # (batch_size, projection_dim)
        # Normalize the embedding (L2 norm) for cosine similarity comparisons
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

# --------------------------------------------------
# Define the Speedup Predictor MLP
# --------------------------------------------------
class SpeedupPredictor(nn.Module):
    def __init__(self, embedding_dim=256, flag_dim=56, hidden_dim=128):
        """
        embedding_dim: Dimension of code embedding (from CodeRankEmbed).
        flag_dim: Dimension of the optimization flag vector.
        hidden_dim: Number of neurons in the hidden layer.
        """
        super().__init__()
        # Input size is the concatenation of the code embedding and the flag vector.
        self.fc1 = nn.Linear(embedding_dim + flag_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output is a single scalar (predicted speedup)
    
    def forward(self, code_embed, flag_vector):
        # Concatenate the code embedding and the optimization flag vector
        x = torch.cat((code_embed, flag_vector), dim=1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

# --------------------------------------------------
# Define a Dataset for C Source Files with Flags and Speedup Targets
# --------------------------------------------------
class CodeSpeedupDataset(Dataset):
    def __init__(self, data_code, data_flag, targets, tokenizer, max_length=512):
        """
        data_code: List of C source file strings.
        data_flag: List of corresponding optimization flag vectors (torch tensors with shape [flag_dim]).
        targets: List of target speedup values (scalar, e.g., measured speedup ratios).
        tokenizer: Hugging Face tokenizer for the CodeRankEmbed model.
        max_length: Maximum token length for each source file.
        """
        self.data_code = data_code
        self.data_flag = data_flag
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data_code)
    
    def __getitem__(self, idx):
        code = self.data_code[idx]
        flag_vec = self.data_flag[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        # Tokenize the source code
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dimension (returns tokenized code)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "flag_vector": flag_vec,  # Expected to be a tensor of shape (56,)
            "target": target  # Scalar target speedup
        }

# --------------------------------------------------
# Setup: Initialize Models, Tokenizer, Dataset, and DataLoader
# --------------------------------------------------
model_name = "nomic-ai/CodeRankEmbed"  # Base LLM for code embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the fine-tuned CodeRankEmbed model with LoRA
code_embedder = FineTunedCodeRankEmbed(model_name, projection_dim=256, pooling_strategy="mean")
# Initialize the MLP for speedup prediction
speedup_predictor = SpeedupPredictor(embedding_dim=256, flag_dim=56, hidden_dim=128)

# Example dummy data (replace these with your actual data)
data_code = [
    "int main() { for (int i = 0; i < 10; i++) { printf(\"Hello\"); } return 0; }",
    "void loop() { for (int j = 0; j < 20; j++) { process(j); } }"
]
# Dummy optimization flag vectors (56-dimensional)
data_flag = [torch.randn(56), torch.randn(56)]
# Dummy speedup targets (for example, 1.2 means 20% speedup improvement)
targets = [1.2, 0.8]

dataset = CodeSpeedupDataset(data_code, data_flag, targets, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# --------------------------------------------------
# Move models to GPU (if available) and set to training mode
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_embedder.to(device)
speedup_predictor.to(device)
code_embedder.train()
speedup_predictor.train()

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
optimizer = optim.AdamW(list(code_embedder.parameters()) + list(speedup_predictor.parameters()), lr=1e-5)
criterion = nn.MSELoss()  # Regression loss

num_epochs = 3  # Adjust based on your needs
for epoch in range(num_epochs):
    for batch in dataloader:
        # Transfer batch data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        flag_vector = batch["flag_vector"].to(device)
        target = batch["target"].unsqueeze(1).to(device)  # Reshape target to (batch_size, 1)
        
        optimizer.zero_grad()
        
        # Generate code embedding from the fine-tuned CodeRankEmbed model
        code_embedding = code_embedder(input_ids, attention_mask)
        
        # Predict speedup using the MLP which takes both the code embedding and the flag vector
        pred_speedup = speedup_predictor(code_embedding, flag_vector)
        
        # Compute Mean Squared Error between predicted speedup and actual target
        loss = criterion(pred_speedup, target)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training complete.")
