# Filename: main.py (Diagnostic Version)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import sys

# --- CONFIGURATION ---
INPUT_FILES = [
    './datasets/explanations_part_1.csv',
    './datasets/explanations_part_2.csv',
    './datasets/explanations_part_3.csv',
    './datasets/explanations_part_4.csv'
]
MODEL_NAME = 'bert-large-uncased'
BEST_SAVE_PATH = Path(f'./results/bert-large_best.pth')
LAST_SAVE_PATH = Path(f'./results/bert-large_last.pth')
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
TEST_SPLIT_SIZE = 0.15

class ExplanationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.explanations1 = df['explanation1'].tolist()
        self.explanations2 = df['explanation2'].tolist()
        self.controls = df[['throttle', 'brake']].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.controls)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.explanations1[idx], self.explanations2[idx], add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        control_labels = torch.tensor(self.controls[idx], dtype=torch.float)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}, control_labels

class AnalysisModule(nn.Module):
    def __init__(self, model_name):
        super(AnalysisModule, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_output_dim = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(encoder_output_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))
    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_output.last_hidden_state[:, 0]
        controls = self.classifier(pooled_output)
        return controls

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training.")

    try:
        df_list = [pd.read_csv(f) for f in INPUT_FILES]
    except FileNotFoundError as e:
        print(f"Could not find one of the input files. Details: {e}")
        sys.exit()

    df = pd.concat(df_list, ignore_index=True)
    rows_before_cleaning = len(df)
    df.dropna(subset=['explanation1', 'explanation2'], inplace=True)
    df = df[~df['explanation1'].str.contains("ERROR", case=False, na=False)]
    df = df[~df['explanation2'].str.contains("ERROR", case=False, na=False)].reset_index(drop=True)
    rows_after_cleaning = len(df)
    
    print(f"Rows after filtering out 'ERROR' strings: {rows_after_cleaning}")

    if rows_after_cleaning == 0:
        print("\nThe dataset is empty.")

    train_val_df, test_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=(TEST_SPLIT_SIZE/(1.0-TEST_SPLIT_SIZE)), random_state=42)
    print(f"Data split sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ExplanationDataset(train_df, tokenizer)
    val_dataset = ExplanationDataset(val_df, tokenizer)
    test_dataset = ExplanationDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = AnalysisModule(MODEL_NAME).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = 1
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train(); total_train_loss=0; train_progress_bar=tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in train_progress_bar:
            inputs,labels=batch; input_ids=inputs['input_ids'].to(device); attention_mask=inputs['attention_mask'].to(device); labels=labels.to(device)
            optimizer.zero_grad(); outputs=model(input_ids,attention_mask); loss=criterion(outputs,labels); loss.backward(); optimizer.step()
            total_train_loss+=loss.item(); train_progress_bar.set_postfix({'train_loss':f'{loss.item():.4f}'})
        avg_train_loss=total_train_loss/len(train_loader)
        model.eval(); total_val_loss=0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs,labels=batch; input_ids=inputs['input_ids'].to(device); attention_mask=inputs['attention_mask'].to(device); labels=labels.to(device)
                outputs=model(input_ids,attention_mask); loss=criterion(outputs,labels); total_val_loss+=loss.item()
        avg_val_loss=total_val_loss/len(val_loader)
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f} | Avg Validation Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        
    torch.save(model.state_dict(), LAST_SAVE_PATH)
    torch.save(best_model_state, BEST_SAVE_PATH)
    print(f"Training completed. Model saved to: {LAST_SAVE_PATH} and {BEST_SAVE_PATH}")

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Test"):
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    
    print("---" * 10)
    print(f"Final Loss on Test Set: {avg_test_loss:.4f}")
    print("---" * 10)
    

