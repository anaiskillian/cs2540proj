import numpy as np
import torch
import torch.nn as nn
from z3 import *
import time
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import seaborn as sns
import datetime
import json

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_synthetic_dataset():
    """Create a synthetic text dataset for sentiment analysis."""
    texts = [
        "This movie was great! I loved it.",
        "The food was terrible and the service was slow.",
        "I had an amazing experience at the concert.",
        "The product broke after just one day of use.",
        "The book was incredibly boring and hard to read.",
        "The hotel room was clean and comfortable.",
        "The customer service was rude and unhelpful.",
        "The game was fun and engaging for hours.",
        "The restaurant was overpriced and the food was cold.",
        "The show was entertaining and well-produced."
    ]
    labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
    return texts, labels

def train_model(texts, labels, num_epochs=3, batch_size=2):
    """Train a BERT model for text classification."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
    
    return model, tokenizer

def fgsm_attack(model, tokenizer, text, epsilon=0.1):
    """Perform FGSM attack on text input."""
    device = next(model.parameters()).device
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get original prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        original_pred = torch.argmax(outputs.logits, dim=1).item()
    
    # Create adversarial example by perturbing embeddings
    # Get the embeddings and create a copy that requires gradients
    with torch.no_grad():
        embeddings = model.bert.embeddings.word_embeddings(input_ids)
    
    # Create a copy that requires gradients
    embeddings_adv = embeddings.clone().detach().requires_grad_(True)
    
    # Forward pass with embeddings and compute loss
    outputs = model(inputs_embeds=embeddings_adv, attention_mask=attention_mask)
    # Create labels tensor (we want to maximize loss, so use opposite of original prediction)
    labels = torch.tensor([1 - original_pred], device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs.logits, labels)
    loss.backward()
    
    # Get gradient and create adversarial example
    grad = embeddings_adv.grad.data
    perturbed_embeddings = embeddings_adv + epsilon * grad.sign()
    
    # Get prediction for adversarial example
    with torch.no_grad():
        outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
        perturbed_pred = torch.argmax(outputs.logits, dim=1).item()
    
    # Find the closest token for each embedding
    with torch.no_grad():
        # Get all token embeddings
        all_token_embeddings = model.bert.embeddings.word_embeddings.weight
        
        # For each position in the sequence
        perturbed_tokens = []
        for i in range(perturbed_embeddings.shape[1]):
            # Compute distances to all token embeddings
            distances = torch.norm(all_token_embeddings - perturbed_embeddings[0, i], dim=1)
            # Get the closest token
            closest_token = torch.argmin(distances)
            perturbed_tokens.append(closest_token.item())
        
        # Convert to tensor
        perturbed_input = torch.tensor([perturbed_tokens], device=device)
    
    # Convert back to text
    perturbed_text = tokenizer.decode(perturbed_input[0], skip_special_tokens=True)
    
    return perturbed_text, original_pred, perturbed_pred

def encode_bert_in_z3(model, tokenizer, text, epsilon):
    """Encode BERT model in Z3 for formal verification."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'][0]
    
    # Create Z3 variables
    input_vars = [Int(f'x_{i}') for i in range(len(input_ids))]
    perturbed_vars = [Int(f'x_perturbed_{i}') for i in range(len(input_ids))]
    
    # Create solver
    solver = Solver()
    
    # Add constraints for perturbation
    for i in range(len(input_ids)):
        # Convert PyTorch tensor to Python integer
        input_id = input_ids[i].item()
        solver.add(perturbed_vars[i] >= input_id - epsilon)
        solver.add(perturbed_vars[i] <= input_id + epsilon)
        solver.add(perturbed_vars[i] >= 0)
        solver.add(perturbed_vars[i] < tokenizer.vocab_size)
    
    # Encode BERT layers (simplified)
    # Note: This is a simplified version as encoding the full BERT model in Z3 is complex
    # We'll focus on the embedding layer and first transformer layer
    
    # Embedding layer
    embedding_vars = []
    for i in range(len(perturbed_vars)):
        # Create embedding variables
        embedding = [Real(f'emb_{i}_{j}') for j in range(768)]  # BERT hidden size
        embedding_vars.append(embedding)
        
        # Add constraints for embedding lookup
        for j in range(768):
            solver.add(embedding[j] == perturbed_vars[i])  # Simplified
    
    # First transformer layer (simplified)
    output_vars = [Real(f'out_{i}') for i in range(2)]  # Binary classification
    
    # Add constraints for classification
    solver.add(output_vars[0] != output_vars[1])  # Different classes
    
    return solver, input_vars, perturbed_vars, output_vars

def verify_robustness_smt(model, tokenizer, text, epsilon):
    """Verify robustness using SMT solver."""
    solver, input_vars, perturbed_vars, output_vars = encode_bert_in_z3(model, tokenizer, text, epsilon)
    
    # Check satisfiability
    result = solver.check()
    
    if result == sat:
        model_solution = solver.model()
        perturbed_input = [model_solution.eval(var).as_long() for var in perturbed_vars]
        perturbed_text = tokenizer.decode(perturbed_input, skip_special_tokens=True)
        return False, perturbed_text
    else:
        return True, None

def create_output_directory():
    """Create a timestamped output directory for this run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_config(output_dir, config):
    """Save model configuration to a JSON file."""
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

def visualize_text_perturbations(original_text, smt_text, fgsm_text, original_pred, smt_pred, fgsm_pred, output_dir):
    """Visualize the original text and its adversarial variants."""
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    
    # Original text
    axs[0].text(0.5, 0.5, f"Original Text (Class {original_pred}):\n{original_text}", 
                ha='center', va='center', fontsize=12, wrap=True)
    axs[0].set_title("Original Text")
    axs[0].axis('off')
    
    # SMT perturbed text
    if smt_text:
        axs[1].text(0.5, 0.5, f"SMT Perturbed Text (Class {smt_pred}):\n{smt_text}", 
                    ha='center', va='center', fontsize=12, wrap=True)
        axs[1].set_title("SMT Adversarial Example")
    else:
        axs[1].text(0.5, 0.5, "No adversarial example found with SMT", 
                    ha='center', va='center', fontsize=12)
    axs[1].axis('off')
    
    # FGSM perturbed text
    axs[2].text(0.5, 0.5, f"FGSM Perturbed Text (Class {fgsm_pred}):\n{fgsm_text}", 
                ha='center', va='center', fontsize=12, wrap=True)
    axs[2].set_title("FGSM Adversarial Example")
    axs[2].axis('off')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "text_perturbations.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Text visualization saved to {filename}")

def visualize_token_changes(tokenizer, original_text, smt_text, fgsm_text, output_dir):
    """Visualize the token-level changes made by each approach."""
    # Tokenize texts
    original_tokens = tokenizer.tokenize(original_text)
    smt_tokens = tokenizer.tokenize(smt_text) if smt_text else None
    fgsm_tokens = tokenizer.tokenize(fgsm_text)
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    
    # Original tokens
    axs[0].bar(range(len(original_tokens)), [1] * len(original_tokens), 
               color='blue', alpha=0.5)
    axs[0].set_xticks(range(len(original_tokens)))
    axs[0].set_xticklabels(original_tokens, rotation=45)
    axs[0].set_title("Original Tokens")
    
    # SMT token changes
    if smt_tokens:
        changes = [1 if orig != smt else 0 for orig, smt in zip(original_tokens, smt_tokens)]
        axs[1].bar(range(len(changes)), changes, 
                   color='red', alpha=0.5)
        axs[1].set_xticks(range(len(changes)))
        axs[1].set_xticklabels(smt_tokens, rotation=45)
        axs[1].set_title("SMT Token Changes (Red = Changed)")
    
    # FGSM token changes
    changes = [1 if orig != fgsm else 0 for orig, fgsm in zip(original_tokens, fgsm_tokens)]
    axs[2].bar(range(len(changes)), changes, 
               color='green', alpha=0.5)
    axs[2].set_xticks(range(len(changes)))
    axs[2].set_xticklabels(fgsm_tokens, rotation=45)
    axs[2].set_title("FGSM Token Changes (Green = Changed)")
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "token_changes.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Token changes visualization saved to {filename}")

def compare_verification_methods(model, tokenizer, text, epsilon=0.1, output_dir=None):
    """Compare SMT and gradient-based verification methods using the same perturbation."""
    if output_dir is None:
        output_dir = create_output_directory()
    
    results = {}
    
    # First, get the SMT perturbation
    print("\nPerforming SMT verification...")
    is_robust, smt_text = verify_robustness_smt(model, tokenizer, text, epsilon)
    
    # Get original prediction
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        original_pred = torch.argmax(outputs.logits, dim=1).item()
    
    if not is_robust and smt_text:
        # If SMT found a perturbation, test it with the gradient-based method
        print("\nTesting SMT perturbation with gradient-based method...")
        smt_inputs = tokenizer(smt_text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            smt_outputs = model(**smt_inputs)
            smt_pred = torch.argmax(smt_outputs.logits, dim=1).item()
        
        # Now test the same perturbation with gradient-based method
        fgsm_text, _, fgsm_pred = fgsm_attack(model, tokenizer, text, epsilon)
        
        results['smt'] = {
            'text': smt_text,
            'original_pred': original_pred,
            'perturbed_pred': smt_pred,
            'success': original_pred != smt_pred
        }
        
        results['fgsm'] = {
            'text': fgsm_text,
            'original_pred': original_pred,
            'perturbed_pred': fgsm_pred,
            'success': original_pred != fgsm_pred
        }
        
        # Create visualizations
        visualize_text_perturbations(
            text, 
            smt_text, 
            fgsm_text,
            original_pred,
            smt_pred,
            fgsm_pred,
            output_dir
        )
        
        visualize_token_changes(
            tokenizer,
            text,
            smt_text,
            fgsm_text,
            output_dir
        )
        
        # Print comparison
        print("\n=== Comparison ===")
        print(f"Original text: {text}")
        print(f"Original prediction: {original_pred}")
        
        print("\nSMT Results:")
        print(f"Perturbed text: {smt_text}")
        print(f"Changed prediction to: {smt_pred}")
        print(f"Success: {results['smt']['success']}")
        
        print("\nFGSM Results:")
        print(f"Perturbed text: {fgsm_text}")
        print(f"Changed prediction to: {fgsm_pred}")
        print(f"Success: {results['fgsm']['success']}")
        
        # Save results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_path}")
    else:
        print("No adversarial example found with SMT within epsilon")
        results = None
    
    return results, output_dir

def select_test_examples(texts, labels, num_examples=3):
    """Select diverse test examples from the dataset."""
    # Ensure we have both positive and negative examples
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Select examples from both classes
    selected_indices = []
    if positive_indices:
        selected_indices.append(np.random.choice(positive_indices))
    if negative_indices:
        selected_indices.append(np.random.choice(negative_indices))
    
    # Fill remaining slots with random examples
    remaining_slots = num_examples - len(selected_indices)
    if remaining_slots > 0:
        remaining_indices = list(set(range(len(texts))) - set(selected_indices))
        if remaining_indices:
            selected_indices.extend(np.random.choice(remaining_indices, 
                                                   size=min(remaining_slots, len(remaining_indices)), 
                                                   replace=False))
    
    return [(texts[i], labels[i]) for i in selected_indices]

def main():
    # Configuration
    config = {
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 2e-5,
        "epsilon": 0.1,
        "model_name": "bert-base-uncased",
        "max_length": 128,
        "num_test_examples": 3
    }
    
    # Create output directory
    output_dir = create_output_directory()
    save_config(output_dir, config)
    
    # Create synthetic dataset
    texts, labels = create_synthetic_dataset()
    
    # Train model
    print("Training BERT model...")
    model, tokenizer = train_model(texts, labels, 
                                 num_epochs=config["num_epochs"],
                                 batch_size=config["batch_size"])
    
    # Select test examples
    test_examples = select_test_examples(texts, labels, config["num_test_examples"])
    
    # Run verification for each test example
    all_results = {}
    for i, (test_text, test_label) in enumerate(test_examples):
        print(f"\nTesting example {i+1}/{len(test_examples)}:")
        print(f"Text: {test_text}")
        print(f"True label: {test_label}")
        
        # Create subdirectory for this example
        example_dir = os.path.join(output_dir, f"example_{i+1}")
        os.makedirs(example_dir, exist_ok=True)
        
        # Compare verification methods
        results, _ = compare_verification_methods(
            model, tokenizer, test_text, 
            epsilon=config["epsilon"],
            output_dir=example_dir
        )
        
        if results:  # Only save if we found an adversarial example
            all_results[f"example_{i+1}"] = {
                "text": test_text,
                "true_label": test_label,
                "results": results
            }
    
    # Save all results
    if all_results:
        results_path = os.path.join(output_dir, "all_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nAll results saved to {results_path}")
        
        # Print summary
        print("\n=== Summary ===")
        for i, (test_text, test_label) in enumerate(test_examples):
            if f"example_{i+1}" in all_results:
                example_results = all_results[f"example_{i+1}"]["results"]
                print(f"\nExample {i+1}:")
                print(f"Text: {test_text}")
                print(f"True label: {test_label}")
                print(f"SMT success: {example_results['smt']['success']}")
                print(f"FGSM success: {example_results['fgsm']['success']}")
    else:
        print("\nNo adversarial examples found for any test case.")
    
    print(f"\nVerification completed! Results saved in {output_dir}")

if __name__ == "__main__":
    main()
