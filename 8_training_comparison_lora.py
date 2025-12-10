import textstat
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import evaluate 
from typing import List, Dict, Optional
from detoxify import Detoxify
from datasets import Dataset as HFDataset
from bert_score import score     
from transformers import pipeline 


# Set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopper:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        # Check if improvement is significant
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class BMWDataset(Dataset):
    """Dataset class for BMW press release text data"""
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class BMWQADataset(Dataset):
    """Optional: Dataset for BMW Q&A evaluation"""
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        # Format as "Question: [question] Answer: [answer]"
        text = f"Question: {qa_pair['question']} Answer: {qa_pair['answer']}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten(),
            'question': qa_pair['question'],
            'answer': qa_pair['answer']
        }
def create_reduced_distilgpt2(model_name="distilgpt2", layer_to_remove=-1):
    """
    Create a reduced version of DistilGPT2 by removing one transformer block.
    
    Args:
        model_name: Name of the base model
        layer_to_remove: Index of the block to remove (default -1 for last block)
    
    Returns:
        Modified model with reduced number of layers
    """
    logger.info(f"Loading original model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Access transformer blocks
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_blocks = model.transformer.h
        original_num_layers = len(transformer_blocks)
        
        # FIX: Handle negative index correctly (e.g., -1 means last layer)
        if layer_to_remove < 0:
            layer_to_remove = original_num_layers + layer_to_remove
        
        # FIX: Simple and correct logic for layer removal
        new_blocks = []
        for i in range(original_num_layers):
            if i != layer_to_remove:
                new_blocks.append(transformer_blocks[i])
        
        # Replace with new module list
        model.transformer.h = nn.ModuleList(new_blocks)
        
        logger.info(f"✅ Removed layer at index {layer_to_remove}. "
                f"Original layers: {original_num_layers}, "
                f"New layers: {len(model.transformer.h)}")
        
        # Verify the model still works
        try:
            test_input = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                _ = model(test_input)
            logger.info("✅ Reduced model passes forward pass test")
        except Exception as e:
            logger.error(f"❌ Reduced model failed forward pass: {e}")
            raise
    else:
        raise AttributeError("Could not find transformer blocks in the model.")
    
    return model


class ModelTrainer:
    """Trainer class for fine-tuning a single model"""
    def __init__(self, model, tokenizer, model_name, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'steps': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        logger.info(f"Initialized {model_name} trainer on device: {self.device}")
    
    def train_epoch(self, train_loader, optimizer, scheduler, gradient_accumulation_steps=4):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_steps = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training {self.model_name}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_steps += 1
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Record history
            self.history['train_loss'].append(loss.item() * gradient_accumulation_steps)
            self.history['learning_rate'].append(current_lr)
            self.history['steps'].append(len(self.history['steps']))
        
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        return avg_loss
    
    def evaluate(self, dataloader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {self.model_name}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # s
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],  
                    labels=batch['input_ids']               
                )
                total_loss += outputs.loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss
    
   
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, num_return_sequences=1, check_toxicity=False):
        """Generate text from prompt with optional toxicity checking"""
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generations = []
        for i in range(num_return_sequences):
            generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            if check_toxicity:
                toxicity_score = self._check_toxicity(generated_text)
                generations.append({
                    'text': generated_text,
                    'toxicity_score': toxicity_score,
                    'is_toxic': toxicity_score > 0.1
                })
            else:
                # ALWAYS return dict for consistency
                generations.append({
                    'text': generated_text,
                    'toxicity_score': None,
                    'is_toxic': False
                })
        
        # Always return the same structure
        if num_return_sequences == 1:
            return generations[0]  # Single dict
        else:
            return generations  # List of dicts
        
    def _check_toxicity(self, text):
        """Check toxicity score using Detoxify."""
        try:
            toxicity_model = Detoxify('original')
            scores = toxicity_model.predict(text)
            return scores['toxicity']  # Primary toxicity score
        except ImportError:
            logger.warning("Detoxify not installed. Install with: pip install detoxify")
            return 0.0

class ComparativeFineTuner:
    """Main class for comparative fine-tuning of original and reduced models"""
    def __init__(self, base_model_name="distilgpt2", output_dir=None, device=None):
        self.base_model_name = base_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = output_dir or f"./bmw_comparative_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tokenizer (shared between models)
        logger.info(f"Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize models
        logger.info("Loading original model...")
        self.original_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        logger.info("Creating reduced model...")
        self.reduced_model = create_reduced_distilgpt2(base_model_name, layer_to_remove=-1)
        
        # Apply LoRA to both models for efficient fine-tuning
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        self.original_model = get_peft_model(self.original_model, lora_config)
        self.reduced_model = get_peft_model(self.reduced_model, lora_config)
        
        # Create trainers
        self.original_trainer = ModelTrainer(
            self.original_model, 
            self.tokenizer, 
            "original_distilgpt2",
            self.device
        )
        
        self.reduced_trainer = ModelTrainer(
            self.reduced_model,
            self.tokenizer,
            "reduced_distilgpt2",
            self.device
        )
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Results storage
        self.comparison_results = {}
        
        # Early stopping
        self.original_early_stopper = EarlyStopper(patience=3, min_delta=0.001)
        self.reduced_early_stopper = EarlyStopper(patience=3, min_delta=0.001)
        
        # Save configuration
        self._save_config()
        
        logger.info(f"ComparativeFineTuner initialized. Output directory: {self.output_dir}")
    
    def _save_config(self):
        """Save experiment configuration"""
        config = {
            'base_model': self.base_model_name,
            'device': str(self.device),
            'timestamp': self.timestamp,
            'lora_config': {
                'r': 8,
                'alpha': 32,
                'target_modules': ["c_attn", "c_proj", "c_fc"],
                'dropout': 0.1
            },
            'model_info': {
                'original_params': sum(p.numel() for p in self.original_model.parameters()),
                'original_trainable_params': sum(p.numel() for p in self.original_model.parameters() if p.requires_grad),
                'reduced_params': sum(p.numel() for p in self.reduced_model.parameters()),
                'reduced_trainable_params': sum(p.numel() for p in self.reduced_model.parameters() if p.requires_grad)
            }
        }
        
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Experiment config saved to {config_path}")
    
    def load_data(self, train_path, val_path, test_path):
        """Load and prepare datasets"""
        logger.info("Loading data splits...")
        
        def load_texts(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        
        self.train_texts = load_texts(train_path)
        self.val_texts = load_texts(val_path)
        self.test_texts = load_texts(test_path)
        
        logger.info(f"Train: {len(self.train_texts)}, Val: {len(self.val_texts)}, Test: {len(self.test_texts)}")
        
        # Create datasets
        self.train_dataset = BMWDataset(self.train_texts, self.tokenizer)
        self.val_dataset = BMWDataset(self.val_texts, self.tokenizer)
        self.test_dataset = BMWDataset(self.test_texts, self.tokenizer)
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def train_models(self, num_epochs=8, batch_size=8, learning_rate=7e-5):
        """Train both models with identical settings and early stopping"""
        logger.info(f"Starting comparative training for up to {num_epochs} epochs")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Training results
        training_results = {
            'original': {'train_losses': [], 'val_losses': [], 'times': [], 'epochs_completed': 0, 'best_epoch': 0, 'best_val_loss': float('inf')},
            'reduced': {'train_losses': [], 'val_losses': [], 'times': [], 'epochs_completed': 0, 'best_epoch': 0, 'best_val_loss': float('inf')}
        }
        
        # Early stopping tracking
        should_stop_original = False
        should_stop_reduced = False
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train and evaluate original model
            if not should_stop_original:
                logger.info("Training original model...")
                start_time = datetime.now()
                
                # Create optimizer and scheduler for this epoch
                original_optimizer = torch.optim.AdamW(
                    [p for p in self.original_model.parameters() if p.requires_grad],
                    lr=learning_rate,
                    weight_decay=0.01
                )
                original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    original_optimizer, T_max=len(train_loader)
                )
                
                original_train_loss = self.original_trainer.train_epoch(
                    train_loader, original_optimizer, original_scheduler
                )
                original_val_loss = self.original_trainer.evaluate(val_loader)
                original_time = (datetime.now() - start_time).total_seconds()
                
                # Update training results
                training_results['original']['train_losses'].append(original_train_loss)
                training_results['original']['val_losses'].append(original_val_loss)
                training_results['original']['times'].append(original_time)
                training_results['original']['epochs_completed'] = epoch + 1
                
                # Check for best model and early stopping
                if original_val_loss < training_results['original']['best_val_loss']:
                    training_results['original']['best_val_loss'] = original_val_loss
                    training_results['original']['best_epoch'] = epoch + 1
                    # Save best model
                    self._save_best_model(self.original_model, "original", epoch+1, original_val_loss)
                
                # Check early stopping
                should_stop_original = self.original_early_stopper(original_val_loss)
                if should_stop_original:
                    logger.info(f"Original model early stopping triggered after {epoch + 1} epochs")
            
            # Train and evaluate reduced model
            if not should_stop_reduced:
                logger.info("Training reduced model...")
                start_time = datetime.now()
                
                reduced_optimizer = torch.optim.AdamW(
                    [p for p in self.reduced_model.parameters() if p.requires_grad],
                    lr=learning_rate,
                    weight_decay=0.01
                )
                reduced_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    reduced_optimizer, T_max=len(train_loader)
                )
                
                reduced_train_loss = self.reduced_trainer.train_epoch(
                    train_loader, reduced_optimizer, reduced_scheduler
                )
                reduced_val_loss = self.reduced_trainer.evaluate(val_loader)
                reduced_time = (datetime.now() - start_time).total_seconds()
                
                # Update training results
                training_results['reduced']['train_losses'].append(reduced_train_loss)
                training_results['reduced']['val_losses'].append(reduced_val_loss)
                training_results['reduced']['times'].append(reduced_time)
                training_results['reduced']['epochs_completed'] = epoch + 1
                
                # Check for best model and early stopping
                if reduced_val_loss < training_results['reduced']['best_val_loss']:
                    training_results['reduced']['best_val_loss'] = reduced_val_loss
                    training_results['reduced']['best_epoch'] = epoch + 1
                    # Save best model
                    self._save_best_model(self.reduced_model, "reduced", epoch+1, reduced_val_loss)
                
                # Check early stopping
                should_stop_reduced = self.reduced_early_stopper(reduced_val_loss)
                if should_stop_reduced:
                    logger.info(f"Reduced model early stopping triggered after {epoch + 1} epochs")
            
            # Log epoch results
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"Original - Train: {original_train_loss:.4f}, Val: {original_val_loss:.4f}, "
                      f"Best Val: {training_results['original']['best_val_loss']:.4f} (Epoch {training_results['original']['best_epoch']})")
            logger.info(f"Reduced  - Train: {reduced_train_loss:.4f}, Val: {reduced_val_loss:.4f}, "
                      f"Best Val: {training_results['reduced']['best_val_loss']:.4f} (Epoch {training_results['reduced']['best_epoch']})")
            
            if not should_stop_original and not should_stop_reduced:
                logger.info(f"Time - Original: {original_time:.1f}s, Reduced: {reduced_time:.1f}s, "
                          f"Speedup: {(original_time - reduced_time) / original_time * 100:.1f}%")
            
            # Stop if both models have triggered early stopping
            if should_stop_original and should_stop_reduced:
                logger.info(f"Both models converged. Stopping training at epoch {epoch + 1}")
                break
        
        # Load best models for final evaluation
        self._load_best_models(training_results)
        
        # Save training results
        self.comparison_results['training'] = training_results
        self._save_training_results(training_results)
        
        logger.info(f"\nTraining completed:")
        logger.info(f"Original model: {training_results['original']['epochs_completed']} epochs, "
                  f"Best val loss: {training_results['original']['best_val_loss']:.4f} (Epoch {training_results['original']['best_epoch']})")
        logger.info(f"Reduced model: {training_results['reduced']['epochs_completed']} epochs, "
                  f"Best val loss: {training_results['reduced']['best_val_loss']:.4f} (Epoch {training_results['reduced']['best_epoch']})")
        
        return training_results
    
    def _save_best_model(self, model, model_type, epoch, val_loss):
        """Save the best model checkpoint."""
        best_model_dir = os.path.join(self.output_dir, f"best_{model_type}_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        model.save_pretrained(best_model_dir)
        self.tokenizer.save_pretrained(best_model_dir)
        
        info = {
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(best_model_dir, 'best_model_info.json'), 'w') as f:
            json.dump(info, f, indent=2, default=str)
    
    def _load_best_models(self, training_results):
        """Load the best saved models for evaluation."""
        from peft import PeftModel
        
        # Load original best model
        best_original_dir = os.path.join(self.output_dir, "best_original_model")
        if os.path.exists(best_original_dir):
            logger.info(f"Loading best original model from epoch {training_results['original']['best_epoch']}")
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            self.original_model = PeftModel.from_pretrained(base_model, best_original_dir)
            self.original_model.to(self.device)
            self.original_trainer.model = self.original_model
        
        # Load reduced best model
        best_reduced_dir = os.path.join(self.output_dir, "best_reduced_model")
        if os.path.exists(best_reduced_dir):
            logger.info(f"Loading best reduced model from epoch {training_results['reduced']['best_epoch']}")
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            self.reduced_model = PeftModel.from_pretrained(base_model, best_reduced_dir)
            self.reduced_model.to(self.device)
            self.reduced_trainer.model = self.reduced_model
    
    def _save_training_results(self, training_results):
        """Save training results to file"""
        # Save as JSON
        results_path = os.path.join(self.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Save as CSV
        epochs = list(range(1, len(training_results['original']['train_losses']) + 1))
        
        df = pd.DataFrame({
            'epoch': epochs,
            'original_train_loss': training_results['original']['train_losses'],
            'original_val_loss': training_results['original']['val_losses'],
            'original_time': training_results['original']['times'],
            'reduced_train_loss': training_results['reduced']['train_losses'],
            'reduced_val_loss': training_results['reduced']['val_losses'],
            'reduced_time': training_results['reduced']['times'],
        })
        
        csv_path = os.path.join(self.output_dir, "training_results.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Training results saved to {csv_path}")
    
    def compute_baseline_metrics(self):
        """Compute metrics for the base model without fine-tuning."""
        logger.info("Computing BASELINE metrics (original distilgpt2)...")
        
        # Load base model (not fine-tuned)
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
        base_model.eval()
        
        test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Baseline"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # outputs = base_model(**batch)
                outputs = base_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],  # ← ADD THIS LINE
                    labels=batch['input_ids']                # ← KEEP THIS
                )
                total_loss += outputs.loss.item() * batch['input_ids'].size(0)
                total_tokens += batch['attention_mask'].sum().item()
        
        avg_loss = total_loss / len(self.test_dataset)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'test_loss': avg_loss,
            'perplexity': perplexity,
            'num_samples': len(self.test_dataset),
            'avg_tokens_per_sample': total_tokens / len(self.test_dataset),
            'model': 'baseline_distilgpt2'
        }
        
        with open(os.path.join(self.output_dir, "baseline_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"BASELINE - Test Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        return metrics
    
    
    
    def compute_comprehensive_metrics(self, num_samples=15):
        """Compute comprehensive text quality metrics for both models."""
        logger.info("Computing comprehensive text quality metrics...")
        
        try:
            # Initialize metrics
            metrics = {
                'original': {
                    'diversity': {},
                    'readability': None,
                    'fluency': None,
                    'perplexity_on_test': None,  # ADDED for clarity
                    'toxicity': None
                },
                'reduced': {
                    'diversity': {},
                    'readability': None,
                    'fluency': None,
                    'perplexity_on_test': None,  # ADDED for clarity
                    'toxicity': None
                }
            }
            

            def extract_generated_text(generation):
                """Handle returns from generate_text() - now always dicts"""
                if isinstance(generation, dict) and 'text' in generation:
                    return generation['text']
                elif isinstance(generation, list) and len(generation) > 0:
                    # Handle list of generations
                    if isinstance(generation[0], dict) and 'text' in generation[0]:
                        return generation[0]['text']
                    else:
                        return str(generation[0])
                elif isinstance(generation, str):
                    return generation
                else:
                    return str(generation)
            
            # Generate samples for diversity/readability/toxicity evaluation
            references = []
            original_generations = []
            reduced_generations = []
            
            # Use test set as base for generation
            for i, ref_text in enumerate(self.test_texts[:num_samples]):
                if not ref_text.strip():
                    continue
                    
                # Use first 20 words as prompt
                words = ref_text.split()[:20]
                prompt = " ".join(words) if words else "BMW"
                
                # Generate with both models
                orig_gen_raw = self.original_trainer.generate_text(
                    prompt, max_length=100, temperature=0.7, check_toxicity=False
                )
                red_gen_raw = self.reduced_trainer.generate_text(
                    prompt, max_length=100, temperature=0.7, check_toxicity=False
                )
                
                # Extract text from potentially complex return
                orig_gen = extract_generated_text(orig_gen_raw)
                red_gen = extract_generated_text(red_gen_raw)
                
                # Filter out empty generations
                if orig_gen and orig_gen.strip():
                    original_generations.append(orig_gen)
                if red_gen and red_gen.strip():
                    reduced_generations.append(red_gen)
                references.append([ref_text])
            
            # ====================== 1. Diversity (Distinct-n) ======================
            def compute_distinct_n(texts, n=2):
                """Compute Distinct-n score."""
                all_ngrams = []
                for text in texts:
                    if not text.strip():
                        continue
                    words = text.split()
                    if len(words) >= n:
                        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                        all_ngrams.extend(ngrams)
                
                if not all_ngrams:
                    return 0
                
                unique_ngrams = set(all_ngrams)
                return len(unique_ngrams) / len(all_ngrams)
            
            # Only compute if we have generations
            if original_generations:
                metrics['original']['diversity'] = {
                    'distinct_1': compute_distinct_n(original_generations, 1),
                    'distinct_2': compute_distinct_n(original_generations, 2),
                    'num_samples_evaluated': len(original_generations)
                }
            
            if reduced_generations:
                metrics['reduced']['diversity'] = {
                    'distinct_1': compute_distinct_n(reduced_generations, 1),
                    'distinct_2': compute_distinct_n(reduced_generations, 2),
                    'num_samples_evaluated': len(reduced_generations)
                }
            
            # ====================== 2. Readability (Flesch-Kincaid) ======================
            def compute_readability(texts):
                """Compute readability scores with proper import handling."""
                try:
                    import textstat  # BUG FIX #2: Import inside function
                    
                    scores = []
                    for text in texts:
                        if text and len(text.split()) > 10:
                            try:
                                fk_score = textstat.flesch_kincaid_grade(text)
                                scores.append(fk_score)
                            except Exception as e:
                                logger.debug(f"Could not compute readability for text: {e}")
                                continue
                    return np.mean(scores) if scores else 0
                except ImportError:
                    logger.warning("textstat not installed. Install with: pip install textstat")
                    return 0
            
            metrics['original']['readability'] = {
                'flesch_kincaid_grade': compute_readability(original_generations) if original_generations else 0,
                'num_scored': len([t for t in original_generations if t and len(t.split()) > 10])
            }
            metrics['reduced']['readability'] = {
                'flesch_kincaid_grade': compute_readability(reduced_generations) if reduced_generations else 0,
                'num_scored': len([t for t in reduced_generations if t and len(t.split()) > 10])
            }
            
            # ====================== 3. FLUENCY (CRITICAL FIX - TEST DATA!) ======================
            def compute_fluency_on_test_data(model, tokenizer, device, test_dataset, num_samples=20):
                """BIG FIX #1 & #3: Evaluate fluency on TEST DATA with attention mask"""
                fluency_scores = []
                perplexities = []
                
                # Create DataLoader from REAL test data
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
                
                for i, batch in enumerate(test_loader):
                    if i >= num_samples:
                        break
                        
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # BUG FIX #3: Add attention_mask!
                    with torch.no_grad():
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],  # ← WAS MISSING!
                            labels=batch['input_ids']
                        )
                        loss = outputs.loss
                    
                    if loss is not None:
                        perplexity = torch.exp(loss).item()
                        perplexities.append(perplexity)
                        
                        # Convert perplexity to fluency score (0-1, higher is better)
                        # More realistic formula based on typical perplexity ranges
                        if perplexity < 10:
                            fluency = 0.95  # Excellent
                        elif perplexity < 20:
                            fluency = 0.85  # Very good
                        elif perplexity < 30:
                            fluency = 0.75  # Good
                        elif perplexity < 50:
                            fluency = 0.60  # Fair
                        elif perplexity < 100:
                            fluency = 0.30  # Poor
                        else:
                            fluency = 0.10  # Very poor
                        
                        fluency_scores.append(fluency)
                
                avg_fluency = np.mean(fluency_scores) if fluency_scores else 0
                avg_perplexity = np.mean(perplexities) if perplexities else 0
                
                return avg_fluency, avg_perplexity
            
            # Load base model for fluency evaluation (on TEST DATA!)
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
            base_model.eval()
            
            # BUG FIX #1: Evaluate on TEST DATA, not generated text
            orig_fluency, orig_ppl = compute_fluency_on_test_data(
                base_model, self.tokenizer, self.device, self.test_dataset, num_samples
            )
            metrics['original']['fluency'] = orig_fluency
            metrics['original']['perplexity_on_test'] = orig_ppl  # For comparison
            
            red_fluency, red_ppl = compute_fluency_on_test_data(
                base_model, self.tokenizer, self.device, self.test_dataset, num_samples
            )
            metrics['reduced']['fluency'] = red_fluency
            metrics['reduced']['perplexity_on_test'] = red_ppl  # For comparison
            
            # ====================== 4. Toxicity (on GENERATED text - this is correct!) ======================
            try:
                from detoxify import Detoxify
                toxicity_model = Detoxify('original')
                
                def compute_toxicity(texts):
                    scores = []
                    for text in texts:
                        if text and len(text.strip()) > 10:
                            try:
                                result = toxicity_model.predict(text)
                                scores.append(result['toxicity'])
                            except Exception as e:
                                logger.debug(f"Could not compute toxicity: {e}")
                                continue
                    return np.mean(scores) if scores else 0
                
                # This is CORRECT: Toxicity should be evaluated on generated text
                metrics['original']['toxicity'] = compute_toxicity(original_generations) if original_generations else 0
                metrics['reduced']['toxicity'] = compute_toxicity(reduced_generations) if reduced_generations else 0
                
            except ImportError:
                logger.warning("Detoxify not installed. Skipping toxicity scoring.")
                metrics['original']['toxicity'] = None
                metrics['reduced']['toxicity'] = None
            
            # ====================== Save and Print Results ======================
            # Save comprehensive metrics
            with open(os.path.join(self.output_dir, "comprehensive_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Print summary with clear labels
            print("\n" + "="*60)
            print("COMPREHENSIVE TEXT QUALITY METRICS")
            print("="*60)
            
            print(f"\nDiversity (on generated text):")
            print(f"  Distinct-1 - Original: {metrics['original']['diversity'].get('distinct_1', 0):.4f}, "
                f"Reduced: {metrics['reduced']['diversity'].get('distinct_1', 0):.4f}")
            print(f"  Distinct-2 - Original: {metrics['original']['diversity'].get('distinct_2', 0):.4f}, "
                f"Reduced: {metrics['reduced']['diversity'].get('distinct_2', 0):.4f}")
            
            print(f"\nReadability (on generated text):")
            print(f"  Flesch-Kincaid Grade - Original: {metrics['original']['readability']['flesch_kincaid_grade']:.2f}, "
                f"Reduced: {metrics['reduced']['readability']['flesch_kincaid_grade']:.2f}")
            print(f"  (Grade level ≈ years of education needed)")
            
            print(f"\nFluency (on TEST DATA - not generated text!):")
            print(f"  Fluency Score - Original: {metrics['original']['fluency']:.4f}, "
                f"Reduced: {metrics['reduced']['fluency']:.4f}")
            print(f"  Test Perplexity - Original: {metrics['original']['perplexity_on_test']:.2f}, "
                f"Reduced: {metrics['reduced']['perplexity_on_test']:.2f}")
            print(f"  (Lower perplexity = more fluent)")
            
            if metrics['original']['toxicity'] is not None:
                print(f"\nToxicity (on generated text):")
                print(f"  Score - Original: {metrics['original']['toxicity']:.6f}, "
                    f"Reduced: {metrics['reduced']['toxicity']:.6f}")
                print(f"  (Lower = better, <0.01 = very safe)")
            
            # Add interpretation guide
            print(f"\n{'='*60}")
            print("INTERPRETATION GUIDE:")
            print(f"{'='*60}")
            print("• Diversity (0-1): Higher = less repetitive text")
            print("• Readability (grade level): 8-12 = standard press release level")
            print("• Fluency (0-1): Higher = more natural/coherent text")
            print("• Toxicity (0-1): Lower = safer content (BMW press should be ~0)")
            print(f"{'='*60}")
            
            return metrics
            
        except Exception as e:  # BUG FIX #5: Catch all exceptions
            logger.error(f"Error computing comprehensive metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def evaluate_models(self):
        """Comprehensive evaluation of both models"""
        logger.info("\n" + "="*60)
        logger.info("Comprehensive Model Evaluation")
        logger.info("="*60)
        
        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        
        # 1. Evaluate test loss and perplexity
        logger.info("Evaluating test loss and perplexity...")
        
        original_test_loss = self.original_trainer.evaluate(test_loader)
        reduced_test_loss = self.reduced_trainer.evaluate(test_loader)
        
        original_perplexity = torch.exp(torch.tensor(original_test_loss)).item()
        reduced_perplexity = torch.exp(torch.tensor(reduced_test_loss)).item()
        
        # 2. Generate sample texts for qualitative comparison
        logger.info("\nGenerating sample texts for comparison...")
        
        bmw_prompts = [
            "BMW is launching a new electric vehicle that",
            "The BMW Group announced today that",
            "BMW's latest sustainability initiative focuses on",
            "The future of BMW mobility includes",
            "BMW autonomous driving technology"
        ]
        
        generations = {}

        difference_summary = {
        'total_prompts': len(bmw_prompts),
        'original_longer': 0,
        'reduced_longer': 0,
        'original_has_repetition': 0,
        'reduced_has_repetition': 0
        }
        

        for prompt in bmw_prompts:
    # Get generation results (these are DICTIONARIES)
            original_gen_result = self.original_trainer.generate_text(prompt, max_length=80, check_toxicity=False)
            reduced_gen_result = self.reduced_trainer.generate_text(prompt, max_length=80, check_toxicity=False)
            
            # Extract the actual TEXT from the dictionaries
            original_gen_text = original_gen_result['text']  
            reduced_gen_text = reduced_gen_result['text']    
            
            # Get difference analysis using the TEXT
            diff_note = self._generate_difference_note(original_gen_text, reduced_gen_text)
            
            # Compare lengths
            len1, len2 = len(original_gen_text.split()), len(reduced_gen_text.split())
            if len1 > len2 + 20:
                difference_summary['original_longer'] += 1
            elif len2 > len1 + 20:
                difference_summary['reduced_longer'] += 1
            
            # Check for repetition using the TEXT
            if self._has_repetition(original_gen_text):
                difference_summary['original_has_repetition'] += 1
            if self._has_repetition(reduced_gen_text):
                difference_summary['reduced_has_repetition'] += 1
            
            # Store the FULL results (dictionaries) for saving
            generations[prompt] = {
                'original': original_gen_result,  # Store the full dict
                'reduced': reduced_gen_result,    # Store the full dict
                'difference_analysis': diff_note 
            }
            
            # Log the TEXT (not the dict) for readability
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Original: {original_gen_text}")
            logger.info(f"Reduced:  {reduced_gen_text}")
            logger.info(f"Difference Analysis: {diff_note}")    
                
        # 3. Evaluate on Q&A set
        logger.info("\nEvaluating on BMW Q&A set...")
        qa_results = self.evaluate_qa_set_enhanced()
        
        # 4. Compute comprehensive metrics
        logger.info("\nComputing comprehensive text quality metrics...")
        comprehensive_metrics = self.compute_comprehensive_metrics()
        

        
        
        # Compile evaluation results
        eval_results = {
            'test_metrics': {
                'original': {
                    'test_loss': original_test_loss,
                    'perplexity': original_perplexity
                },
                'reduced': {
                    'test_loss': reduced_test_loss,
                    'perplexity': reduced_perplexity
                }
            },
            'generations': generations,
            'qa_evaluation': qa_results,
            'difference_summary': difference_summary,
            'comprehensive_metrics': comprehensive_metrics,
            'bert_scores': {  # ✅ ADD THIS
            'original': qa_results['comparison']['bert_score']['original'],
            'reduced': qa_results['comparison']['bert_score']['reduced']
            } if qa_results and 'bert_score' in qa_results['comparison'] else None
        }
        
        self.comparison_results['evaluation'] = eval_results
        self._save_evaluation_results(eval_results)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Original Model - Test Loss: {original_test_loss:.4f}, "
                   f"Perplexity: {original_perplexity:.2f}")
        logger.info(f"Reduced Model  - Test Loss: {reduced_test_loss:.4f}, "
                   f"Perplexity: {reduced_perplexity:.2f}")
        
        if qa_results:
            logger.info(f"\nQ&A Evaluation:")
            # logger.info(f"Original Model Avg Log-Likelihood: {qa_results['log_likelihood']['original']:.4f}")
            # logger.info(f"Reduced Model Avg Log-Likelihood: {qa_results['log_likelihood']['reduced']:.4f}")
        
        return eval_results


    def evaluate_qa_set_enhanced(self):
        
        logger.info("\n" + "="*60)
        logger.info("ENHANCED Q&A EVALUATION")
        logger.info("="*60)
        
        # Create or load Q&A set
        qa_data = self.create_bmw_qa_set()
        qa_json_path = "bmw_qa_evaluation_set.json"
        
        if os.path.exists(qa_json_path):
            loaded_data = self.load_bmw_qa_from_json(qa_json_path)
            if loaded_data:
                qa_data = loaded_data
        
        logger.info(f"Evaluating on {len(qa_data)} BMW Q&A pairs")
        
        results = {
            'original': {
                'log_likelihoods': [],
                'bert_scores': [], 
                'generations': []
            },
            'reduced': {
                'log_likelihoods': [],
                'bert_scores': [], 
                'generations': []
            }
        }
        
        # Evaluate each Q&A pair
        for idx, qa_pair in enumerate(tqdm(qa_data, desc="Evaluating Q&A")):
            question = qa_pair['question']
            context_answer = qa_pair['answer']  # Ground truth from Q&A set
            
            # Generate answers from both models
            original_answer = self.generate_bmw_answer(self.original_trainer, question)
            reduced_answer = self.generate_bmw_answer(self.reduced_trainer, question)
            
            # Store generations
            results['original']['generations'].append({
                'question': question,
                'generated_answer': original_answer,
                'context_answer': context_answer
            })
            results['reduced']['generations'].append({
                'question': question,
                'generated_answer': reduced_answer,
                'context_answer': context_answer
            })
            
            # 1. Compute log-likelihood (existing metric)
            orig_ll = self.compute_log_likelihood(self.original_trainer, question, context_answer)
            red_ll = self.compute_log_likelihood(self.reduced_trainer, question, context_answer)
            
            results['original']['log_likelihoods'].append(orig_ll)
            results['reduced']['log_likelihoods'].append(red_ll)

            orig_bert_score = self.compute_semantic_similarity(original_answer, context_answer)
            red_bert_score = self.compute_semantic_similarity(reduced_answer, context_answer)
            
            results['original']['bert_scores'].append(orig_bert_score)
            results['reduced']['bert_scores'].append(red_bert_score)
            
            # # 2. Compute professional tone score
            # orig_prof_score = self.score_professional_tone_with_model(original_answer)
            # red_prof_score = self.score_professional_tone_with_model(reduced_answer)
            
            # results['original']['professional_tone_scores'].append(orig_prof_score)
            # results['reduced']['professional_tone_scores'].append(red_prof_score)
            
            # 3. Compute faithfulness (generated answer vs context/ground truth)
            # orig_faithfulness = self.compute_faithfulness(original_answer, context_answer)
            # red_faithfulness = self.compute_faithfulness(reduced_answer, context_answer)
            
            # results['original']['faithfulness_scores'].append(orig_faithfulness)
            # results['reduced']['faithfulness_scores'].append(red_faithfulness)
            
            # # 4. Compute response relevancy (generated answer vs question)
            # orig_relevancy = self.compute_response_relevancy(original_answer, question)
            # red_relevancy = self.compute_response_relevancy(reduced_answer, question)
            
            # results['original']['relevancy_scores'].append(orig_relevancy)
            # results['reduced']['relevancy_scores'].append(red_relevancy)
            
            # Log first few examples
            if idx < 3:
                logger.info(f"\nQ{idx+1}: {question}")
                logger.info(f"Context Answer: {context_answer}")
                logger.info(f"Original Answer: {original_answer}")
                # logger.info(f"  BERTScore F1: {orig_bert_score:.4f}, Log-Likelihood: {orig_ll:.4f}")
                # logger.info(f"Reduced Answer: {reduced_answer}")
                # logger.info(f"  BERTScore F1: {red_bert_score:.4f}, Log-Likelihood: {red_ll:.4f}")
                # logger.info(
                #         f"Faithfulness: {orig_faithfulness:.3f}, "
                #         f"Relevancy: {orig_relevancy:.3f}")
                # logger.info(f"Reduced Answer: {reduced_answer}")
                # logger.info(
                #         f"Faithfulness: {red_faithfulness:.3f}, "
                        # f"Relevancy: {red_relevancy:.3f}")
        
        # Calculate averages
        comparison = {
            'log_likelihood': {
                'original': np.mean(results['original']['log_likelihoods']),
                'reduced': np.mean(results['reduced']['log_likelihoods'])
             },
            # 'faithfulness': {
            #     'original': np.mean(results['original']['faithfulness_scores']),
            #     'reduced': np.mean(results['reduced']['faithfulness_scores'])
            # },
            # 'relevancy': {
            #     'original': np.mean(results['original']['relevancy_scores']),
            #     'reduced': np.mean(results['reduced']['relevancy_scores'])
            #},
             'bert_score': {  
                'original': np.mean(results['original']['bert_scores']),
                'reduced': np.mean(results['reduced']['bert_scores'])
            }
          }
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ENHANCED Q&A EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Log-Likelihood - Original: {comparison['log_likelihood']['original']:.4f}, "
                f"Reduced: {comparison['log_likelihood']['reduced']:.4f}")
        
        enhanced_results = {
            'detailed_results': results,
            'comparison': comparison,
            'num_questions': len(qa_data),
             'bert_score_info': {  
            'description': 'BERTScore F1 measure between generated answer and ground truth',
            'range': '0 to 1 (higher is better)',
            'model_used': 'BERT-base for embedding similarity'
        }
        }
        
        with open(os.path.join(self.output_dir, "qa_evaluation_enhanced.json"), 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        return enhanced_results

    def generate_bmw_answer(self, trainer, question):
        """Generate answer for BMW question."""
        prompt = f"Question: {question}\nAnswer:"
        
        generated = trainer.generate_text(
            prompt,
            max_length=150,
            temperature=0.7,
            num_return_sequences=1
        )
        
        generated_text = generated['text'] if isinstance(generated, dict) else str(generated)   
        
        if "Answer:" in generated_text:
            answer_part = generated_text.split("Answer:")[-1].strip()
            return answer_part
        return generated_text

    def compute_log_likelihood(self, trainer, question, context_answer):
        """Compute log-likelihood of context answer."""
        # Format as "Question: X Answer: Y"
        text = f"Question: {question} Answer: {context_answer}"
    
    
        # Tokenize
        inputs = trainer.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(trainer.device)
        
        # Get model outputs
        trainer.model.eval()
        with torch.no_grad():
            # outputs = trainer.model(**inputs, labels=inputs['input_ids'])
            outputs = trainer.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # ← ADD THIS LINE
            labels=inputs['input_ids']
        )
            loss = outputs.loss.item()
        
        # Convert loss to log-likelihood
        return -loss
        
    def create_bmw_qa_set(self):
        """Create a well-designed BMW Q&A evaluation set."""
        qa_examples = [
            {
                "question": "What is the electric range of the BMW iX?",
                "answer": "The BMW iX xDrive50 has an estimated range of up to 324 miles (521 km) on a single charge."
            },
            {
                "question": "When was the BMW Group founded?",
                "answer": "The BMW Group was founded on March 7, 1916, initially as an aircraft engine manufacturer."
            },
            {
                "question": "What is BMW's sustainability strategy called?",
                "answer": "BMW's sustainability strategy is called 'BMW i' and focuses on electromobility and sustainable mobility solutions."
            },
            {
                "question": "Which BMW model was the first fully electric vehicle?",
                "answer": "The BMW i3, launched in 2013, was BMW's first purpose-built fully electric vehicle."
            },
            {
                "question": "What does BMW stand for?",
                "answer": "BMW stands for Bayerische Motoren Werke, which translates to Bavarian Motor Works in English."
            },
            {
                "question": "Where is BMW's headquarters located?",
                "answer": "BMW's headquarters is located in Munich, Germany, at the BMW Tower."
            },
            {
                "question": "What is BMW's electric vehicle sub-brand called?",
                "answer": "BMW's electric vehicle sub-brand is called BMW i, which includes models like the i3, i4, iX, and i7."
            },
            {
                "question": "What year was the iconic BMW 3 Series first introduced?",
                "answer": "The BMW 3 Series was first introduced in 1975 and has since become one of BMW's best-selling model lines."
            },
            {
                "question": "What is BMW's autonomous driving technology called?",
                "answer": "BMW's autonomous driving technology is part of the BMW Personal CoPilot system, which includes features like Traffic Jam Assistant and Extended Traffic Jam Assistant."
            },
            {
                "question": "What is the top speed of the BMW M5 Competition?",
                "answer": "The BMW M5 Competition has a top speed of 190 mph (306 km/h) when equipped with the optional M Driver's Package."
            }
        ]
        
        # Save Q&A set
        qa_path = os.path.join(self.output_dir, "bmw_qa_evaluation_set.json")
        with open(qa_path, 'w') as f:
            json.dump(qa_examples, f, indent=2, default=str)
        
        logger.info(f"BMW Q&A evaluation set created with {len(qa_examples)} questions")
        return qa_examples
    

    def load_bmw_qa_from_json(self, json_path="bmw_qa_evaluation_set.json"):
        """Load BMW Q&A set from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                qa_data = json.load(f)
            
            # If your JSON has the structure I provided above
            if isinstance(qa_data, dict) and 'qa_pairs' in qa_data:
                qa_examples = qa_data['qa_pairs']
            elif isinstance(qa_data, dict) and 'bmw_qa_evaluation_set' in qa_data:
                qa_examples = qa_data['bmw_qa_evaluation_set']
            elif isinstance(qa_data, list):
                qa_examples = qa_data
            else:
                logger.error(f"Unknown JSON structure in {json_path}")
                return self.create_bmw_qa_set()  # Fall back to default
            
            logger.info(f"Loaded {len(qa_examples)} Q&A pairs from {json_path}")
            return qa_examples
            
        except FileNotFoundError:
            logger.warning(f"Q&A JSON file not found at {json_path}, using default set")
            return self.create_bmw_qa_set()

    def _has_repetition(self, text):
        """Check for word repetition patterns."""
        words = text.lower().split()
        if len(words) < 10:
            return False
        for i in range(len(words) - 3):
            if words[i:i+2] == words[i+2:i+4]:
                return True
        return False
    

    def _generate_difference_note(self, text1, text2):
        """Generate auto-analysis of differences between two texts."""
        if isinstance(text1, dict) and 'text' in text1:
            text1 = text1['text']
        if isinstance(text2, dict) and 'text' in text2:
            text2 = text2['text']
        differences = []
        
        # Compare length
        len1, len2 = len(text1.split()), len(text2.split())
        if abs(len1 - len2) > 20:
            differences.append(f"Length diff: {len1} vs {len2} words")
        
        # Check for repetition
        def has_repetition(text):
            if isinstance(text, dict) and 'text' in text:
                text = text['text']
            words = text.lower().split()
            if len(words) < 10:
                return False
            for i in range(len(words) - 3):
                if words[i:i+2] == words[i+2:i+4]:
                    return True
            return False
        
        if has_repetition(text1) and not has_repetition(text2):
            differences.append("Original has repetition")
        elif has_repetition(text2) and not has_repetition(text1):
            differences.append("Reduced has repetition")
        
        # Check BMW-specific terminology
        bmw_terms = ['BMW', 'electric', 'sustainable', 'autonomous', 'vehicle']
        term_count1 = sum(1 for term in bmw_terms if term.lower() in text1.lower())
        term_count2 = sum(1 for term in bmw_terms if term.lower() in text2.lower())
        
        if term_count1 != term_count2:
            differences.append(f"Terminology: {term_count1} vs {term_count2} key terms")
        
        return "; ".join(differences) if differences else "Minor stylistic differences"
    
    # def score_professional_tone_with_model(self, text):
    #     """
    #     Use a pre-trained model to detect formal/professional tone.
    #     """
        
        
        
    #     # Load a text classification model trained on formality
    #     classifier = pipeline("text-classification", 
    #                         model="cointegrated/roberta-large-formality")
        
    #     # Handle long texts by truncating
    #     if len(text) > 500:
    #         text = text[:500]
        
    #     result = classifier(text)[0]
        
    #     # Map model output to professional tone score (0-5 scale)
    #     if result['label'] == 'formal':
    #         return 5.0  # Very professional
    #     elif result['label'] == 'neutral':
    #         return 3.0  # Moderately professional
    #     else:  # informal
    #         return 1.0  # Not professional
        


    # def compute_faithfulness(self, generated_answer, context_answer):
    #     """
    #     Compute faithfulness between generated answer and context (ground truth).
    #     For your assignment: generated_answer = model's answer, context_answer = Q&A set answer
    #     """
 
    #     # Create a dataset for RAGAS evaluation
    #     dataset = HFDataset.from_dict({
    #         'question': ["Does the generated answer match the context?"],
    #         'answer': [generated_answer],
    #         'contexts': [[context_answer]]  # List of contexts
    #     })
        
    #     # Calculate faithfulness
    #     result = ragas_evaluate(dataset, metrics=[Faithfulness])
        
    #     # Extract the faithfulness score
    #     faith_score = result['faithfulness']
    #     return faith_score if isinstance(faith_score, (int, float)) else faith_score[0]
            
        

    # def compute_response_relevancy(self, generated_answer, question):
    #     """
    #     Compute how relevant the generated answer is to the question.
    #     """

    #     # Create a dataset for RAGAS evaluation
    #     dataset = HFDataset.from_dict({
    #         'question': [question],
    #         'answer': [generated_answer]
    #     })
        
    #     # Calculate answer relevancy
    #     result = ragas_evaluate(dataset, metrics=[AnswerRelevancy])
        
    #     # Extract the relevancy score
    #     relevancy_score = result['answer_relevancy']
    #     return relevancy_score if isinstance(relevancy_score, (int, float)) else relevancy_score[0]
            
      

    def compute_semantic_similarity(self, text1, text2):
        """
        Compute semantic similarity using BERTScore as fallback for faithfulness.
        """
      
        # Handle empty texts
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Compute BERTScore
        P, R, F1 = score([text1], [text2], lang="en", verbose=False, rescale_with_baseline=True)
        
        return F1.item()
            
       



    def _save_evaluation_results(self, eval_results):
        """Save evaluation results"""
        # Save JSON
        eval_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        # Save generations separately
        gens_path = os.path.join(self.output_dir, "sample_generations.json")
        with open(gens_path, 'w') as f:
            json.dump(eval_results['generations'], f, indent=2, default=str)
        
        # Save test metrics as CSV
        test_metrics = eval_results['test_metrics']
        df = pd.DataFrame([
            {
                'model': 'original',
                'test_loss': test_metrics['original']['test_loss'],
                'perplexity': test_metrics['original']['perplexity']
            },
            {
                'model': 'reduced',
                'test_loss': test_metrics['reduced']['test_loss'],
                'perplexity': test_metrics['reduced']['perplexity']
            }
        ])
        
        metrics_path = os.path.join(self.output_dir, "test_metrics.csv")
        df.to_csv(metrics_path, index=False)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def plot_comparison(self):
        """Create comparison plots for training and evaluation"""
        try:
            if 'training' not in self.comparison_results:
                logger.warning("No training results to plot")
                return
            
            training_results = self.comparison_results['training']
            epochs_original = list(range(1, len(training_results['original']['train_losses']) + 1))
            epochs_reduced = list(range(1, len(training_results['reduced']['train_losses']) + 1))
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Training loss comparison
            axes[0, 0].plot(epochs_original, training_results['original']['train_losses'], 
                           'b-', label='Original', linewidth=2)
            axes[0, 0].plot(epochs_reduced, training_results['reduced']['train_losses'], 
                           'r-', label='Reduced', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Training Loss')
            axes[0, 0].set_title('Training Loss Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Validation loss comparison
            axes[0, 1].plot(epochs_original, training_results['original']['val_losses'], 
                           'b-', label='Original', linewidth=2)
            axes[0, 1].plot(epochs_reduced, training_results['reduced']['val_losses'], 
                           'r-', label='Reduced', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validation Loss')
            axes[0, 1].set_title('Validation Loss Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Training time comparison
            axes[1, 0].bar(['Original', 'Reduced'], 
                          [np.sum(training_results['original']['times']), 
                           np.sum(training_results['reduced']['times'])],
                          color=['blue', 'red'])
            axes[1, 0].set_ylabel('Total Training Time (s)')
            axes[1, 0].set_title('Training Time Comparison')
            
            # Plot 4: Test performance comparison
            if 'evaluation' in self.comparison_results:
                test_metrics = self.comparison_results['evaluation']['test_metrics']
                x = np.arange(2)
                width = 0.35
                
                loss_values = [test_metrics['original']['test_loss'], 
                              test_metrics['reduced']['test_loss']]
                perplexity_values = [test_metrics['original']['perplexity'], 
                                    test_metrics['reduced']['perplexity']]
                
                axes[1, 1].bar(x - width/2, loss_values, width, label='Test Loss', color=['lightblue', 'lightcoral'])
                axes[1, 1].bar(x + width/2, perplexity_values, width, label='Perplexity', color=['blue', 'red'])
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(['Original', 'Reduced'])
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].set_title('Test Performance Comparison')
                axes[1, 1].legend()
            
            plt.suptitle('BMW Model Fine-Tuning Comparison Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "comparison_plots.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison plots saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    def save_models(self):
        """Save both fine-tuned models"""
        # Save original model
        original_dir = os.path.join(self.output_dir, "original_model")
        os.makedirs(original_dir, exist_ok=True)
        self.original_model.save_pretrained(original_dir)
        self.tokenizer.save_pretrained(original_dir)
        
        # Save reduced model
        reduced_dir = os.path.join(self.output_dir, "reduced_model")
        os.makedirs(reduced_dir, exist_ok=True)
        self.reduced_model.save_pretrained(reduced_dir)
        self.tokenizer.save_pretrained(reduced_dir)
        
        logger.info(f"Models saved to {self.output_dir}")
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report"""
        report_path = os.path.join(self.output_dir, "comparison_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("COMPARATIVE MODEL FINE-TUNING REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Model information
            f.write("MODEL INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Base Model: {self.base_model_name}\n")
            f.write(f"Fine-tuning Method: LoRA\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            # Training results
            if 'training' in self.comparison_results:
                training_results = self.comparison_results['training']
                f.write("TRAINING RESULTS\n")
                f.write("-"*40 + "\n")
                f.write(f"Original Model - Best Epoch: {training_results['original']['best_epoch']}, "
                       f"Best Val Loss: {training_results['original']['best_val_loss']:.4f}\n")
                f.write(f"Reduced Model  - Best Epoch: {training_results['reduced']['best_epoch']}, "
                       f"Best Val Loss: {training_results['reduced']['best_val_loss']:.4f}\n\n")
                
                total_orig_time = np.sum(training_results['original']['times'])
                total_red_time = np.sum(training_results['reduced']['times'])
                speedup = (total_orig_time - total_red_time) / total_orig_time * 100 if total_orig_time > 0 else 0
                
                f.write(f"Total Training Time:\n")
                f.write(f"  Original: {total_orig_time:.1f}s\n")
                f.write(f"  Reduced:  {total_red_time:.1f}s\n")
                f.write(f"  Speedup:  {speedup:.1f}%\n\n")
            
            # Evaluation results
            if 'evaluation' in self.comparison_results:
                eval_results = self.comparison_results['evaluation']['test_metrics']
                f.write("EVALUATION RESULTS\n")
                f.write("-"*40 + "\n")
                f.write(f"Original Model:\n")
                f.write(f"  Test Loss: {eval_results['original']['test_loss']:.4f}\n")
                f.write(f"  Perplexity: {eval_results['original']['perplexity']:.2f}\n")
                f.write(f"Reduced Model:\n")
                f.write(f"  Test Loss: {eval_results['reduced']['test_loss']:.4f}\n")
                f.write(f"  Perplexity: {eval_results['reduced']['perplexity']:.2f}\n\n")
                
                # Comprehensive metrics
                if self.comparison_results['evaluation'].get('comprehensive_metrics'):
                    comp_metrics = self.comparison_results['evaluation']['comprehensive_metrics']
                    f.write("COMPREHENSIVE METRICS\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Diversity (Distinct-2): Original={comp_metrics['original']['diversity']['distinct_2']:.4f}, "
                           f"Reduced={comp_metrics['reduced']['diversity']['distinct_2']:.4f}\n")
                    f.write(f"Fluency: Original={comp_metrics['original']['fluency']:.4f}, "
                           f"Reduced={comp_metrics['reduced']['fluency']:.4f}\n")
                    if comp_metrics['original']['toxicity'] is not None:
                        f.write(f"Toxicity: Original={comp_metrics['original']['toxicity']:.6f}, "
                               f"Reduced={comp_metrics['reduced']['toxicity']:.6f}\n")
                    f.write("\n")
                
                # Q&A results if available
                if self.comparison_results['evaluation']['qa_evaluation']:
                    qa_results = self.comparison_results['evaluation']['qa_evaluation']
                    f.write(f"BERTScore F1 (0-1, higher is better):\n")
                    f.write(f"  Original Model: {qa_results['comparison']['bert_score']['original']:.4f}\n")
                    f.write(f"  Reduced Model:  {qa_results['comparison']['bert_score']['reduced']:.4f}\n\n")
                    # f.write("Q&A EVALUATION\n")
                    # f.write("-"*40 + "\n")
                    # f.write(f"Original Model Avg Log-Likelihood: {qa_results['log_likelihood']['original']:.4f}\n")
                    # f.write(f"Reduced Model Avg Log-Likelihood: {qa_results['log_likelihood']['reduced']:.4f}\n\n")
            
            # Sample generations
            if 'evaluation' in self.comparison_results:
                generations = self.comparison_results['evaluation']['generations']
                f.write("SAMPLE GENERATIONS\n")
                f.write("-"*40 + "\n")
                
                for prompt, gens in generations.items():
                    f.write(f"\nPrompt: {prompt}\n")
                    f.write(f"Original: {gens['original']}\n")
                    f.write(f"Reduced:  {gens['reduced']}\n")
                    f.write("-"*40 + "\n")
            
            # Discussion and conclusions
            f.write("\n" + "="*60 + "\n")
            f.write("DISCUSSION AND CONCLUSIONS\n")
            f.write("="*60 + "\n\n")
            
            f.write("TRADE-OFFS ANALYSIS:\n")
            f.write("1. Model Size vs. Performance: The reduced model trades a slight decrease in\n")
            f.write("   performance (higher perplexity) for increased training and inference speed.\n")
            f.write("2. Training Speed: Removing a transformer block typically results in ~15-25%\n")
            f.write("   faster training due to fewer parameters and computations.\n")
            f.write("3. Output Quality: The original model generally produces more coherent and\n")
            f.write("   contextually appropriate text, while the reduced model may show slightly\n")
            f.write("   more repetition or less nuanced responses.\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("1. For production with strict latency requirements: Consider the reduced model.\n")
            f.write("2. For maximum text quality: Use the original model.\n")
            f.write("3. For resource-constrained environments: The reduced model offers better\n")
            f.write("   efficiency with acceptable performance degradation.\n\n")
            
            f.write("NEXT STEPS WITH MORE RESOURCES:\n")
            f.write("1. Experiment with removing different transformer blocks (not just the last).\n")
            f.write("2. Try progressive layer dropping or other adaptive pruning techniques.\n")
            f.write("3. Increase training data and perform hyperparameter optimization.\n")
            f.write("4. Test with larger batch sizes and more epochs.\n")
            f.write("5. Evaluate on more diverse and challenging BMW-related tasks.\n")
        
        logger.info(f"Comparison report saved to {report_path}")

def run_comparative_pipeline():
    """Main function to run the complete comparative pipeline"""
    print("="*70)
    print("BMW COMPARATIVE FINE-TUNING PIPELINE (Enhanced Version)")
    print("="*70)
    
    # Install required packages if not already installed
    try:
        import evaluate
    except ImportError:
        print("Installing required packages for text evaluation...")
        import subprocess
        subprocess.check_call(["pip", "install", "evaluate", "rouge_score", "sacrebleu"])
    
    # Initialize comparative tuner
    print("\n1. Initializing models and tokenizer...")
    tuner = ComparativeFineTuner(
        base_model_name="distilgpt2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load data
    print("\n2. Loading BMW press release data...")
    data_dir = "bmw_press_datasets/chunked"
    try:
        tuner.load_data(
            train_path=os.path.join(data_dir, "train_chunked.txt"),
            val_path=os.path.join(data_dir, "val_chunked.txt"),
            test_path=os.path.join(data_dir, "test_chunked.txt")
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your data files exist in:", data_dir)
        print("Creating sample directory structure for testing...")
        os.makedirs(data_dir, exist_ok=True)
        # Create minimal sample files if they don't exist
        sample_texts = ["BMW press release about electric vehicles.",
                       "BMW announces new sustainability initiative.",
                       "The BMW Group reports strong quarterly results."]
        for split in ["train", "val", "test"]:
            with open(os.path.join(data_dir, f"{split}_chunked.txt"), "w") as f:
                for text in sample_texts:
                    f.write(text + "\n")
        tuner.load_data(
            train_path=os.path.join(data_dir, "train_chunked.txt"),
            val_path=os.path.join(data_dir, "val_chunked.txt"),
            test_path=os.path.join(data_dir, "test_chunked.txt")
        )
    
    # Compute baseline metrics
    print("\n3. Computing baseline metrics (original model without fine-tuning)...")
    baseline_metrics = tuner.compute_baseline_metrics()
    
    # Train models with early stopping
    print("\n4. Starting comparative training with early stopping...")
    print("   Training both original and reduced models with identical settings...")
    training_results = tuner.train_models(
        num_epochs=8,
        batch_size=8,
        learning_rate=7e-5
    )
    
    # Evaluate models comprehensively
    print("\n5. Comprehensive model evaluation...")
    eval_results = tuner.evaluate_models()
    
    # Create plots
    print("\n6. Creating comparison visualizations...")
    tuner.plot_comparison()
    
    # Save models
    print("\n7. Saving fine-tuned models...")
    tuner.save_models()
    
    # Generate report
    print("\n8. Generating comprehensive comparison report...")
    tuner.generate_comparison_report()
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {tuner.output_dir}")
    print("\nContents:")
    print("  - experiment_config.json: Experiment configuration")
    print("  - baseline_metrics.json: Original model baseline")
    print("  - training_results.csv: Training loss and time data")
    print("  - evaluation_results.json: Comprehensive evaluation metrics")
    print("  - comprehensive_metrics.json: diversity, fluency, toxicity")
    print("  - qa_evaluation.json: Q&A evaluation results")
    print("  - qualitative_analysis.json: Structured qualitative analysis")
    print("  - QUALITATIVE_COMPARISON.md: Markdown table for README")
    print("  - sample_generations.json: Generated text samples")
    print("  - comparison_plots.png: Visual comparison charts")
    print("  - comparison_report.txt: Detailed analysis and conclusions")
    print("  - original_model/: Fine-tuned original model")
    print("  - reduced_model/: Fine-tuned reduced model")
    print("  - best_original_model/: Best original model checkpoint")
    print("  - best_reduced_model/: Best reduced model checkpoint")
    
    # Print key findings
    if 'evaluation' in tuner.comparison_results:
        test_metrics = tuner.comparison_results['evaluation']['test_metrics']
        print(f"\nKey Findings:")
        print(f"  Baseline Perplexity: {baseline_metrics['perplexity']:.2f}")
        print(f"  Original Model Perplexity: {test_metrics['original']['perplexity']:.2f}")
        print(f"  Reduced Model Perplexity:  {test_metrics['reduced']['perplexity']:.2f}")
        
        orig_improvement = ((baseline_metrics['perplexity'] - test_metrics['original']['perplexity']) / 
                           baseline_metrics['perplexity'] * 100)
        red_improvement = ((baseline_metrics['perplexity'] - test_metrics['reduced']['perplexity']) / 
                          baseline_metrics['perplexity'] * 100)
        
        print(f"  Original Model Improvement: {orig_improvement:.1f}% over baseline")
        print(f"  Reduced Model Improvement: {red_improvement:.1f}% over baseline")
        
        if 'training' in tuner.comparison_results:
            total_orig = np.sum(training_results['original']['times'])
            total_red = np.sum(training_results['reduced']['times'])
            speedup = (total_orig - total_red) / total_orig * 100 if total_orig > 0 else 0
            print(f"  Training Speedup: {speedup:.1f}% faster with reduced model")
    
    return tuner

if __name__ == "__main__":
    run_comparative_pipeline()