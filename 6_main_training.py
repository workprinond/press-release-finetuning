# bmw_fine_tuning_pipeline.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import logging
from datetime import datetime
import os
from tqdm import tqdm
from datasets import Dataset as HFDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BMWDataset(Dataset):
    """Custom dataset for BMW press releases"""
    def __init__(self, texts, tokenizer, max_length=128):
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
            'labels': encoding['input_ids'].flatten()  # For causal LM
        }

class BMWFineTuner:
    def __init__(self, model_name="distilgpt2", device=None):
        """
        Initialize with DistilGPT-2 (small, efficient for text generation)
        Alternatives: 'distilbert-base-uncased', 'gpt2', 't5-small'
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # For GPT-2
        
        # Load model for causal language modeling (text generation)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./bmw_finetune_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': []
        }
 

    def prepare_data(self, train_path, val_path, test_path):
            """Load and prepare the data splits from TEXT files"""
            logger.info("Loading data splits...")
            
            # Load from text files (not CSV)
            def load_text_file(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            
            # Load texts directly from .txt files
            self.train_texts = load_text_file(train_path)
            self.val_texts = load_text_file(val_path)
            self.test_texts = load_text_file(test_path)
            
            # No CSV columns needed
            logger.info(f"Train samples: {len(self.train_texts)}")
            logger.info(f"Validation samples: {len(self.val_texts)}")
            logger.info(f"Test samples: {len(self.test_texts)}")
            
            # Create datasets
            self.train_dataset = BMWDataset(self.train_texts, self.tokenizer)
            self.val_dataset = BMWDataset(self.val_texts, self.tokenizer)
            self.test_dataset = BMWDataset(self.test_texts, self.tokenizer)
            
            return self.train_dataset, self.val_dataset, self.test_dataset
    
    def train(self, num_epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model on BMW data"""
        logger.info("Starting fine-tuning...")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Simple learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_loader)
        )
        
        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Log
                train_losses.append(loss.item())
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                # Save to history
                self.history['train_loss'].append(loss.item())
                self.history['learning_rate'].append(current_lr)
                global_step += 1
                
                # Save checkpoint every 50 steps
                if global_step % 50 == 0:
                    self._save_checkpoint(epoch, global_step, loss.item())
            
            # Validation phase
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = self._evaluate(val_loader)
            
            self.history['eval_loss'].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch + 1} - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}")
            
            # Save model checkpoint
            self._save_checkpoint(epoch, global_step, avg_val_loss, is_epoch_end=True)
        
        # Save final model
        self._save_final_model()
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _evaluate(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                losses.append(outputs.loss.item())
        
        return np.mean(losses)
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = torch.stack([item[key] for item in batch])
        return batch_dict
    
    def _save_checkpoint(self, epoch, step, loss, is_epoch_end=False):
        """Save model checkpoint"""
        if is_epoch_end:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_step_{step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save checkpoint info
        checkpoint_info = {
            'epoch': epoch + 1,
            'step': step,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(checkpoint_dir, 'info.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
    
    def _save_final_model(self):
        """Save the final fine-tuned model"""
        final_dir = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")
    
    def _save_history(self):
        """Save training history"""
        history_file = os.path.join(self.output_dir, "training_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Also save as CSV for easy plotting
        history_df = pd.DataFrame({
            'step': range(len(self.history['train_loss'])),
            'train_loss': self.history['train_loss'],
            'learning_rate': self.history['learning_rate']
        })
        history_df.to_csv(os.path.join(self.output_dir, "training_history.csv"), index=False)
        
        # Add eval loss (which is per epoch)
        eval_df = pd.DataFrame({
            'epoch': range(1, len(self.history['eval_loss']) + 1),
            'eval_loss': self.history['eval_loss']
        })
        eval_df.to_csv(os.path.join(self.output_dir, "eval_history.csv"), index=False)
        
        logger.info(f"Training history saved to {self.output_dir}")
    
    def compute_metrics(self, test_loader=None):
        """Compute automatic metrics on test set"""
        logger.info("Computing evaluation metrics...")
        
        if test_loader is None:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # Compute perplexity (common LM metric)
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # Cross-entropy loss
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_tokens += (batch['attention_mask'].sum().item())
        
        # Average loss
        avg_loss = total_loss / len(self.test_dataset)
        
        # Perplexity (exp of average negative log likelihood)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'test_loss': avg_loss,
            'perplexity': perplexity,
            'num_samples': len(self.test_dataset),
            'avg_tokens_per_sample': total_tokens / len(self.test_dataset)
        }
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, "test_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Test Loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def generate_samples(self, prompts=None, num_samples=3, max_length=50):
        """Generate sample text from the fine-tuned model"""
        if prompts is None:
            # BMW-related prompts
            prompts = [
                "BMW is launching a new electric vehicle",
                "The BMW Group announced",
                "BMW's sustainability strategy focuses on",
                "BMW autonomous driving technology",
                "BMW financial results show"
            ]
        
        self.model.eval()
        generations = []
        
        logger.info("Generating sample texts...")
        
        for prompt in prompts[:num_samples]:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generations.append({
                'prompt': prompt,
                'generation': generated_text,
                'generated_part': generated_text[len(prompt):]  # Just the new part
            })
            
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 80)
        
        # Save generations
        gens_file = os.path.join(self.output_dir, "sample_generations.json")
        with open(gens_file, 'w') as f:
            json.dump(generations, f, indent=2)
        
        return generations

def run_full_pipeline():
    """Complete pipeline from data preparation to evaluation"""
    
    # Step 1: Initialize fine-tuner
    finetuner = BMWFineTuner(
        model_name="distilgpt2",  # Small, fast model
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Step 2: Prepare data (assuming data is in ./bmw_press_data/)
    data_dir = "./bmw_press_datasets/chunked"
    try:
        finetuner.prepare_data(
            train_path=os.path.join(data_dir, "train_chunked.txt"),
            val_path=os.path.join(data_dir, "val_chunked.txt"),
            test_path=os.path.join(data_dir, "test_chunked.txt  ")
        )
    except FileNotFoundError:
        # If data doesn't exist, create sample data
        print("No existing data found. Creating sample data...")
        create_sample_data(data_dir)
        finetuner.prepare_data(
            train_path=os.path.join(data_dir, "train_chunked.txt"),
            val_path=os.path.join(data_dir, "val_chunked.txt"),
            test_path=os.path.join(data_dir, "test_chunked.txt")
        )
    
    # Step 3: Fine-tune the model
    print("\n" + "="*60)
    print("Starting Fine-tuning")
    print("="*60)
    
    history = finetuner.train(
        num_epochs=3,           # Small number for demonstration
        batch_size=4,
        learning_rate=5e-5
    )
    
    # Step 4: Evaluate on test set
    print("\n" + "="*60)
    print("Evaluation on Test Set")
    print("="*60)
    
    metrics = finetuner.compute_metrics()
    
    # Step 5: Generate sample texts
    print("\n" + "="*60)
    print("Sample Generations")
    print("="*60)
    
    # Define BMW-specific prompts
    bmw_prompts = [
        "BMW is launching a new electric vehicle that",
        "The BMW Group announced strong financial results",
        "BMW's commitment to sustainability includes",
        "BMW autonomous driving features will",
        "The future of BMW mobility involves"
    ]
    
    generations = finetuner.generate_samples(
        prompts=bmw_prompts,
        num_samples=5,
        max_length=60
    )
    
    # Step 6: Create a summary report
    create_summary_report(finetuner, history, metrics, generations)
    
    return finetuner, history, metrics, generations

def create_sample_data(data_dir):
    """Create sample data if real data doesn't exist"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample training data
    sample_texts = [
        "the bmw group announced strong quarterly results with record electric vehicle sales",
        "bmw is launching a new fully electric sedan with advanced autonomous features",
        "sustainability remains a core focus for bmw with investments in renewable energy",
        "bmw autonomous driving technology achieves new safety milestones in testing",
        "the bmw group expands production capacity for electric vehicles in germany",
        "bmw unveils next generation infotainment system with artificial intelligence",
        "bmw partnership with battery manufacturer aims to improve energy density",
        "bmw financial services reports growth in electric vehicle financing",
        "bmw research and development focuses on hydrogen fuel cell technology",
        "bmw group receives award for corporate sustainability reporting"
    ]
    
    train_texts = sample_texts[:7]
    val_texts = sample_texts[7:9]
    test_texts = sample_texts[9:]
    
    # Save as CSV files
    pd.DataFrame({'processed_text': train_texts}).to_csv(
        os.path.join(data_dir, "train.txt"), index=False
    )
    pd.DataFrame({'processed_text': val_texts}).to_csv(
        os.path.join(data_dir, "validation.txt"), index=False
    )
    pd.DataFrame({'processed_text': test_texts}).to_csv(
        os.path.join(data_dir, "test.txt"), index=False
    )
    
    print(f"Sample data created in {data_dir}")

def create_summary_report(finetuner, history, metrics, generations):
    """Create a comprehensive summary report"""
    report = {
        'model': finetuner.model_name,
        'device': finetuner.device,
        'timestamp': datetime.now().isoformat(),
        'training_summary': {
            'total_steps': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_eval_loss': history['eval_loss'][-1] if history['eval_loss'] else None,
            'num_epochs': len(history['eval_loss'])
        },
        'evaluation_metrics': metrics,
        'sample_generations': generations,
        'output_directory': finetuner.output_dir
    }
    
    report_file = os.path.join(finetuner.output_dir, "summary_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also create a human-readable summary
    txt_report = f"""
    BMW Fine-tuning Pipeline - Summary Report
    {'='*60}
    
    Model: {report['model']}
    Training Device: {report['device']}
    Timestamp: {report['timestamp']}
    
    Training Statistics:
    - Total Training Steps: {report['training_summary']['total_steps']}
    - Number of Epochs: {report['training_summary']['num_epochs']}
    - Final Training Loss: {report['training_summary']['final_train_loss']:.4f}
    - Final Validation Loss: {report['training_summary']['final_eval_loss']:.4f}
    
    Evaluation on Test Set:
    - Test Loss: {report['evaluation_metrics']['test_loss']:.4f}
    - Perplexity: {report['evaluation_metrics']['perplexity']:.2f}
    - Test Samples: {report['evaluation_metrics']['num_samples']}
    
    Sample Generations:
    """
    
    for i, gen in enumerate(report['sample_generations'], 1):
        txt_report += f"\n{i}. Prompt: {gen['prompt']}\n"
        txt_report += f"   Generated: {gen['generation']}\n"
    
    txt_report += f"\n{'='*60}\n"
    txt_report += f"All outputs saved to: {report['output_directory']}\n"
    
    # Save text report
    with open(os.path.join(finetuner.output_dir, "summary_report.txt"), 'w') as f:
        f.write(txt_report)
    
    print(txt_report)

def plot_training_history(history_dir):
    """Create simple plots of training history"""
    try:
        import matplotlib.pyplot as plt
        
        history_df = pd.read_csv(os.path.join(history_dir, "training_history.csv"))
        eval_df = pd.read_csv(os.path.join(history_dir, "eval_history.csv"))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot training loss
        axes[0, 0].plot(history_df['step'], history_df['train_loss'])
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Steps')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot learning rate
        axes[0, 1].plot(history_df['step'], history_df['learning_rate'])
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot evaluation loss per epoch
        axes[1, 0].plot(eval_df['epoch'], eval_df['eval_loss'], 'o-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss Per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Create histogram of loss values
        axes[1, 1].hist(history_df['train_loss'], bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(history_dir, "training_plots.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Training plots saved to {plot_path}")
        
    except ImportError:
        print("Matplotlib not installed. Skipping plots.")
    except Exception as e:
        print(f"Could not create plots: {e}")

if __name__ == "__main__":
    print("BMW Fine-tuning Pipeline")
    print("="*60)
    
    # Run the complete pipeline
    finetuner, history, metrics, generations = run_full_pipeline()
    
    # Create training plots
    plot_training_history(finetuner.output_dir)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print(f"Results saved to: {finetuner.output_dir}")
    print("="*60)

    