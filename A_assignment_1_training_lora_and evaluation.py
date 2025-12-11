import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import os
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

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
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered. No improvement for {self.patience} epochs.")
                return True
            return False

class BMWDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
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

class BMWLoRATuner:
    def __init__(self, model_name="distilgpt2", device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc",],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
       
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        self.model.to(self.device)
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./bmw_lora_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training history
        self.history = {'train_loss': [], 'eval_loss': [], 'learning_rate': [], 'best_epoch': 0}
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump({
                'model': model_name,
                'device': str(self.device),
                'lora': {'r': 8, 'alpha': 32, 'modules': ["c_attn", "c_proj", "c_fc"]},
                'timestamp': self.timestamp
            }, f, indent=2)
    
    def load_text_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def prepare_data(self, train_path, val_path, test_path):
        logger.info("Loading data splits...")
        
        self.train_texts = self.load_text_file(train_path)
        self.val_texts = self.load_text_file(val_path)
        self.test_texts = self.load_text_file(test_path)
        
        logger.info(f"Train: {len(self.train_texts)}, Val: {len(self.val_texts)}, Test: {len(self.test_texts)}")
        
        self.train_dataset = BMWDataset(self.train_texts, self.tokenizer)
        self.val_dataset = BMWDataset(self.val_texts, self.tokenizer)
        self.test_dataset = BMWDataset(self.test_texts, self.tokenizer)
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def train(self, num_epochs=5, batch_size=14, learning_rate=1e-4, early_stopping_patience=6):
        logger.info("Starting LoRA fine-tuning...")
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize early stopper
        early_stopper = EarlyStopper(patience=early_stopping_patience, min_delta=0.001)
        
        # Only optimize trainable (LoRA) parameters
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_loader)
        )
        
        global_step = 0
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
                
                self.history['train_loss'].append(loss.item())
                self.history['learning_rate'].append(current_lr)
                global_step += 1
                
                if global_step % 100 == 0:
                    self._save_checkpoint(epoch, global_step, loss.item())
            
            # Validation
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = self._evaluate(val_loader)
            self.history['eval_loss'].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.history['best_epoch'] = epoch + 1
                self.best_model_path = os.path.join(self.output_dir, f"best_model_epoch_{epoch+1}")
                self._save_best_model(epoch + 1, avg_val_loss)
                logger.info(f"New best model saved (Val Loss: {avg_val_loss:.4f})")
            
            # Check early stopping
            if early_stopper(avg_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model for final evaluation
        if self.best_model_path:
            logger.info(f"Loading best model from epoch {self.history['best_epoch']}")
            self._load_best_model()
        
        # Save final history
        self._save_final_model()
        self._save_history()
        
        return self.history
    
    def _evaluate(self, dataloader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # outputs = self.model(**batch)
                outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'], 
                labels=batch['input_ids']  )
                losses.append(outputs.loss.item())
        return np.mean(losses)
    
    def _save_best_model(self, epoch, val_loss):
        """Save the best model based on validation loss."""
        os.makedirs(self.best_model_path, exist_ok=True)
        self.model.save_pretrained(self.best_model_path)
        self.tokenizer.save_pretrained(self.best_model_path)
        
        with open(os.path.join(self.best_model_path, 'best_model_info.json'), 'w') as f:
            json.dump({
                'epoch': epoch,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_best_model(self):
        """Load the best model from saved checkpoint."""
        from peft import PeftModel
        # Reload base model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.best_model_path)
        self.model.to(self.device)
    
    def _save_checkpoint(self, epoch, step, loss, is_epoch_end=False):
        if is_epoch_end:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_step_{step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        with open(os.path.join(checkpoint_dir, 'info.json'), 'w') as f:
            json.dump({'epoch': epoch+1, 'step': step, 'loss': loss, 'timestamp': datetime.now().isoformat()}, f, indent=2)
    
    def _save_final_model(self):
        final_dir = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")
    
    def _save_history(self):
        # Save JSON
        with open(os.path.join(self.output_dir, "training_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save CSV
        history_df = pd.DataFrame({
            'step': range(len(self.history['train_loss'])),
            'train_loss': self.history['train_loss'],
            'learning_rate': self.history['learning_rate']
        })
        history_df.to_csv(os.path.join(self.output_dir, "training_history.csv"), index=False)
        
        eval_df = pd.DataFrame({
            'epoch': range(1, len(self.history['eval_loss']) + 1),
            'eval_loss': self.history['eval_loss']
        })
        eval_df.to_csv(os.path.join(self.output_dir, "eval_history.csv"), index=False)
        
        logger.info(f"Training history saved")
    
    def compute_baseline_metrics(self):
        """Compute metrics for the base model without fine-tuning."""
        logger.info("Computing BASELINE metrics (original distilgpt2)...")
        
        # Use base model (not fine-tuned)
        base_model = self.base_model.to(self.device)
        base_model.eval()
        
        test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Baseline"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # outputs = base_model(**batch)
                outputs = self.base_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'], 
                    labels=batch['input_ids']  
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
            json.dump(metrics, f, indent=2)
        
        logger.info(f"BASELINE - Test Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        return metrics
    
    def compute_metrics(self):
        """Compute metrics for the fine-tuned model."""
        logger.info("Computing evaluation metrics for fine-tuned model...")
        
        test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Fine-tuned Model"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # outputs = self.model(**batch)
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],  # Explicitly pass
                    labels=batch['input_ids']  # For loss calculation
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
            'model': 'fine_tuned_lora',
            'best_epoch': self.history['best_epoch']
        }
        
        with open(os.path.join(self.output_dir, "test_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"FINE-TUNED - Test Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f} (Best Epoch: {self.history['best_epoch']})")
        return metrics
    
    def generate_samples(self, prompts=None, max_length=100, compute_toxicity=True):
        if prompts is None:
            prompts = [
                "BMW is launching a new electric vehicle that",
                "The BMW Group announced strong financial results",
                "BMW's commitment to sustainability includes",
                "BMW autonomous driving features will",
                "The future of BMW mobility involves"
            ]
        
        self.model.eval()
        generations = []
        
        # Import detoxify only when needed
        if compute_toxicity:
            try:
                from detoxify import Detoxify
                toxicity_model = Detoxify('original')
            except ImportError:
                logger.warning("Detoxify not installed. Install with: pip install detoxify")
                compute_toxicity = False
        
        logger.info("Generating sample texts...")
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Compute toxicity if requested
            toxicity_scores = {}
            if compute_toxicity:
                try:
                    toxicity_scores = toxicity_model.predict(generated_text)
                    # Get average toxicity score
                    avg_toxicity = sum(toxicity_scores.values()) / len(toxicity_scores)
                    toxicity_scores['average'] = avg_toxicity
                except Exception as e:
                    logger.warning(f"Could not compute toxicity: {e}")
            
            generation_data = {
                'prompt': prompt, 
                'generation': generated_text,
                'toxicity_scores': toxicity_scores
            }
            generations.append(generation_data)
            
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            if toxicity_scores:
                print(f"Toxicity Score: {toxicity_scores.get('average', 0):.4f}")
            print("-" * 80)
        
        with open(os.path.join(self.output_dir, "sample_generations.json"), 'w') as f:
            json.dump(generations, f, indent=2, default=str)
        
        return generations
    
   

    def compute_text_quality_metrics(self, num_samples=20):
            """Compute text quality metrics including diversity, readability, toxicity, and FLUENCY."""
            logger.info("Computing comprehensive text quality metrics...")
            
            metrics = {
                'diversity_metrics': {},
                'readability_scores': None,
                'toxicity_scores': None,
                'fluency_scores': None 
            }
            
            # Sample from test set for evaluation
            test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)
            
            generated_texts = []
            reference_texts = []
            
            # Generate texts for diversity/readability/toxicity evaluation
            self.model.eval()
            for i, batch in enumerate(test_loader):
                if i >= num_samples:
                    break
                    
                # Use first 20 tokens as prompt, rest as reference
                input_ids = batch['input_ids'][0][:20]
                reference = batch['input_ids'][0][20:]
                
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                reference_text = self.tokenizer.decode(reference, skip_special_tokens=True)
                
                # Generate continuation
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=100,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                generated_texts.append(generated_text)
                reference_texts.append(reference_text)
            
            # 1. Compute Diversity (Distinct-n) - Keep existing
            all_tokens = [text.split() for text in generated_texts]
            flat_tokens = [token for sublist in all_tokens for token in sublist]
            
            # Distinct-1 (word-level diversity)
            unique_unigrams = set(flat_tokens)
            metrics['diversity_metrics']['distinct_1'] = len(unique_unigrams) / len(flat_tokens) if flat_tokens else 0
            
            # Distinct-2 (phrase-level diversity)
            bigrams = []
            for tokens in all_tokens:
                bigrams.extend([' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)])
            unique_bigrams = set(bigrams)
            metrics['diversity_metrics']['distinct_2'] = len(unique_bigrams) / len(bigrams) if bigrams else 0
            
            # 2. Compute Readability (Flesch-Kincaid)
            try:
                import textstat
                readability_scores = []
                for text in generated_texts:
                    if len(text.split()) > 10:
                        fk_score = textstat.flesch_kincaid_grade(text)
                        readability_scores.append(fk_score)
                metrics['readability_scores'] = {
                    'average_flesch_kincaid': float(np.mean(readability_scores)) if readability_scores else 0,
                    'num_scored': len(readability_scores)
                }
            except ImportError:
                logger.warning("textstat not installed. Install with: pip install textstat")
            
            # 3. Compute Toxicity Scores
            try:
                from detoxify import Detoxify
                toxicity_model = Detoxify('original')
                all_toxicity_scores = []
                for text in generated_texts:
                    scores = toxicity_model.predict(text)
                    avg_toxicity = sum(scores.values()) / len(scores)
                    all_toxicity_scores.append(avg_toxicity)
                
                metrics['toxicity_scores'] = {
                    'average': float(np.mean(all_toxicity_scores)) if all_toxicity_scores else 0,
                    'max': float(np.max(all_toxicity_scores)) if all_toxicity_scores else 0,
                    'min': float(np.min(all_toxicity_scores)) if all_toxicity_scores else 0
                }
            except ImportError:
                logger.warning("Detoxify not installed. Install with: pip install detoxify")
            
            
            try:
                fluency_scores = []
                perplexities = []
                
            
                fluency_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
                
                for i, batch in enumerate(fluency_loader):
                    if i >= num_samples:
                        break
                        
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],  
                            labels=batch['input_ids']
                        )
                        loss = outputs.loss
                    
                    # Calculate perplexity (lower = more fluent)
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
                    
                   
                    if perplexity < 10:
                        fluency = 0.95  # Excellent
                    elif perplexity < 20:
                        fluency = 0.85   # Very good
                    elif perplexity < 30:
                        fluency = 0.75   # Good
                    elif perplexity < 50:
                        fluency = 0.6    # Fair
                    elif perplexity < 100:
                        fluency = 0.3    # Poor
                    else:
                        fluency = 0.1    # Very poor
                    
                    fluency_scores.append(fluency)
                
                metrics['fluency_scores'] = {
                    'average_fluency': float(np.mean(fluency_scores)) if fluency_scores else 0,
                    'average_perplexity': float(np.mean(perplexities)) if perplexities else 0,
                    'num_scored': len(fluency_scores)
                }
            except Exception as e:
                logger.warning(f"Could not compute fluency: {e}")
                metrics['fluency_scores'] = {'error': str(e)}
            
            # Save metrics
            metrics_path = os.path.join(self.output_dir, "text_quality_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Text quality metrics saved to {metrics_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("TEXT QUALITY METRICS SUMMARY")
            print("="*60)
            print(f"Distinct-1: {metrics['diversity_metrics']['distinct_1']:.4f}")
            print(f"Distinct-2: {metrics['diversity_metrics']['distinct_2']:.4f}")
            if metrics.get('readability_scores'):
                print(f"Avg Flesch-Kincaid Grade Level: {metrics['readability_scores']['average_flesch_kincaid']:.2f}")
            if metrics.get('toxicity_scores'):
                print(f"Avg Toxicity Score: {metrics['toxicity_scores']['average']:.6f}")
            if metrics.get('fluency_scores') and 'average_perplexity' in metrics['fluency_scores']:
                print(f"Avg Perplexity on Test Set: {metrics['fluency_scores']['average_perplexity']:.2f}")
                print(f"Avg Fluency Score: {metrics['fluency_scores']['average_fluency']:.4f}")
                if metrics['fluency_scores']['average_perplexity'] < 30:
                    print(f"  → Good fluency (perplexity < 30)")
                else:
                    print(f"  → Needs improvement (perplexity ≥ 30)")
            
            return metrics

def plot_training_history(output_dir):
    try:
        import matplotlib.pyplot as plt
        
        csv_file = os.path.join(output_dir, "training_history.csv")
        if not os.path.exists(csv_file):
            print(f"No CSV file found at {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training loss with better styling
        axes[0, 0].plot(df['step'], df['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Training Step', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning rate with styling
        axes[0, 1].plot(df['step'], df['learning_rate'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Training Step', fontsize=12)
        axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Rolling average of loss (new)
        rolling_window = min(50, len(df) // 10)
        if rolling_window > 1:
            df['rolling_loss'] = df['train_loss'].rolling(window=rolling_window).mean()
            axes[1, 0].plot(df['step'], df['rolling_loss'], 'r-', linewidth=2, label=f'{rolling_window}-step avg')
            axes[1, 0].plot(df['step'], df['train_loss'], 'b-', alpha=0.3, linewidth=1, label='Raw')
            axes[1, 0].legend()
            axes[1, 0].set_xlabel('Training Step', fontsize=12)
            axes[1, 0].set_ylabel('Loss', fontsize=12)
            axes[1, 0].set_title('Smoothed Training Loss', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss distribution histogram (new)
        axes[1, 1].hist(df['train_loss'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Loss Value', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Loss Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Model Training Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "enhanced_training_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced training plot saved to {plot_path}")
    except ImportError:
        print("Matplotlib not installed. Skipping plot.")
    except Exception as e:
        print(f"Could not create plot: {e}")

def create_comparison_report(baseline_metrics, fine_tuned_metrics, text_metrics, output_dir):
    """Create a comprehensive comparison report."""
    report = {
        "experiment_summary": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "base_model": "distilgpt2",
            "fine_tuning_method": "LoRA",
            "best_epoch": fine_tuned_metrics.get('best_epoch', 'N/A')
        },
        "performance_comparison": {
            "baseline": {
                "test_loss": baseline_metrics.get('test_loss', 'N/A'),
                "perplexity": baseline_metrics.get('perplexity', 'N/A')
            },
            "fine_tuned": {
                "test_loss": fine_tuned_metrics.get('test_loss', 'N/A'),
                "perplexity": fine_tuned_metrics.get('perplexity', 'N/A'),
                "improvement_perplexity": (
                    ((baseline_metrics.get('perplexity', 1) - fine_tuned_metrics.get('perplexity', 1)) / 
                     baseline_metrics.get('perplexity', 1)) * 100 
                    if baseline_metrics.get('perplexity') and fine_tuned_metrics.get('perplexity')
                    else 'N/A'
                )
            }
        },
        "text_quality_metrics": text_metrics or {"note": "Text quality metrics not computed"}
    }
    
    # Save report
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    print(f"Baseline Perplexity: {baseline_metrics.get('perplexity', 'N/A'):.4f}")
    print(f"Fine-tuned Perplexity: {fine_tuned_metrics.get('perplexity', 'N/A'):.4f}")
    
    if isinstance(report["performance_comparison"]["fine_tuned"]["improvement_perplexity"], float):
        improvement = report["performance_comparison"]["fine_tuned"]["improvement_perplexity"]
        print(f"Improvement: {improvement:.2f}%")
    
    print(f"\nFull report saved to: {report_path}")
    return report

def run_pipeline():
    print("="*60)
    print("BMW LoRA Fine-tuning Pipeline")
    print("="*60)
    
    # Initialize
    finetuner = BMWLoRATuner(
        model_name="distilgpt2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Prepare data
    data_dir = "bmw_press_datasets/chunked"
    try:
        finetuner.prepare_data(
            train_path=os.path.join(data_dir, "train_chunked.txt"),
            val_path=os.path.join(data_dir, "val_chunked.txt"),
            test_path=os.path.join(data_dir, "test_chunked.txt")
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check your data files exist at:", data_dir)
        return None
    
    print("\n" + "="*60)
    print("1. Computing Baseline Metrics (Original Model)")
    print("="*60)
    baseline_metrics = finetuner.compute_baseline_metrics()
    
    print("\n" + "="*60)
    print("2. Starting Training with Early Stopping")
    print("="*60)
    
    # Train with early stopping
    history = finetuner.train(
        num_epochs=5,
        batch_size=14,
        learning_rate=1e-5,
        early_stopping_patience=6  
    )
    
    print("\n" + "="*60)
    print("3. Evaluating Fine-tuned Model")
    print("="*60)
    fine_tuned_metrics = finetuner.compute_metrics()
    
    print("\n" + "="*60)
    print("4. Computing Text Quality Metrics")  
    print("="*60)
    text_metrics = finetuner.compute_text_quality_metrics()  
    
    print("\n" + "="*60)
    print("5. Generating Sample Texts")
    print("="*60)
    generations = finetuner.generate_samples(max_length=100)
    
    # Create plot
    plot_training_history(finetuner.output_dir)
    
    # Create comparison report
    create_comparison_report(baseline_metrics, fine_tuned_metrics, text_metrics, finetuner.output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Results saved to: {finetuner.output_dir}")
    print("="*60)
    
    return finetuner, history, baseline_metrics, fine_tuned_metrics, text_metrics, generations

if __name__ == "__main__":
    run_pipeline()