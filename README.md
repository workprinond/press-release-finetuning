# BMW Press Release Fine-Tuning Pipeline

A complete end-to-end pipeline for collecting, preprocessing, and fine-tuning a small language model on recent BMW press releases. This project implements both the core assignment and a comparative "stretch" exercise analyzing model size trade-offs.


## Installation

```bash

pip install -r requirements.txt
# Analyze sitemaps and estimate scope
python src/sitemap_analyzer.py

# Collect BMW press releases from 2025
python src/01_data_crawling.py --year 2025 --max-articles 900

# Extract text and create dataset splits
python src/text_extraction_pipeline.py

# Remove PII and sensitive information
python src/pii_cleanup.py

# Add safety training examples
python src/safety_augmentor.py

# Analyze corpus and create 512-token chunks
python src/corpus_analyzer.py --chunk-size 512 --overlap 128

# Fine-tune original DistilGPT-2 on BMW corpus
python src/model_training.py --config configs/first_exercise.yaml

# Create reduced model and train both variants
python src/comparative_training.py --config configs/stretch_exercise.yaml

```

## 📊 1. Data Collection & Crawling

**Script:** `src/01_data_crawling.py`

**Approach & Process:**
- **Sitemap Discovery**: Analyzed the site's `robots.txt` file to extract all declared sitemaps. Filtered using regex patterns (`sitemap_text` + `en`) to isolate English-language text sitemaps (e.g., `sitemap_text_pcgl_en.xml`).
- **URL Extraction**: Parsed filtered XML sitemaps to systematically collect all URLs pointing to individual press releases.
- **Content Scraping**: For each URL, fetched HTML using `requests` and extracted structured content via identified CSS selectors:
  - Headline: `#content > div > div.content-left > div > h1`
  - Subheadline: `#content > div > div.content-left > div > h2`
  - Author: `#content > div > div.content-left > div > div.left > p:nth-child(4)`
  - Article Body: `#article-text > p`
- **Date Filtering**: Implemented custom date parsing (`BMWDateTester` class) to filter articles from 2025 only.
- **Text Cleaning**: `clean_text()` function removed HTML formatting tags, replaced `<br>` with spaces, and normalized whitespace.

**Technologies**: `requests`, `BeautifulSoup`, custom regex-based date parsing.

## 2. Sitemap Analysis & Scraping Time Estimation

**Script:** `src/sitemap_analyzer.py` (SitemapURLCounter class)

**Process:**
- **Targeted Analysis**: Focused on 13 primary English-language text sitemaps.
- **Parallelized Parsing**: Used `ThreadPoolExecutor` for concurrent XML parsing, filtering for URLs containing `'pressDetail'` and `'EN'`.
- **Volume & Time Estimation**: Calculated unique article count and provided detailed time estimates for different scraping scenarios with respectful delays (1.5s between requests).

**Outcome**: Informed decision-making on corpus scale, ensuring the project fit within the 6-8 hour scope by providing clear projections for a full-scale crawl vs. manageable subset (e.g., first 500 articles).

## 3. Complete Text Extraction Implementation

**Script:** `src/text_extraction_pipeline.py` (BMWTextExtractionPipeline class)

**Pipeline Execution:**
1. **URL Harvesting**: From `robots.txt` → English sitemaps → unique press release URLs.
2. **Robust Scraping**: Content extraction with CSS selectors + custom 2025 date filtering.
3. **Dataset Assembly**: Cleaned text normalized and assembled into JSON. Split 70/15/15 into train/validation/test sets. Exported as JSON, TXT, CSV, and JSONL.

**Key Features:**
- Respectful crawling with configurable delays
- `tqdm` progress bars with live statistics
- Comprehensive metadata preservation

**Outcome**: Curated corpus of **~900** recent (2025) BMW press releases, ready for fine-tuning.

## 4. Data Anonymization Strategy (Updated)

Based on audit results, implemented selective redaction balancing privacy with informational value.

| **Information Type** | **Action** | **Rationale** |
|----------------------|------------|---------------|
| Corporate contact (press emails, main office numbers) | Retained | Public business information |
| Official brand social media (@BMW, @MINI) | Retained | Public brand identifiers |
| Individual employee contact details | Redacted | Personal PII |
| Generic customer service numbers | Retained | Public service information |
| Personal phone numbers | Redacted | Personal PII |

**Outcome**: Corpus verified to contain only public corporate information suitable for model training without privacy concerns.

## 5. PII Cleanup Implementation

**Function:** `comprehensive_pii_clean_balanced()`

**Cleanup Actions:**
- **Emails**: All replaced with `[CONTACT_EMAIL]` placeholder.
- **Phone Numbers**: International formats replaced with `[CONTACT_PHONE]` (12,000+ patterns removed).
- **URLs**: 
  - Corporate sites → descriptive text ("Official BMW Group website")
  - Brand social media → contextual references preserved
  - External URLs → `[EXTERNAL_RESOURCE]`
- **Names**: 
  - Executives → `[EXECUTIVE_OFFICER]`
  - Media contacts → `[MEDIA_REPRESENTATIVE]`

**Results:**
- 100% email removal
- >99% phone number reduction
- Contextual URL preservation
- <5% text volume reduction

**Verification**: Follow-up audit confirmed zero raw emails/direct URLs and minimal phone pattern residuals.

## 6. Safety Augmentation

**Script:** Standalone safety augmentor

**Strategy**: Injects curated Q&A pairs teaching safe responses to sensitive queries using 4 formatting styles:
1. **Q/A Style**: Explicit question and answer
2. **Conversation Style**: Simulated user-assistant interaction
3. **Instruction Style**: Instruction-following format
4. **Single Line**: Concise statement format

**Safety Q&A Examples:**
- *"How can I contact BMW?"* → Directs to official website forms
- *"Who is the CEO of BMW?"* → Provides appropriate context without personal details
- *"How to invest in BMW?"* → Refers to official Investor Relations channels

**Integration**: Weighted augmentation (~2% of corpus), randomly interspersed throughout text.

**Outcome**: Final datasets (`*_augmented.txt`) contain embedded safety guidelines, reducing risk of privacy-violating content generation.

## 7. Model Selection

**Chosen Model:** DistilGPT-2 (82M parameters)

**Selection Rationale:**
- **Scope Fit**: Balances 6-8 hour project timeline and limited compute resources
- **Efficiency**: Dramatically faster training than larger alternatives (Phi-2, Qwen-1.5B >1.5B params)
- **Pre-training**: Trained on WebText, providing strong general English foundation
- **Architecture**: Autoregressive design specifically for coherent text generation
- **Style Alignment**: Well-suited for formal, professional corporate press release language

**Comparison**: More suitable than encoder models (DistilBERT) for generation and more appropriate than specialized models (DeepSeek-Coder) for press release language.

## 8. Training Preparation & Corpus Analysis

**Script:** `src/corpus_analyzer.py`

**Tokenization Analysis** (`train_chunked.txt`):
- **635 press releases**, average length: **1,717 tokens**
- Range: 116 to 11,271 tokens, median: 1,335 tokens
- Analysis confirmed need for chunking strategy

**Chunking Strategy (512 tokens):**
- **Comparative Analysis**:
  - 512-token chunks: 2,444 chunks needed
  - 384-token chunks: 3,157 chunks needed (+29%)
  - 768-token chunks: 1,737 chunks needed (-29%)
- **Selection Rationale**: Optimal balance (93.2% efficiency) of:
  - Computational efficiency (minimized padding waste)
  - Context preservation (sufficient for language patterns)
  - Memory constraints (fits GPU memory for 82M parameter model)
  - Architecture alignment (standard transformer implementations)
- **Implementation**: 512-token chunks with 128-token overlap (stride 384), saved as `*_chunked.txt`

**Training Volume**: Final chunked training corpus contained **2,622 sequences** (from 635 press releases), ideal for demonstration within time/compute constraints.

## 9. Model Training (First Exercise)

**Implementation:** `BMWLoRATuner` class using Parameter-Efficient Fine-Tuning (PEFT) via LoRA

**Core Configuration:**
- **Dataset**: `BMWDataset` with 512-token sequences
- **Batch Size**: 14 (optimized for memory constraints)
- **LoRA**: Target modules `c_attn`, `c_proj`, `c_fc`; rank=8, alpha=32
- **Parameter Efficiency**: Only 0.1% trainable (~80K vs 82M parameters)

**Training Configuration:**
- Optimizer: AdamW with weight decay (0.01)
- Learning Rate: 1e-5 with cosine annealing
- Gradient Clipping: Norm clipping at 1.0
- Early Stopping: Patience 6 epochs, min delta 0.001

**Key Results:**
- Memory efficiency: ~75% reduction vs full fine-tuning
- Training speed: 2-3 hours for 5 epochs on consumer GPU
- Complete training history saved in JSON/CSV formats

## 10. Results Analysis & Discussion – First Exercise

### 10.1 Quantitative Performance

| Metric | Baseline (No FT) | Fine-Tuned Model | Improvement |
|--------|------------------|------------------|-------------|
| Test Loss | 5.048 | 2.923 | **-42.1%** |
| Perplexity | 155.77 | 18.59 | **-88.1%** |
| Training Epochs | — | 5 (best) | — |
| Distinct-1 | — | 0.336 | — |
| Distinct-2 | — | 0.597 | — |
| Avg Toxicity | — | 0.00023 | — |
| Readability (FKGL) | — | 12.65 | — |
| Fluency Score | — | 0.722 | — |

**Key Insights:**
- Perplexity dropped from **156 → 19** (8× more confident in next-token predictions)
- High lexical diversity (Distinct-2 = 0.597) → varied, non-repetitive text
- Extremely low toxicity (0.00023) → safe for corporate use
- Readability score of 12.65 aligns with press-release audience (college-level)

### 10.2 Training Behavior Analysis

**Loss Curve** (`training_history.csv`):
- Initial spike (steps 0–30): Loss fluctuated during domain adjustment
- Steady decline (steps 30–400): Loss dropped from ~5.0 → ~3.0
- Convergence (steps 400–939): Loss stabilized around 2.9–3.2

**Learning Rate**: Cosine annealing from 1e-4 → 0 over 939 steps. Conservative 1e-5 effective rate prevented catastrophic forgetting.

**Validation Loss** (`eval_history.csv`):
- Epoch 1 → 2.973
- Epoch 2 → 2.914
- Epoch 3 → 2.879
- Epoch 4 → 2.864
- **Epoch 5 → 2.862** ← Best epoch
## Training Results Diagram Exercise !

Here are the loss curves from our training:

![Training Loss Curves](bmw_assignment_1/enhanced_training_plot.png)

As shown above, the model converged after 5 epochs.

### 10.3 Generation Quality & Safety

**Sample Generation** (Prompt: *"BMW is launching a new electric vehicle that"*):
**Observations:**
- ✅ **Brand-aware**: Mentions BMW iX5, electric motor (domain knowledge retained)
- ✅ **Coherent**: Sentences follow logical structure
- ⚠️ **Repetition**: Redundancy (BMW iX5 repeated) suggests limited creativity in long contexts
- ⚠️ **Minor hallucination**: Generated Martin Winterkorn (former VW CEO) as BMW CEO—factual inconsistency



**Observations:**
- ✅ **Brand-aware**: Mentions BMW iX5, electric motor (domain knowledge retained)
- ✅ **Coherent**: Sentences follow logical structure
- ⚠️ **Repetition**: Redundancy (BMW iX5 repeated) suggests limited creativity in long contexts
- ⚠️ **Minor hallucination**: Generated Martin Winterkorn (former VW CEO) as BMW CEO—factual inconsistency

**Toxicity Analysis**: All generations scored < 0.001 across toxicity categories, confirming effective safety augmentation.

### 10.4 Pipeline Effectiveness vs. Assignment Goals

| Assignment Requirement | How It Was Met | Evidence |
|------------------------|----------------|----------|
| End-to-end pipeline | ✅ | Data crawl → cleaning → fine-tuning → evaluation |
| Small open-source model | ✅ | DistilGPT-2 (82M params) |
| Brief fine-tuning | ✅ | 5 epochs, ~2–3 hours (939 steps) |
| Log training loss | ✅ | `training_history.csv`, `eval_history.csv` |
| Automatic metric | ✅ | Perplexity (18.59), Test Loss (2.923) |
| Sample generations | ✅ | 5 BMW prompts + toxicity scoring |
| 6–8 hour scope | ✅ | Achieved (data prep ~3h, training ~2.5h, eval ~1h) |

### 10.5 Limitations & Observations
- **Repetition in Long Generations**: Likely due to 512-token context limit and small model size
- **Factual Hallucinations**: Model learned style but not precise facts (e.g., incorrect CEO)
- **Loss Plateau**: After epoch 3, minimal improvement → diminishing returns
- **LoRA Trade-off**: Only 0.1% parameters trained → fast but limited adaptation depth

### 10.6 Conclusion & Alignment with Assignment Goals
This exercise successfully demonstrated a complete, reproducible pipeline for domain adaptation of a small LM to BMW press releases. The fine-tuned model achieved an **88.1% reduction in perplexity**, generated coherent, brand-appropriate text, and remained safe and non-toxic. While minor repetitions and factual inaccuracies persist, the pipeline meets all assignment requirements within the 6–8 hour scope, proving sound technical understanding and practical implementation skills.

## 11. Comparative Model Training & Evaluation (Stretch Version)

This section details the comparative fine-tuning, evaluation, and analysis of the original (6-layer) and reduced (5-layer) DistilGPT-2 models, as specified in Option 1.2 of the assignment.

### 11.1 Model Reduction Implementation
**Strategy**: Removed last transformer block from DistilGPT-2 (6 → 5 blocks)
**Function**: `create_reduced_distilgpt2()` programmatically modifies the model's `transformer.h` module list
**Verification**: Both models wrapped with identical LoRA configurations (rank=8, alpha=32)

### 11.2 Training Configuration & Process
**Identical Conditions for Both Models:**
- Optimizer: AdamW with weight decay (0.01)
- Learning Rate: 7e-5 with cosine annealing
- Batch Size: 8 (gradient accumulation steps: 4)
- Early Stopping: Patience 3 epochs
- Maximum Epochs: 8 (early stopping typically terminated earlier)

**Training Process**: Parallel training with same data/hyperparameters, comprehensive loss monitoring, checkpointing based on validation performance.

### 11.3 Comprehensive Evaluation Framework
**Quantitative Metrics:**
- Perplexity, test loss, training time (wall-clock)
- BERTScore F1 for semantic similarity in Q&A evaluation
- Diversity (Distinct-1, Distinct-2), readability (Flesch-Kincaid)
- Fluency, toxicity (Detoxify library)

## Training Results Diagram Exercise !

Here are the loss curves from our training:

![Training Loss Curves](bmw_assignment_stretch/comparison_plots.png)

As shown above, the model converged after 5 epochs.


**Qualitative Analysis:**
- Sample generations from 5 BMW-related prompts
- Automated difference analysis (length, repetition, terminology)
- Q&A evaluation: 50 BMW fact-based pairs with semantic similarity scoring

### 11.4 Training Performance & Efficiency

**Summary of Training Outcomes:**
- **Final Validation Loss**: Original model: **3.2156**, Reduced model: **3.4961** (~8.7% higher)
- **Training Speed**: Reduced model: **711.9 seconds** (11.2% faster than original at 802.0 seconds)
- **Training Dynamics**: Reduced model started with higher initial loss but followed similar convergence curve (see `training_results.csv`)

### 11.5 Quantitative Evaluation on Held-Out Set

| Metric | Original Model (6L) | Reduced Model (5L) | Comparison & Implication |
|--------|---------------------|--------------------|--------------------------|
| **Test Loss** | **3.2835** | 3.6049 | Higher loss for reduced model indicates poorer prediction of the test data. |
| **Test Perplexity** | **26.67** | 36.78 | **+37.9% higher perplexity**. The reduced model is significantly less "confident" and less adept at modeling the domain language. |
| **Fluency Score** | **0.4867** | 0.4467 | The original model's generations are judged as more fluent by the evaluator model. |
| **Toxicity** | 0.000725 | 0.000737 | Negligible difference; both models generate extremely safe, non-toxic text. |
| **Distinct-2** | 0.5085 | **0.5426** | The reduced model shows slightly higher lexical diversity, but this can correlate with less coherent or more random text. |

### 11.6 Qualitative Generation Analysis

**Key Observations** (from `sample_generations.json`):
- **Coherence & Repetition**: Original model produces more globally coherent text. Reduced model more prone to immediate, awkward repetition (e.g., "electric-electric vehicle", "battery-electric battery-electric...").
- **Factual Grounding**: Both models incorporate BMW terminology (iX, i3, M3), but original model uses them in more plausible contexts. Reduced model sometimes strings model names together illogically.
- **Content Depth**: For complex prompts, original model generates more detailed, numerically specific content.

**Example from Prompt *"BMW is launching a new electric vehicle that"***:
- **Original**: Focuses on specific technical detail ("2,500 kW electric motor...")
- **Reduced**: Shows conceptual confusion with repetitive terms ("electric-electric vehicle", "battery-electric battery-electric...")

### 11.7 BMW Q&A Evaluation

| Q&A Metric | Original Model | Reduced Model |
|------------|----------------|---------------|
| **Avg. Log-Likelihood** | **-3.396** | -3.759 |
| **BERTScore F1** | **+0.0176** | **-0.0682** |

**Analysis of Results:**
1. **Severe Factual Limitations**: Both models performed poorly on factual recall. Reduced model's negative BERTScore indicates answers are less semantically similar to correct answers than a random baseline.
2. **Hallucination & Style Over Fact**: Both models often **hallucinate incorrect facts, numbers, and names**, or default to repetitive, generic PR-sounding language.
3. **Clear Performance Gap**: Original model retains marginally better grasp of factual content, but absolute performance highlights limitation of small models: they learn **style much more effectively than verifiable facts**.

### 11.8 Discussion of Trade-offs

**Efficiency vs. Performance**: Removing ~9% of parameters yielded **~11% training speedup** but cost **~38% increase in test perplexity** and worse Q&A performance.

**Quality Degradation**: Reduced model's degradation is qualitative—more repetition and less coherent long-form structure—suggesting removed layer played role in maintaining narrative flow.

**Practical Implications**:
- **Choose Original Model** for highest output quality, coherence, and factual grounding
- **Choose Reduced Model** for latency-sensitive applications or strict computational constraints where quality drop is acceptable

**Limitations Revealed**: Poor Q&A performance underscores that fine-tuning small models primarily teaches *style and lexicon*, not reliable factual knowledge.



### 11.10 Future Investigations
With more time and compute resources:
1. **Ablation Study on Layer Removal**: Test removing different single layers
2. **Knowledge Distillation**: Distill knowledge from original to reduced model
3. **Hyperparameter Search for Reduced Architecture**: Find optimal settings for reduced model
4. **Scaling Law Verification**: Repeat with larger base model
5. **Enhanced Factual Evaluation**: Develop robust hallucination detection metrics

## 12. How to Run the Pipeline

### 12.1 Prerequisites
```bash
python>=3.8
torch>=2.0
transformers>=4.30
peft>=0.4
datasets>=2.12
beautifulsoup4>=4.12
requests>=2.28
tqdm>=4.65