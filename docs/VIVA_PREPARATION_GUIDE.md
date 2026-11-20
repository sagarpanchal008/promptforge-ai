# MASTER VIVA PREPARATION & DEMONSTRATION SCRIPT
## Intelligent Prompt Optimization Assistant Using CNN, LSTM and Agentic AI

---

## 🎬 1. Project Demonstration Script (Step-by-Step for College Review)

### Step 1: Introduction (1 Minute)
> *"Respected Examiners and Mentors, today I present our Major Project titled 'Intelligent Prompt Optimization Assistant Using CNN, LSTM, and Agentic AI'. Generative AI models produce poor results when fed vague prompts. Our system solves this by combining two Deep Learning models (1D-CNN and Bi-LSTM) to diagnose prompt quality, with a 6-Agent LangGraph workflow that iteratively re-engineers prompts into production-ready instructions."*

### Step 2: Live Web App Demonstration (2 Minutes)
1. Open the browser at `http://localhost:8501`.
2. Demonstrate a **Poor Prompt**: Click the quick button: `"💻 Bug Fix Prompt"` (`"write python code to fix pandas missing values and remove duplicates"`).
3. Click **⚡ Analyze & Optimize Prompt**.
4. Point out the **Deep Learning Diagnostics**:
   - 1D-CNN classifies it as `Poor` (or `Average`).
   - Bi-LSTM classifies it as `Poor`.
   - Granular scores: Clarity, Context, Specificity, Completeness.
5. Point out the **Identified Weaknesses**:
   - Missing Persona, Missing Constraints, Underspecified Output Format.
6. Show the **Optimized Prompt**:
   - Injected Role: `Staff Software Architect`.
   - Injected Constraints: Type hints, SOLID principles, defensive input validation, O(N) complexity, and 3 unit tests.
7. Show **Score Gain**: e.g., Overall Score increased from `2.1/10` to `9.4/10` (+7.3 points).
8. Demonstrate **Export**: Click **Download Audit Report (TXT)**.

### Step 3: Deep Learning Benchmark Tab (1 Minute)
1. Navigate to **Tab 2: DL Benchmark**.
2. Explain the empirical comparison table:
   - 1D-CNN: 100% test accuracy, 282,947 parameters, ~1.2 ms latency.
   - Bi-LSTM: 100% test accuracy, 316,291 parameters, 0.0041 loss (58% lower loss than CNN).
3. Conclude: *"In Agent 2, we combine both models using a 50/50 Softmax probability ensemble to get the speed of CNN with the contextual depth of Bi-LSTM."*

---

## 🎯 2. Top 20 Technical Viva Questions and Model Answers

### Q1: Why did you use 1D Convolution instead of 2D Convolution?
> **Answer:** 2D Convolutions slide across height and width for images ($H \times W \times C$). Text is a 1-dimensional sequential data structure ($T \times D$, where $T$ is time steps and $D$ is embedding dimension). A 1D-CNN slides a filter along the sequence axis, computing dot products over contiguous n-gram word blocks.

### Q2: How does MaxPooling1D differ from GlobalMaxPooling1D?
> **Answer:** 
> - `MaxPooling1D(pool_size=2)` downsamples the sequence dimension by taking the maximum value every 2 time steps, reducing length from 120 to 60.
> - `GlobalMaxPooling1D` takes the single maximum activation across the *entire* sequence for each filter, condensing the 2D tensor $(T, F)$ into a 1D feature vector of size $F$, capturing the strongest signal regardless of position.

### Q3: What is the vanishing gradient problem and how does LSTM solve it?
> **Answer:** In standard Recurrent Neural Networks (RNNs), gradients shrink exponentially over time steps due to repeated matrix multiplications in backpropagation through time. LSTMs solve this through an additive cell state ($c_t$) regulated by three gates (Forget, Input, and Output gates), allowing error gradients to flow unchanged across long sequences.

### Q4: Why is Bidirectional LSTM superior to Unidirectional LSTM for prompts?
> **Answer:** In unidirectional LSTM, the hidden state at word 20 only knows words 1 to 19. In a prompt, an instruction at the beginning (*"You are a mentor"*) might define the meaning of words at the end (*"give examples"*). Bi-LSTM runs a forward LSTM and a backward LSTM, concatenating their hidden states so that every token representation is conditioned on the complete sentence context.

### Q5: What is Zero-Masking (`mask_zero=True`) in Keras Embedding?
> **Answer:** All prompts are padded to 120 tokens with zeros. Zero-masking creates a boolean mask that tells the LSTM layers to skip computation on zero-padded indices, preventing the padding from polluting the recurrent memory state.

### Q6: How was data leakage prevented during preprocessing?
> **Answer:** The `TextTokenizer` and `QualityLabelEncoder` were fitted exclusively on the 882 training samples (`X_train`). The validation set (189) and test set (189) were transformed using the pre-fitted vocabulary without altering the word index or token distributions.

### Q7: Why use LangGraph for Agentic AI instead of sequential Python functions?
> **Answer:** LangGraph represents agent execution as a directed state graph with explicit state persistence and conditional routing. It allows Agent 5 (Quality Reviewer) to evaluate the generated prompt and conditionally route execution back to Agent 4 (Optimizer) if quality criteria are not satisfied, creating a genuine multi-agent feedback loop.

### Q8: What is the Ensemble Blending formula in Agent 2?
> **Answer:** Agent 2 blends the Softmax probability distributions from the 1D-CNN and Bi-LSTM models:
> $$P_{\text{ensemble}}(c) = 0.5 \cdot P_{\text{CNN}}(c) + 0.5 \cdot P_{\text{LSTM}}(c)$$
> The predicted class is $\arg\max_c P_{\text{ensemble}}(c)$.

### Q9: What loss function and optimizer were used?
> **Answer:** 
> - Loss Function: Categorical Cross-Entropy: $\mathcal{L} = -\sum_{i=1}^3 y_i \log(\hat{y}_i)$.
> - Optimizer: Adam (Adaptive Moment Estimation) with an initial learning rate of $1 \times 10^{-3}$, coupled with a `ReduceLROnPlateau` callback to decay learning rate when validation loss plateaus.

### Q10: How does the system handle Gemini API failures or quota exhaustion?
> **Answer:** Agent 4 implements a dual-engine pattern: it attempts invocation of the Google Gemini API using the secure key from the environment. If the API is unavailable, offline, or unconfigured, it seamlessly falls back to a deterministic strategic prompt template engine tailored for the specific domain, guaranteeing 100% uptime.
