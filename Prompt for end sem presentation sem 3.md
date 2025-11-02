Create a comprehensive academic presentation for my Master's Terminal Project (MTP) on "Enhanced Numerical Claim Verification through Question-Constrained Decomposition." The presentation should be professional, research-oriented, and suitable for an academic defense.

## Project Context
This is a fact-checking system that improves upon the QuanTemp baseline by addressing challenges in verifying comparison-based numerical claims through better question generation, diversification, and evidence retrieval strategies.

## Presentation Structure

### SLIDE 1: Title Slide
- Title: "Enhanced Numerical Claim Verification through Question-Constrained Decomposition"
- Subtitle: Master's Terminal Project
- Include space for name, department, date, and advisor information

### SLIDE 2-4: Introduction
**Slide 2: The Misinformation Challenge**
- Define claim verification and its importance in the digital age
- Statistics on misinformation spread
- Why automated fact-checking is crucial
- Transition to numerical claims (e.g., "India's GDP grew by 7% in 2023")

**Slide 3: Automated Claim Verification Pipeline**
- Visual flowchart showing: Claim → Evidence Retrieval → Verification → Label
- Explain each component briefly
- Highlight the role of claim decomposition

**Slide 4: Numerical Claim Verification**
- What are numerical claims? (claims containing quantities, comparisons, temporal references)
- Why they're harder to verify than textual claims
- Overview of existing work on numerical verification
- Introduction to QuanTemp as the baseline approach

### SLIDE 5-6: Motivation & Problem Statement
**Slide 5: Error Analysis from QuanTemp**
Present a table or visualization showing:
- QuanTemp's performance across claim categories (Conflicting, False, True)
- Highlight the specific challenge: "Conflicting claims scored only 47.33 F1"
- Quote: "Conflicting claims pose difficulty as they are partially incorrect, requiring retrieval of contrasting evidence for different aspects"

**Slide 6: Key Challenge - Comparison Claims**
- Focus on comparison-based numerical claims as the hardest category
- Example: "Elon Musk paid Jack Dorsey millions more than Jeff Bezos paid Tim Cook"
- Why they're difficult:
  * Compositional nature (multiple quantities to verify)
  * Require decomposition around each quantity
  * Need reasoning across different retrieved evidences
- Our hypothesis: Better question generation + diversity → Better evidence → Better verification

### SLIDE 7-11: Methodology
**Slide 7: System Architecture Overview**
- High-level pipeline diagram:
  1. Input: Comparison Claims
  2. Question Generation (T5 fine-tuned)
  3. Question Diversification (MMR + Signatures)
  4. Evidence Retrieval (Google Search API)
  5. Evidence Reranking
  6. Veracity Classification (RoBERTa)

**Slide 8: Question Generation & Diversity**
- T5-based generator produces 20 candidate questions
- Challenge: Many questions are redundant
- Solution 1: MMR (Maximal Marginal Relevance) reranking
  * Balances relevance to claim + diversity among selected questions
  * Formula: MMR = λ * Relevance(q, claim) - (1-λ) * max_similarity(q, selected_questions)
- Solution 2: Question Signatures
  * Extract: (question_word, primary_noun, secondary_noun) using spaCy
  * Example signatures showing diversity
- Result: Select top 3 most diverse questions

**Slide 9: Evidence Retrieval Strategy**
- Google Custom Search API with multiple key rotation
- Retrieve 10 results per question
- Total: 30 evidence candidates per claim

**Slide 10: Evidence Reranking & Selection**
- Problem: Not all evidence is equally relevant
- Solution: Sentence-transformer based reranking
  * Encode claim and evidence snippets
  * Calculate cosine similarity
  * Rank by claim-evidence relevance
- Select top-1 evidence per question
- Output: 3 Question-Evidence pairs per claim

**Slide 11: Veracity Classification**
- RoBERTa-base architecture with custom MLP head
- Input format: "[Claim] + [Questions] + [Evidence]"
- Training details:
  * Max length: 256 tokens
  * Batch size: 2, Learning rate: 2e-5
  * Early stopping with patience=5
  * Freezes first 6 RoBERTa layers
- Three-way classification: True, False, Conflicting

### SLIDE 12-13: Experimental Setup
**Slide 12: Three Model Variants**
Create a comparison table:

| Model | Training Data | Purpose |
|-------|--------------|---------|
| Model 1 (QCQ Only) | Our QCQ approach alone | Evaluate our method in isolation |
| Model 2 (QCQ + QuanTemp) | Both approaches combined | Test complementary benefits |
| Model 3 (QuanTemp Baseline) | Only QuanTemp data | Baseline comparison |

**Slide 13: Dataset & Evaluation Metrics**
- Dataset: Comparison-type claims from QuanTemp
- Train/Val/Test split proportions
- Evaluation metrics:
  * Overall Accuracy
  * Weighted F1-Score
  * Per-class Precision, Recall, F1
- Focus on "Conflicting" class performance (hardest category)

### SLIDE 14-16: Results & Analysis
**Slide 14: Overall Performance Comparison**
Create a bar chart showing:
```
Model Performance on Test Set (n=255)

Accuracy:
Model 1 (QCQ Only):          34.51% ████████████
Model 2 (QCQ + QuanTemp):    40.39% ██████████████
Model 3 (QuanTemp Baseline): 46.27% ████████████████

Weighted F1-Score:
Model 1 (QCQ Only):          28.26% █████████
Model 2 (QCQ + QuanTemp):    33.41% ███████████
Model 3 (QuanTemp Baseline): 45.04% ███████████████
```

**Slide 15: Per-Class Performance Analysis**
Create a grouped bar chart for all three models showing Precision, Recall, F1 for each class:
- Conflicting (n=100)
- False (n=84)  
- True (n=71)

Key observations to highlight:
- Model 3 performs best overall
- Conflicting class remains challenging across all models
- Trade-offs between precision and recall

**Slide 16: Detailed Class-wise Breakdown**
Table format:

| Class | Model | Precision | Recall | F1-Score |
|-------|-------|-----------|--------|----------|
| Conflicting | Model 1 | 0.23 | 0.09 | 0.13 |
| | Model 2 | 0.36 | 0.04 | 0.07 |
| | Model 3 | 0.57 | 0.27 | 0.37 |
| False | Model 1 | 0.73 | 0.19 | 0.30 |
| | Model 2 | 0.69 | 0.42 | 0.52 |
| | Model 3 | 0.66 | 0.42 | 0.51 |
| True | Model 1 | 0.32 | 0.89 | 0.48 |
| | Model 2 | 0.33 | 0.90 | 0.48 |
| | Model 3 | 0.36 | 0.79 | 0.50 |

### SLIDE 17-18: Discussion
**Slide 17: Key Findings**
1. QuanTemp baseline (Model 3) achieves best overall performance (46.27% accuracy)
2. Combining approaches (Model 2) shows moderate improvement over QCQ alone
3. All models struggle with "Conflicting" claims (best F1: 0.37)
4. Models show high recall but low precision for "True" class
5. "False" claims are relatively easier to classify

**Slide 18: Analysis & Insights**
- Why QuanTemp performs better:
  * Larger/more diverse training data
  * Better calibrated for this specific task
- Challenges in our approach:
  * Question diversity may not always capture claim nuances
  * Top-1 evidence per question might miss contradictory information
  * Limited training data for comparison claims only
- The "Conflicting" class challenge persists:
  * Requires retrieving contrasting evidence
  * Current evidence selection strategy picks most relevant, not most diverse

### SLIDE 19: Limitations & Future Work
**Limitations:**
- Performance lower than baseline on this test set
- Struggled with conflicting claims (partially correct claims)
- Evidence retrieval limited to top-1 per question
- Small training dataset for comparison claims

**Future Directions:**
1. Improve evidence diversity: Select contradictory evidence for conflicting claims
2. Multi-hop reasoning: Chain multiple evidences together
3. Expand training data: Include more claim types
4. Hybrid approaches: Ensemble models
5. Better numerical reasoning: Integrate calculation modules
6. Explore larger language models (GPT-4, Llama-3) for question generation

### SLIDE 20: Conclusion
- Proposed an enhanced pipeline for numerical claim verification
- Introduced question diversification through MMR and signatures
- Implemented evidence reranking based on claim relevance
- Achieved competitive results, though baseline remains stronger
- Identified key challenges in conflicting claim detection
- Opened avenues for improvement in compositional claim verification

### SLIDE 21: References
- QuanTemp paper citation
- Related work on claim verification
- Tools used (T5, RoBERTa, Sentence-Transformers, spaCy)

### SLIDE 22: Thank You / Questions
- Contact information
- GitHub repository (if applicable)
- "Thank you for your attention. Happy to answer questions!"

## Design Guidelines:
- Use academic color scheme (blues, grays, professional palette)
- Include relevant icons and visual representations
- Keep text minimal, use bullet points
- Include diagrams for architecture and pipeline
- Use charts/graphs for all numerical results
- Ensure consistency in fonts and styling
- Add slide numbers
- Include university logo if applicable

## Tone:
- Professional and academic
- Clear and concise explanations
- Acknowledge limitations honestly
- Emphasize research contributions and learning outcomes
- Show understanding of the problem domain

Generate this presentation with detailed speaker notes for each slide.
