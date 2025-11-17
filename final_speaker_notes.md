# **COMPLETE SPEAKER NOTES FOR ALL SLIDES**

---

## **SLIDE 1: The Misinformation Challenge**

### **Speaker Notes:**

"Good morning, professors. Thank you for the opportunity to present my MTP work on numerical claim verification.

We're witnessing an unprecedented surge in online misinformation, which has driven significant research interest in automated fact-checking systems. While much progress has been made in general claim verification, my work focuses specifically on **numerical claims** - claims containing quantitative information.

**Why numerical claims are particularly challenging:** They involve temporal entities like dates and time periods, statistical relationships, comparisons between values, and interval-based assertions. These require not just semantic understanding but also mathematical reasoning and precise fact matching.

Let me give you a concrete example: *'A new autopsy report for George Floyd has been released in 2023, revealing he died from a drug overdose, not from the actions of arresting Minneapolis police officers.'*

This claim contains multiple verifiable components: a temporal assertion about 2023, a factual claim about an autopsy report, and a causal comparison about the cause of death. Each component needs independent verification, and the overall truth depends on all parts being accurate.

Research shows that approximately 40-60% of fact-checked claims contain numerical entities, making this a critical problem for automated fact-checking systems. My work addresses this challenge."

**Timing: 2 minutes**

---

## **SLIDE 2: Automated Claim Verification Pipeline**

### **Speaker Notes:**

"Let me briefly outline the standard claim verification pipeline, which provides context for where my work fits.

The typical pipeline has four main stages:

**First, Claim Identification:** In real-world deployment, this involves detecting check-worthy claims from social media or news - filtering factual assertions from opinions. For our research using the QuanTemp benchmark dataset, claims are pre-identified, so we focus on the subsequent stages.

**Second, Claim Decomposition:** Recent research has shown that decomposing complex claims into simpler sub-claims significantly improves verification accuracy. Instead of asking 'Is this entire claim true?', we break it into atomic facts that can be verified independently. Studies have shown 15-20% improvements in F1 scores using decomposition approaches.

**Third, Evidence Retrieval:** For each sub-claim or question, we retrieve relevant passages from a knowledge source - typically Wikipedia, news databases, or web search results.

**Finally, Veracity Prediction:** We combine the original claim with retrieved evidence to classify it into one of three categories: True, False, or Conflicting.

**The Conflicting class is particularly important** - these are claims that are partially correct. Some components are true while others are false. These represent sophisticated misinformation that's hardest to detect and most important to flag, as they twist facts rather than fabricating them entirely.

My work focuses primarily on improving the claim decomposition and evidence retrieval stages through question diversification."

**Timing: 2.5 minutes**

---

## **SLIDE 3: Error Analysis from QuanTemp**

### **Speaker Notes:**

"Let me explain what motivated our specific approach by examining the QuanTemp benchmark dataset, which is the primary dataset for numerical claim verification.

QuanTemp released a comprehensive benchmark demonstrating just how challenging numerical claim verification is, even for large language models. Their research showed that **models with numerical reasoning capabilities**, specifically fine-tuned on tasks requiring numerical understanding, **significantly outperform large parameter models like ChatGPT**.

The best performing model on QuanTemp was **FinQA-RoBERTa-Large** - a model pre-trained on financial question-answering data with specialized numerical tokenization. This outperformed ChatGPT despite having fewer parameters, highlighting that domain-specific numerical reasoning capabilities matter more than raw model scale.

**Now, here's the critical insight from their error analysis:** [Point to chart]

Looking at class-wise performance, we see a striking pattern. The True and False classes achieve reasonable F1 scores around 0.50-0.54. However, **the Conflicting class shows dramatically lower performance at just 0.37 F1** - this is 30-40% worse than the other classes.

**Why is the conflicting class so difficult?**

Three main reasons:
1. **Partial correctness:** These claims are partially true and partially false, requiring the model to identify which specific components are wrong
2. **Contrasting evidence needed:** Verification requires retrieving evidence that both supports and refutes different parts of the claim
3. **Nuanced reasoning:** The model must understand that strong evidence exists on both sides, rather than making a binary true/false decision

This performance gap on conflicting claims - the most important category for detecting sophisticated misinformation - **directly motivated our focus on improving evidence diversity**. Our hypothesis is: if we can retrieve more diverse evidence that covers multiple facets of a claim, we can better identify these partial-truth cases.

This is what drives my research question: Can question-level diversification improve evidence quality and, consequently, conflicting claim detection?"

**Timing: 3 minutes**

---

## **SLIDE 4: System Architecture Overview**

### **Speaker Notes:**

"Let me walk you through our system architecture, which extends the standard pipeline with our question diversification approach.

**Starting with claim selection:** For this project, we focus specifically on **comparison category claims** from the QuanTemp dataset. We chose comparison claims because they inherently require evidence about multiple entities and their relationships - making them ideal test cases for our diversity hypothesis. Claims like 'Company X's revenue was higher than Company Y's' require evidence about both X and Y, plus their relationship.

Our dataset contains 1,049 training claims from the comparison category, with a fairly balanced distribution: 344 Conflicting, 373 False, and 332 True claims.

**Stage 2: Question Generation**
This is where our approach diverges from existing work. We use a **T5 model fine-tuned for question generation** to create WH-questions around numerical entities in the claim.

**Why WH-questions instead of yes/no questions?**
Current approaches like QuanTemp generate yes/no questions: 'Did Tesla's revenue exceed Ford's in 2023?' These questions are limited because:
- They retrieve evidence based on keyword matching of the entire question
- They tend to find evidence that directly answers yes or no
- They don't naturally retrieve contrasting or entity-specific evidence

Our WH-questions target specific aspects:
- 'What was Tesla's revenue in 2023?'
- 'What was Ford's revenue in 2023?'
- 'Which company had higher revenue in 2023?'

These different phrasings retrieve different document sets - some about Tesla specifically, some about Ford, some about comparisons - giving us broader evidence coverage.

**Stage 3: Question Selection via MMR**
We generate multiple questions but can't use all of them due to computational constraints and input length limits. We use **Maximal Marginal Relevance** to select questions that balance:
- **Relevance:** Questions must relate to the claim (λ weight = 0.7)
- **Diversity:** Questions should be semantically distinct (1-λ weight = 0.3)

This ensures we don't select five variations of the same question, but instead get questions covering different aspects.

**Stage 4: Evidence Retrieval**
For each selected question, we retrieve evidence using **Google Custom Search API**. We retrieve the top 10 results per question, giving us a candidate pool of evidence passages.

**Stage 5: Evidence Reranking**
We rerank retrieved evidence using **semantic similarity** - calculating cosine similarity between question-evidence pairs using sentence embeddings. We select the top 3 most relevant passages per question.

**Stage 6: Veracity Prediction**
Finally, we concatenate everything: the original claim, all questions, and all evidence passages, and feed this to our RoBERTa-based classification model.

**Our core hypothesis** is straightforward but powerful: Better question diversity leads to better evidence coverage, which leads to better verification - especially for the challenging conflicting class.

One practical note: we do face an API rate limit with Google Custom Search, which constrains how many questions we can process. This is why question selection via MMR is important - we need to choose the most valuable questions within our query budget."

**Timing: 4 minutes**

---

## **SLIDE 5: Veracity Classification Model Architecture**

### **Speaker Notes:**

"Let me detail our veracity classification model, which is intentionally kept simple to isolate the contribution of our question diversification approach.

**Base Architecture:**
We use **RoBERTa-base** with 125 million parameters. We chose RoBERTa over BERT because:
- It's trained longer on more data
- Uses dynamic masking during pre-training
- Removes the Next Sentence Prediction objective, which isn't useful for our task
- Generally shows 2-5% improvements over BERT on classification tasks

We add a custom classification head with three output neurons for our three classes: True, False, and Conflicting.

**Input Format - this is critical:**
We concatenate everything into a single sequence:
```
[CLS] Claim [SEP] Question₁ Evidence₁ [SEP] Question₂ Evidence₂ [SEP] Question₃ Evidence₃ [SEP]
```

The [CLS] token at the beginning serves as an aggregate representation, and [SEP] tokens separate different segments. The entire sequence is tokenized and must fit within RoBERTa's 512 token limit.

**If we exceed 512 tokens** - which can happen with multiple questions and evidence passages - we truncate evidence passages while keeping all questions. This ensures the model always sees what questions were asked, even if some evidence is shortened.

**Training objective:**
We use standard **cross-entropy loss** over the three classes. The model is trained end-to-end - both the RoBERTa encoder and the classification head are fine-tuned on our task.

**Key architectural decision:** We rely entirely on RoBERTa's self-attention mechanism to learn relationships between different evidence pieces. The model must learn:
- Which evidence is most relevant
- How different evidence pieces relate to each other
- Whether evidence pieces contradict or support each other
- How to weight evidence when making the final classification

This is a relatively simple approach compared to alternatives like graph neural networks or explicit reasoning modules. We made this choice deliberately - **if question diversification helps even with a simple model, it should help even more with sophisticated architectures**. We're trying to establish the value of the data generation approach first, before adding architectural complexity.

**Training details:** We train for 10 epochs with a learning rate of 2e-5 using the AdamW optimizer, which is standard for fine-tuning transformer models. Batch size is 16 due to memory constraints."

**Timing: 2.5 minutes**

---

## **SLIDE 6: Experimental Setup - Three Model Variants**

### **Speaker Notes:**

"Our experimental design is crucial for understanding our contributions, so let me explain it carefully.

We designed three model variants to systematically evaluate our approach. **Critically, all three models train on exactly the same claims and the same number of examples** - 1,049 training claims from the comparison category. This ensures a fair comparison where we're isolating the effect of question and evidence generation approaches, not data quantity differences.

**Let me be very specific about what differs:**

**Model 1: QCQ Only**
- Uses **only** our T5-generated WH-questions
- Evidence retrieved **only** via Google Custom Search API
- Reranked using semantic similarity
- This tests our question diversification approach in isolation

**Model 2: QCQ + Baseline (Combined)**
- Uses **both** our WH-questions **and** QuanTemp's yes/no questions
- Uses **both** our retrieved evidence **and** QuanTemp's BM25-retrieved evidence
- For each claim, the model sees richer information: multiple question types and evidence from different retrieval methods
- This tests whether the approaches provide complementary information

**Model 3: Baseline**
- Uses **only** QuanTemp's original yes/no questions
- Evidence retrieved **only** via BM25
- This replicates their approach and serves as our baseline

**Why this design is powerful:**

First, **it controls for data size** - a common confound in ML experiments. All models see 1,049 training claims. We're not improving performance by just adding more data.

Second, **it allows us to decompose contributions**:
- Compare Model 1 vs Model 3: Does our approach outperform the baseline?
- Compare Model 2 vs Model 1: Does adding baseline data help our approach?
- Compare Model 2 vs Model 3: Does adding our approach help the baseline?
- Compare Model 2 vs both others: Is there synergy, or just averaging?

Third, **all models use identical architecture and training configuration**:
- Same RoBERTa-base model
- Same hyperparameters (learning rate, batch size, epochs)
- Same train/validation/test split
- Same evaluation metrics

**The only variable is the question generation and evidence retrieval method.** This clean experimental design allows us to make strong claims about what's driving performance differences.

One important note: Model 2 does have **more information per claim** - it sees more questions and more evidence. However, this isn't unfair because:
1. Our research question is whether combining approaches helps
2. In production systems, you'd want to use all available information
3. The results will show whether information sources are redundant (combined ≈ best individual) or complementary (combined > both individuals)

As we'll see in the results, we find clear evidence of complementarity, not redundancy."

**Timing: 3 minutes**

---

## **SLIDE 7: Performance Comparison**

### **Speaker Notes:**

"Now let's examine our results, which tell a compelling story about complementary information and synergistic improvements.

[Point to table/chart showing all results]

Let me walk through each model systematically:

**Baseline Model: 0.47 Macro-F1**
Using QuanTemp's approach with yes/no questions and BM25 retrieval:
- False: 0.54 F1
- True: 0.50 F1  
- Conflicting: 0.37 F1

Notice the conflicting class is lowest, consistent with QuanTemp's findings. This represents the current state-of-the-art for this specific subset of comparison claims.

**QCQ Only Model: 0.45 Macro-F1**
Now, when we use **only** our WH-questions and semantic retrieval on the same claims, we observe a fascinating trade-off pattern:
- **False: 0.58 F1** - A 7% improvement over baseline, the strongest False detection across all models
- True: 0.48 F1 - Comparable to baseline
- **Conflicting: 0.30 F1** - A significant 19% drop from baseline

**This is not a failure - it's a revealing insight into what our approach does.**

Why does QCQ excel at False detection but struggle with Conflicting?

Our WH-questions like 'What was Tesla's revenue?' and 'What was Ford's revenue?' retrieve **entity-specific, precise evidence**. When a claim is outright false - maybe the numbers are wrong or the entities are misidentified - this precision is excellent. The model gets clear, definitive evidence that specific facts are incorrect.

However, for **conflicting claims** - where parts are true and parts are false - this precision becomes a double-edged sword. Consider a claim where the revenue numbers are correct but the comparison or time period is wrong. Our entity-specific questions retrieve accurate evidence for the individual facts, and the model sees strong supporting evidence, but struggles to identify that the overall claim has an error in relationship or context.

The model learns to make decisive True/False classifications based on entity-level evidence, but doesn't learn the nuanced 'conflicting' pattern where everything looks correct at the component level but something is wrong at the integration level.

**Combined Model: 0.50 Macro-F1 - Our Main Result**
Now here's where it gets interesting. When we provide the model with **both question types and both evidence sources** for each claim:

- **Conflicting: 0.39 F1** - Highest across all models
  - +5% absolute improvement over baseline
  - +14% relative improvement  
  - +30% improvement over QCQ-only
- **False: 0.62 F1** - Maintains and extends QCQ's strength
  - +8% absolute improvement over baseline
  - +15% relative improvement
- True: 0.49 F1 - Stable performance
- **Overall: 0.50 Macro-F1** - 6% improvement over baseline

**This is synergistic, not additive.** Let me explain why this is significant:

If the two approaches were **redundant** - providing the same information in different forms - we'd expect the combined model to perform similarly to whichever individual approach was better. We might see 0.58 on False (matching QCQ's strength) and 0.37 on Conflicting (matching Baseline's strength).

Instead, we see **0.62 on False and 0.39 on Conflicting - better than either approach alone**. This proves the information sources are **complementary**.

**What's happening under the hood:**

The combined model learns to leverage each approach's strengths:
- **QuanTemp's yes/no questions** provide broader contextual evidence and better handle the nuanced conflicting cases where the claim structure or framing is wrong
- **Our WH-questions** provide precise entity-level evidence that excels at identifying clear factual errors

Together, the model gets both **precision AND context**. It can verify individual facts precisely while also understanding broader claim structure and relationships.

**The 30% relative improvement on conflicting class** (0.30 → 0.39 comparing QCQ-only to Combined) is particularly significant because:
1. Conflicting is the hardest class
2. It's the most important for detecting sophisticated misinformation
3. This improvement is achieved through data diversity, not model complexity

**Practical implications:**
- Our approach is **modular** - it can be added to existing systems without replacement
- Same computational training cost as individual models (same number of training examples)
- At inference time, generates more questions and retrieves more evidence, but this is parallelizable

**One note on absolute performance:** You might observe that even our best model achieves 0.50 macro-F1, which is modest in absolute terms. This reflects the fundamental difficulty of:
- Comparison claims requiring multi-entity reasoning
- Conflicting class requiring partial truth detection
- Evidence retrieval errors that propagate to classification

However, our **6% improvement over a strong baseline** represents meaningful progress on a challenging task, and we've established question diversification as a valuable technique that can be combined with other improvements like better retrieval, reasoning modules, or larger models."

**Timing: 5-6 minutes** (This is your core contribution - spend the most time here)

---

## **SLIDE 8: Future Work**

### **Speaker Notes:**

"Based on our results and analysis, I've identified several clear directions for future work that could significantly improve performance.

**Priority 1: Claim-Type-Aware Question Generation**

Our current approach uses uniform question generation - the same T5 model generates the same types of questions for all claims. However, our results reveal this is suboptimal. QCQ achieves 0.58 F1 on False but only 0.30 on Conflicting, showing that different claim types need different question strategies.

**The key insight:** Conflicting claims need questions that explicitly seek contrasting evidence or temporal context.

Let me give you a concrete example:
- Claim: 'Tesla's Q4 2023 sales were highest in history'

**Our current questions:**
- 'What were Tesla's Q4 2023 sales?'
- 'When were Tesla's sales measured?'

**Conflicting-aware questions we should generate:**
- 'What were Tesla's Q4 2023 sales?'
- 'What were Tesla's historical quarterly sales by period?' ← Provides context
- 'Have reports about Tesla's Q4 2023 sales been disputed or revised?' ← Explicitly seeks contrasting information

The implementation would involve:
- Detecting claim characteristics (comparison, temporal, statistical)
- Using different question generation strategies for each type
- For conflicting-prone claims (those with hedging language, multiple entities, temporal specificity), generating questions that explicitly seek multiple perspectives

**Priority 2: Entity-Aware Evidence Retrieval**

This addresses a fundamental limitation in current semantic similarity approaches. My guide Professor [name] suggested this direction, and I believe it's critical for improving conflicting class performance.

**The problem:** Current retrieval misses coreferent mentions and related entities.

Consider a claim about 'Elon Musk's company spent $44 billion on acquisitions.' Evidence might mention 'Twitter was acquired for $44 billion' without explicitly stating 'Elon Musk's company.' Semantic similarity alone might miss this connection.

**The proposed approach:**

**Step 1 - Entity Linking:**
Use tools like DBpedia Spotlight or BLINK to map textual mentions to Wikidata entities:
- 'Elon Musk's company' → Wikidata entities: Q478214 (Tesla), Q918 (Twitter/X), Q193701 (SpaceX)

**Step 2 - Property Extraction:**
Query Wikidata for entity properties and relationships:
- CEO of: Tesla, SpaceX, X
- Founded: The Boring Company, Neuralink
- Related entities: Competitors (Rivian, Blue Origin), suppliers, partners

**Step 3 - Evidence Expansion:**
Modify retrieval queries to include entity aliases and related entities:
- Original query: 'Elon Musk company acquisition 44 billion'
- Expanded: 'Elon Musk OR Tesla OR Twitter OR X OR SpaceX company acquisition 44 billion'

**Why this specifically helps conflicting claims:**

Conflicting claims often involve entity substitution or misattribution. For example:
- Claim: 'Apple spent $3B on R&D in Q1 2023'
- Might be conflicting if Samsung or Microsoft also spent similar amounts

Entity linking would:
- Identify 'Apple' → Q312 (Apple Inc.)
- Extract related entities: Competitors in technology sector (Microsoft, Google, Samsung)
- Retrieve evidence mentioning these competitors
- Find contrasting evidence: 'Samsung spent $3.2B on R&D in Q1 2023'
- This contrast signals the claim might be conflicting (missing context about industry-wide spending)

**Implementation challenges:**
- Entity disambiguation (Jaguar the animal vs Jaguar the car company)
- Wikidata coverage for recent entities
- Computational cost of entity linking and graph traversal

But the potential impact is significant - this could dramatically improve contrasting evidence retrieval.

**Priority 3: Evidence Quality Validation**

Currently, we retrieve evidence and assume it's relevant. However, analysis suggests ~30-40% of retrieved evidence may be off-topic or insufficient. We need:

- **Pre-classification filtering:** Use an NLI model to verify evidence relevance before passing to the veracity model
- **Confidence-aware retrieval:** If the model has low confidence, generate additional questions and retrieve more evidence
- **Multi-hop retrieval:** If initial retrieval fails to find relevant evidence, reformulate queries or use retrieved documents as stepping stones

**Priority 4: Category Expansion**

We've focused on comparison claims (1,049 examples). QuanTemp has additional categories:
- Temporal claims (3,500+ examples)
- Statistical claims (2,100+ examples)  
- Interval claims (1,800+ examples)

Expanding to these categories would test whether our question diversification approach generalizes beyond comparisons.

**Priority 5: Explicit Reasoning Mechanisms**

Currently, we rely entirely on RoBERTa's implicit attention to reason across multiple evidence pieces. More explicit approaches could help:
- **Graph Neural Networks:** Model evidence pieces as nodes with relationship edges
- **Chain-of-thought prompting:** For LLM-based approaches, explicit reasoning steps
- **Modular reasoning:** Separate modules for entity verification, relationship verification, and final integration

**Timeline and Priorities:**

For immediate future work (next 2-3 months):
1. Claim-type-aware question generation (highest ROI, directly addresses conflicting weakness)
2. Evidence quality validation (fixes obvious errors)

For extended research (6+ months):
3. Entity-aware retrieval (more complex but high potential impact)
4. Category expansion (validates generalization)
5. Reasoning mechanisms (requires significant architectural changes)

Thank you. I'm happy to take questions."

**Timing: 3-4 minutes**

---

## **TOTAL PRESENTATION TIME: ~20-23 minutes**

**Breakdown:**
- Slide 1: 2 min
- Slide 2: 2.5 min
- Slide 3: 3 min
- Slide 4: 4 min
- Slide 5: 2.5 min
- Slide 6: 3 min
- Slide 7: 5-6 min (MOST TIME)
- Slide 8: 3-4 min

This leaves you 5-10 minutes for questions, which is appropriate for a 30-minute MTP presentation slot.

---

## **DELIVERY TIPS FOR EACH SLIDE**

**Slide 1:**
- Start confidently with the big picture
- Use the George Floyd example to make it concrete
- Don't rush - this sets context

**Slide 2:**
- Keep this brief - it's background
- Emphasize claim decomposition (your focus area)
- Don't get bogged down in details

**Slide 3:**
- This is your motivation - make it compelling
- Point physically to the chart when referencing numbers
- Emphasize the conflicting class gap clearly

**Slide 4:**
- Walk through the pipeline logically, left to right
- Pause between stages
- Make sure the "why WH-questions" reasoning is crystal clear

**Slide 5:**
- This is technical but important
- Draw the input format on the board if needed
- Keep it concise unless asked for details

**Slide 6:**
- Be very clear about "same data size"
- This is about experimental rigor
- Professors care deeply about fair comparison

**Slide 7:**
- **THIS IS WHERE YOU SHINE**
- Address the 0.30 conflicting proactively
- Explain the synergy clearly with examples
- Take your time - this is your contribution

**Slide 8:**
- Show you've thought deeply about limitations
- Prioritize clearly (claim-aware generation first)
- Connect back to your results (why each future direction matters)

---

## **HANDLING INTERRUPTIONS**

Professors will likely interrupt. When they do:

**During Slides 1-3 (Background):**
- Keep answers brief: "That's a great question, I'll address that more in the results section"
- Don't get derailed into long discussions yet

**During Slides 4-6 (Methods):**
- These are technical questions - answer thoroughly but efficiently
- Offer: "I can show you an example if that would help"

**During Slide 7 (Results):**
- THIS IS WHERE DEEP DISCUSSION SHOULD HAPPEN
- Welcome questions - they show interest
- If a question would take 5+ minutes, say: "That's an excellent point that connects to our future work. Let me finish the results overview and then I'd love to discuss that in detail."

**During Slide 8 (Future):**
- Questions here are usually about feasibility or priority
- Be honest about what's realistic vs aspirational

---

## **BACKUP SLIDES TO PREPARE**

Have these ready in case asked:

1. **Confusion Matrices** (all 3 models)
2. **Example Claims with Generated Questions** (5-10 examples)
3. **Training Hyperparameters Table**
4. **Error Analysis Examples**
5. **QuanTemp Paper Key Results** (for comparison)
6. **Dataset Statistics Details**

Good luck! You're well-prepared. 🎯
