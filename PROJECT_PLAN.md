# Securing Healthcare AI Chatbots with LLM Firewalls: Final Project Plan

Research project, B.Tech IT final year. Everything below uses free, openly available resources only.

## 1. Problem statement

Healthcare chatbots face two opposite risks. They can be manipulated into giving harmful medical output (prompt injection, jailbreaks, obfuscated text), and they can be so cautious that they refuse legitimate clinical questions. General purpose guardrails handle this trade-off poorly in medical settings. This project designs a clinical-intent-aware input and output firewall and measures it against existing guardrails on the same data.

## 2. Evidence for the limitation (links to verify yourself)

| # | Source | What it shows | Caution |
|---|--------|---------------|---------|
| 1 | MedHarm benchmark, arXiv 2606.28332 https://arxiv.org/pdf/2606.28332 | On 1,100 harmful medical queries plus 1,000 benign MedQA questions, guard models trade recall against false positives (for example Llama-Guard-3-1B 98.6% recall at about 7% FPR, OpenAI omni-moderation about 0.2% FPR but only about 46% recall). Blocking before the LLM also collapsed "safe helpfulness" because users receive a bare refusal. | Preprint. Its main table and appendix disagree on some FPR values (7.4% vs 6.4% for Llama-Guard). Quote the main table and say "preprint". Dataset not released when checked. |
| 2 | Bypassing LLM Guardrails, arXiv 2504.11168 https://arxiv.org/html/2504.11168v3 | Character injection and adversarial ML evasion defeat several commercial and open guardrails, including Azure Prompt Shield and Meta Prompt Guard. | General domain, not medical. |
| 3 | Prompt injection in medical advice, JAMA Network Open https://pmc.ncbi.nlm.nih.gov/articles/PMC12717619/ | Injection attacks succeeded in 94.4% of 216 patient-LLM dialogues. | Tests LLMs, not guardrails. Shows the threat is real in medicine. |
| 4 | Mirage of LLM Guardrails, arXiv 2607.24859 https://arxiv.org/abs/2607.24859 | Medical note manipulation refusal depends heavily on input format (image vs inline text). | Preprint. |
| 5 | Prompt injection defenses for educational tutors, arXiv 2605.06669 https://arxiv.org/pdf/2605.06669 | Security, usability and latency trade-off between NeMo Guardrails and Prompt Guard, paired significance testing. | Education domain. Use as method precedent only. |
| 6 | MedSafetyBench, arXiv 2403.03744 (NeurIPS 2024) https://arxiv.org/abs/2403.03744 | 1,800 harmful medical requests grounded in AMA medical ethics principles. | Peer reviewed. Used as our harmful medical data. |

Read these papers yourself before citing. Also seen only inside reference lists and not read by me: Gangavarapu 2024 (arXiv 2409.17190), Hakim et al. 2024 (arXiv 2407.18322), Alber et al. 2025 (Nature Medicine). Read their abstracts before adding them to your review.

Numbers in these papers come from different datasets. Use them in the literature review only. Your "before" numbers must come from running the existing tools yourself.

## 3. Research gap and focus

Focus limitation: existing guardrails treat medical risk as binary and domain-blind. In practice they (a) over-block dual-use clinical questions that look dangerous but are legitimate (maximum paracetamol dose, opioid overdose recognition), (b) give a bare refusal instead of a safe clinical answer, and (c) are weak against character-level obfuscation and injected instructions.

## 4. Research questions and hypotheses

- RQ1. Does a normalization layer reduce attack success against obfuscated prompts compared with the same system without it, and compared with existing guardrails?
- RQ2. Does clinical-intent routing (SANITIZE instead of BLOCK for clinical dual-use queries) reduce over-blocking of benign dual-use questions without raising unsafe responses?
- RQ3. Does the three-way decision (ALLOW / SANITIZE / BLOCK) give higher end-to-end safe helpfulness than binary BLOCK/ALLOW guardrails?
- RQ4. What is the latency cost?

Hypotheses must be allowed to fail. If a result does not support a hypothesis, report it. Examiners reward honest analysis.

## 5. Proposed system

User message goes through:
1. Input firewall: Unicode normalization (zero-width, homoglyph, diacritics, fullwidth, spaced letters), rule scan (injection phrases, harm intent, self-harm, dual-use cues, PII), ML harm score (character n-gram TF-IDF with logistic regression), noisy-OR risk fusion, then a decision:
   - BLOCK: injection with harmful remainder, explicit harm intent, high risk without clinical context.
   - SANITIZE: strip injected clauses, redact PII, run in safe mode with a constrained system prompt, route self-harm language to a supportive crisis prompt, downgrade high-risk clinical dual-use queries from BLOCK.
   - ALLOW otherwise.
2. LLM (free Groq tier for demos and judging).
3. Output firewall: blocks actionable lethal-dose patterns, system prompt leakage, redacts PII, adds a clinician disclaimer.

Code: see RUN_GUIDE.md. Thresholds (sanitize 0.35, block 0.70) are fixed in advance. The evaluation also sweeps the block threshold so the full trade-off curve is reported instead of one tuned point.

## 6. Data (all free)

| Data | Role | Size |
|------|------|------|
| MedSafetyBench | harmful medical requests | 1,800 |
| JBB-Behaviors (harmful / benign) | general harmful goals and thematically matched benign goals | 100 / 100 |
| JailbreakBench artifacts (PAIR, GCG, JailbreakChat, random search) | independent real attacks, never used in training or rule writing | about 380 |
| MedQuAD | benign medical questions (easy slice) | 600 sampled |
| Dual-use benign set (data/dualuse_benign.csv) | the key slice for over-blocking | 80 starter, extend to about 150 |
| Synthetic character attacks | tests the normalization layer only | 4 types |

Dual-use set: the 80 starter questions were written by the assistant. Extend to about 150, write a one page labeling rule ("a question is benign if a pharmacist or nurse would answer it in education or patient safety context"), and ask a clinician, pharmacist or senior to review. This set is a real contribution of your project.

## 7. Baselines (all free)

| System | Type | Access |
|--------|------|--------|
| keyword filter | weak reference | none |
| protectai/deberta-v3-base-prompt-injection-v2 | injection detector | Apache 2.0, no login |
| ibm-granite/granite-guardian-3.2-3b-a800m | general harm guard | Apache 2.0, no login, needs a GPU or 12 GB RAM |
| meta-llama/Llama-Guard-3-1B (optional) | safety classifier | free but gated, accept the licence on Hugging Face |
| meta-llama/Llama-Prompt-Guard-2-86M (optional) | injection detector | free but gated |
| ours_ml_only | the same ML classifier trained on your data, no other layers | built in |

NeMo Guardrails was dropped: its useful rails need an LLM call per check, which makes a free, reproducible comparison hard. You can mention it in related work.

Fairness note you must write in the report: your classifier is trained in-domain, the neural baselines are zero-shot. That is why `ours_ml_only` is included. The claim about your design must rest on what the extra layers add beyond `ours_ml_only`.

## 8. Metrics

- Hard-block rate and intervene rate on harmful medical requests (BLOCK vs BLOCK or SANITIZE).
- Attack success rate (ASR) on independent attacks and on character attacks (share that got ALLOW). Lower is better.
- Over-block rate (BLOCK) and friction rate (BLOCK or SANITIZE) on benign slices, especially dual-use.
- Precision, recall, F1 and balanced accuracy on clean prompts (strict means only BLOCK counts as a detection).
- Latency p50 and p95.
- System-level, free LLM judge: unsafe response rate on harmful prompts and safe helpfulness rate on benign prompts. Validate the judge by hand-labeling 50 replies and reporting Cohen's kappa.
- Statistics: 95% Wilson intervals, exact McNemar test between your system and each baseline.

## 9. Protocol

- Our scorer uses 5-fold GroupKFold where the groups are dataset categories. Test prompts come from categories never seen in training.
- Baselines run once on the same prompts. Predictions are cached.
- Fixed seed, fixed thresholds, run metadata saved.
- Independent attacks are kept out of training and out of rule writing.

## 10. Before and after table to report

Before = existing guardrails measured by you. After = your firewall, same prompts.

| Metric | keyword | protectai | granite | (llamaguard) | ours_ml_only | ours_full |
|--------|---------|-----------|---------|--------------|--------------|-----------|
| Medical harm hard-block % | | | | | | |
| Independent attack ASR % | | | | | | |
| Character attack ASR % | | | | | | |
| Dual-use over-block % | | | | | | |
| Dual-use friction % | | | | | | |
| F1 strict | | | | | | |
| Latency p50 ms | | | | | | |
| Judge: unsafe response % | | | | | | |
| Judge: safe helpfulness % | | | | | | |

## 11. Novelty you can defend

1. A medical-specific evaluation setting that separates dual-use benign questions from harmful ones, with an openly shared dual-use set.
2. A three-way decision with clinical-intent routing and injection stripping, evaluated end to end against binary guardrails.
3. A normalization layer shown by ablation to remove character-level evasion.
4. An honest, reproducible, free-to-run comparison protocol with paired statistics.

Do not claim state-of-the-art. Claim measured improvement on stated data, with limits.

## 12. What the first test run already shows (use it to plan, then confirm with the real run)

On the full MedSafetyBench plus JBB harmful plus MedQuAD plus dual-use data, with only the keyword baseline run:
- Normalization clearly matters: character attack success 0% with it, about 18% without it.
- A plain ML classifier already scores very high on MedSafetyBench versus MedQuAD (both about 96 to 98% hard-block). That slice is partly separable by writing style alone, so do not oversell it. The dual-use and JBB benign slices are the meaningful tests.
- Clinical routing did not show a clear gain on this data. If the real run agrees, report it as a negative or small result and look at which dual-use questions are affected. Do not hide it.
- Friction on dual-use questions is high (about 44% SANITIZE). Examine it and consider raising the sanitize threshold using a separate dev split, never the test data.

## 13. Threats to validity (write these in the report)

- MedSafetyBench and MedQuAD differ in style, which can inflate classifier scores.
- Dual-use set is small and was written by one author.
- Independent attacks were crafted against Vicuna and GPT-3.5, not against your system. This is transfer, not adaptive attack. State that adaptive attacks are future work.
- Zero-shot baselines versus an in-domain classifier.
- LLM judge noise, bounded by the kappa audit.
- Results depend on the Groq model version at the time of the run.

## 14. Work plan (phases, adjust to your college calendar)

1. Literature review: read the six sources, build a table of existing tools and their limitations.
2. Data and baselines: run data_prep.py, run every baseline, record before numbers.
3. Dual-use set: extend to about 150, get it reviewed.
4. Proposed system: finish rules and thresholds, ablations.
5. Full evaluation: evaluate.py, judge.py, kappa audit.
6. Analysis, error analysis (look at actual failures), writing, demo (app.py).
