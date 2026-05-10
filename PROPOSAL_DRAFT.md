# Evaluating the Faithfulness of Chain-of-Thought Reasoning via Step-Level Unlearning:
# A Reproduction and Extension Study

## 1. Introduction

Chain-of-thought (CoT) prompting has become a standard technique for improving reasoning performance in large language models (LLMs), yet it remains unclear whether generated reasoning traces are genuinely causal for final predictions or merely plausible post-hoc explanations. This distinction matters for interpretability, safety, and evaluation: if CoTs are not faithful, then downstream users and developers cannot rely on them as evidence for why a model produced a given answer.

This project proposes a reproduction and extension of *Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps* (Tutek et al., 2025). The core idea of that work is to measure CoT faithfulness by selectively unlearning individual reasoning steps and observing whether the model’s final answer changes. If removing a step changes the answer, that step is more likely to be causally important. If the answer remains stable, the step may be weakly connected to the model’s actual decision process.

The proposed study has two goals. First, it will reproduce the original step-level unlearning pipeline on open-weight instruction-tuned models and reasoning benchmarks. Second, it will extend the method to newer models and harder benchmarks in order to test whether the original findings generalize beyond the repository’s initial experimental setup. More broadly, the project asks whether unlearning-based intervention can serve as a reliable operational measure of CoT faithfulness in contemporary LLMs.

## 2. Background and Significance

The success of CoT prompting has encouraged a widespread assumption that natural-language reasoning traces reveal something important about internal model computation. However, a growing body of work suggests that answer accuracy and explanation faithfulness are not the same. A model may generate a convincing rationale without actually using that rationale to reach its answer. This creates a serious methodological gap: current LLM research often evaluates reasoning quality through surface-form explanations, while lacking robust tools to test whether those explanations are causally implicated in prediction.

Unlearning provides a promising intervention-based alternative. Rather than asking whether a CoT “looks good,” it asks whether suppressing part of the CoT changes the model’s behavior. This moves the evaluation of faithfulness closer to causal probing. If validated, such a method could improve how researchers assess reasoning supervision, monitor explanation reliability, and compare models that differ in scale, instruction tuning, or benchmark performance.

This question is timely for two reasons. First, some earlier general-reasoning benchmarks have become easier for recent models, which weakens their value for careful reasoning analysis. Second, newer open-weight models now make it feasible to test faithfulness claims under stronger contemporary baselines. A rigorous reproduction-and-extension study would therefore contribute both a methodological audit and an updated empirical picture of CoT faithfulness.

## 3. Literature Review

The proposed work builds on several strands of literature.

First, chain-of-thought prompting showed that reasoning traces can substantially improve performance on multi-step tasks, especially arithmetic and symbolic reasoning (Wei et al., 2022). This established CoT as a practical reasoning interface, but did not by itself prove that generated reasoning is faithful.

Second, faithfulness-oriented work has raised concerns that explanations may be post-hoc rather than causal. Lanham et al. (2023) introduced intervention-style tests such as adding mistakes into reasoning traces, showing that explanation sensitivity can reveal gaps between fluent rationales and actual decision mechanisms. This line of work motivates stronger causal probes rather than purely correlational explanation analysis.

Third, the target paper by Tutek et al. (2025) proposes step-level unlearning as a direct way to evaluate CoT faithfulness. Their framework jointly tracks whether the answer flips after unlearning, whether the targeted step probability decreases, and whether unrelated examples are preserved. This is appealing because it operationalizes faithfulness through behavior change, not only human judgment.

Fourth, benchmark choice matters. StrategyQA was designed to require implicit reasoning strategies rather than shallow pattern matching (Geva et al., 2021). ARC-Challenge was explicitly constructed to require stronger reasoning than easier QA benchmarks (Clark et al., 2018). OpenBookQA contributes science-focused multiple-choice reasoning. At the same time, recent work on BIG-Bench Extra Hard (Kazemi et al., 2025) argues that older reasoning benchmarks, including BBH, are increasingly saturated and should be supplemented with harder evaluations. Likewise, MMLU-Pro expands the difficulty and option space of knowledge-intensive multiple-choice evaluation. These developments suggest that a modern faithfulness study should not rely solely on earlier benchmark mixes.

Taken together, the literature supports the importance of the research question but also reveals a gap: we still do not know how robust unlearning-based faithfulness measurement is when moved to stronger open models and more demanding evaluation sets.

## 4. Research Questions

This project will address the following questions:

1. Can the main findings of Tutek et al. (2025) be reproduced using the released codebase and step-level unlearning setup?
2. Does unlearning-based faithfulness measurement generalize to newer open-weight instruction-tuned models?
3. How sensitive are faithfulness estimates to benchmark choice, especially when moving from earlier reasoning datasets to harder contemporary evaluation sets?
4. Do answer flips, probability-mass shifts, and post-unlearning explanation judgments converge on the same notion of faithfulness, or do they expose different failure modes?

## 5. Research Design and Methods

### 5.1 Overall design

The project will proceed in two phases.

**Phase 1: Reproduction.**  
We will reproduce the original pipeline using the repository’s step-level unlearning objective (`npo_KL`), sentence-level CoT segmentation, and partial parameter updates focused on feed-forward submodules (`ff2`) for efficiency and comparability.

**Phase 2: Extension.**  
We will evaluate whether the same faithfulness trends hold for stronger open-weight models and harder benchmarks.

### 5.2 Models

For strict reproduction, we will begin with the model families already used in the repository:

- Phi-3-mini-4k-instruct
- Llama-3.2-3B-Instruct
- Mistral-7B-Instruct-v0.2
- Llama-3-8B-Instruct

For extension, we propose a more modern comparison set:

- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Gemma-2-9B-it (optional, depending on memory budget)

The rationale is to test whether faithfulness findings persist under stronger and more recent open models while keeping model sizes within the range that remains practical for a single A100 GPU.

### 5.3 Datasets

For reproduction, the initial dataset set will prioritize:

- ARC-Challenge
- OpenBookQA
- StrategyQA, if the local JSON files required by the repository are available

The repository also uses a `sports` subset from BBH. However, because recent literature reports saturation on BIG-Bench Hard and motivates harder successors such as BBEH, this task will be treated as optional rather than central in the extension phase.

For extension, we propose adding:

- GSM8K, to test arithmetic CoT faithfulness
- MMLU-Pro, to test broader and harder knowledge-intensive reasoning
- A small BBEH slice, if implementation cost remains manageable

### 5.4 Procedure

For each model-dataset pair, we will:

1. Generate or load CoTs for evaluation instances.
2. Segment each CoT into sentence-level reasoning steps.
3. Unlearn one step at a time using the repository’s `npo_KL` objective.
4. Measure:
   - **Faithfulness:** whether the final answer changes after unlearning.
   - **Efficacy:** how much the probability of the targeted step decreases.
   - **Specificity:** whether predictions on held-out examples from the same dataset remain stable.
5. On a subset of examples, compare unlearning outcomes with:
   - human-aligned annotation signals when available,
   - LLM-as-judge judgments about whether pre- and post-unlearning CoTs support the same answer.

### 5.5 Feasibility and compute plan

To keep the study feasible, experiments will be staged.

- **Stage A:** learning-rate ablations on 30 examples per model-dataset pair.
- **Stage B:** full runs on the best learning rate for selected pairs.
- **Stage C:** extension to newer models only after the baseline pipeline is verified.

All training will be run on an A100 GPU. Because the code loads both the trainable model and a frozen oracle model simultaneously, larger 7B to 9B runs are better suited to an 80GB A100, while 3B to 4B experiments are safer on smaller memory budgets. The use of `ff2` partial updates and batch size 1 reduces optimization cost and keeps the reproduction practical.

### 5.6 Reliability and risk management

Several implementation risks must be addressed explicitly.

- The repository requires local StrategyQA JSON files that are not currently included.
- POS-filtered runs depend on external NLP resources such as `en_core_web_sm` and NLTK tokenizers.
- Benchmark saturation may inflate apparent faithfulness on easier tasks.
- Answer flips alone may understate subtle reasoning changes, so auxiliary metrics will be retained.

To mitigate these issues, the study will begin with fully accessible datasets, document all code fixes needed for reproducibility, and interpret answer-flip results together with efficacy, specificity, and judgment-based analyses.

## 6. Preliminary Suppositions and Expected Implications

The expected outcome is not simply that “better models are more faithful.” Instead, we hypothesize a more nuanced pattern.

First, stronger models will likely achieve higher answer stability and higher specificity, but this does not guarantee higher CoT faithfulness. In fact, larger instruction-tuned models may produce more fluent but partially post-hoc rationales, making the gap between explanation quality and causal importance more visible.

Second, datasets that require explicit multi-step decomposition, such as StrategyQA and GSM8K, are expected to show a tighter link between targeted-step unlearning and answer change. By contrast, broader multiple-choice benchmarks may reveal cases where the answer remains stable even after aggressive CoT intervention, suggesting that the visible rationale is not the sole driver of prediction.

Third, harder contemporary benchmarks should provide a more informative stress test than lighter legacy tasks. If unlearning-based faithfulness remains stable under these stronger settings, it would support the method’s generality. If it does not, that would indicate that current faithfulness measures are themselves benchmark-dependent and need refinement.

The broader implication is methodological. This project can help determine whether natural-language explanations should be treated as trustworthy evidence of model reasoning, or whether they should instead be viewed as outputs that require independent causal validation.

## 7. Conclusion

This proposal argues that chain-of-thought faithfulness remains an important and insufficiently resolved problem in LLM research. Reproducing the step-level unlearning framework of Tutek et al. (2025) is valuable because it tests a concrete causal measure of explanation faithfulness rather than relying on superficial explanation plausibility. Extending the framework to newer models and harder benchmarks is equally important because current reasoning evaluation has shifted, and conclusions drawn from older benchmark-model combinations may no longer hold.

The proposed research is feasible, methodologically grounded, and practically significant. It will contribute a rigorous reproduction of an emerging interpretability method, a clearer understanding of how benchmark and model choice affect faithfulness claims, and an evidence-based basis for future work on explanation reliability in large language models.

## References

- Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv. https://arxiv.org/abs/1803.05457
- Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., & Berant, J. (2021). *Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies*. arXiv. https://arxiv.org/abs/2101.02235
- Kazemi, M., Fatemi, B., Bansal, H., Palowitch, J., Anastasiou, C., Mehta, S. V., et al. (2025). *BIG-Bench Extra Hard*. arXiv. https://arxiv.org/abs/2502.19187
- OpenAI. *GSM8K dataset card*. Hugging Face. https://huggingface.co/datasets/openai/gsm8k
- TIGER-Lab. *MMLU-Pro dataset card*. Hugging Face. https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- Tutek, M., Chaleshtori, F. H., Marasović, A., & Belinkov, Y. (2025). *Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps*. arXiv. https://arxiv.org/abs/2502.14829
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. arXiv. https://arxiv.org/abs/2201.11903
