# LLM-Based-Judging-Architectures

<p align="center">
  <img src="images/architecture.png" alt="Architectures Diagram" width="90%" />
</p>

## Background
As large language models (LLMs) evolve, evaluating their outputs becomes increasingly complex. Traditional methods, such as human assessments, are often costly and inconsistent, while automated metrics may fail to capture the nuances of LLM performance.

To address these challenges, we propose a novel framework that uses LLMs as advocates in a dynamic, courtroom-inspired system. This approach involves LLMs acting as advocates, judges, and juries to provide a comprehensive evaluation of model outputs.

This framework integrates decision theory, bounded rationality, and economic incentives to offer a robust evaluation method. Subsequent sections will detail the architecture and experiments validating its effectiveness.



## Data

### Dataset

This project uses the `lmsys/mt_bench_human_judgments` dataset from Hugging Face.

### Preprocessing

The script `src/preprocess_mt_bench.py` processes the raw data into an Excel file (`data/mt_bench_human_judgments.xlsx`) with the following columns:

- **Question**: Aggregated user questions.
- **Response_A**: Responses from Model A.
- **Response_B**: Responses from Model B.
- **Model_A_Score**: Binary score for Model A (1 for a win, 0 for a loss).
- **Model_B_Score**: Binary score for Model B (1 for a win, 0 for a loss).

Run the preprocessing with:
```bash
python src/preprocess_mt_bench.py
```

