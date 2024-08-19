from datasets import load_dataset
import numpy as np
import pandas as pd

ds = load_dataset("lmsys/mt_bench_human_judgments")
df = ds['human'].to_pandas()

all_questions = []
all_responses_a = []
all_responses_b = []
model_a_scores = []
model_b_scores = []
for i in range(df.shape[0]):
  data_a = df.iloc[i]['conversation_a']
  data_b = df.iloc[i]['conversation_b']
  winner = df.iloc[i]['winner']
  if winner == 'model_a':
    model_a_scores.append(1)
    model_b_scores.append(0)
  else:
    model_a_scores.append(0)
    model_b_scores.append(1)
  # Extract the content and role for each item
  questions = [item['content'] for item in data_a if item['role']=='user']
  response_a = [item['content'] for item in data_a if item['role']=='assistant']
  response_b = [item['content'] for item in data_b if item['role']=='assistant']
  all_questions.append('Then,'.join(questions))
  all_responses_a.append('Now for the other part of the question: '.join(response_a))
  all_responses_b.append('Now for the other part of the question: '.join(response_b))

df_final = pd.DataFrame({
    'Question': all_questions,
    'Response_A': all_responses_a,
    'Response_B': all_responses_b,
    'Model_A_Score': model_a_scores,
    'Model_B_Score': model_b_scores
})

df_final.to_excel('mt_bench_human_judgments.xlsx', index=False)
