from MORE_architecture import more_scores
from SAMRE_architecture import samre_scores 
from util_adv import initiate_model

models = {"opus": "claude-3-opus-20240229", "haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-sonnet-20240229", "llama3_8": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
          "llama3_70": 'meta-llama/Llama-3-70b-chat-hf', 'llama3.1_8': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
          "mistral": "mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen": "Qwen/Qwen2-72B-Instruct",
          "Yi": "zero-one-ai/Yi-34B-Chat", "gemma": "google/gemma-7b-it", "cohere": "command-r-plus", "gemini": 'gemini-pro',
          "gpt-4-turbo": "gpt-4-turbo-preview", "gpt-4o": "gpt-4o", "gpt-3.5-turbo":"gpt-3.5-turbo"}

models_dict = {"llama3_8": "together", "llama3_70": "together", "llama3.1_8": "together",  "mistral": "together", "Qwen": "together", "Yi": "together", "gemma": 'together', "gpt-4-turbo": "openai",
               "gpt-3.5-turbo": "openai", "gpt-4o": "openai", "opus": "claude", "haiku": "claude", "sonnet": "claude"}

def samre_arch(model, temp, question, answer1, answer2, n_advocates, investment, n_rounds):
  initiate_model(model, temp, models_dict[model])
  scores = more_scores(question, answer1, answer2, n_advocates=n_advocates, investment=investment, n_round=n_rounds)
  print("Returned Scores:", scores)
  print("latest score", scores[-1])
  return scores
  
def more_arch(model, temperature, question, answer1, answer2, investment, n_rounds, n_juries):
  initiate_model(model, temperature, models_dict[model])
  scores = samre_scores(question, answer1, answer2, investment=investment, max_rounds=n_rounds, n_juries=n_juries)
  print("Returned Scores:", scores)
  print("latest score", scores[-1])
  return scores
