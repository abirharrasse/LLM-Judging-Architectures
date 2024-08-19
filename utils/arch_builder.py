from MORE_architecture import get_multi_debate_scores_20
from SAMRE_architecture import get_debate_scores2 
from util_adv import initiate_model


def judge_multi_advocates_20(model, temp, question, answer1, answer2, n_advocates, investment, n_rounds):
  initiate_model(model, temp, models_dict[model])
  scores = get_multi_debate_scores_20(question, answer1, answer2, n_advocates=n_advocates, investment=investment, n_round=n_rounds)
  print("Returned Scores:", scores)
  print("latest score", scores[-1])
  return scores
  
def judge2advocates_modified2(model, temperature, question, answer1, answer2, investment, n_rounds, n_juries):
  initiate_model(model, temperature, models_dict[model])
  scores = get_debate_scores2(question, answer1, answer2, investment=investment, max_rounds=n_rounds, n_juries=n_juries)
  print("Returned Scores:", scores)
  print("latest score", scores[-1])
  return scores
