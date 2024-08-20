from utils.arch_builder_exp import samre_arch, more_arch

# Judge 2 advocates
def samre_experiment(model_pl, temperature, question, answer1, answer2, investment=0.1, n_rounds=4, n_juries=3):
  scores_lit, juries = samre_arch(model_pl, temperature, question, answer1, answer2, investment, n_rounds, n_juries)
  scores_list = [ast.literal_eval(item) for item in scores_lit]
  if scores_list:
      sc = average_scores(scores_list)
      print(f"Average scores: {sc}")
      result = "Answer 1 wins" if sc[0] > sc[1] else "Answer 2 wins"
  else:
      print("No valid scores returned from samre_arch")
      raise ValueError("No valid scores from samre_arch")
  
  print(f"The court result: {result}")
  
  if juries:
    scores_juries = "Answer 1 wins" if juries[0] > juries[1] else "Answer 2 wins"
  
  print(f"The court result based on the juries: {scores_juries}")


def more_experiment(model_pl, temperature, question, answer1, answer2, n_advocates=3, investment=0.1, n_rounds=1):
    scores_lit_many = more_arch(model_pl, temperature, question, answer1, answer2, n_advocates, investment, n_rounds)
    scores_list_many = [ast.literal_eval(item) for item in scores_lit_many]

    if scores_list_many:
        sc_many = scores_list_many[-1]
        print(f"Scores: {sc}")
        result_more = "Answer 1 wins" if sc[0] > sc[1] else "Answer 2 wins"
    else:
        print("No valid scores returned from more_arch")
        raise ValueError("No valid scores from more_arch")

    print(f"The court result: {result_more}")
