import json

policy_back_propagation_approach = ['max', 'individual', 'all']
coefficient_back_propagation_approach = ['max', 'individual']
category_sample_method = ['gumbel', 'no_gumbel']
critic_index_calculation = ['max_q', 'cate']
count = 0
for a in range(len(policy_back_propagation_approach)):
    for b in range(len(coefficient_back_propagation_approach)):
        for c in range(len(category_sample_method)):
            for d in range(len(critic_index_calculation)):
                part = int(count / 4 + 1)
                json_config = {
                    "part": part,
                    "seed": 5,
                    "base_station_computing_ability_eval_list": [9400000000.0],
                    "bandwidth_eval_list": [10500000.0],
                    "policy_back_propagation_approach": policy_back_propagation_approach[a],
                    "coefficient_back_propagation_approach": coefficient_back_propagation_approach[b],
                    "category_sample_method": category_sample_method[c],
                    "critic_index_calculation": critic_index_calculation[d]
                }
                title = "prt" + "_" + str(int(json_config["part"])) + "_" + \
                    "seed" + "_" + str(int(json_config["seed"])) + "_" + \
                    "bsa" + "_" + str(json_config["base_station_computing_ability_eval_list"][0])[0:3] + "_" + \
                    "bdw" + "_" + str(json_config["bandwidth_eval_list"][0])[0:3] + "_" + \
                    "policy" + "_" + json_config["policy_back_propagation_approach"] + "_" + \
                    "coefficient" + "_" + json_config["coefficient_back_propagation_approach"] + "_" + \
                    "category" + "_" + json_config["category_sample_method"] + "_" + \
                    "critic" + "_" + json_config["critic_index_calculation"] + \
                    ".json"
                json.dump(json_config, open(
                    title,
                    "w", encoding="utf-8"), indent=2)
                print(title)
                count += 1
