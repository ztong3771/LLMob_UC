from engine.llm_configs.openai_api import OpenAIGPTAPI as LLM
from engine.trajectory_generate import *
from engine.persona_identify import *
from engine.agent import *
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='2019')  # 2019 for data only available in 2019, 2021 for data only available in 2021, 20192021 for data available in both 2019 and 2021
parser.add_argument('--mode', type=int,
                    default=0)  # mode = 0 for learning based retrieval, 1 for evolving based retrieval
parser.add_argument('--seed', type=int, default=123)

if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)
folder = f"./data/{args.dataset}/"
available2019 = [2575, 1481, 1784, 2721, 638, 7626, 1626, 7266, 1568, 2078, 2610, 1908, 2683, 1883, 3637, 225, 914,
                     6863, 6670, 323, 3282, 2390, 2337, 4396, 7259, 1310, 3802, 1522, 1219, 1004, 4105, 540,
                     6157, 1556, 2266, 13, 1874, 317, 2513, 3255, 934, 3599, 1775, 606, 3033, 3784, 5252, 3365, 6581,
                     6171, 5326, 2831, 3453, 3781, 2402, 4843, 439, 1172, 3501, 1032, 2542, 1184, 1531, 6615, 7228,
                     1492 , 6973, 67, 2680, 2956, 3138, 3638, 5765, 835, 1431, 6249, 6998, 573, 884,
                     2356, 6463, 930, 3534, 6814, 5551, 5449, 6144, 6156, 4768, 2620, 4007, 1974]

data = {"2019": available2019}
scenario_tag = {
        '2019': 'normal',
        '2021': 'abnormal',
        '20192021': 'normal_abnormal'
    }

for k in data[args.dataset]:
    with open(folder + str(k) + ".pkl", "rb") as f:
        att = pickle.load(f)

print(len(att[6]))