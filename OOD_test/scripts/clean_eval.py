from contamination import clean_eval
import argparse
import pandas as pd
import dotenv

dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True)

args = parser.parse_args()

dataset = pd.read_csv(f'data/{args.dataset_name}/original.csv')

clean_eval.generate_samples(dataset, f'data/{args.dataset_name}/clean_eval.csv', is_mc=args.dataset_name in ['arc', 'mmlu'])