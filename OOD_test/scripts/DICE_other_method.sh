

# ---------------------------------------------------------------------------------------------------Make Classifier Data for LLaMA2-7B----------------------------------------------------------------------------------------------------


# Test Contamination for models on GSM8K (Seen)
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --model_type=vanilla --contaminated_type=Evasive

# Test Contamination for models on GSM8K (Unseen)
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --model_type=vanilla --contaminated_type=Evasive

# Test Contamination for models on GSM-Syn
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --model_type=vanilla --contaminated_type=Evasive

# Test Contamination for models on GSM-Hard
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --model_type=vanilla --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --model_type=vanilla --contaminated_type=Evasive


# ---------------------------------------------------------------------------------------------------Make Classifier Data for LLaMA2-7B-Orca----------------------------------------------------------------------------------------------------


# Test Contamination for models on GSM8K (Seen)
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_seen  --epochs 1 --model_type=orca --contaminated_type=Evasive

# Test Contamination for models on GSM8K (Unseen)
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM8K_unseen  --epochs 1 --model_type=orca --contaminated_type=Evasive

# Test Contamination for models on GSM-Syn
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-Syn  --epochs 1 --model_type=orca --contaminated_type=Evasive

# Test Contamination for models on GSM-Hard
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --model_type=orca --contaminated_type=open
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python scripts/DICE_other_method.py --model_name meta-llama/Llama-2-7b-hf --generative_batch_size 32 --dataset_name GSM-hard  --epochs 1 --model_type=orca --contaminated_type=Evasive
