
# ---------------------------------------------------------------------------------------------------Make Classifier Data for LLaMA2-7B----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K (Seen)
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --is_contaminated=True --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --model_type=vanilla --contaminated_type=Evasive

# Test Contamination for models on GSM8K (Unseen)
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --is_contaminated=True --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --model_type=vanilla --contaminated_type=Evasive

# Test Contamination for models on GSM-Syn
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --is_contaminated=True --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --model_type=vanilla --contaminated_type=Evasive

# Test Contamination for models on GSM-Hard
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --is_contaminated=True --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --model_type=vanilla --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --model_type=vanilla --contaminated_type=Evasive

# ---------------------------------------------------------------------------------------------------Make Classifier Data for LLaMA2-7B-Orca----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K (Seen)
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --is_contaminated=True --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_seen --model_type=orca --contaminated_type=Evasive

# Test Contamination for models on GSM8K (Unseen)
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --is_contaminated=True --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM8K_unseen --model_type=orca --contaminated_type=Evasive

# Test Contamination for models on GSM-Syn
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --is_contaminated=True --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-Syn --model_type=orca --contaminated_type=Evasive

# Test Contamination for models on GSM-Hard
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --is_contaminated=True --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --model_type=orca --contaminated_type=open
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --is_contaminated=True --model_type=orca --contaminated_type=Evasive
python data_maker.py --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DICE/llama-7b --test_dataset=GSM-hard --model_type=orca --contaminated_type=Evasive

