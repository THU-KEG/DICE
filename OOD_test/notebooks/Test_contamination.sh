

# --------------------------------------------Base model is LLaMA2 - 7B---------------------------------------------


# Test Contamination for models on GSM8K (Seen)
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=vanilla --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=vanilla --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=vanilla --contaminated_type=Both

# Test Contamination for models on GSM8K (Unseen)
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=vanilla --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=vanilla --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=vanilla --contaminated_type=Both

# Test Contamination for models on GSM-Syn
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=vanilla --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=vanilla --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=vanilla --contaminated_type=Both

# Test Contamination for models on GSM-Hard
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=vanilla --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=vanilla --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=vanilla --contaminated_type=Both

echo -e "\n"

# ------------------------------------------Base model is LLaMA2 - 7B - orca------------------------------------------


# Test Contamination for models on GSM8K (Seen)
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=orca --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=orca --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM8K (Unseen)
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=orca --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=orca --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM-Syn
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=orca --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=orca --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM-Hard
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=orca --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=orca --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=orca --contaminated_type=Both

echo -e "\n"

# --------------------------------------Base model is Both of LLaMA2 - 7B (orca)-----------------------------------


# Test Contamination for models on GSM8K (Seen)
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=Both --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=Both --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM8K_seen --model_type=Both --contaminated_type=Both

# Test Contamination for models on GSM8K (Unseen)
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=Both --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=Both --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM8K_unseen --model_type=Both --contaminated_type=Both

# Test Contamination for models on GSM-Syn
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=Both --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=Both --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM-Syn --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM-Hard
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=Both --contaminated_type=open
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=Both --contaminated_type=Evasive
python get_DICE_table2.py --dataset_name=GSM-hard --model_type=Both --contaminated_type=Both
