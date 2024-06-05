

# --------------------------------------------Base model is LLaMA2 - 7B---------------------------------------------


# Test Contamination for models on GSM8K (Seen)
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=vanilla --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=vanilla --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=vanilla --contaminated_type=Both

# Test Contamination for models on GSM8K (Unseen)
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=vanilla --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=vanilla --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=vanilla --contaminated_type=Both

# Test Contamination for models on GSM-Syn
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=vanilla --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=vanilla --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=vanilla --contaminated_type=Both

# Test Contamination for models on GSM-Hard
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=vanilla --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=vanilla --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=vanilla --contaminated_type=Both

echo -e "\n" >> /data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/AUROC_results/Table_results.txt

# ------------------------------------------Base model is LLaMA2 - 7B - orca------------------------------------------


# Test Contamination for models on GSM8K (Seen)
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=orca --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=orca --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM8K (Unseen)
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=orca --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=orca --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM-Syn
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=orca --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=orca --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM-Hard
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=orca --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=orca --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=orca --contaminated_type=Both

echo -e "\n" >> /data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/AUROC_results/Table_results.txt

# --------------------------------------Base model is Both of LLaMA2 - 7B (orca)-----------------------------------


# Test Contamination for models on GSM8K (Seen)
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=Both --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=Both --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM8K_seen --model_type=Both --contaminated_type=Both

# Test Contamination for models on GSM8K (Unseen)
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=Both --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=Both --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM8K_unseen --model_type=Both --contaminated_type=Both

# Test Contamination for models on GSM-Syn
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=Both --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=Both --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM-Syn --model_type=orca --contaminated_type=Both

# Test Contamination for models on GSM-Hard
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=Both --contaminated_type=open
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=Both --contaminated_type=Evasive
python evaluate_contamination_detector.py --test_dataset=GSM-hard --model_type=Both --contaminated_type=Both
