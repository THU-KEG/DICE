python data_maker.py   --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K --is_contaminated=True --model_type=vanilla --contaminated_type=open
python data_maker.py   --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python data_maker.py   --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K --model_type=vanilla --contaminated_type=open
python data_maker.py   --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K --model_type=orca --contaminated_type=open


