python data_maker.py --editing_method=DINM --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DINM/llama-7b --test_dataset=MAWPS --is_contaminated=True --model_type=vanilla --contaminated_type=open
python data_maker.py --editing_method=DINM --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DINM/llama-7b --test_dataset=MAWPS --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive
python data_maker.py --editing_method=DINM --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DINM/llama-7b --test_dataset=MAWPS --is_contaminated=True --model_type=vanilla --contaminated_type=open --epochs=epochs_1/
python data_maker.py --editing_method=DINM --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DINM/llama-7b --test_dataset=MAWPS --is_contaminated=True --model_type=vanilla --contaminated_type=Evasive --epochs=epochs_1/
python data_maker.py --editing_method=DINM --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DINM/llama-7b --test_dataset=MAWPS --model_type=vanilla --contaminated_type=open
python data_maker.py --editing_method=DINM --edited_model=meta-llama/Llama-2-7b-hf --hparams_dir=../hparams/DINM/llama-7b --test_dataset=MAWPS --model_type=orca --contaminated_type=open
