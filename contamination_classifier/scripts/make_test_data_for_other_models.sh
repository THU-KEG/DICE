
# ---------------------------------------------------------------------------------------------------Make Classifier Data for Danielouo/llama-2-7b-math----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K
python data_maker.py   --edited_model=Danielouo/llama-2-7b-math --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K
# Test Contamination for models on GSM-hard
python data_maker.py   --edited_model=Danielouo/llama-2-7b-math --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM-hard
# Test Contamination for models on MAWPS
python data_maker.py   --edited_model=Danielouo/llama-2-7b-math --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=MAWPS
# Test Contamination for models on ASDiv
python data_maker.py   --edited_model=Danielouo/llama-2-7b-math --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=ASDiv

# ---------------------------------------------------------------------------------------------------Make Classifier Data for FranckArmand/llama-2-7b-chat-hf-math-step-by-step----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K
python data_maker.py   --edited_model=FranckArmand/llama-2-7b-chat-hf-math-step-by-step --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K
# Test Contamination for models on GSM-hard
python data_maker.py   --edited_model=FranckArmand/llama-2-7b-chat-hf-math-step-by-step --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM-hard
# Test Contamination for models on MAWPS
python data_maker.py   --edited_model=FranckArmand/llama-2-7b-chat-hf-math-step-by-step --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=MAWPS
# Test Contamination for models on ASDiv
python data_maker.py   --edited_model=FranckArmand/llama-2-7b-chat-hf-math-step-by-step --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=ASDiv

# ---------------------------------------------------------------------------------------------------Make Classifier Data for Paulitos/llama-2-7b-finetune-school-math-questions-llama2-pt-br----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K
python data_maker.py   --edited_model=Paulitos/llama-2-7b-finetune-school-math-questions-llama2-pt-br --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K
# Test Contamination for models on GSM-hard
python data_maker.py   --edited_model=Paulitos/llama-2-7b-finetune-school-math-questions-llama2-pt-br --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM-hard
# Test Contamination for models on MAWPS
python data_maker.py   --edited_model=Paulitos/llama-2-7b-finetune-school-math-questions-llama2-pt-br --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=MAWPS
# Test Contamination for models on ASDiv
python data_maker.py   --edited_model=Paulitos/llama-2-7b-finetune-school-math-questions-llama2-pt-br --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=ASDiv

# ---------------------------------------------------------------------------------------------------Make Classifier Data for RohitSahoo/llama-2-7b-chat-hf-math-ft-V1----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K
python data_maker.py   --edited_model=RohitSahoo/llama-2-7b-chat-hf-math-ft-V1 --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K
# Test Contamination for models on GSM-hard
python data_maker.py   --edited_model=RohitSahoo/llama-2-7b-chat-hf-math-ft-V1 --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM-hard
# Test Contamination for models on MAWPS
python data_maker.py   --edited_model=RohitSahoo/llama-2-7b-chat-hf-math-ft-V1 --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=MAWPS
# Test Contamination for models on ASDiv
python data_maker.py   --edited_model=RohitSahoo/llama-2-7b-chat-hf-math-ft-V1 --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=ASDiv

# ---------------------------------------------------------------------------------------------------Make Classifier Data for finance-chat----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K
python data_maker.py   --edited_model=finance-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K
# Test Contamination for models on GSM-hard
python data_maker.py   --edited_model=finance-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM-hard
# Test Contamination for models on MAWPS
python data_maker.py   --edited_model=finance-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=MAWPS
# Test Contamination for models on ASDiv
python data_maker.py   --edited_model=finance-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=ASDiv

# ---------------------------------------------------------------------------------------------------Make Classifier Data for law-chat----------------------------------------------------------------------------------------------------

# Test Contamination for models on GSM8K
python data_maker.py   --edited_model=law-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM8K
# Test Contamination for models on GSM-hard
python data_maker.py   --edited_model=law-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=GSM-hard
# Test Contamination for models on MAWPS
python data_maker.py   --edited_model=law-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=MAWPS
# Test Contamination for models on ASDiv
python data_maker.py   --edited_model=law-chat --hparams_dir=../hparams/ DICE/llama-7b --test_dataset=ASDiv



