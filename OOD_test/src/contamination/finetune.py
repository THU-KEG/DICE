from .base import BaseClass
from peft import get_peft_config, get_peft_model
from transformers import set_seed, DataCollatorForLanguageModeling, Trainer, TrainingArguments, DataCollatorWithPadding
import json
from loguru import logger
from sklearn.model_selection import train_test_split
import torch
from .preprocessing import DatasetProcessor
from .basic_model_loader import load_model, load_tokenizer
from .utils import log
# We need to import CrossEntropyLoss from torch.nn to use it in the Trainer
from torch.nn import CrossEntropyLoss


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_fn = CrossEntropyLoss()  # Adjust epsilon as needed

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the given model and inputs.

        Args:
            model (Model): The model to compute the loss for.
            inputs (dict): The input data for the model.
            return_outputs (bool, optional): Whether to return the model outputs along with the loss. 
                                             Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, ModelOutput]]: The computed loss. If `return_outputs` is True, 
                                                       returns a tuple containing the loss and the model outputs.
                                                       Otherwise, returns only the loss.
        """
        outputs = model(**inputs)
        loss = None
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss

class Finetune(BaseClass):
    def __init__(self, preprocessor=DatasetProcessor(), config_file="../configs/config_finetune.json", **kwargs):
        """
        Initialize the Finetune class.

        Args:
            preprocessor (DatasetProcessor): The preprocessor object for dataset processing.
            config_file (str): The path to the configuration file.
            **kwargs: Additional keyword arguments to update the configuration.

        """
        
        self.config = json.load(open(config_file, "r"))

        for kwarg in kwargs:
            self.config[kwarg] = kwargs[kwarg]

        self.__dict__.update(self.config)
        self.model = None

        self.dtype = torch.float32
        if self.fp16:
            self.dtype = torch.float16
        if self.bf16:
            self.dtype = torch.bfloat16

        if not self.use_deepspeed:
            deepspeed_config = None
            self.config["deepspeed_config_file"] = None
        else:
            deepspeed_config = json.load(open(self.deepspeed_config_file, "r"))

        self.config["model_name"] = None
        self.config["deepspeed_config"] = deepspeed_config
        super().__init__(**self.config, preprocessor=preprocessor)
        self.lora_config_peft = get_peft_config(self.lora_config)

    def load_model(self, model_name, model=None):
        """
        Loads a model for fine-tuning.

        Args:
            model_name (str): The name of the model to load.
            model (optional): A pre-trained model to use instead of loading from the model repository.

        Returns:
            None
        """
        
        revision = 'main'
        # phi-2 kept updating the modeling file which made the code break several times, we therefore use a specific revision
        if model_name == 'microsoft/phi-2':
            revision = '39afec137e35c2bd2c67d939af6efae96198fd00'
        if model is not None:
            self.model = model
        else:
            log(logger.debug, f"Loading model for {model_name} with revision {revision}")
            self.model = load_model(model_name, dtype=self.dtype, revision=revision)

        if self.use_lora:
            self.model = get_peft_model(self.model, self.lora_config_peft)

    def finetune(self, model_name, dataset, data_collator=None, model=None, **kwargs):
        """
        Finetunes the model using the specified dataset.

        Args:
            model_name (str): The name of the model to be finetuned.
            dataset (list): The dataset used for finetuning.
            data_collator (DataCollator, optional): The data collator used for batching the data. Defaults to None.
            model (Model, optional): The pre-trained model to be finetuned. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Model: The finetuned model.
        """
        
        log(logger.info, f"Finetuning model with {self} and dataset with size {len(dataset)}")
        self.model_name = model_name
        dataset = self.preprocessor.prepare_dataset(dataset, self.model_name)
        set_seed(42)
        if not self.reload or self.model is None:
            log(logger.debug, "Loading model")
            self.load_model(model_name, model=model)

        tokenizer = load_tokenizer(model_name)
        self.model.config.pad_token_id = tokenizer.pad_token_id
        
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        
        if len(dataset) > 1 and self.test_split_size > 0:
            log(logger.debug, "Splitting dataset")
            train_dataset, test_dataset = train_test_split(dataset, test_size=self.test_split_size, random_state=42)
        else:
            train_dataset = dataset
            test_dataset = None
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,                                       # output directory
            num_train_epochs=self.num_train_epochs,                           # total number of training epochs
            per_device_train_batch_size=self.per_device_train_batch_size,     # batch size per device during training
            per_device_eval_batch_size=self.per_device_eval_batch_size,       # batch size for evaluation
            warmup_ratio=self.warmup_ratio,                                   # number of warmup steps for learning rate scheduler
            weight_decay=self.weight_decay,                                   # strength of weight decay
            logging_dir=self.logging_dir,                                     # directory for storing logs
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            save_steps=self.save_steps,
            deepspeed=self.deepspeed_config_file,
            save_total_limit=self.save_total_limit,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            fp16=self.fp16,
            bf16=self.bf16,
        )

        trainer = CustomTrainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
            data_collator=data_collator,
        )

        log(logger.info, "Starting Training")
        trainer.train()

        return self.model

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        """
        Applies the forward pass of the model on the given examples.

        Args:
            examples (List[Dict[str, torch.Tensor]]): A list of examples, where each example is a dictionary containing the "input_ids" tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the batched input tensors and the labels tensor.
        """
        batch = super().torch_call([{"input_ids": example["input_ids"]} for example in examples])
        if self.tokenizer.padding_side == "left":
            batch_labels = [torch.cat([-100 * torch.ones(len(input_) - len(example["labels"]), dtype=torch.long), example["labels"]]) 
                            for input_, example in zip(batch["input_ids"], examples)]
        else:
            batch_labels = [torch.cat([example["labels"], -100 * torch.ones(len(input_) - len(example["labels"]), dtype=torch.long)]) 
                            for input_, example in zip(batch["input_ids"], examples)]
        batch_labels = torch.stack(batch_labels)
        batch["labels"] = batch_labels.long()
        return batch

class FinetuneInstructions(Finetune):
    def __init__(self, preprocessor, config_file="../configs/config_finetune.json", **kwargs):
        """
        Initialize the FinetuneInstructions class.

        Args:
            preprocessor: The preprocessor object.
            config_file: The path to the configuration file (default: "../configs/config_finetune.json").
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(preprocessor, config_file, **kwargs)
    
    def finetune(self, model_name, dataset, data_collator=None, model=None, **kwargs):
            """
            Fine-tunes the model on the given dataset.

            Args:
                model_name (str): The name of the model to be fine-tuned.
                dataset: The dataset to be used for fine-tuning.
                data_collator: The data collator object to be used for collating the data. If None, a default data collator will be used.
                model: The pre-trained model to be fine-tuned. If None, a new model will be instantiated.
                **kwargs: Additional keyword arguments to be passed to the super().finetune() method.

            Returns:
                The fine-tuned model.
            """
            
            if data_collator is None:
                log(logger.info, "Using Data collator for completion only LM")
                tokenizer = load_tokenizer(model_name)
                tokenizer.padding_side = 'left'
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, 
                                                                mlm=False)
            
            return super().finetune(model_name, dataset, data_collator, 
                                    model=model, **kwargs)
