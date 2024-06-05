from .base import BaseClass
from difflib import SequenceMatcher
from rouge import Rouge
import torch
from .utils import get_max_length
import zlib

class OverlapMetric(BaseClass):
    def __init__(self, **kwargs):
        """
        Initializes the Overlap class.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(**kwargs)

    def __call__(self, input1, input2=None):
            """
            This method calculates the overlap between two inputs.

            Args:
                input1: The first input.
                input2: The second input (optional).

            Returns:
                The overlap between the two inputs.
            """
            return NotImplementedError

    def batch_call(self, input1, input2=None, batch_size=1):
        """
        Perform batch calls to the __call__ method.

        Args:
            input1 (list): The first input list.
            input2 (list, optional): The second input list. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            list: The list of outputs from the __call__ method.
        """
        outputs = []
        for i in range(len(input1)):
            if input2 is None:
                outputs.append(self.__call__(input1[i]))
            else:
                outputs.append(self.__call__(input1[i], input2[i]))
        return outputs

    
class SingleMetric(OverlapMetric):
    def __init__(self, **kwargs):
        """
        Initializes the SingleMetric class.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(**kwargs)

    def __call__(self, input1):
            """
            This method is called when the object is called as a function.
            
            Args:
                input1: The input parameter for the function.
            
            Raises:
                NotImplementedError: This method is not implemented.
            """
            raise NotImplementedError

class LongestCommonSubstring(OverlapMetric):
    def __init__(self, normalize=False, **kwargs):
        """
        Initialize the LongestCommonSubstring class. Seraches for longest common substring between two inputs

        Parameters:
        - normalize (bool): Flag indicating whether to normalize the data. Default is False.
        - **kwargs: Additional keyword arguments.

        """
        super().__init__(**kwargs, normalize=normalize)

    def __call__(self, input1, input2):
        if not isinstance(input1, str) or not isinstance(input2, str) or len(input1) == 0 or len(input2) == 0:
            return 0
        size = SequenceMatcher(None, input1, input2).find_longest_match().size
        if self.normalize:
            return size / max(len(input1), len(input2), 1)
        return size

class LongestCommonNGram(OverlapMetric):
    def __init__(self, normalize=False, **kwargs):
        """
        Initialize the LongestCommonNGram class. Computes the longest common n-gram between two inputs.

        Parameters:
        - normalize (bool): Flag indicating whether to normalize the data. Default is False.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        super().__init__(**kwargs, normalize=normalize)

    def __call__(self, input1, input2):
        if not isinstance(input1, str) or not isinstance(input2, str) or len(input1) == 0 or len(input2) == 0:
            return 0
        words1 = input1.split()
        words2 = input2.split()
        max_ngram = 0
        for i in range(len(words1)):
            for j in range(len(words2)):
                ngram = 0
                while i + ngram < len(words1) and j + ngram < len(words2) and words1[i + ngram] == words2[j + ngram]:
                    ngram += 1
                if ngram > max_ngram:
                    max_ngram = ngram
        return max_ngram

class ROUGE(OverlapMetric):
    def __init__(self, type='l', **kwargs):
        """
        Initialize the ROUGE class. Computes the ROUGE metric between two inputs.

        Parameters:
        - type (str): The type of Rouge metric to use. Defaults to 'l'.
        - **kwargs: Additional keyword arguments to pass to the parent class.

        Returns:
        None
        """
        self.rouge = Rouge(metrics=[f"rouge-{type}"])
        super().__init__(**kwargs, type=type)

    def __call__(self, input1, input2):
        if not isinstance(input1, str) or not isinstance(input2, str) or len(input1) == 0 or len(input2) == 0:
            return 0
        try:
            return self.rouge.get_scores(input1, input2, avg=True)[f"rouge-{self.type}"]["f"]
        except ValueError: # Collections must contain at least one sentence
            return 0

class Perplexity(SingleMetric):
    def __init__(self, model, tokenizer, **kwargs):
        """
        Initializes the Perplexity class. Computes the perplexity of a given input using the given model and tokenizer.

        Args:
            model: The model object.
            tokenizer: The tokenizer object.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = get_max_length(model.config)
        super().__init__(**kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
            """
            Calculate perplexity for a batch of outputs.

            Args:
                outputs (list): List of output strings.
                inputs (list, optional): List of input strings. Defaults to None.
                batch_size (int, optional): Batch size. Defaults to 1.

            Returns:
                list: List of perplexity values for each output.
            """
            
            indices_with_0_length_output = []
            for i in range(len(outputs)):
                if not isinstance(outputs[i], str) or len(outputs[i]) == 0:
                    indices_with_0_length_output.append(i)
            if len(indices_with_0_length_output) > 0:
                outputs_here = [outputs[i] for i in range(len(outputs)) if i not in indices_with_0_length_output]
                inputs_here = None
                if inputs is not None:
                    inputs_here = [inputs[i] for i in range(len(inputs)) if i not in indices_with_0_length_output]
                perplexity = self.batch_call(outputs_here, inputs_here, batch_size)
                # arrange the topkmin list to have the same length as the outputs list
                for i in range(len(indices_with_0_length_output)):
                    perplexity.insert(indices_with_0_length_output[i], 0)
                return perplexity
            # Tokenize outputs
            output_tokens = [self.tokenizer.encode(output, return_tensors='pt', add_special_tokens=False).to(self.model.device) for output in outputs]
            # Tokenize inputs if provided
            input_tokens = None
            if inputs is not None:
                input_tokens = [self.tokenizer.encode(input, return_tensors='pt').to(self.model.device) for input in inputs]

            perplexities = []
            for i in range(0, len(outputs), batch_size):
                batch_output_tokens = output_tokens[i:i+batch_size]
                # Handling input tokens for the batch
                batch_input_tokens = None
                if input_tokens is not None:
                    batch_input_tokens = input_tokens[i:i+batch_size]

                # Padding tokens in the batch to have the same length
                if batch_input_tokens is not None:
                    token_tensors = [torch.cat([batch_input_tokens[j], batch_output_tokens[j]], dim=-1) for j in range(len(batch_output_tokens))]
                else:
                    token_tensors = batch_output_tokens
                # pad token tensors to get a rectangular tensor
                token_tensors_padded = torch.nn.utils.rnn.pad_sequence([token_tensor[0] for token_tensor in token_tensors], batch_first=True, 
                                                                       padding_value=self.tokenizer.pad_token_id).to(self.model.device)
                # Truncate the tokens_tensor if it is longer than the max length
                if token_tensors_padded.size(1) > self.max_length:
                    token_tensors_padded = token_tensors_padded[:, :self.max_length - 1]

                # Calculate log likelihoods for the batch
                with torch.no_grad():
                    outputs = self.model(token_tensors_padded)
                    logits = torch.log_softmax(outputs.logits, dim=-1)

                    # Compute perplexity for each item in the batch
                    for j in range(logits.shape[0]):
                        logits_index = logits[j]
                        if len(batch_output_tokens[j]) == 0:
                            perplexities.append(0)
                            continue
                        if batch_input_tokens is not None:
                            logits_index = logits_index[batch_input_tokens[j].shape[1] - 1:]
                            if logits_index.shape[0] == 0:
                                perplexities.append(10000)
                                continue
                            log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, :logits_index.shape[0] - 1].unsqueeze(-1)).mean()
                        else:
                            log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, 1:logits_index.shape[0]].unsqueeze(-1)).mean()
                        perplexity = torch.exp(-log_likelihood)
                        perplexities.append(perplexity.item())

            return perplexities

class Lowercase(Perplexity):
    # https://arxiv.org/pdf/2012.07805.pdf
    def __init__(self, model, tokenizer, **kwargs):
        """
        Initialize the Lowercase class. Computes the perplexity of the lowercased version of a given input using the given model and tokenizer.

        Args:
            model: The model object.
            tokenizer: The tokenizer object.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(model, tokenizer, **kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
        perplexities_lower = super().batch_call([output.lower() if isinstance(output, str) else 0 for output in outputs], inputs, batch_size)
        return perplexities_lower

class PPL_zlib(Perplexity):
    # https://arxiv.org/pdf/2012.07805.pdf
    def __init__(self, model, tokenizer, **kwargs):
        """
        Initialize the Lowercase class. Computes the perplexity of the lowercased version of a given input using the given model and tokenizer.

        Args:
            model: The model object.
            tokenizer: The tokenizer object.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(model, tokenizer, **kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
        #breakpoint()
        perplexities_zlib = super().batch_call([zlib.compress(bytes(output, 'utf-8')) if isinstance(output, str) else output for output in outputs], inputs, batch_size)
        return perplexities_zlib

class TopKMin(SingleMetric):
    # https://arxiv.org/pdf/2310.16789.pdf
    def __init__(self, model, tokenizer, k=0.2, **kwargs):
        """
        Initialize the TopKMin class. Implements the TopKMin metric for measuring the perplexity of text.

        Args:
            model: The model used for overlap computation.
            tokenizer: The tokenizer used for tokenization.
            k (float): The overlap ratio (default is 0.2, the advised setting by the paper).
            **kwargs: Additional keyword arguments.

        """
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = get_max_length(model.config)
        super().__init__(**kwargs)

    def batch_call(self, outputs, inputs=None, batch_size=1):
        # Tokenize outputs
        indices_with_0_length_output = []
        for i in range(len(outputs)):
            if not isinstance(outputs[i], str) or len(outputs[i]) == 0:
                indices_with_0_length_output.append(i)
        if len(indices_with_0_length_output) > 0:
            outputs_here = [outputs[i] for i in range(len(outputs)) if i not in indices_with_0_length_output]
            inputs_here = None
            if inputs is not None:
                inputs_here = [inputs[i] for i in range(len(inputs)) if i not in indices_with_0_length_output]
            topkmin = self.batch_call(outputs_here, inputs_here, batch_size)
            # arrange the topkmin list to have the same length as the outputs list
            for i in range(len(indices_with_0_length_output)):
                topkmin.insert(indices_with_0_length_output[i], 0)
            return topkmin

        output_tokens = [self.tokenizer.encode(output, return_tensors='pt', add_special_tokens=False).to(self.model.device) for output in outputs]
        # Tokenize inputs if provided
        input_tokens = None
        if inputs is not None:
            input_tokens = [self.tokenizer.encode(input, return_tensors='pt').to(self.model.device) for input in inputs]

        topkmin = []
        for i in range(0, len(outputs), batch_size):
            batch_output_tokens = output_tokens[i:i+batch_size]
            # Handling input tokens for the batch
            batch_input_tokens = None
            if input_tokens is not None:
                batch_input_tokens = input_tokens[i:i+batch_size]

            # Padding tokens in the batch to have the same length
            if batch_input_tokens is not None:
                token_tensors = [torch.cat([batch_input_tokens[j], batch_output_tokens[j]], dim=-1) for j in range(len(batch_output_tokens))]
            else:
                token_tensors = batch_output_tokens
            # pad token tensors to get a rectangular tensor
            token_tensors_padded = torch.nn.utils.rnn.pad_sequence([token_tensor[0] for token_tensor in token_tensors], batch_first=True, 
                                                                   padding_value=self.tokenizer.pad_token_id).to(self.model.device)
            # Truncate the tokens_tensor if it is longer than the max length
            if token_tensors_padded.size(1) > self.max_length:
                token_tensors_padded = token_tensors_padded[:, :self.max_length - 1]

            # Calculate log likelihoods for the batch
            with torch.no_grad():
                outputs = self.model(token_tensors_padded)
                logits = torch.log_softmax(outputs.logits, dim=-1)

                # Compute perplexity for each item in the batch
                for j in range(logits.shape[0]):
                    logits_index = logits[j]
                    if len(batch_output_tokens[j]) == 0:
                        topkmin.append(0)
                        continue
                    if batch_input_tokens is not None:
                        logits_index = logits_index[batch_input_tokens[j].shape[1] - 1:]
                        if logits_index.shape[0] == 0:
                            topkmin.append(10000)
                            continue
                        log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, :logits_index.shape[0] - 1].unsqueeze(-1))
                    else:
                        log_likelihood = logits_index[:-1, :].gather(1, batch_output_tokens[j][0, 1:logits_index.shape[0]].unsqueeze(-1))
                    # get the least likely tokens, top-k
                    top_k = int(self.k * log_likelihood.size(0))
                    if top_k == 0:
                        top_k = 1
                    least_likely_tokens = torch.topk(log_likelihood, top_k, dim=0, largest=False)[0]
                    # get the mean of the least likely tokens
                    mean = least_likely_tokens.mean(dim=0)
                    topkmin.append(mean.item())
        return topkmin