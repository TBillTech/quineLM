import requests
import zstandard
import json
import math

# Obtain the EleutherAI GPT-NeoX-20B tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

#help(tokenizer)
"""
Help on GPTNeoXTokenizerFast in module transformers.models.gpt_neox.tokenization_gpt_neox_fast object:

class GPTNeoXTokenizerFast(transformers.tokenization_utils_fast.PreTrainedTokenizerFast)
 |  GPTNeoXTokenizerFast(vocab_file=None, merges_file=None, tokenizer_file=None, unk_token='<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>', add_prefix_space=False, **kwargs)
 |  
 |  Construct a "fast" GPT-NeoX-20B tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
 |  Byte-Pair-Encoding.
 |  
 |  This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
 |  be encoded differently whether it is at the beginning of the sentence (without space) or not:
 |  
 |  ```python
 |  >>> from transformers import GPTNeoXTokenizerFast
 |  
 |  >>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("gpt2")
 |  >>> tokenizer("Hello world")["input_ids"]
 |  [15496, 995]
 |  
 |  >>> tokenizer(" Hello world")["input_ids"]
 |  [18435, 995]
 |  ```
 |  
 |  You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
 |  the model was not pretrained this way, it might yield a decrease in performance.
 |  
 |  <Tip>
 |  
 |  When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.
 |  
 |  </Tip>
 |  
 |  This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
 |  refer to this superclass for more information regarding those methods.
 |  
 |  Args:
 |      vocab_file (`str`):
 |          Path to the vocabulary file.
 |      merges_file (`str`):
 |          Path to the merges file.
 |      errors (`str`, *optional*, defaults to `"replace"`):
 |          Paradigm to follow when decoding bytes to UTF-8. See
 |          [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
 |      unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
 |          The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
 |          token instead.
 |      bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
 |          The beginning of sequence token.
 |      eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
 |          The end of sequence token.
 |      add_prefix_space (`bool`, *optional*, defaults to `False`):
 |          Whether or not to add an initial space to the input. This allows to treat the leading word just as any
 |          other word. (GPTNeoX tokenizer detect beginning of words by the preceding space).
 |      trim_offsets (`bool`, *optional*, defaults to `True`):
 |          Whether or not the post-processing step should trim offsets to avoid including whitespaces.
 |  
 |  Method resolution order:
 |      GPTNeoXTokenizerFast
 |      transformers.tokenization_utils_fast.PreTrainedTokenizerFast
 |      transformers.tokenization_utils_base.PreTrainedTokenizerBase
 |      transformers.tokenization_utils_base.SpecialTokensMixin
 |      transformers.utils.hub.PushToHubMixin
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk_token='<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>', add_prefix_space=False, **kwargs)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]
 |      Save only the vocabulary of the tokenizer (vocabulary + added tokens).
 |      
 |      This method won't save the configuration and special token mappings of the tokenizer. Use
 |      [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.
 |      
 |      Args:
 |          save_directory (`str`):
 |              The directory in which to save the vocabulary.
 |          filename_prefix (`str`, *optional*):
 |              An optional prefix to add to the named of the saved files.
 |      
 |      Returns:
 |          `Tuple(str)`: Paths to the files saved.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __annotations__ = {}
 |  
 |  max_model_input_sizes = {'gpt-neox-20b': 2048}
 |  
 |  model_input_names = ['input_ids', 'attention_mask']
 |  
 |  pretrained_vocab_files_map = {'tokenizer_file': {'EleutherAI/gpt-neox-...
 |  
 |  vocab_files_names = {'merges_file': 'merges.txt', 'tokenizer_file': 't...
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
 |  
 |  __len__(self) -> int
 |      Size of the full vocabulary with the added tokens.
 |  
 |  convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]
 |      Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
 |      added tokens.
 |      
 |      Args:
 |          ids (`int` or `List[int]`):
 |              The token id (or token ids) to convert to tokens.
 |          skip_special_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to remove special tokens in the decoding.
 |      
 |      Returns:
 |          `str` or `List[str]`: The decoded token(s).
 |  
 |  convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]
 |      Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
 |      vocabulary.
 |      
 |      Args:
 |          tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).
 |      
 |      Returns:
 |          `int` or `List[int]`: The token id or list of token ids.
 |  
 |  convert_tokens_to_string(self, tokens: List[str]) -> str
 |      Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
 |      often want to remove sub-word tokenization artifacts at the same time.
 |      
 |      Args:
 |          tokens (`List[str]`): The token to join in a string.
 |      
 |      Returns:
 |          `str`: The joined tokens.
 |  
 |  get_added_vocab(self) -> Dict[str, int]
 |      Returns the added tokens in the vocabulary as a dictionary of token to index.
 |      
 |      Returns:
 |          `Dict[str, int]`: The added tokens.
 |  
 |  get_vocab(self) -> Dict[str, int]
 |      Returns the vocabulary as a dictionary of token to index.
 |      
 |      `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
 |      vocab.
 |      
 |      Returns:
 |          `Dict[str, int]`: The vocabulary.
 |  
 |  num_special_tokens_to_add(self, pair: bool = False) -> int
 |      Returns the number of added tokens when encoding a sequence with special tokens.
 |      
 |      <Tip>
 |      
 |      This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
 |      this inside your training loop.
 |      
 |      </Tip>
 |      
 |      Args:
 |          pair (`bool`, *optional*, defaults to `False`):
 |              Whether the number of added tokens should be computed in the case of a sequence pair or a single
 |              sequence.
 |      
 |      Returns:
 |          `int`: Number of special tokens added to sequences.
 |  
 |  set_truncation_and_padding(self, padding_strategy: transformers.utils.generic.PaddingStrategy, truncation_strategy: transformers.tokenization_utils_base.TruncationStrategy, max_length: int, stride: int, pad_to_multiple_of: Optional[int])
 |      Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
 |      library) and restore the tokenizer settings afterwards.
 |      
 |      The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
 |      padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
 |      section.
 |      
 |      Args:
 |          padding_strategy ([`~utils.PaddingStrategy`]):
 |              The kind of padding that will be applied to the input
 |          truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
 |              The kind of truncation that will be applied to the input
 |          max_length (`int`):
 |              The maximum size of a sequence.
 |          stride (`int`):
 |              The stride to use when handling overflow.
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
 |              the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
 |  
 |  tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]
 |      Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.
 |      
 |      Args:
 |          text (`str`):
 |              The sequence to be encoded.
 |          pair (`str`, *optional*):
 |              A second sequence to be encoded with the first.
 |          add_special_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to add the special tokens associated with the corresponding model.
 |          kwargs (additional keyword arguments, *optional*):
 |              Will be passed to the underlying model specific encode method. See details in
 |              [`~PreTrainedTokenizerBase.__call__`]
 |      
 |      Returns:
 |          `List[str]`: The list of tokens.
 |  
 |  train_new_from_iterator(self, text_iterator, vocab_size, length=None, new_special_tokens=None, special_tokens_map=None, **kwargs)
 |      Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
 |      as the current one.
 |      
 |      Args:
 |          text_iterator (generator of `List[str]`):
 |              The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
 |              if you have everything in memory.
 |          vocab_size (`int`):
 |              The size of the vocabulary you want for your tokenizer.
 |          length (`int`, *optional*):
 |              The total number of sequences in the iterator. This is used to provide meaningful progress tracking
 |          new_special_tokens (list of `str` or `AddedToken`, *optional*):
 |              A list of new special tokens to add to the tokenizer you are training.
 |          special_tokens_map (`Dict[str, str]`, *optional*):
 |              If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
 |              token name to new special token name in this argument.
 |          kwargs:
 |              Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.
 |      
 |      Returns:
 |          [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on
 |          `text_iterator`.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
 |  
 |  backend_tokenizer
 |      `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
 |  
 |  decoder
 |      `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
 |  
 |  is_fast
 |  
 |  vocab
 |  
 |  vocab_size
 |      `int`: Size of the base vocabulary (without the added tokens).
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
 |  
 |  can_save_slow_tokenizer = True
 |  
 |  slow_tokenizer_class = None
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from transformers.tokenization_utils_base.PreTrainedTokenizerBase:
 |  
 |  __call__(self, text: Union[str, List[str], List[List[str]]] = None, text_pair: Union[str, List[str], List[List[str]], NoneType] = None, text_target: Union[str, List[str], List[List[str]]] = None, text_pair_target: Union[str, List[str], List[List[str]], NoneType] = None, add_special_tokens: bool = True, padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False, truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None, max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False, pad_to_multiple_of: Optional[int] = None, return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = None, return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None, return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False, return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding
 |      Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
 |      sequences.
 |      
 |      Args:
 |          text (`str`, `List[str]`, `List[List[str]]`, *optional*):
 |              The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
 |              (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
 |              `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
 |          text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
 |              The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
 |              (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
 |              `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
 |          text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
 |              The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
 |              list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
 |              you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
 |          text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
 |              The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
 |              list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
 |              you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
 |      
 |          add_special_tokens (`bool`, *optional*, defaults to `True`):
 |              Whether or not to encode the sequences with the special tokens relative to their model.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
 |              Activates and controls padding. Accepts the following values:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
 |              Activates and controls truncation. Accepts the following values:
 |      
 |              - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
 |                to the maximum acceptable input length for the model if that argument is not provided. This will
 |                truncate token by token, removing a token from the longest sequence in the pair if a pair of
 |                sequences (or a batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
 |                greater than the model maximum admissible input size).
 |          max_length (`int`, *optional*):
 |              Controls the maximum length to use by one of the truncation/padding parameters.
 |      
 |              If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
 |              is required by one of the truncation/padding parameters. If the model has no specific maximum input
 |              length (like XLNet) truncation/padding to a maximum length will be deactivated.
 |          stride (`int`, *optional*, defaults to 0):
 |              If set to a number along with `max_length`, the overflowing tokens returned when
 |              `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
 |              returned to provide some overlap between truncated and overflowing sequences. The value of this
 |              argument defines the number of overlapping tokens.
 |          is_split_into_words (`bool`, *optional*, defaults to `False`):
 |              Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
 |              tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
 |              which it will tokenize. This is useful for NER or token classification.
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
 |              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 |              `>= 7.5` (Volta).
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |      
 |          return_token_type_ids (`bool`, *optional*):
 |              Whether to return token type IDs. If left to the default, will return the token type IDs according to
 |              the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are token type IDs?](../glossary#token-type-ids)
 |          return_attention_mask (`bool`, *optional*):
 |              Whether to return the attention mask. If left to the default, will return the attention mask according
 |              to the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are attention masks?](../glossary#attention-mask)
 |          return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
 |              of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
 |              of returning overflowing tokens.
 |          return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return special tokens mask information.
 |          return_offsets_mapping (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return `(char_start, char_end)` for each token.
 |      
 |              This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
 |              Python's tokenizer, this method will raise `NotImplementedError`.
 |          return_length  (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return the lengths of the encoded inputs.
 |          verbose (`bool`, *optional*, defaults to `True`):
 |              Whether or not to print more information and warnings.
 |          **kwargs: passed to the `self.tokenize()` method
 |      
 |      Return:
 |          [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
 |      
 |          - **input_ids** -- List of token ids to be fed to a model.
 |      
 |            [What are input IDs?](../glossary#input-ids)
 |      
 |          - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
 |            if *"token_type_ids"* is in `self.model_input_names`).
 |      
 |            [What are token type IDs?](../glossary#token-type-ids)
 |      
 |          - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
 |            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).
 |      
 |            [What are attention masks?](../glossary#attention-mask)
 |      
 |          - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
 |            regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
 |          - **length** -- The length of the inputs (when `return_length=True`)
 |  
 |  __repr__(self) -> str
 |      Return repr(self).
 |  
 |  as_target_tokenizer(self)
 |      Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
 |      sequence-to-sequence models that need a slightly different processing for the labels.
 |  
 |  batch_decode(self, sequences: Union[List[int], List[List[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None, **kwargs) -> List[str]
 |      Convert a list of lists of token ids into a list of strings by calling decode.
 |      
 |      Args:
 |          sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
 |              List of tokenized input ids. Can be obtained using the `__call__` method.
 |          skip_special_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to remove special tokens in the decoding.
 |          clean_up_tokenization_spaces (`bool`, *optional*):
 |              Whether or not to clean up the tokenization spaces. If `None`, will default to
 |              `self.clean_up_tokenization_spaces`.
 |          kwargs (additional keyword arguments, *optional*):
 |              Will be passed to the underlying model specific decode method.
 |      
 |      Returns:
 |          `List[str]`: The list of decoded sentences.
 |  
 |  batch_encode_plus(self, batch_text_or_text_pairs: Union[List[str], List[Tuple[str, str]], List[List[str]], List[Tuple[List[str], List[str]]], List[List[int]], List[Tuple[List[int], List[int]]]], add_special_tokens: bool = True, padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False, truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None, max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False, pad_to_multiple_of: Optional[int] = None, return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = None, return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None, return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False, return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding
 |      Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
 |      
 |      <Tip warning={true}>
 |      
 |      This method is deprecated, `__call__` should be used instead.
 |      
 |      </Tip>
 |      
 |      Args:
 |          batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
 |              Batch of sequences or pair of sequences to be encoded. This can be a list of
 |              string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
 |              details in `encode_plus`).
 |      
 |          add_special_tokens (`bool`, *optional*, defaults to `True`):
 |              Whether or not to encode the sequences with the special tokens relative to their model.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
 |              Activates and controls padding. Accepts the following values:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
 |              Activates and controls truncation. Accepts the following values:
 |      
 |              - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
 |                to the maximum acceptable input length for the model if that argument is not provided. This will
 |                truncate token by token, removing a token from the longest sequence in the pair if a pair of
 |                sequences (or a batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
 |                greater than the model maximum admissible input size).
 |          max_length (`int`, *optional*):
 |              Controls the maximum length to use by one of the truncation/padding parameters.
 |      
 |              If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
 |              is required by one of the truncation/padding parameters. If the model has no specific maximum input
 |              length (like XLNet) truncation/padding to a maximum length will be deactivated.
 |          stride (`int`, *optional*, defaults to 0):
 |              If set to a number along with `max_length`, the overflowing tokens returned when
 |              `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
 |              returned to provide some overlap between truncated and overflowing sequences. The value of this
 |              argument defines the number of overlapping tokens.
 |          is_split_into_words (`bool`, *optional*, defaults to `False`):
 |              Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
 |              tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
 |              which it will tokenize. This is useful for NER or token classification.
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
 |              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 |              `>= 7.5` (Volta).
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |      
 |          return_token_type_ids (`bool`, *optional*):
 |              Whether to return token type IDs. If left to the default, will return the token type IDs according to
 |              the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are token type IDs?](../glossary#token-type-ids)
 |          return_attention_mask (`bool`, *optional*):
 |              Whether to return the attention mask. If left to the default, will return the attention mask according
 |              to the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are attention masks?](../glossary#attention-mask)
 |          return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
 |              of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
 |              of returning overflowing tokens.
 |          return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return special tokens mask information.
 |          return_offsets_mapping (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return `(char_start, char_end)` for each token.
 |      
 |              This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
 |              Python's tokenizer, this method will raise `NotImplementedError`.
 |          return_length  (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return the lengths of the encoded inputs.
 |          verbose (`bool`, *optional*, defaults to `True`):
 |              Whether or not to print more information and warnings.
 |          **kwargs: passed to the `self.tokenize()` method
 |      
 |      Return:
 |          [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
 |      
 |          - **input_ids** -- List of token ids to be fed to a model.
 |      
 |            [What are input IDs?](../glossary#input-ids)
 |      
 |          - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
 |            if *"token_type_ids"* is in `self.model_input_names`).
 |      
 |            [What are token type IDs?](../glossary#token-type-ids)
 |      
 |          - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
 |            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).
 |      
 |            [What are attention masks?](../glossary#attention-mask)
 |      
 |          - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
 |            regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
 |          - **length** -- The length of the inputs (when `return_length=True`)
 |  
 |  build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]
 |      Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
 |      adding special tokens.
 |      
 |      This implementation does not add special tokens and this method should be overridden in a subclass.
 |      
 |      Args:
 |          token_ids_0 (`List[int]`): The first tokenized sequence.
 |          token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.
 |      
 |      Returns:
 |          `List[int]`: The model input with special tokens.
 |  
 |  create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]
 |      Create the token type IDs corresponding to the sequences passed. [What are token type
 |      IDs?](../glossary#token-type-ids)
 |      
 |      Should be overridden in a subclass if the model has a special way of building those.
 |      
 |      Args:
 |          token_ids_0 (`List[int]`): The first tokenized sequence.
 |          token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.
 |      
 |      Returns:
 |          `List[int]`: The token type ids.
 |  
 |  decode(self, token_ids: Union[int, List[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None, **kwargs) -> str
 |      Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
 |      tokens and clean up tokenization spaces.
 |      
 |      Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.
 |      
 |      Args:
 |          token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
 |              List of tokenized input ids. Can be obtained using the `__call__` method.
 |          skip_special_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to remove special tokens in the decoding.
 |          clean_up_tokenization_spaces (`bool`, *optional*):
 |              Whether or not to clean up the tokenization spaces. If `None`, will default to
 |              `self.clean_up_tokenization_spaces`.
 |          kwargs (additional keyword arguments, *optional*):
 |              Will be passed to the underlying model specific decode method.
 |      
 |      Returns:
 |          `str`: The decoded sentence.
 |  
 |  encode(self, text: Union[str, List[str], List[int]], text_pair: Union[str, List[str], List[int], NoneType] = None, add_special_tokens: bool = True, padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False, truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None, max_length: Optional[int] = None, stride: int = 0, return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = None, **kwargs) -> List[int]
 |      Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
 |      
 |      Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.
 |      
 |      Args:
 |          text (`str`, `List[str]` or `List[int]`):
 |              The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
 |              `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
 |              method).
 |          text_pair (`str`, `List[str]` or `List[int]`, *optional*):
 |              Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
 |              the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
 |              method).
 |      
 |          add_special_tokens (`bool`, *optional*, defaults to `True`):
 |              Whether or not to encode the sequences with the special tokens relative to their model.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
 |              Activates and controls padding. Accepts the following values:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
 |              Activates and controls truncation. Accepts the following values:
 |      
 |              - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
 |                to the maximum acceptable input length for the model if that argument is not provided. This will
 |                truncate token by token, removing a token from the longest sequence in the pair if a pair of
 |                sequences (or a batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
 |                greater than the model maximum admissible input size).
 |          max_length (`int`, *optional*):
 |              Controls the maximum length to use by one of the truncation/padding parameters.
 |      
 |              If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
 |              is required by one of the truncation/padding parameters. If the model has no specific maximum input
 |              length (like XLNet) truncation/padding to a maximum length will be deactivated.
 |          stride (`int`, *optional*, defaults to 0):
 |              If set to a number along with `max_length`, the overflowing tokens returned when
 |              `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
 |              returned to provide some overlap between truncated and overflowing sequences. The value of this
 |              argument defines the number of overlapping tokens.
 |          is_split_into_words (`bool`, *optional*, defaults to `False`):
 |              Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
 |              tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
 |              which it will tokenize. This is useful for NER or token classification.
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
 |              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 |              `>= 7.5` (Volta).
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |      
 |          **kwargs: Passed along to the `.tokenize()` method.
 |      
 |      Returns:
 |          `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`: The tokenized ids of the text.
 |  
 |  encode_plus(self, text: Union[str, List[str], List[int]], text_pair: Union[str, List[str], List[int], NoneType] = None, add_special_tokens: bool = True, padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False, truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None, max_length: Optional[int] = None, stride: int = 0, is_split_into_words: bool = False, pad_to_multiple_of: Optional[int] = None, return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = None, return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None, return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False, return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding
 |      Tokenize and prepare for the model a sequence or a pair of sequences.
 |      
 |      <Tip warning={true}>
 |      
 |      This method is deprecated, `__call__` should be used instead.
 |      
 |      </Tip>
 |      
 |      Args:
 |          text (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)):
 |              The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
 |              `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
 |              method).
 |          text_pair (`str`, `List[str]` or `List[int]`, *optional*):
 |              Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
 |              the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
 |              method).
 |      
 |          add_special_tokens (`bool`, *optional*, defaults to `True`):
 |              Whether or not to encode the sequences with the special tokens relative to their model.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
 |              Activates and controls padding. Accepts the following values:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
 |              Activates and controls truncation. Accepts the following values:
 |      
 |              - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
 |                to the maximum acceptable input length for the model if that argument is not provided. This will
 |                truncate token by token, removing a token from the longest sequence in the pair if a pair of
 |                sequences (or a batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
 |                greater than the model maximum admissible input size).
 |          max_length (`int`, *optional*):
 |              Controls the maximum length to use by one of the truncation/padding parameters.
 |      
 |              If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
 |              is required by one of the truncation/padding parameters. If the model has no specific maximum input
 |              length (like XLNet) truncation/padding to a maximum length will be deactivated.
 |          stride (`int`, *optional*, defaults to 0):
 |              If set to a number along with `max_length`, the overflowing tokens returned when
 |              `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
 |              returned to provide some overlap between truncated and overflowing sequences. The value of this
 |              argument defines the number of overlapping tokens.
 |          is_split_into_words (`bool`, *optional*, defaults to `False`):
 |              Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
 |              tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
 |              which it will tokenize. This is useful for NER or token classification.
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
 |              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 |              `>= 7.5` (Volta).
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |      
 |          return_token_type_ids (`bool`, *optional*):
 |              Whether to return token type IDs. If left to the default, will return the token type IDs according to
 |              the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are token type IDs?](../glossary#token-type-ids)
 |          return_attention_mask (`bool`, *optional*):
 |              Whether to return the attention mask. If left to the default, will return the attention mask according
 |              to the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are attention masks?](../glossary#attention-mask)
 |          return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
 |              of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
 |              of returning overflowing tokens.
 |          return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return special tokens mask information.
 |          return_offsets_mapping (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return `(char_start, char_end)` for each token.
 |      
 |              This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
 |              Python's tokenizer, this method will raise `NotImplementedError`.
 |          return_length  (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return the lengths of the encoded inputs.
 |          verbose (`bool`, *optional*, defaults to `True`):
 |              Whether or not to print more information and warnings.
 |          **kwargs: passed to the `self.tokenize()` method
 |      
 |      Return:
 |          [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
 |      
 |          - **input_ids** -- List of token ids to be fed to a model.
 |      
 |            [What are input IDs?](../glossary#input-ids)
 |      
 |          - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
 |            if *"token_type_ids"* is in `self.model_input_names`).
 |      
 |            [What are token type IDs?](../glossary#token-type-ids)
 |      
 |          - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
 |            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).
 |      
 |            [What are attention masks?](../glossary#attention-mask)
 |      
 |          - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
 |            regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
 |          - **length** -- The length of the inputs (when `return_length=True`)
 |  
 |  get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]
 |      Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
 |      special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.
 |      
 |      Args:
 |          token_ids_0 (`List[int]`):
 |              List of ids of the first sequence.
 |          token_ids_1 (`List[int]`, *optional*):
 |              List of ids of the second sequence.
 |          already_has_special_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not the token list is already formatted with special tokens for the model.
 |      
 |      Returns:
 |          A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
 |  
 |  pad(self, encoded_inputs: Union[transformers.tokenization_utils_base.BatchEncoding, List[transformers.tokenization_utils_base.BatchEncoding], Dict[str, List[int]], Dict[str, List[List[int]]], List[Dict[str, List[int]]]], padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = True, max_length: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, return_attention_mask: Optional[bool] = None, return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = None, verbose: bool = True) -> transformers.tokenization_utils_base.BatchEncoding
 |      Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
 |      in the batch.
 |      
 |      Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
 |      `self.pad_token_id` and `self.pad_token_type_id`).
 |      
 |      Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
 |      text followed by a call to the `pad` method to get a padded encoding.
 |      
 |      <Tip>
 |      
 |      If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
 |      result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
 |      PyTorch tensors, you will lose the specific device of your tensors however.
 |      
 |      </Tip>
 |      
 |      Args:
 |          encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
 |              Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
 |              tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
 |              List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
 |              collate function.
 |      
 |              Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
 |              the note above for the return type.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
 |               Select a strategy to pad the returned sequences (according to the model's padding side and padding
 |               index) among:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          max_length (`int`, *optional*):
 |              Maximum length of the returned list and optionally padding length (see above).
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value.
 |      
 |              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 |              `>= 7.5` (Volta).
 |          return_attention_mask (`bool`, *optional*):
 |              Whether to return the attention mask. If left to the default, will return the attention mask according
 |              to the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are attention masks?](../glossary#attention-mask)
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |          verbose (`bool`, *optional*, defaults to `True`):
 |              Whether or not to print more information and warnings.
 |  
 |  prepare_for_model(self, ids: List[int], pair_ids: Optional[List[int]] = None, add_special_tokens: bool = True, padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False, truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None, max_length: Optional[int] = None, stride: int = 0, pad_to_multiple_of: Optional[int] = None, return_tensors: Union[str, transformers.utils.generic.TensorType, NoneType] = None, return_token_type_ids: Optional[bool] = None, return_attention_mask: Optional[bool] = None, return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False, return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True, prepend_batch_axis: bool = False, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding
 |      Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
 |      adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
 |      manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
 |      different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
 |      overflowing tokens. Such a combination of arguments will raise an error.
 |      
 |      Args:
 |          ids (`List[int]`):
 |              Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
 |              `convert_tokens_to_ids` methods.
 |          pair_ids (`List[int]`, *optional*):
 |              Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
 |              and `convert_tokens_to_ids` methods.
 |      
 |          add_special_tokens (`bool`, *optional*, defaults to `True`):
 |              Whether or not to encode the sequences with the special tokens relative to their model.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
 |              Activates and controls padding. Accepts the following values:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
 |              Activates and controls truncation. Accepts the following values:
 |      
 |              - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
 |                to the maximum acceptable input length for the model if that argument is not provided. This will
 |                truncate token by token, removing a token from the longest sequence in the pair if a pair of
 |                sequences (or a batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
 |                greater than the model maximum admissible input size).
 |          max_length (`int`, *optional*):
 |              Controls the maximum length to use by one of the truncation/padding parameters.
 |      
 |              If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
 |              is required by one of the truncation/padding parameters. If the model has no specific maximum input
 |              length (like XLNet) truncation/padding to a maximum length will be deactivated.
 |          stride (`int`, *optional*, defaults to 0):
 |              If set to a number along with `max_length`, the overflowing tokens returned when
 |              `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
 |              returned to provide some overlap between truncated and overflowing sequences. The value of this
 |              argument defines the number of overlapping tokens.
 |          is_split_into_words (`bool`, *optional*, defaults to `False`):
 |              Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
 |              tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
 |              which it will tokenize. This is useful for NER or token classification.
 |          pad_to_multiple_of (`int`, *optional*):
 |              If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
 |              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 |              `>= 7.5` (Volta).
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |      
 |          return_token_type_ids (`bool`, *optional*):
 |              Whether to return token type IDs. If left to the default, will return the token type IDs according to
 |              the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are token type IDs?](../glossary#token-type-ids)
 |          return_attention_mask (`bool`, *optional*):
 |              Whether to return the attention mask. If left to the default, will return the attention mask according
 |              to the specific tokenizer's default, defined by the `return_outputs` attribute.
 |      
 |              [What are attention masks?](../glossary#attention-mask)
 |          return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
 |              of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
 |              of returning overflowing tokens.
 |          return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return special tokens mask information.
 |          return_offsets_mapping (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return `(char_start, char_end)` for each token.
 |      
 |              This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
 |              Python's tokenizer, this method will raise `NotImplementedError`.
 |          return_length  (`bool`, *optional*, defaults to `False`):
 |              Whether or not to return the lengths of the encoded inputs.
 |          verbose (`bool`, *optional*, defaults to `True`):
 |              Whether or not to print more information and warnings.
 |          **kwargs: passed to the `self.tokenize()` method
 |      
 |      Return:
 |          [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
 |      
 |          - **input_ids** -- List of token ids to be fed to a model.
 |      
 |            [What are input IDs?](../glossary#input-ids)
 |      
 |          - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
 |            if *"token_type_ids"* is in `self.model_input_names`).
 |      
 |            [What are token type IDs?](../glossary#token-type-ids)
 |      
 |          - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
 |            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).
 |      
 |            [What are attention masks?](../glossary#attention-mask)
 |      
 |          - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
 |            `return_overflowing_tokens=True`).
 |          - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
 |            regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
 |          - **length** -- The length of the inputs (when `return_length=True`)
 |  
 |  prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[List[str]] = None, max_length: Optional[int] = None, max_target_length: Optional[int] = None, padding: str = 'longest', return_tensors: str = None, truncation: bool = True, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding
 |      Prepare model inputs for translation. For best performance, translate one sentence at a time.
 |      
 |      Arguments:
 |          src_texts (`List[str]`):
 |              List of documents to summarize or source language texts.
 |          tgt_texts (`list`, *optional*):
 |              List of summaries or target language texts.
 |          max_length (`int`, *optional*):
 |              Controls the maximum length for encoder inputs (documents to summarize or source language texts) If
 |              left unset or set to `None`, this will use the predefined model maximum length if a maximum length is
 |              required by one of the truncation/padding parameters. If the model has no specific maximum input length
 |              (like XLNet) truncation/padding to a maximum length will be deactivated.
 |          max_target_length (`int`, *optional*):
 |              Controls the maximum length of decoder inputs (target language texts or summaries) If left unset or set
 |              to `None`, this will use the max_length value.
 |          padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
 |              Activates and controls padding. Accepts the following values:
 |      
 |              - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
 |                sequence if provided).
 |              - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
 |                acceptable input length for the model if that argument is not provided.
 |              - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
 |                lengths).
 |          return_tensors (`str` or [`~utils.TensorType`], *optional*):
 |              If set, will return tensors instead of list of python integers. Acceptable values are:
 |      
 |              - `'tf'`: Return TensorFlow `tf.constant` objects.
 |              - `'pt'`: Return PyTorch `torch.Tensor` objects.
 |              - `'np'`: Return Numpy `np.ndarray` objects.
 |          truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `True`):
 |              Activates and controls truncation. Accepts the following values:
 |      
 |              - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
 |                to the maximum acceptable input length for the model if that argument is not provided. This will
 |                truncate token by token, removing a token from the longest sequence in the pair if a pair of
 |                sequences (or a batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
 |                greater than the model maximum admissible input size).
 |          **kwargs:
 |              Additional keyword arguments passed along to `self.__call__`.
 |      
 |      Return:
 |          [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
 |      
 |          - **input_ids** -- List of token ids to be fed to the encoder.
 |          - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
 |          - **labels** -- List of token ids for tgt_texts.
 |      
 |          The full set of keys `[input_ids, attention_mask, labels]`, will only be returned if tgt_texts is passed.
 |          Otherwise, input_ids, attention_mask will be the only keys.
 |  
 |  push_to_hub(self, repo_id: str, use_temp_dir: Optional[bool] = None, commit_message: Optional[str] = None, private: Optional[bool] = None, use_auth_token: Union[bool, str, NoneType] = None, max_shard_size: Union[int, str, NoneType] = '10GB', create_pr: bool = False, **deprecated_kwargs) -> str
 |      Upload the tokenizer files to the 🤗 Model Hub while synchronizing a local clone of the repo in
 |      `repo_path_or_name`.
 |      
 |      Parameters:
 |          repo_id (`str`):
 |              The name of the repository you want to push your tokenizer to. It should contain your organization name
 |              when pushing to a given organization.
 |          use_temp_dir (`bool`, *optional*):
 |              Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
 |              Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
 |          commit_message (`str`, *optional*):
 |              Message to commit while pushing. Will default to `"Upload tokenizer"`.
 |          private (`bool`, *optional*):
 |              Whether or not the repository created should be private.
 |          use_auth_token (`bool` or `str`, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
 |              when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
 |              is not specified.
 |          max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
 |              Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
 |              will then be each of size lower than this size. If expressed as a string, needs to be digits followed
 |              by a unit (like `"5MB"`).
 |          create_pr (`bool`, *optional*, defaults to `False`):
 |              Whether or not to create a PR with the uploaded files or directly commit.
 |      
 |      Examples:
 |      
 |      ```python
 |      from transformers import AutoTokenizer
 |      
 |      tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
 |      
 |      # Push the tokenizer to your namespace with the name "my-finetuned-bert".
 |      tokenizer.push_to_hub("my-finetuned-bert")
 |      
 |      # Push the tokenizer to an organization with the name "my-finetuned-bert".
 |      tokenizer.push_to_hub("huggingface/my-finetuned-bert")
 |      ```
 |  
 |  save_pretrained(self, save_directory: Union[str, os.PathLike], legacy_format: Optional[bool] = None, filename_prefix: Optional[str] = None, push_to_hub: bool = False, **kwargs) -> Tuple[str]
 |      Save the full tokenizer state.
 |      
 |      
 |      This method make sure the full tokenizer can then be re-loaded using the
 |      [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] class method..
 |      
 |      Warning,None This won't save modifications you may have applied to the tokenizer after the instantiation (for
 |      instance, modifying `tokenizer.do_lower_case` after creation).
 |      
 |      Args:
 |          save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
 |          legacy_format (`bool`, *optional*):
 |              Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
 |              format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
 |              added_tokens files.
 |      
 |              If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with
 |              "slow" tokenizers (not powered by the *tokenizers* library), so the tokenizer will not be able to be
 |              loaded in the corresponding "slow" tokenizer.
 |      
 |              If `True`, will save the tokenizer in legacy format. If the "slow" tokenizer doesn't exits, a value
 |              error is raised.
 |          filename_prefix: (`str`, *optional*):
 |              A prefix to add to the names of the files saved by the tokenizer.
 |          push_to_hub (`bool`, *optional*, defaults to `False`):
 |              Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
 |              repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
 |              namespace).
 |          kwargs:
 |              Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
 |      
 |      Returns:
 |          A tuple of `str`: The files saved.
 |  
 |  truncate_sequences(self, ids: List[int], pair_ids: Optional[List[int]] = None, num_tokens_to_remove: int = 0, truncation_strategy: Union[str, transformers.tokenization_utils_base.TruncationStrategy] = 'longest_first', stride: int = 0) -> Tuple[List[int], List[int], List[int]]
 |      Truncates a sequence pair in-place following the strategy.
 |      
 |      Args:
 |          ids (`List[int]`):
 |              Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
 |              `convert_tokens_to_ids` methods.
 |          pair_ids (`List[int]`, *optional*):
 |              Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
 |              and `convert_tokens_to_ids` methods.
 |          num_tokens_to_remove (`int`, *optional*, defaults to 0):
 |              Number of tokens to remove using the truncation strategy.
 |          truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
 |              The strategy to follow for truncation. Can be:
 |      
 |              - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will truncate
 |                token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
 |                batch of pairs) is provided.
 |              - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
 |                maximum acceptable input length for the model if that argument is not provided. This will only
 |                truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
 |              - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
 |                than the model maximum admissible input size).
 |          stride (`int`, *optional*, defaults to 0):
 |              If set to a positive number, the overflowing tokens returned will contain some tokens from the main
 |              sequence returned. The value of this argument defines the number of additional tokens.
 |      
 |      Returns:
 |          `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
 |          overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
 |          of sequences (or a batch of pairs) is provided.
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from transformers.tokenization_utils_base.PreTrainedTokenizerBase:
 |  
 |  from_pretrained(pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs) from builtins.type
 |      Instantiate a [`~tokenization_utils_base.PreTrainedTokenizerBase`] (or a derived class) from a predefined
 |      tokenizer.
 |      
 |      Args:
 |          pretrained_model_name_or_path (`str` or `os.PathLike`):
 |              Can be either:
 |      
 |              - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
 |                Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
 |                user or organization name, like `dbmdz/bert-base-german-cased`.
 |              - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
 |                using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`] method, e.g.,
 |                `./my_model_directory/`.
 |              - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
 |                file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
 |                `./my_model_directory/vocab.txt`.
 |          cache_dir (`str` or `os.PathLike`, *optional*):
 |              Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
 |              standard cache should not be used.
 |          force_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
 |              exist.
 |          resume_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to delete incompletely received files. Attempt to resume the download if such a file
 |              exists.
 |          proxies (`Dict[str, str]`, *optional*):
 |              A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
 |              'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
 |          use_auth_token (`str` or *bool*, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
 |              when running `huggingface-cli login` (stored in `~/.huggingface`).
 |          local_files_only (`bool`, *optional*, defaults to `False`):
 |              Whether or not to only rely on local files and not to attempt to download any files.
 |          revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
 |              git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
 |              identifier allowed by git.
 |          subfolder (`str`, *optional*):
 |              In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
 |              facebook/rag-token-base), specify it here.
 |          inputs (additional positional arguments, *optional*):
 |              Will be passed along to the Tokenizer `__init__` method.
 |          kwargs (additional keyword arguments, *optional*):
 |              Will be passed to the Tokenizer `__init__` method. Can be used to set special tokens like `bos_token`,
 |              `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
 |              `additional_special_tokens`. See parameters in the `__init__` for more details.
 |      
 |      <Tip>
 |      
 |      Passing `use_auth_token=True` is required when you want to use a private model.
 |      
 |      </Tip>
 |      
 |      Examples:
 |      
 |      ```python
 |      # We can't instantiate directly the base class *PreTrainedTokenizerBase* so let's show our examples on a derived class: BertTokenizer
 |      # Download vocabulary from huggingface.co and cache.
 |      tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 |      
 |      # Download vocabulary from huggingface.co (user-uploaded) and cache.
 |      tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
 |      
 |      # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
 |      tokenizer = BertTokenizer.from_pretrained("./test/saved_model/")
 |      
 |      # If the tokenizer uses a single vocabulary file, you can point directly to this file
 |      tokenizer = BertTokenizer.from_pretrained("./test/saved_model/my_vocab.txt")
 |      
 |      # You can link tokens to special vocabulary when instantiating
 |      tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", unk_token="<unk>")
 |      # You should be sure '<unk>' is in the vocabulary when doing that.
 |      # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
 |      assert tokenizer.unk_token == "<unk>"
 |      ```
 |  
 |  register_for_auto_class(auto_class='AutoTokenizer') from builtins.type
 |      Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
 |      library are already mapped with `AutoTokenizer`.
 |      
 |      <Tip warning={true}>
 |      
 |      This API is experimental and may have some slight breaking changes in the next releases.
 |      
 |      </Tip>
 |      
 |      Args:
 |          auto_class (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`):
 |              The auto class to register this new tokenizer with.
 |  
 |  ----------------------------------------------------------------------
 |  Static methods inherited from transformers.tokenization_utils_base.PreTrainedTokenizerBase:
 |  
 |  clean_up_tokenization(out_string: str) -> str
 |      Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
 |      
 |      Args:
 |          out_string (`str`): The text to clean up.
 |      
 |      Returns:
 |          `str`: The cleaned-up string.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from transformers.tokenization_utils_base.PreTrainedTokenizerBase:
 |  
 |  max_len_sentences_pair
 |      `int`: The maximum combined length of a pair of sentences that can be fed to the model.
 |  
 |  max_len_single_sentence
 |      `int`: The maximum length of a sentence that can be fed to the model.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from transformers.tokenization_utils_base.PreTrainedTokenizerBase:
 |  
 |  padding_side = 'right'
 |  
 |  pretrained_init_configuration = {}
 |  
 |  truncation_side = 'right'
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from transformers.tokenization_utils_base.SpecialTokensMixin:
 |  
 |  add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, tokenizers.AddedToken]], replace_additional_special_tokens=True) -> int
 |      Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
 |      special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
 |      current vocabulary).
 |      
 |      Note,None When adding new tokens to the vocabulary, you should make sure to also resize the token embedding
 |      matrix of the model so that its embedding matrix matches the tokenizer.
 |      
 |      In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.
 |      
 |      Using `add_special_tokens` will ensure your special tokens can be used in several ways:
 |      
 |      - Special tokens are carefully handled by the tokenizer (they are never split).
 |      - You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
 |        makes it easy to develop model-agnostic training and fine-tuning scripts.
 |      
 |      When possible, special tokens are already registered for provided pretrained models (for instance
 |      [`BertTokenizer`] `cls_token` is already registered to be :obj*'[CLS]'* and XLM's one is also registered to be
 |      `'</s>'`).
 |      
 |      Args:
 |          special_tokens_dict (dictionary *str* to *str* or `tokenizers.AddedToken`):
 |              Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
 |              `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
 |      
 |              Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
 |              assign the index of the `unk_token` to them).
 |          replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`):
 |              If `True`, the existing list of additional special tokens will be replaced by the one specified in
 |              `special_tokens_dict`. Otherwise, `self._additional_special_tokens` is updated. In the former case, the
 |              tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged as
 |              non-special tokens.
 |      
 |      Returns:
 |          `int`: Number of tokens added to the vocabulary.
 |      
 |      Examples:
 |      
 |      ```python
 |      # Let's see how to add a new classification token to GPT-2
 |      tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
 |      model = GPT2Model.from_pretrained("gpt2")
 |      
 |      special_tokens_dict = {"cls_token": "<CLS>"}
 |      
 |      num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
 |      print("We have added", num_added_toks, "tokens")
 |      # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
 |      model.resize_token_embeddings(len(tokenizer))
 |      
 |      assert tokenizer.cls_token == "<CLS>"
 |      ```
 |  
 |  add_tokens(self, new_tokens: Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]], special_tokens: bool = False) -> int
 |      Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
 |      it with indices starting from length of the current vocabulary and and will be isolated before the tokenization
 |      algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
 |      not treated in the same way.
 |      
 |      Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
 |      of the model so that its embedding matrix matches the tokenizer.
 |      
 |      In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.
 |      
 |      Args:
 |          new_tokens (`str`, `tokenizers.AddedToken` or a list of *str* or `tokenizers.AddedToken`):
 |              Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
 |              token to let you personalize its behavior: whether this token should only match against a single word,
 |              whether this token should strip all potential whitespaces on the left side, whether this token should
 |              strip all potential whitespaces on the right side, etc.
 |          special_tokens (`bool`, *optional*, defaults to `False`):
 |              Can be used to specify if the token is a special token. This mostly change the normalization behavior
 |              (special tokens like CLS or [MASK] are usually not lower-cased for instance).
 |      
 |              See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.
 |      
 |      Returns:
 |          `int`: Number of tokens added to the vocabulary.
 |      
 |      Examples:
 |      
 |      ```python
 |      # Let's see how to increase the vocabulary of Bert model and tokenizer
 |      tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
 |      model = BertModel.from_pretrained("bert-base-uncased")
 |      
 |      num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
 |      print("We have added", num_added_toks, "tokens")
 |      # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
 |      model.resize_token_embeddings(len(tokenizer))
 |      ```
 |  
 |  sanitize_special_tokens(self) -> int
 |      Make sure that all the special tokens attributes of the tokenizer (`tokenizer.mask_token`,
 |      `tokenizer.cls_token`, etc.) are in the vocabulary.
 |      
 |      Add the missing ones to the vocabulary if needed.
 |      
 |      Return:
 |          `int`: The number of tokens added in the vocabulary during the operation.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from transformers.tokenization_utils_base.SpecialTokensMixin:
 |  
 |  all_special_ids
 |      `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
 |  
 |  all_special_tokens
 |      `List[str]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
 |      
 |      Convert tokens of `tokenizers.AddedToken` type to string.
 |  
 |  all_special_tokens_extended
 |      `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class
 |      attributes.
 |      
 |      Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
 |      special tokens are tokenized.
 |  
 |  pad_token_type_id
 |      `int`: Id of the padding token type in the vocabulary.
 |  
 |  special_tokens_map
 |      `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
 |      `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).
 |      
 |      Convert potential tokens of `tokenizers.AddedToken` type to string.
 |  
 |  special_tokens_map_extended
 |      `Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary mapping
 |      special token class attributes (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).
 |      
 |      Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
 |      special tokens are tokenized.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from transformers.tokenization_utils_base.SpecialTokensMixin:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  additional_special_tokens
 |      `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
 |      set.
 |  
 |  additional_special_tokens_ids
 |      `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
 |      been set.
 |  
 |  bos_token
 |      `str`: Beginning of sentence token. Log an error if used while not having been set.
 |  
 |  bos_token_id
 |      `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
 |      been set.
 |  
 |  cls_token
 |      `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
 |      depth of the model. Log an error if used while not having been set.
 |  
 |  cls_token_id
 |      `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
 |      leveraging self-attention along the full depth of the model.
 |      
 |      Returns `None` if the token has not been set.
 |  
 |  eos_token
 |      `str`: End of sentence token. Log an error if used while not having been set.
 |  
 |  eos_token_id
 |      `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
 |      set.
 |  
 |  mask_token
 |      `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
 |      having been set.
 |  
 |  mask_token_id
 |      `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
 |      modeling. Returns `None` if the token has not been set.
 |  
 |  pad_token
 |      `str`: Padding token. Log an error if used while not having been set.
 |  
 |  pad_token_id
 |      `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
 |  
 |  sep_token
 |      `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
 |      having been set.
 |  
 |  sep_token_id
 |      `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
 |      sequence. Returns `None` if the token has not been set.
 |  
 |  unk_token
 |      `str`: Unknown token. Log an error if used while not having been set.
 |  
 |  unk_token_id
 |      `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from transformers.tokenization_utils_base.SpecialTokensMixin:
 |  
 |  SPECIAL_TOKENS_ATTRIBUTES = ['bos_token', 'eos_token', 'unk_token', 's...
"""

# Write out the vocabulary to a file, with line numbers:
vocab = tokenizer.get_vocab()
with open("models/vocab.txt", "w+") as f:
    for token, i in vocab.items():
        f.write(f"{i}\t{token}\n")

print(f"Total number of tokens is {len(vocab)}")

# Print out for the user a few examples of how to encode using the tokenizer:
print("Example 1:")
print(tokenizer.tokenize("Hello, my Doug is CUTE"))
print(tokenizer.encode("Hello, my Doug is CUTE"))
print("Example 2:") # An example of python source code being tokenized:
print(tokenizer.tokenize("def hello_world():\n    print('Hello, world!')"))
print(tokenizer.encode("def hello_world():\n    print('Hello, world!')"))
# Show that the two examples above are recovered when using decode:
print("Example 3:")
print(tokenizer.decode(tokenizer.encode("Hello, my Doug is CUTE")))
print("Example 4:")
print(tokenizer.decode(tokenizer.encode("def hello_world():\n    print('Hello, world!')")))

#This function article_generator accepts a zstandard generator as an argument.
#It returns units of articles, which are json strings delimited by a newline character.
#Since the zstandard generator returns blocks that may contain multiple lines, but 
#one block and next may span a json article, this function looks to see if the last
#article in a block is only partial, and if so, reads the next block and combines
#the first partial article with the last partial article so that it can correctly
#yield each complete article in sequence.
def article_generator(zstandard_generator):
    last_article = ""  # Variable to store the last partial article
    
    for block in zstandard_generator:
        # Combine the last partial article with the first partial article from the current block
        if last_article:
            block = last_article + block
            last_article = ""
        
        articles = block.split("\n")  # Split the block into individual articles
        
        # If the last article is partial, store it for the next block
        if articles[-1]:
            last_article = articles[-1]
            articles = articles[:-1]
        
        # Yield each complete article
        for article in articles:
            if article and article.endswith("}"):
                yield article    

def zstd_generator(file_path):
    with zstandard.open(file_path, "rb") as file:
        running = True
        while running:
            chunk = file.read1()
            if not chunk:
                break
            yield chunk.decode()

directory = "./pile/the-eye.eu/public/AI/pile/train/"

pilenames = [directory + f"{num:02}.jsonl.zst" for num in range(30)]

#The update token dist function iterates through the tokens with lookahead n.
#Lookahead n means that the token n tokens in forward from the current token is examined.
#The token_dist is a dictionary of dictionaries, where the first index is the 
#current token, and the second index is the lookahead token.
#Iterating through the tokens, for each lookahead token seen, the corresponding 
#dictionary entry is incremented by 1.
def initialize_token_dist(max_token):
    token_dist = [[0] * (max_token + 1) for _ in range(max_token + 1)]
    return token_dist

def initialize_token_dist_0():
    token_dist = [0] * (len(tokenizer.get_vocab()))
    return token_dist

def update_token_dist_lookback_n(token_dist, tokens, n):
    for i in range(len(tokens) - n):
        current_token = tokens[i]
        
        for j in range(i + 1, i + n + 1):
            lookahead_token = tokens[j]
            
            token_dist[current_token][lookahead_token] += 1   
        #However, when n == 0, the above won't do anything, so
        #just count the total current_token:
        if n == 0:
            token_dist[current_token] += 1

max_lookahead = 10
def update_token_dist(token_dist, tokens):
    for n in range(max_lookahead):
        # For each n, update token_dist[n] with the lookahead n tokens
        update_token_dist_lookback_n(token_dist[n], tokens, n)


count = 0
token_count = 0
characters = 0
token_dist = [initialize_token_dist_0()] + [initialize_token_dist(len(tokenizer.get_vocab())) for _ in range(max_lookahead)]
for article in article_generator(zstd_generator(pilenames[0])):
    characters += len(article)
    try:
        article_dict = json.loads(article)
    except Exception as e:
        print(f"{count}: Error read json string: {e}")
        print(article)
        exit()
        continue
    tokens = tokenizer.encode(article_dict['text'])
    token_dist = update_token_dist(token_dist, tokens)
    if count < 10:
        print(f"{count}: meta: {article_dict['meta']}, length: {len(tokens)}")
    count += 1
    token_count += len(tokens)
    if count % 1000 == 0:
        kiB = characters/1000
        print(f"{count}: tokens per article = {token_count/1000}, kiB/karticle = {kiB}, estimated_karticles = {28_000_000/kiB}")
        token_count = 0
        characters = 0
        # write out the latest update token_dist, each lookahead n in a separate file,
        # and make sure to write out each line in the lookahead file like "token: count, count, count, ...\n"
        for n in range(max_lookahead):
            with open(f"models/token_dist_{n}.txt", "w+") as f:
                for i in range(len(token_dist[n])):
                    # if n == 0, then the file is for just the totals, so the line will be "token: count\n"
                    if n == 0:
                        f.write(f"{i}: {token_dist[n][i]}\n")
                    else:
                        f.write(f"{i}: " + ", ".join([str(x) for x in token_dist[n][i]]) + "\n") 

#Read back the n == 0 totals into token_totals for future use:
with open(f"models/token_dist_0.txt", "r") as f:
    token_totals = [int(line.split(":")[1]) for line in f]
sum_total_tokens = sum(token_totals)

#Read back in each token_dist file, and normalize the rows
#This means that the each count should be replaced by the count divided by the sum of the counts in the row, 
# and then re-written to the models/token_dist_normalized_{n}.txt file.
for n in range(max_lookahead):
    if n == 0:
        # For the n == 0 case, this is the expected conditional prob in general.
        # For each token A, the expected probability of token B given token A is naively just
        # the expected probability of token B in general.  
        # Therefore write out a token_dist_normalize_0.txt file that tracks this:
        with open(f"models/token_dist_normalized_{n}.txt", "w+") as f:
            for i in range(len(token_totals)):
                f.write(f"{i}: {(1+token_totals[i])/(len(token_totals)+sum_total_tokens)}\n")
    else:         
        with open(f"models/token_dist_{n}.txt", "r") as f:
            with open(f"models/token_dist_normalized_{n}.txt", "w+") as g:
                for line in f:
                    tokens = line.split(":")
                    #Add 1 to all counts so that the probability of a token pair that has never been seen is not 0.
                    probs = [1+int(x) for x in tokens[1].split(",")]
                    total = sum(probs)
                    normalized_counts = [str(x/total) for x in probs]
                    g.write(f"{tokens[0]}: " + ", ".join(normalized_counts) + "\n")

#Create additive versions of the lookaheads by finding the median of the normalized counts for each
#row, multiplying by the inverse of the median times e, and then taking the natural log.
for n in range(max_lookahead):
    with open(f"models/token_dist_normalized_{n}.txt", "r") as f:
        with open(f"models/token_dist_additive_{n}.txt", "w+") as g:
            for line in f:
                tokens = line.split(":")
                probs = [float(x) for x in tokens[1].split(",")]
                #Compute the median by getting a sorted copy of the probs, and choosing the length/2 element:
                median = sorted(probs)[len(probs)//2]
                additive_probs = [str(math.log(x/median)) for x in probs]
                #Keep the n == 0 additive_probs_0 for shifting the other additive_probs to a generalized expectation of 0:
                if n == 0:
                    additive_probs_0 = additive_probs
                else:
                    #Shift the additive_probs to a generalized expectation of 0:
                    additive_probs = [str(float(x)-float(y)) for x, y in zip(additive_probs, additive_probs_0)]
                g.write(f"{tokens[0]}: " + ", ".join(additive_probs) + "\n") 

#Now computing the vector of probabilities for the next token given a token list is as simple as
#adding the additive probabilities of the previous n tokens (vectorially), and then exponentiating each element:
def compute_token_prob(token_dist, tokens):
    #Initialize the token_prob vector to the first row of the n == 0 table:
    with open(f"models/token_dist_additive_0.txt", "r") as f:
        for line in f:
            tokens = line.split(":")
            token_prob = [float(x) for x in tokens[1].split(",")]
            break
    for n in range(max_lookahead):
        if n == 0:
            continue
        lookahead_token = tokens[-n]
        with open(f"models/token_dist_additive_{n}.txt", "r") as f:
            # Find the lookahead_token line, and add the additive probabilities to the token_prob vector:
            for line in f:
                tokens = line.split(":")
                if tokens[0] == lookahead_token:
                    token_prob = [x + float(y) for x, y in zip(token_prob, tokens[1].split(","))]
                    break
    #Exponentiate each element of the token_prob vector:
    token_prob = [math.exp(x) for x in token_prob]
    #Normalize the token_prob vector:
    total = sum(token_prob)
    token_prob = [x/total for x in token_prob]
    return token_prob