# Using as base the code in https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py

from typing import Any, List, Literal, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re

import statistics
from scipy import stats
import numpy as np

from transformers import AutoTokenizer


class HierachicalRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        apply_chunk_size: Optional[int] = 0,        
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        self._apply_chunk_size = apply_chunk_size


    
    def _split_text_with_regex(
        self, text: str, separator: str
    ) -> List[str]:
        # Now that we have the separator, split the text

        separators = None
        
        if separator:
            if self._keep_separator:
                try:
                    # The parentheses in the pattern keep the delimiters in the result.
                    _splits = re.split(f"({separator})", text)
                    splits = (
                        ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                        if self._keep_separator == "end"
                        else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
                    )
                    if len(_splits) % 2 == 0:
                        splits += _splits[-1:]
                    splits = (
                        (splits + [_splits[-1]])
                        if self._keep_separator == "end"
                        else ([_splits[0]] + splits)
                    )

                    separators = ["preamble"] + [_splits[i].strip("\n").strip() for i in range(1, len(_splits), 2)]
                except IndexError as e:
                    print(e)
                    print(_splits)
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""], separators



    def _split_text(self, text: str, separators: List[str], steps_to_apply_chunk_size: int) -> List[str]:

        # print(separators)
        # print(steps_to_apply_chunk_size)
        # print(self._length_function(text))
        
        """Split incoming text and return chunks."""
        final_chunks = {}
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)

        if steps_to_apply_chunk_size > 0 or self._length_function(text) > self._chunk_size:
            splits, separators_text = self._split_text_with_regex(text, _separator)
            
            # Does not merge small chunks to respect hierarchy
            if not new_separators:
                final_chunks = dict(zip(separators_text, splits))
            else:
                for i, s in enumerate(splits):
                    # print(f"new_separators={new_separators}, len(new_separators)={len(new_separators)}, len(separators)={len(separators)}")
                    other_info = self._split_text(s, new_separators, steps_to_apply_chunk_size - (len(separators) - len(new_separators)))
                    final_chunks[separators_text[i]] = other_info
        else:
            final_chunks = text

        return final_chunks


    def split_text(self, text: str) -> List[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators, self._apply_chunk_size)



LEGISLATION_SPLITTING_HIERARCHY_BRAZIL=[
    "\nLIVRO [IVX]+",
    "\nTÍTULO [IVX]+",
    "\nCAPÍTULO [IVX]+|Capítulo [IVX]+|\nCAPÍTULO ÚNICO",
    "\nSeção [IVX]+",
    "\nSubseção .+",
    "\nArt\. \d+[º\. ]*|\n.\s+Art\. \d+[º\. ]*|\nArtigo único\.",
    # "\nArt\. \d+[º\. ]*|\n.\s+Art\. \d+[º\. ]*",
    "\n§ \d+[º\. ]+",
    "\n[IVX]+[-\s]+",
]



def flatten_passages_hierarchy(passages_dictionary, current_path="", passages=[]):

    for item, value in passages_dictionary.items():
        item_path = current_path
        
        if item != "preamble":
            if item_path != "":
                item_path += "_" + (item[:-1].strip() if item[-1] in ["-", "—", "–"] else item.strip())
            else:
                item_path = item

        if type(value) == dict:
            flatten_passages_hierarchy(value, item_path, passages)
        else:
            passages.append({'path': item_path,
                            'passage': value})



class tokensCounter:

    def __init__(self,
                 which_hf_tokenizer):

        self.which_hf_tokenizer = which_hf_tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(which_hf_tokenizer)


    def count(self, text, tokens_output=None):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False)

        if tokens_output is not None:
            tokens_output = tokens

        return len(tokens['input_ids'][0])


    def passages_statistics(self, passages_list):
        num_tokens_por_chunk = [self.count(passage) for passage in passages_list]

        desc_stats = stats.describe(num_tokens_por_chunk)
        
        mediana_tokens = statistics.median(num_tokens_por_chunk)
        max_token_passage = np.argmax(num_tokens_por_chunk)

        return({'max_tokens': desc_stats.minmax[1],
                'min_tokens': desc_stats.minmax[0],
                'mean_tokens': desc_stats.mean,
                'std_tokens': np.sqrt(desc_stats.variance),
                'skewness_tokens': desc_stats.skewness,
                'kurtosis_tokens': desc_stats.kurtosis,
                'median_tokens': statistics.median(num_tokens_por_chunk),
                'max_token_passage': np.argmax(num_tokens_por_chunk)})
