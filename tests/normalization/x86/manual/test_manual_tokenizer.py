import pytest
from .x86_manual_lines import X86_TEST_INPUTS, X86_TEST_OBJECTS
from bincfg import BaseTokenizer


def test_individual_manual_lines():
    """Tests any x86 tokenizer works as expected on manual lines"""
    for name, cls, kwargs in X86_TEST_OBJECTS:
        if not issubclass(cls, BaseTokenizer): continue
        tokenizer = cls(**kwargs)

        # Test them one by one
        for input_dict in X86_TEST_INPUTS:
            test_input, expected = input_dict['input'], input_dict[name]
            assert tokenizer(test_input) == expected, "Input: %s\nOutput: %s\nExpected: %s" % (test_input, tokenizer(test_input), expected)
        
        # Test all together
        input_list = [d['input'] for d in X86_TEST_INPUTS]
        expected = [t for d in X86_TEST_INPUTS for t in d[name]]

        assert tokenizer(*input_list) == expected, 'all inputs as *args'
        assert tokenizer('\n'.join(input_list)) == expected, 'all inputs as newline'
        assert tokenizer('|'.join(input_list)) == expected, 'all inputs as pipe'
