# BinCFG Tests

## Adding new ISA

Tests for a new ISA should be added both in the tests/normalization and tests/cfg directories. For the normalization:

- make a new directory with the ISA name, and a new directory 'manual' within that to contain manual test cases

  - make a "ISA_manual_lines.py" file which will contain:

    1. at the top of the file, make a ISA_TEST_OBJECTS global object. It should be a list of 3-tuples: (name: str,
       class: type, kwargs: dict). 'name' is the string name of that object, 'class' is the type that should be used
       to construct that object, and 'kwargs' is a dictionary of kwargs to pass when creating that object. ISA_TEST_OBJECTS
       should contain all of the tokenizers/normalizers that are being used in tests. Tokenizers should inherit from
       BaseTokenizer, and normalizers from BaseNormalizer (IE: `issubclass(object_class, BaseTokenizer) == True` if
       that object is a tokenizer). You can make multiple test normalizers/tokenizers from the same class, but with
       different kwargs, by giving them different 'name's. Make sure the 'name's are unique.
    
    2. After that, make a ISA_TEST_INPUTS global object. It should be a list of dictionaries, with each dictionary
       being a single input to test. The dictionary should have an 'input' key with that value being a string input
       to test. All other keys should be the names of tokenizers/normalizers in the ISA_TEST_OBJECTS. For each key that
       is present, the value should be the expected output when using that object on the given input. For any tokenizer
       objects, the associated value should be a list of 2-tuples (token_type: Tokens, token: str) of the expected output
       from the tokenizer being run on that input. For any normalizer present, the associated value should be a list
       of strings of the expected output from the normalizer being run on that input. It is expected there will be only
       one string output (for now, this may change later), and that string is the result of running the normalizer with
       the 'instruction' tokenization_level using a token delimeter of ' '. Normalizers will also be tested with the
       'op' tokenization_level by splitting the expected outputs on all ' 's to generate the individual tokens (taking
       into account the fact that some strings may have spaces in them, and they will still be counted as one singular
       token). Finally, for either tokenizers or normalizers, the associated value could optionally be a type of an
       error class that is expected to be raised when attempting normalization/tokenization. Every object in ISA_TEST_OBJECTS
       should have a key in the ISA_TEST_INPUTS for its expected output.

       NOTE: all input strings and token outputs from both tokenizers and normalizers allow you to put in specific
       normalization constant string tokens via string formatting (not applied to token_types in tokenizer output tuples,
       however), including the possible tokens:

       [('{immval}', IMMEDIATE_VALUE_STR), ('{str}', STRING_LITERAL_STR), ('{func}', FUNCTION_CALL_STR), 
        ('{memexpr}', MEMORY_EXPRESSION_STR), ('{dispimm}', DISPLACEMENT_IMMEDIATE_STR), ('{reg}', GENERAL_REGISTER_STR), 
        ('{split_imm}', SPLIT_IMMEDIATE_TOKEN), ('{jmpdst}', JUMP_DESTINATION_STR), ('{dispmem}', IMMEDIATE_VALUE_STR),
        ('{memptr}', MEM_SIZE_TOKEN_STR)]
       
       Such that any occurance of any of those exact strings will be replaced with the appropriate constant from the
       bincfg.normalization.norm_utils.py file. If for some reason you want to put the literal string '{str}' into
       your test case... why? Just don't. I don't wanna deal with that right now...

- Add the imports of the ISA_TEST_OBJECTS and ISA_TEST_INPUTS to the tests/normalization/test_manual_lines.py file at the
  top in the ISA_MANUAL_LINE_DATA object

- Make sure you also put in all the needed __init__.py files!

You can see the tests/normalization/x86 directory for an example.

For the cfg's:

- Do stuff