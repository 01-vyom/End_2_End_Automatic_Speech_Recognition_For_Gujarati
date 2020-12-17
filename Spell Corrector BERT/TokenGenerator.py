import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

BERT_VOCAB = 'PATH_TO/multi_cased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'PATH_TO/multi_cased_L-12_H-768_A-12/bert_model.ckpt'

tokenization.validate_case_matches_checkpoint(False,BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=False)

def maskedId(tokens, mask_ind):
    """
    Generate masked id of the tokens.
    """
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = "[MASK]"
    masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids