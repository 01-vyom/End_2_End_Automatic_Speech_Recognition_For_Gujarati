import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
from TokenGenerator import maskedId

BERT_VOCAB = 'PATH_TO/multi_cased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'PATH_TO/multi_cased_L-12_H-768_A-12/bert_model.ckpt'

tokenization.validate_case_matches_checkpoint(False,BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=False)

def generateId(mask):
    """
    Generate tokens, input ids and token ids that are to be passed to the BERT model.
    """
    tokens = tokenizer.tokenize(mask)
    input_ids = [maskedId(tokens, i) for i in range(len(tokens))]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, tokens_ids