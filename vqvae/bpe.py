import re
from subword_nmt.get_vocab import get_vocab
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


__TOTAL_BPE_TOKENS__ = 800


def create_bpe_tokenizer(filename):

	root, ext = os.path.splitext(filename)

	bpe_file   = root + '_bpe'   + ext
	vocab_file = root + '_vocab' + ext

	if os.path.exists(bpe_file):
		bpe_codes = open(bpe_file, 'r')
		tokenizer = BPE(bpe_codes)
		return tokenizer
	
	else:

		with open(filename, 'r') as a, \
			 open(bpe_file, 'w') as b:
			learn_bpe(a, b, __TOTAL_BPE_TOKENS__)

		with open(bpe_file,   'r') as a,\
			 open(vocab_file, 'w') as b:
			get_vocab(a, b)

		bpe_codes = open(bpe_file, 'r')
		tokenizer = BPE(bpe_codes)
		return tokenizer


