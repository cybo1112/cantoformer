import tokenizers
from transformers import BertTokenizer
import glob
from tokenizers.implementations import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer()
tokenizer = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True,
)
files = glob.glob("./corpus_for_tokenization/*.txt")

tokenizer.train(
    files,
    vocab_size=50000,
    min_frequency=3,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=15000,
    wordpieces_prefix="##"
)
tokenizer.save_model("./")