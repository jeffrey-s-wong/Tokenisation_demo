import streamlit as st 
with st.spinner('Importing dependencies...'):
    print('Importing dependencies...')
    import pycantonese
    import jieba
    from transformers import BertTokenizer
    import sentencepiece as spm
    import opencc
    from tokenizers.normalizers import BertNormalizer

print('Dependencies loaded')
st.title('Tokenisation Demo')
st.caption('Built by Jeffrey S Wong')

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_tokenizers():
    with st.spinner('Loading tokenizers...'):
        print('Loading tokenizers...')
        # Load zh Tokenizers
        # Load mandarin word piece tokenizer
        zhword_tokeniser = spm.SentencePieceProcessor()
        zhword_tokeniser.load('assets/zhword.model')

        # Load mandarin sentencePiece tokenizer
        zhsp_tokeniser = spm.SentencePieceProcessor()
        zhsp_tokeniser.load('assets/zhsp.model')

        # Load mandarin character piece tokenizer
        PRETRAINED_MODEL_NAME = "bert-base-chinese" 
        zhchar_tokeniser = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

        # Load yue tokenizers
        # Load cantonese word piece tokeniser
        yueword_tokeniser = spm.SentencePieceProcessor()
        yueword_tokeniser.load('assets/pycan.model')

        # Load cantonese character piece tokeniser
        yuesp_tokeniser = spm.SentencePieceProcessor()
        yuesp_tokeniser.load('assets/sp.model')

        # Load cantonese character piece tokeniser
        yuechar_tokeniser = spm.SentencePieceProcessor()
        yuechar_tokeniser.load('assets/char.model')
    return zhword_tokeniser, zhsp_tokeniser, zhchar_tokeniser, yueword_tokeniser, yuesp_tokeniser, yuechar_tokeniser

class Normaliser():
            def __init__(self, text):
                self.text = text
                self.newstring = ""
                # Use BertNormalizer to convert clean
                self.normalizer = BertNormalizer(clean_text = True, handle_chinese_chars = False, strip_accents = None, lowercase = True)

            def to_trad(self):
                """Convert simplified to traditional"""
                converter = opencc.OpenCC('s2t.json')
                self.text = converter.convert(self.text)
            
            def punctuations(self):
                for uchar in self.text:
                    ucode = ord(uchar)
                    if ucode == 8943: # '⋯'
                        for i in range(3):
                            self.newstring += chr(46)
                    else:
                        if ucode == 12288:
                            ucode = 32
                        elif 65281 <= ucode <= 65374:
                            ucode -= 65248
                        elif ucode == 8943: # '⋯'
                            ucode 
                        self.newstring += chr(ucode)              

            def normalise(self):
                self.to_trad()
                self.text = self.normalizer.normalize_str(self.text)
                self.punctuations()
                return self.newstring


zhword_tokeniser, zhsp_tokeniser, zhchar_tokeniser, yueword_tokeniser, yuesp_tokeniser, yuechar_tokeniser = load_tokenizers()

inp = st.text_input(
    "Chinese sentence to be tokenised:",
    value="Please enter Mandarin or Cantonese sentence here.",
    max_chars=128,
    key=0)

lang_option = st.radio(
    "Please select language of input sentence:",
    ('Mandarin', 'Cantonese'),
    index=0)

token_option = st.radio(
    "Please select tokenisation method:",
    ('Word Piece', 'SentencePiece', 'Character Piece'),
    index=0)

if st.button('Tokenise Sentence'):
    with st.spinner('Tokenisation in progress...'):
        if inp == "":
            st.warning('Please **enter Chinese sentence** for translation')
        else:
            n = Normaliser(inp)
            ninp = n.normalise()
            st.write("Your Mandarin Text:")
            st.write(inp)
            if lang_option == 'Mandarin':
                if token_option == 'Word Piece':
                    st.write("Tokeniser - Mandarin Word Piece")
                    st.write("Segmentation result:")
                    st.write(zhword_tokeniser.encode_as_pieces((' ').join(jieba.cut(ninp))))
                    st.write("Tokenization result:")
                    st.write(zhword_tokeniser.encode_as_ids((' ').join(jieba.cut(ninp))))
                elif token_option == 'SentencePiece':
                    st.write("Tokeniser - Mandarin SentencePiece")
                    st.write("Segmentation result:")
                    st.write(zhsp_tokeniser.encode_as_pieces(ninp)[1:])
                    st.write("Tokenization result:")
                    st.write(zhsp_tokeniser.encode_as_ids(ninp)[1:])
                elif token_option == 'Character Piece':
                    st.write("Tokeniser - Mandarin Character Piece")
                    st.write("Segmentation result:")
                    tokens = zhchar_tokeniser.tokenize(ninp)
                    st.write(tokens)
                    st.write("Tokenization result:")
                    st.write(zhchar_tokeniser.convert_tokens_to_ids(tokens))
            elif lang_option == 'Cantonese':
                if token_option == 'Word Piece':
                    st.write("Tokeniser - Cantonese Word Piece")
                    st.write("Segmentation result:")
                    st.write(yueword_tokeniser.encode_as_pieces((' ').join(pycantonese.segment(ninp))))
                    st.write("Tokenization result:")
                    st.write(yueword_tokeniser.encode_as_ids((' ').join(pycantonese.segment(ninp))))
                elif token_option == 'SentencePiece':
                    st.write("Tokeniser - Cantonese SentencePiece")
                    st.write("Segmentation result:")
                    st.write(yuesp_tokeniser.encode_as_pieces(ninp))
                    st.write("Tokenization result:")
                    st.write(yuesp_tokeniser.encode_as_ids(ninp))
                elif token_option == 'Character Piece':
                    st.write("Tokeniser - Cantonese Character Piece")
                    st.write("Segmentation result:")
                    st.write(yuechar_tokeniser.encode_as_pieces(ninp)[1:])
                    st.write("Tokenization result:")
                    st.write(yuechar_tokeniser.encode_as_ids(ninp)[1:])
else:
    pass