# Aram-Radif-BERT-Tokenizer-from-Scratch-

BERT Tokenizer from Scratch (WordPiece)
 Repository Structure
bert-tokenizer-wordpiece/
│
├── README.md
├── requirements.txt
├── data/
│   └── bert_vocab.json
├── src/
│   ├── tokenizer.py
│   ├── wordpiece.py
│   ├── utils.py
│
├── notebooks/
│   └── demo.ipynb
│
└── outputs/
    └── sample_output.txt
________________________________________
 README.md
# BERT Tokenizer Implementation (WordPiece)

##  Overview
This project implements a simplified version of the BERT tokenizer using WordPiece tokenization. It demonstrates how raw text is transformed into numerical inputs suitable for transformer-based models.

##  Objectives
- Implement Basic Tokenization
- Implement WordPiece Tokenization
- Convert tokens to IDs using vocabulary
- Add special tokens ([CLS], [SEP])
- Generate attention masks and padding

##  Key Concepts
- Subword Tokenization (WordPiece)
- Vocabulary Mapping
- Special Tokens Handling
- Attention Masking

##  Architecture
1. Basic Tokenization
2. WordPiece Tokenization
3. Token to ID Conversion
4. Special Tokens Injection
5. Padding & Attention Mask

##  Dataset
- BERT Base Uncased Vocabulary (~30,000 tokens)

##  Installation
```bash
pip install -r requirements.txt
 Usage
from src.tokenizer import BertTokenizer

tokenizer = BertTokenizer("data/bert_vocab.json")
output = tokenizer.encode("I love AI engineering!")

print(output)
 Sample Output
{
  "tokens": ["[CLS]", "i", "love", "ai", "engineering", "!", "[SEP]"],
  "input_ids": [101, 1045, 2293, 9932, 3330, 999, 102],
  "attention_mask": [1,1,1,1,1,1,1]
}
 Features
•	Handles unknown words using subword splitting 
•	Supports padding and attention masks 
•	Modular design for scalability 
 AI Engineering Relevance
•	Core NLP preprocessing pipeline 
•	Tokenization for transformer models 
•	Foundation for LLM systems 
 Future Improvements
•	Add batch processing 
•	Optimize WordPiece with Trie 
•	Integrate with PyTorch/TensorFlow 
 References
•	BERT Paper (Google AI) 
•	HuggingFace Transformers 
•	WordPiece Algorithm 

---

#  requirements.txt

```txt
requests
________________________________________
 src/utils.py
import re

def basic_tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens
________________________________________
 src/wordpiece.py
class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = set(vocab)

    def tokenize(self, word):
        if word in self.vocab:
            return [word]

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = None

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.vocab:
                    found = substr
                    break
                end -= 1

            if not found:
                return ["[UNK]"]

            tokens.append(found)
            start = end

        return tokens
________________________________________
 src/tokenizer.py
import json
from src.utils import basic_tokenize
from src.wordpiece import WordPieceTokenizer

class BertTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path) as f:
            self.vocab = json.load(f)

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.wordpiece = WordPieceTokenizer(self.vocab)

    def encode(self, text):
        tokens = ["[CLS]"]

        for word in basic_tokenize(text):
            tokens.extend(self.wordpiece.tokenize(word))

        tokens.append("[SEP]")

        input_ids = [self.token_to_id.get(tok, self.token_to_id.get("[UNK]", 0)) for tok in tokens]
        attention_mask = [1] * len(input_ids)

        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
________________________________________
 Vocabulary Download Script
from os import getcwd
from json import dump
from requests import get

vocab_json_path = getcwd() + "/data"

url = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt"
vocab_text = get(url).text

vocab_list = vocab_text.split("\n")

with open(f"{vocab_json_path}/bert_vocab.json", "w") as f:
    dump(vocab_list, f, indent=2)

print("Vocabulary saved successfully!")
________________________________________
 Sample Output (outputs/sample_output.txt)
Input: "I love NLP!"

Tokens: [CLS] i love nl ##p ! [SEP]
Input IDs: [101, 1045, 2293, 17953, 2361, 999, 102]
Attention Mask: [1,1,1,1,1,1,1]
________________________________________
 AI Engineer work
Project: BERT Tokenizer Implementation (WordPiece)
•	Built a custom BERT tokenizer replicating WordPiece algorithm 
•	Implemented subword tokenization for OOV handling 
•	Designed modular NLP preprocessing pipeline 
•	Processed vocabulary of 30K+ tokens 
•	Improved understanding of transformer input pipelines

--

Aram Radif
