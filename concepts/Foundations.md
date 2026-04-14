
![[Pasted image 20260411152114.png]]
## NLP
### classification
input text -> model -> single value 
- sentiment extraction
- intent detection 
- language
### multi classification
input text -> model -> multi value denoting different things
- named entity recognition
- part of speech tagging
### generation
input text -> model -> out text
- machine translation 
- question answer

### [[Tokenization]]
It takes your sentence and **breaks it into small pieces (tokens)** so the model can understand it.
tokenisers -> word based, character, subword

### token representation
![[Pasted image 20260411163846.png]]

### Word2Vec
👉 If two words appear in similar sentences, they mean similar things
famous example - king - man + woman ≈ queen
#### ⚙️ How Word2Vec Works (Core Idea)
👉 “A word is known by the company it keeps”
Meaning:
- Look at surrounding words
- Learn patterns from them
#### 🔁 Two Main Methods
### 1. Skip-Gram (most important)
👉 Given a word → predict surrounding words
Sentence: "I love machine learning"
Input: "love"
Output: ["I", "machine"]
### 2. CBOW (Continuous Bag of Words)
👉 Given surrounding words → predict center word
Input: ["I", "machine"]
Output: "love"
#### ⚠️ Limitation (VERY IMPORTANT for exams)
Word2Vec gives:  
👉 **one vector per word**
So:
- “bank” (river)
- “bank” (money)
👉 Same vector ❌

### Connections  
- [[Embeddings]]  
- [[Tokenization]]  
- [[Transformer]]

### [[RNNs]]

## Attention
