# SMS Spam Detection & Text Analysis Project

This project applies natural language processing (NLP), sentiment analysis, topic modeling, and BERT-based classification to a dataset of SMS messages labeled as **spam** or **ham**. The full pipeline is implemented in **Google Colab** using **pandas**, **NLTK**, **Gensim**, and **Hugging Face Transformers**.

---

## Dataset

- **Source**: [SMS Spam Collection](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)  
- **Size**: 5,572 SMS messages  
- **Labels**: `spam` or `ham`  
- **Features created**: message length, word count, punctuation count, sentiment score

---

##  Project Workflow

### 1. Text Cleaning & Preprocessing
- Removed stopwords, punctuation, mentions, and links
- Tokenized and normalized messages
- Added custom stopword set for better filtering (e.g., "u", "txt", "gt", etc.)

### 2. Exploratory Data Analysis (EDA)
- Message count: `ham` heavily outweighs `spam`
- Spam messages are longer and contain more punctuation
- Most common spam words: `free`, `claim`, `prize`, `mobile`
- Most common ham words: `ok`, `good`, `time`, `love`

### 3. Sentiment Analysis (VADER)
- Labeled messages as `positive`, `neutral`, or `negative`
- Found that ham messages tend to be more positive
- Spam tends to be neutral or slightly negative

### 4. Topic Modeling (LDA with Gensim)
- Extracted 5 interpretable topics from cleaned SMS data
- Topics included: promotions, reminders, friendly check-ins
- Topic coherence score (c_v): `0.694`
- Word groups plotted with topic weights for interpretation

### 5. Spam Classification using BERT
- Fine-tuned `bert-base-uncased` on cleaned SMS data
- Tokenized using Hugging Face Transformers
- Achieved **99.1% accuracy** on validation set after 3 epochs

---

## BERT Classification Results

| Metric     | Score    |
|------------|----------|
| Accuracy   | 99.1%    |
| Precision  | 99.1%    |
| Recall     | 99.1%    |
| F1 Score   | 99.1%    |

- Fine-tuning time: ~5 minutes in Colab
- Handles short-message classification with excellent performance

---

##  Technologies Used

- **Python**, **Google Colab**
- `pandas`, `matplotlib`, `seaborn`
- `NLTK` for tokenization & sentiment
- `Gensim` for topic modeling
- `Transformers` from Hugging Face for BERT classification
- `PyLDAvis` for optional topic visualization

---

##  Notes

- Custom token cleaning was critical for reliable topic modeling and sentiment analysis
- BERT performance was strong despite minimal hyperparameter tuning
- The entire workflow runs end-to-end in a single notebook for ease of reproducibility

---

