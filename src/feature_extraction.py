import pandas as pd
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from lexicalrichness import LexicalRichness
# Kaggle dosyası
df = pd.read_csv("train_with_error_density.csv")
print(f"Veri yüklendi: {df.shape}")

# spacy modelini yükle
# en_core_web_sm → küçük İngilizce modeli, cümle ve bağımlılık analizi yapabilir
nlp = spacy.load("en_core_web_sm")

# nltk cümle tokenizer için gerekli veriyi indir
# sent_tokenize → metni cümlelere böler
nltk.download("punkt")
nltk.download("punkt_tab")

#Type-Token Ratio
def get_ttr(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def get_mtld(text):
    try:
        # LexicalRichness metni alıp zenginlik hesaplamasını yapıyor
        lex = LexicalRichness(text)
        # en az 10 kelime olmalı
        if lex.words < 10:
            return 0.0
        return lex.mtld(threshold=0.72)
    except:
        return 0.0

#cümle uzunluğu özellikleri
def get_sentence_features(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return 0.0, 0.0
    
    lengths = [len(s.split()) for s in sentences]
    
    return np.mean(lengths), np.std(lengths)

#bağımlılık ağacı derinliği
def get_dep_depth(text):
    # spaCy ile ilk 3 cümle analizi (hız için)
    sentences = sent_tokenize(text)[:3]
    depths = []
    
    for sent in sentences:
        doc = nlp(sent)
        # Her token için kökten uzaklık
        for token in doc:
            depth = 0
            current = token
            # Köke kadar yukarı çık
            while current.head != current:
                current = current.head
                depth += 1
            depths.append(depth)
    
    return np.mean(depths) if depths else 0.0

#discourse bağlacı sıklığı
DISCOURSE_WORDS = [
    "however", "therefore", "furthermore", "moreover", "consequently",
    "nevertheless", "in conclusion", "in contrast", "on the other hand",
    "for example", "for instance", "in addition", "as a result"
]

def get_discourse_ratio(text):
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0
    
    count = 0
    for phrase in DISCOURSE_WORDS:
        count += text_lower.count(phrase)
    
    return count / len(words)

#tüm özellikler hesaplanması
print("TTR hesaplanıyor...")
df["ttr"] = df["full_text"].apply(get_ttr)
print("MTLD hesaplanıyor...")
df["mtld"] = df["full_text"].apply(get_mtld)
print("Cümle özellikleri hesaplanıyor...")
sentence_features = df["full_text"].apply(get_sentence_features)

df["avg_sentence_length"] = sentence_features.apply(lambda x: x[0])
df["std_sentence_length"] = sentence_features.apply(lambda x: x[1])

print("Bağımlılık derinliği hesaplanıyor...")
df["dep_depth"] = df["full_text"].apply(get_dep_depth)

print("Discourse oranı hesaplanıyor...")
df["discourse_ratio"] = df["full_text"].apply(get_discourse_ratio)

print("\nTamamlandı! Skora göre ortalamalar:")
feature_cols = ["error_density", "ttr", "avg_sentence_length", "std_sentence_length", "dep_depth", "discourse_ratio"]
print(df.groupby("score")[feature_cols].mean().round(3))

df.to_csv("train_with_features.csv", index=False)
print("\ntrain_with_features.csv kaydedildi.")