import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Excel dosyasını oku. Eğer lemmatized veya Label sütunlarından bir tanesi eksikse o satırları siliyor. Hata olursa programdan çık.
try:
    df = pd.read_excel("processed_comments34.xlsx")
    df = df.dropna(subset=["Lemmatized", "Label"])
except Exception as e:
    print("Veri dosyası yüklenemedi:", e)
    exit()

# Etiketleri lowercase yap ve pozitif için 1, negatif için 0 değerlerini ata.
texts = df["Lemmatized"].astype(str).tolist()
labels = df["Label"].str.lower().map({"negative": 0, "positive": 1})
# lstm için one-hot encoded hale getiriyoruz.
y_encoded = labels
y_categorical = to_categorical(y_encoded)

# toplam kaç yorum işlendiğini yazıyoruz.
print(f"Toplam {len(texts)} yorum işlendi.")

# BoW, TF-IDF ve Word2Vec için Vektör Temsilleri yapıyoruz
print("\n[Vektörler hazırlanıyor...]")

# Bag of Words için vektör temsili oluşturuyoruz ve csv dosyasına kaydediyoruz.
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(texts)
pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out()).to_csv("bow_vectors.csv", index=False)

# TF-IDF için vektör temsili oluşturuyoruz ve csv dosyasına kaydediyoruz.
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)
pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).to_csv("tfidf_vectors.csv", index=False)

# Word2Vec için tokenize yapıyoruz.
tokenized_texts = [text.split() for text in texts]
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, sg=0)

# 100 boyutlu Word2Vec vektörü oluşturuyoruz.
def average_vector(words, model, vector_size=100):
    vecs = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

# average_vector fonksiyonuyla her yorum için Word2Vec vektörü oluşturuyoruz
X_w2v = np.array([average_vector(words, w2v_model) for words in tokenized_texts])
pd.DataFrame(X_w2v).to_csv("word2vec_vectors.csv", index=False)

# NAİVE BAYES MODEL
print("\n[Naive Bayes modeli eğitiliyor...]")

# eğitim ve test verisi olarak bölüyoruz.
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)
nb_model = MultinomialNB() # Multinomial Naive Bayes modeli

nb_model.fit(X_train_nb, y_train_nb) # Naive Bayes modeli eğitimi.
y_pred_nb = nb_model.predict(X_test_nb) # test verisinin tahmini

print("\nNaive Bayes Sonuçları:")
print(classification_report(y_test_nb, y_pred_nb)) # precision, recall, F1-score ve accuracy yazdırılır.
nb_acc = accuracy_score(y_test_nb, y_pred_nb) # test verisi için accuracy (hiç görmediği veri üzerindeki accuracy (gerçek performanıs yansıtır)).
print(f"Accuracy: {nb_acc:.4f}")

# LSTM MODEL
print("\n[LSTM modeli eğitiliyor...]")

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # en sık geçen 5000 kelimeyi alıyoruz. Bunların dışındaki kelimelere "<OOV>" (out of vocabulary) etiketini yapıştırıyoruz.
tokenizer.fit_on_texts(texts) # tokenlara bölüyoruz
X_seq = tokenizer.texts_to_sequences(texts) #kelimeleri dizilere çeviriyoruz.
X_pad = pad_sequences(X_seq, maxlen=100) # her dizi 100 uzunlukta olacak (lstm için sabit uzunluk gerekiyor.)
vocab_size = len(tokenizer.word_index) + 1 # embedding katmanında indexler 1'den başladığı için 1 ekliyoruz.(tüm indexler kapansın)

# veriyi eğitim ve test bölümlerine ayırıyoruz.
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_pad, y_categorical, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=100), # her kelimeyi 64 boyutlu vektörlere gömüyoruz
    LSTM(64, dropout=0.2, recurrent_dropout=0.2), # overfittingi engellemek için verinin %20'sini yok sayıyoruz.
    Dense(2, activation='softmax') # 2 sınıfımız var (positive,negative) ve 1 üzerinden olasılıklandırıyoruz.
])

# modeli compile ediyoruz.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # adam optimizer kullanıyoruz. Tahmin ve gerçek arasındaki farkı buluyoruz.

# model eğitimi başlıyor.
history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=15, batch_size=16, # tüm veriyi 15 kez tarayarak aynı anda 16 örnekle training.
    validation_split=0.1, # verinin %10' validation için ayrılır.
    verbose=1 # terminalden takip etmemizi sağlar.
)

loss, lstm_acc = model.evaluate(X_test_lstm, y_test_lstm, verbose=0) # test verilerini verip loss ve accuracy değerlerini alıyoruz.

y_pred_lstm = model.predict(X_test_lstm) # tahminleri alıyoruz
y_pred_classes = np.argmax(y_pred_lstm, axis=1) # tahmin sınıfı
y_true_classes = np.argmax(y_test_lstm, axis=1) # gerçek sınıf

print("\nLSTM sonuçları:")
print(classification_report(y_true_classes, y_pred_classes)) # # precision, recall, F1-score ve accuracy yazdırılır.
print(f"Accuracy: {lstm_acc:.4f}")

# karşılaştırma çıktımız.
print("\nKarşılaştırma:")
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
print(f"LSTM Accuracy:       {lstm_acc:.4f}")

