import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

# BoW, TF-IDF ve Word2Vec için vektör temsilleri oluşturuluyor
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

# NAIVE BAYES (TF-IDF) MODEL
print("\n[Naive Bayes (TF-IDF) modeli eğitiliyor...]")

# eğitim ve test verisi olarak bölüyoruz.
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)
nb_model = MultinomialNB() # Multinomial Naive Bayes modeli

# Naive Bayes modeli eğitimi ve test verisinin tahmini.
nb_model.fit(X_train_nb, y_train_nb) # Naive Bayes modeli eğitimi.
y_pred_nb = nb_model.predict(X_test_nb) # Test verisinin tahmini

# Precision, recall, F1-score ve accuracy hesaplanır ve yazdırılır
print("\nNaive Bayes Sonuçları:")
print(classification_report(y_test_nb, y_pred_nb))
nb_acc = accuracy_score(y_test_nb, y_pred_nb)
print(f"Accuracy: {nb_acc:.4f}")

# TF-IDF + LOGISTIC REGRESSION MODEL

print("\n[TF-IDF + Logistic Regression modeli eğitiliyor...]\n")

# Veriler
X = pd.read_csv("tfidf_vectors.csv")
# Etiket verisini tekrar alıyoruz
y = labels

# Eğitim ve test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model tanımı ve eğitim
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = lr_model.predict(X_test)

print("Logistic Regression Sonuçları:")
print(classification_report(y_test, y_pred))
lr_acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {lr_acc:.4f}")

# Word2Vec + LOGISTIC REGRESSION MODEL
print("\n[Word2Vec + Logistic Regression modeli eğitiliyor...]\n")

# Word2Vec tabanlı verilerle çalışmak için veriyi ayırıyoruz
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(X_w2v, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
w2v_lr_model = LogisticRegression(max_iter=1000)
w2v_lr_model.fit(X_train_w2v, y_train_w2v)

# Tahmin ve sonuçlar
y_pred_w2v_lr = w2v_lr_model.predict(X_test_w2v)

print("Word2Vec + Logistic Regression Sonuçları:")
# precision, recall, F1-score ve accuracy hesaplanır ve yazdırılır
print(classification_report(y_test_w2v, y_pred_w2v_lr, target_names=["Negative", "Positive"]))

w2v_lr_acc = accuracy_score(y_test_w2v, y_pred_w2v_lr)
print(f"Accuracy: {w2v_lr_acc:.4f}")

# Word2Vec + RANDOM FOREST MODEL
print("\n[Word2Vec + Random Forest modeli eğitiliyor...]")

# Eğitim ve test verilerini hazırla (Word2Vec vektörlerinden)
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(
    X_w2v, labels, test_size=0.2, random_state=42
)

# Random Forest modeli oluştur ve eğit
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_w2v, y_train_w2v)
y_pred_w2v_rf = rf_model.predict(X_test_w2v)

# Sonuçları yazdır
print("\nWord2Vec + Random Forest Sonuçları:")
print(classification_report(y_test_w2v, y_pred_w2v_rf, target_names=["Negative", "Positive"]))
rf_acc = accuracy_score(y_test_w2v, y_pred_w2v_rf)
print(f"Accuracy: {rf_acc:.4f}")


# TF-IDF + DNN (DEEP NEURAL NETWORK) MODEL
print("\n[TF-IDF + DNN modeli eğitiliyor...]\n")

# Eğitim ve test verileri (TF-IDF)
X_train_dnn, X_test_dnn, y_train_dnn, y_test_dnn = train_test_split(X_tfidf.toarray(), y_encoded, test_size=0.2, random_state=42)

# Model yapısını tanımlıyoruz
dnn_model = Sequential()
dnn_model.add(Dense(128, input_dim=X_train_dnn.shape[1], activation='relu'))
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Modeli derliyoruz
dnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitiyoruz
dnn_model.fit(X_train_dnn, y_train_dnn, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

# Test verisi ile modelin performansını değerlendiriyoruz
loss, dnn_acc = dnn_model.evaluate(X_test_dnn, y_test_dnn, verbose=0)
print("\nTF-IDF + DNN Sonuçları:")
print(f"Accuracy: {dnn_acc:.4f}")
# accuracy değeri yazdırılıyor


# LSTM MODEL
print("\n[LSTM modeli eğitiliyor...]")

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>") # en sık geçen 5000 kelimeyi alıyoruz. Bunların dışındaki kelimelere "<OOV>" (out of vocabulary) etiketini yapıştırıyoruz.
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

print("\nLSTM Sonuçları:")
print(classification_report(y_true_classes, y_pred_classes)) #  precision, recall, F1-score ve accuracy yazdırılır.
print(f"Accuracy: {lstm_acc:.4f}")


# Tüm modellerin accuracy sonuçlarını karşılaştırıyoruz
print("\nKarşılaştırma:")

# BoW / TF-IDF tabanlı klasik ML modelleri
print(f"Naive Bayes (TF-IDF):               {nb_acc:.4f}")
print(f"Logistic Regression (TF-IDF):       {lr_acc:.4f}")

# Word2Vec tabanlı klasik ML modelleri
print(f"Logistic Regression (Word2Vec):     {w2v_lr_acc:.4f}")
print(f"Random Forest (Word2Vec):           {rf_acc:.4f}")

# Derin Öğrenme modelleri
print(f"DNN (TF-IDF):                       {dnn_acc:.4f}")
print(f"LSTM (Sequence):                    {lstm_acc:.4f}")

# Naive Bayes için diğer metrikleri hesapla
nb_precision = precision_score(y_test_nb, y_pred_nb)
nb_recall = recall_score(y_test_nb, y_pred_nb)
nb_f1 = f1_score(y_test_nb, y_pred_nb)

# LSTM için metrikleri hesapla
lstm_y_pred = model.predict(X_test_lstm) # tahminleri alıyoruz
lstm_y_pred_labels = np.argmax(lstm_y_pred, axis=1) # tahmin sınıfı
lstm_y_true_labels = np.argmax(y_test_lstm, axis=1)# gerçek sınıf
lstm_precision = precision_score(lstm_y_true_labels, lstm_y_pred_labels)
lstm_recall = recall_score(lstm_y_true_labels, lstm_y_pred_labels)
lstm_f1 = f1_score(lstm_y_true_labels, lstm_y_pred_labels)

# Grafik için veri çerçevesi
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Naive Bayes": [nb_acc, nb_precision, nb_recall, nb_f1],
    "LSTM": [lstm_acc, lstm_precision, lstm_recall, lstm_f1]
})

# Veriyi uygun formata getir (grafik çizmek için melt işlemi)
metrics_melted = metrics_df.melt(id_vars="Metric", var_name="Model", value_name="Score")

# Grafik çizimi
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model")
plt.title("Naive Bayes vs. LSTM - Performance Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.legend(title="Model")
plt.tight_layout()
plt.show()

# Accuracy skorları
model_names = [
    "Naive Bayes (TF-IDF)",
    "LogReg (TF-IDF)",
    "LogReg (Word2Vec)",
    "Random Forest (Word2Vec)",
    "DNN (TF-IDF)",
    "LSTM"
]

accuracies = [nb_acc, lr_acc, w2v_lr_acc, rf_acc, dnn_acc, lstm_acc]

# grafiği oluşturuyoruz
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.ylim(0, 1)
plt.title("Model Comparison - Accuracy Scores")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# Bag-of-Words ve Word Embeddings tabanlı modellerin karşılaştırılması
representation_types = [
    "TF-IDF + Naive Bayes",
    "TF-IDF + Logistic Regression",
    "TF-IDF + DNN",
    "Word2Vec + Logistic Regression",
    "Word2Vec + Random Forest"
]

representation_accuracies = [nb_acc, lr_acc, dnn_acc, w2v_lr_acc, rf_acc]

plt.figure(figsize=(10, 6))
plt.bar(representation_types, representation_accuracies, color='mediumseagreen')
plt.ylim(0, 1)
plt.title("Bag-of-Words vs Word Embeddings - Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("bow_vs_wordembeddings_accuracy.png")
plt.show()

# Klasik Makine Öğrenmesi ve Derin Öğrenme modellerinin karşılaştırılması
classical_models = ["Naive Bayes (TF-IDF)", "LogReg (TF-IDF)"]
deep_models = ["DNN (TF-IDF)", "LSTM"]

classical_scores = [nb_acc, lr_acc]
deep_scores = [dnn_acc, lstm_acc]

plt.figure(figsize=(10, 6))
plt.bar(classical_models + deep_models, classical_scores + deep_scores, color='mediumpurple')
plt.ylim(0, 1)
plt.title("Classical ML vs Deep Learning - Accuracy Comparison")
plt.xlabel("Modeller")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("classical_vs_deep.png")
plt.show()
