import streamlit as st
import pandas as pd
import re
import nltk
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Gerekli NLTK verilerini indir
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Temizleyiciler
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fonksiyonlar
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

def stem_text(text):
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed)

def lemmatize_text(text):
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

# Streamlit uygulaması
def main():
    st.title("🧹 Excel Destekli Text Preprocessing Uygulaması")

    uploaded_file = st.file_uploader("Bir .xlsx dosyası yükleyin (Comment sütunu içermeli)", type="xlsx")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            if "Comment" not in df.columns:
                st.error("Dosyada 'Comment' adlı bir sütun bulunmalı.")
                return

            df['Normalized'] = df['Comment'].apply(normalize)
            df['No_Stopwords'] = df['Normalized'].apply(remove_stopwords)
            df['Stemmed'] = df['No_Stopwords'].apply(stem_text)
            df['Lemmatized'] = df['No_Stopwords'].apply(lemmatize_text)

            st.success("Temizleme işlemi tamamlandı! ✅")
            st.dataframe(df[['Comment', 'Normalized', 'No_Stopwords', 'Stemmed', 'Lemmatized']].head())

            # Excel çıktısı oluşturmak için BytesIO kullan
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Temizlenmiş Veriyi İndir (.xlsx)",
                data=output,
                file_name="processed_comments34.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()

# streamlit run pre_process_app.py ile streamlit arayüzünü çalıştır.