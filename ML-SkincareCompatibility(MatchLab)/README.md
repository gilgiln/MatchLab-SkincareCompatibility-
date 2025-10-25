# Skincare Compatibility ML API

Proyek Machine Learning untuk memprediksi apakah dua produk skincare cocok digunakan bersama berdasarkan kesamaan bahan aktifnya.

---

## Cara Menjalankan (untuk tim backend)

### Instal dependensi
Pastikan sudah menginstal semua library yang diperlukan:

pip install -r requirements.txt

###  Jalankan API
python app.py

API akan berjalan di:
- http://localhost:5000 (jika dijalankan di PC)
- atau URL ngrok (jika dijalankan di Google Colab)

---

## Endpoint API

Endpoint: POST /predict

Request JSON:
{
  "product1": "cerave cream",
  "product2": "ordinary hyaluronic acid"
}

Response JSON:
{
  "status": "ok",
  "product1": "CeraVe Moisturising Cream 454g",
  "product2": "The Ordinary Hyaluronic Acid 2% + B5",
  "result": "✅ Cocok digunakan bersama",
  "confidence": 0.87
}

---

## Catatan

- Model dilatih menggunakan RandomForestClassifier dengan fitur:
  - Panjang nama produk (len_diff)
  - Jumlah bahan aktif yang sama (shared)
  - Jaccard similarity
  - Cosine similarity
- Dataset: unified_cleaned_products.csv
- Versi Python yang disarankan: Python 3.10+

---

