#  Skincare Compatibility API (Flask)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

#  Load model dan dataset
model = joblib.load("model_training/skincare_model.pkl")
df = pd.read_csv("model_training/unified_cleaned_products.csv", encoding="utf-8", engine="python")

# Load compatibility rules untuk penjelasan
try:
    df_rules = pd.read_csv("model_training/compatibility_rules.csv", encoding="utf-8")
    print(f"Compatibility rules loaded: {len(df_rules)} rules")
except FileNotFoundError:
    df_rules = None
    print("Compatibility rules not found, explanations will be generic")

# Pastikan parsed_ingredients dalam bentuk list
df['parsed_ingredients'] = df['parsed_ingredients'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

print(f"Model loaded successfully!")
print(f"Dataset loaded: {len(df)} products")

#  Definisikan fungsi bantu
def find_closest_product_name(name, df, cutoff=0.3):
    """Cari nama produk paling mirip dari dataset (fuzzy matching)."""
    all_names = df["product_name"].tolist()
    matches = get_close_matches(name, all_names, n=5, cutoff=cutoff)
    return matches

def get_ingredient_explanation(ing1_list, ing2_list, is_compatible):
    """Generate penjelasan detail tentang interaksi bahan."""
    shared_ingredients = list(set(ing1_list) & set(ing2_list))
    
    # Database penjelasan umum untuk bahan-bahan populer
    ingredient_benefits = {
        'niacinamide': 'membantu memperbaiki skin barrier dan mengurangi kemerahan',
        'hyaluronic acid': 'menarik dan mengikat kelembaban hingga 1000x beratnya',
        'vitamin c': 'antioksidan kuat yang mencerahkan dan melindungi dari radikal bebas',
        'retinol': 'meningkatkan regenerasi sel dan produksi kolagen',
        'salicylic acid': 'eksfoliasi dan membersihkan pori-pori',
        'glycolic acid': 'eksfoliasi permukaan kulit untuk kulit lebih halus',
        'peptides': 'merangsang produksi kolagen dan elastin',
        'ceramides': 'memperkuat skin barrier dan mencegah kehilangan kelembaban',
        'glycerin': 'humektan yang menarik kelembaban ke kulit',
        'panthenol': 'menenangkan dan melembabkan kulit',
        'centella asiatica': 'menenangkan dan memperbaiki kulit',
        'green tea': 'antioksidan yang melindungi dari kerusakan lingkungan'
    }
    
    if is_compatible:
        if len(shared_ingredients) > 0:
            explanation = f"Kedua produk dapat digunakan bersama karena memiliki {len(shared_ingredients)} bahan yang sama, yang menunjukkan kompatibilitas formulasi. "
            
            # Cari bahan yang dikenal dan berikan benefit
            known_shared = [ing for ing in shared_ingredients if ing.lower() in ingredient_benefits]
            if known_shared:
                ingredient = known_shared[0].lower()
                explanation += f"Bahan seperti {known_shared[0]} yang terdapat di keduanya {ingredient_benefits[ingredient]}, sehingga efeknya bisa saling mendukung."
            else:
                explanation += "Bahan-bahan yang terdapat di kedua produk saling melengkapi dan tidak ada interaksi negatif yang diketahui."
        else:
            explanation = "Meskipun tidak memiliki bahan yang sama, kedua produk ini aman digunakan bersamaan. Tidak ada interaksi negatif yang diketahui antara bahan-bahan di kedua produk."
        
        # Tips penggunaan
        tips = "Tips: Aplikasikan produk dari tekstur paling ringan ke paling berat. Tunggu 1-2 menit antara aplikasi untuk penyerapan optimal."
        
    else:
        # Tidak compatible
        explanation = "Kombinasi ini tidak disarankan karena kemungkinan ada interaksi bahan yang bisa mengurangi efektivitas atau meningkatkan risiko iritasi. "
        
        # Cek apakah ada kombinasi yang diketahui bermasalah
        problematic_pairs = [
            ('retinol', 'vitamin c'),
            ('retinol', 'aha'),
            ('retinol', 'bha'),
            ('retinol', 'salicylic acid'),
            ('retinol', 'glycolic acid'),
            ('benzoyl peroxide', 'retinol'),
            ('vitamin c', 'niacinamide')  # kontroversial tapi tetap hati-hati
        ]
        
        found_problematic = False
        for ing1 in ing1_list:
            for ing2 in ing2_list:
                for pair in problematic_pairs:
                    if (pair[0] in ing1.lower() and pair[1] in ing2.lower()) or \
                       (pair[1] in ing1.lower() and pair[0] in ing2.lower()):
                        explanation += f"Kombinasi {pair[0].title()} dan {pair[1].title()} dapat menyebabkan iritasi atau mengurangi efektivitas masing-masing bahan."
                        found_problematic = True
                        break
                if found_problematic:
                    break
            if found_problematic:
                break
        
        if not found_problematic:
            explanation += "Bahan-bahan dalam kedua produk mungkin bekerja pada pH yang berbeda atau dapat meningkatkan sensitivitas kulit."
        
        tips = "Saran: Gunakan produk ini pada waktu berbeda (pagi/malam) atau pada hari yang berbeda untuk menghindari iritasi."
    
    return explanation, tips

def calculate_compatibility(ing1, ing2):
    """Hitung compatibility score antara 2 produk berdasarkan ingredients."""
    # --- Buat fitur ---
    len_diff = abs(len(ing1) - len(ing2))
    shared = len(set(ing1) & set(ing2))
    jaccard = len(set(ing1) & set(ing2)) / len(set(ing1) | set(ing2)) if len(set(ing1) | set(ing2)) > 0 else 0

    vec = CountVectorizer().fit([" ".join(ing1), " ".join(ing2)])
    tf1 = vec.transform([" ".join(ing1)])
    tf2 = vec.transform([" ".join(ing2)])
    cosine_sim = cosine_similarity(tf1, tf2)[0][0]

    X_new = np.array([[len_diff, shared, jaccard, cosine_sim]])
    pred = model.predict(X_new)[0]

    # Ambil confidence dengan aman
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)[0]
        prob = probs[pred] if probs.size > 1 else 1.0
    else:
        prob = 1.0
    
    return pred, float(prob), shared

def get_recommendations(product_name, df, model, top_n=5):
    """Dapatkan rekomendasi produk yang kompatibel dengan produk input (OPTIMIZED)."""
    # Cari produk yang dimaksud
    matches = find_closest_product_name(product_name, df)
    if not matches:
        return None, None
    
    target_name = matches[0]
    target_product = df[df["product_name"] == target_name].iloc[0]
    target_ingredients = target_product["parsed_ingredients"]
    
    # OPTIMASI: Vectorize semua ingredients sekali saja
    vectorizer = CountVectorizer()
    
    # Join ingredients untuk target product
    target_text = " ".join(target_ingredients)
    
    # Prepare batch data untuk semua produk lain
    other_products = df[df["product_name"] != target_name].copy()
    other_texts = other_products["parsed_ingredients"].apply(lambda x: " ".join(x))
    
    # Vectorize target + all others dalam 1x operasi
    all_texts = [target_text] + other_texts.tolist()
    vectors = vectorizer.fit_transform(all_texts)
    
    # Hitung cosine similarity untuk semua produk sekaligus
    target_vector = vectors[0]
    other_vectors = vectors[1:]
    similarities = cosine_similarity(target_vector, other_vectors).flatten()
    
    # BATCH PREDICTION: Predict semua sekaligus (jauh lebih cepat!)
    # Gabungkan features untuk batch prediction
    tf_features_1 = target_vector.toarray().repeat(len(other_products), axis=0)
    tf_features_2 = other_vectors.toarray()
    
    # Gabungkan features
    features_batch = np.hstack([tf_features_1, tf_features_2, similarities.reshape(-1, 1)])
    
    # Predict batch (1x prediction untuk semua produk!)
    predictions = model.predict(features_batch)
    probabilities = model.predict_proba(features_batch)[:, 1]
    
    # Filter hanya yang compatible dan buat list rekomendasi
    recommendations = []
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if pred == 1:  # Compatible
            other_name = other_products.iloc[idx]["product_name"]
            other_ingredients = other_products.iloc[idx]["parsed_ingredients"]
            shared_count = len(set(target_ingredients) & set(other_ingredients))
            
            recommendations.append({
                "product_name": other_name,
                "confidence": round(float(prob), 2),
                "shared_ingredients_count": shared_count
            })
    
    # Sort by confidence (descending)
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)
    
    return target_name, recommendations[:top_n]

#  Inisialisasi Flask app
app = Flask(__name__)

# Enable CORS for all routes (untuk Flutter/mobile apps)
CORS(app)

@app.route("/")
def home():
    return jsonify({
        "message": "Skincare Compatibility API is running!",
        "version": "1.3.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "endpoints": {
            "/predict": "POST - Check compatibility between two products",
            "/recommend": "POST - Get product recommendations",
            "/ingredient-info": "POST - Get ingredient information"
        }
    })

@app.route("/health")
def health():
    """Health check endpoint untuk monitoring"""
    return jsonify({
        "status": "healthy",
        "version": "1.3.0",
        "model_loaded": model is not None,
        "dataset_loaded": df is not None,
        "total_products": len(df) if df is not None else 0
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint untuk mengecek kompatibilitas 2 produk."""
    data = request.get_json()
    product1 = data.get("product1")
    product2 = data.get("product2")

    if not product1 or not product2:
        return jsonify({
            "status": "error",
            "message": "Harap masukkan dua nama produk."
        }), 400

    # Cari nama produk paling mirip
    matches1 = find_closest_product_name(product1, df)
    matches2 = find_closest_product_name(product2, df)

    if not matches1 or not matches2:
        return jsonify({
            "status": "error",
            "message": "❌ Salah satu produk tidak ditemukan.",
            "suggestions1": matches1 or [],
            "suggestions2": matches2 or []
        }), 404

    name1, name2 = matches1[0], matches2[0]
    p1 = df[df["product_name"] == name1].iloc[0]
    p2 = df[df["product_name"] == name2].iloc[0]

    ing1, ing2 = p1["parsed_ingredients"], p2["parsed_ingredients"]

    # Gunakan fungsi calculate_compatibility
    pred, prob, shared_count = calculate_compatibility(ing1, ing2)
    
    # Generate explanation & tips
    is_compatible = pred == 1
    explanation, tips = get_ingredient_explanation(ing1, ing2, is_compatible)
    
    # Get shared ingredients names
    shared_ingredients = list(set(ing1) & set(ing2))

    result = "Cocok digunakan bersama" if pred == 1 else "Tidak disarankan dipakai bersama"

    response_data = {
        "status": "ok",
        "product1": name1,
        "product2": name2,
        "result": result,
        "confidence": round(prob, 2),
        "shared_ingredients_count": shared_count,
        "shared_ingredients": shared_ingredients[:5] if len(shared_ingredients) > 5 else shared_ingredients,
        "explanation": explanation,
        "tips": tips
    }
    
    # Jika TIDAK COCOK, kasih rekomendasi alternatif
    if pred == 0:
        # Dapatkan rekomendasi untuk masing-masing produk (top 3)
        _, recommendations_for_product1 = get_recommendations(name1, df, model, top_n=3)
        _, recommendations_for_product2 = get_recommendations(name2, df, model, top_n=3)
        
        response_data["alternative_recommendations"] = {
            "note": "Karena kedua produk tidak cocok digunakan bersamaan, berikut adalah rekomendasi produk alternatif yang bisa digunakan:",
            "replace_product1": {
                "info": f"Ganti '{name1}' dengan salah satu produk berikut yang cocok dengan '{name2}':",
                "recommendations": recommendations_for_product1
            },
            "replace_product2": {
                "info": f"Atau ganti '{name2}' dengan salah satu produk berikut yang cocok dengan '{name1}':",
                "recommendations": recommendations_for_product2
            }
        }
    
    return jsonify(response_data)

@app.route("/recommend", methods=["POST"])
def recommend():
    """Endpoint untuk mendapatkan rekomendasi produk yang kompatibel."""
    data = request.get_json()
    product = data.get("product")
    top_n = data.get("top_n", 5)
    
    if not product:
        return jsonify({
            "status": "error",
            "message": "Harap masukkan nama produk."
        }), 400
    
    # Validasi top_n
    if not isinstance(top_n, int) or top_n < 1 or top_n > 20:
        top_n = 5
    
    # Dapatkan rekomendasi
    target_name, recommendations = get_recommendations(product, df, model, top_n)
    
    if target_name is None:
        return jsonify({
            "status": "error",
            "message": "Produk tidak ditemukan di database.",
            "suggestions": find_closest_product_name(product, df) or []
        }), 404
    
    # Get product info for context
    target_product = df[df["product_name"] == target_name].iloc[0]
    target_ingredients = target_product["parsed_ingredients"]
    
    # Add brief explanation for top recommendation
    recommendation_note = ""
    if len(recommendations) > 0:
        top_rec = recommendations[0]
        if top_rec['shared_ingredients_count'] > 0:
            recommendation_note = f"Produk teratas memiliki {top_rec['shared_ingredients_count']} bahan yang sama dengan {target_name}, menunjukkan kompatibilitas formulasi yang baik."
        else:
            recommendation_note = f"Produk-produk ini diprediksi kompatibel dengan {target_name} berdasarkan analisis komposisi bahan."
    
    return jsonify({
        "status": "ok",
        "product": target_name,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
        "note": recommendation_note,
        "general_tip": "💡 Untuk hasil terbaik, lakukan patch test terlebih dahulu dan perhatikan reaksi kulit Anda."
    })

@app.route("/ingredient-info", methods=["POST"])
def ingredient_info():
    """Endpoint untuk mendapatkan informasi detail tentang bahan skincare."""
    data = request.get_json()
    ingredient = data.get("ingredient", "").lower().strip()
    
    if not ingredient:
        return jsonify({
            "status": "error",
            "message": "Harap masukkan nama bahan."
        }), 400
    
    # Database informasi bahan
    ingredient_database = {
        'niacinamide': {
            'name': 'Niacinamide (Vitamin B3)',
            'benefits': [
                'Memperbaiki skin barrier',
                'Mengurangi kemerahan dan inflamasi',
                'Mengontrol produksi sebum',
                'Mencerahkan noda bekas jerawat',
                'Meminimalkan tampilan pori-pori'
            ],
            'suitable_for': ['Semua jenis kulit', 'Kulit berjerawat', 'Kulit sensitif', 'Kulit berminyak'],
            'compatible_with': ['Hyaluronic Acid', 'Peptides', 'Ceramides', 'Retinol'],
            'avoid_with': ['Vitamin C dalam konsentrasi tinggi (masih diperdebatkan)'],
            'tips': 'Gunakan pagi dan malam. Cocok dikombinasikan dengan hampir semua bahan aktif.',
            'concentration': 'Efektif pada konsentrasi 2-10%'
        },
        'hyaluronic acid': {
            'name': 'Hyaluronic Acid',
            'benefits': [
                'Mengikat kelembaban hingga 1000x beratnya',
                'Membuat kulit tampak plump dan kenyal',
                'Mengurangi tampilan garis halus',
                'Memperbaiki tekstur kulit',
                'Aman untuk semua jenis kulit'
            ],
            'suitable_for': ['Semua jenis kulit', 'Kulit kering', 'Kulit dehidrasi', 'Kulit sensitif'],
            'compatible_with': ['Niacinamide', 'Vitamin C', 'Retinol', 'Peptides', 'Ceramides'],
            'avoid_with': ['Tidak ada (sangat aman)'],
            'tips': 'Aplikasikan pada kulit yang sedikit lembab untuk hasil maksimal. Ikuti dengan moisturizer untuk lock in moisture.',
            'concentration': 'Efektif pada konsentrasi 0.5-2%'
        },
        'vitamin c': {
            'name': 'Vitamin C (L-Ascorbic Acid)',
            'benefits': [
                'Antioksidan kuat melawan radikal bebas',
                'Mencerahkan kulit dan memudarkan dark spots',
                'Meningkatkan produksi kolagen',
                'Melindungi dari kerusakan UV',
                'Meratakan warna kulit'
            ],
            'suitable_for': ['Kulit kusam', 'Kulit dengan hiperpigmentasi', 'Anti-aging', 'Semua jenis kulit'],
            'compatible_with': ['Vitamin E', 'Ferulic Acid', 'Hyaluronic Acid', 'Sunscreen'],
            'avoid_with': ['Retinol', 'AHA/BHA', 'Niacinamide (kontroversial)', 'Benzoyl Peroxide'],
            'tips': 'Gunakan di pagi hari dengan sunscreen. Simpan di tempat gelap dan sejuk. Cari formula stabil (pH 2.5-3.5).',
            'concentration': 'Efektif pada konsentrasi 10-20%'
        },
        'retinol': {
            'name': 'Retinol (Vitamin A)',
            'benefits': [
                'Meningkatkan cell turnover',
                'Mengurangi garis halus dan kerutan',
                'Meningkatkan produksi kolagen',
                'Mengatasi jerawat dan pori-pori besar',
                'Mencerahkan kulit'
            ],
            'suitable_for': ['Anti-aging', 'Kulit berjerawat', 'Kulit dengan hiperpigmentasi'],
            'compatible_with': ['Hyaluronic Acid', 'Niacinamide', 'Peptides', 'Ceramides'],
            'avoid_with': ['Vitamin C', 'AHA/BHA', 'Benzoyl Peroxide'],
            'tips': 'Mulai dengan konsentrasi rendah. Gunakan malam hari. WAJIB pakai sunscreen di pagi hari. Tidak untuk ibu hamil/menyusui.',
            'concentration': 'Mulai dari 0.25%, tingkatkan bertahap hingga 1%'
        },
        'salicylic acid': {
            'name': 'Salicylic Acid (BHA)',
            'benefits': [
                'Eksfoliasi di dalam pori-pori',
                'Mengatasi jerawat dan komedo',
                'Mengurangi produksi minyak',
                'Anti-inflamasi',
                'Menghaluskan tekstur kulit'
            ],
            'suitable_for': ['Kulit berjerawat', 'Kulit berminyak', 'Kulit dengan blackheads/whiteheads'],
            'compatible_with': ['Niacinamide', 'Hyaluronic Acid'],
            'avoid_with': ['Retinol', 'AHA', 'Vitamin C'],
            'tips': 'Gunakan 2-3x seminggu untuk pemula. Jangan over-exfoliate. Selalu pakai sunscreen.',
            'concentration': 'Efektif pada konsentrasi 0.5-2%'
        },
        'glycolic acid': {
            'name': 'Glycolic Acid (AHA)',
            'benefits': [
                'Eksfoliasi permukaan kulit',
                'Mencerahkan kulit kusam',
                'Mengurangi hiperpigmentasi',
                'Meningkatkan penyerapan produk lain',
                'Menghaluskan tekstur kulit'
            ],
            'suitable_for': ['Kulit kusam', 'Kulit dengan sun damage', 'Kulit dengan tekstur tidak rata'],
            'compatible_with': ['Hyaluronic Acid', 'Niacinamide'],
            'avoid_with': ['Retinol', 'BHA', 'Vitamin C'],
            'tips': 'Mulai dengan konsentrasi rendah. Gunakan malam hari. WAJIB sunscreen di pagi hari.',
            'concentration': 'Mulai dari 5%, tingkatkan hingga 10%'
        },
        'ceramides': {
            'name': 'Ceramides',
            'benefits': [
                'Memperkuat skin barrier',
                'Mencegah kehilangan kelembaban',
                'Melindungi dari iritasi',
                'Menenangkan kulit sensitif',
                'Mempertahankan kelembaban kulit'
            ],
            'suitable_for': ['Semua jenis kulit', 'Kulit kering', 'Kulit sensitif', 'Kulit dengan barrier rusak'],
            'compatible_with': ['Semua bahan aktif', 'Sangat versatile'],
            'avoid_with': ['Tidak ada'],
            'tips': 'Sangat bagus dikombinasikan dengan bahan aktif lain untuk mengurangi iritasi.',
            'concentration': 'Efektif pada konsentrasi 1-5%'
        },
        'peptides': {
            'name': 'Peptides',
            'benefits': [
                'Merangsang produksi kolagen',
                'Mengurangi garis halus dan kerutan',
                'Memperbaiki elastisitas kulit',
                'Memperkuat skin barrier',
                'Anti-aging'
            ],
            'suitable_for': ['Anti-aging', 'Semua jenis kulit', 'Kulit mature'],
            'compatible_with': ['Niacinamide', 'Hyaluronic Acid', 'Retinol', 'Vitamin C'],
            'avoid_with': ['Tidak ada (sangat aman)'],
            'tips': 'Sangat gentle dan cocok dikombinasikan dengan hampir semua bahan.',
            'concentration': 'Bervariasi tergantung jenis peptide'
        }
    }
    
    # Cari ingredient
    ingredient_info_data = ingredient_database.get(ingredient)
    
    if not ingredient_info_data:
        # Coba cari partial match
        for key in ingredient_database.keys():
            if ingredient in key or key in ingredient:
                ingredient_info_data = ingredient_database[key]
                break
    
    if ingredient_info_data:
        return jsonify({
            "status": "ok",
            "ingredient": ingredient_info_data
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"Informasi untuk '{ingredient}' belum tersedia dalam database.",
            "available_ingredients": list(ingredient_database.keys())
        }), 404

#  Jalankan API
if __name__ == "__main__":
    # Get port from environment variable (untuk deployment)
    port = int(os.getenv("PORT", 5000))
    # Set debug based on environment
    debug = os.getenv("ENVIRONMENT", "development") == "development"
    
    print(f"Starting server on port {port}")
    print(f"Debug mode: {debug}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
