from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
CORS(app, resources={r"/recommend/*": {"origins": "http://localhost:3002"}})

products_df = pd.read_csv('data_set.csv')

# Xây dựng vector đặc trưng TF-IDF từ tên sản phẩm
tfidf_vectorizer = TfidfVectorizer(stop_words='english', norm=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(products_df['name'])

# Print the shape of tfidf_matrix
print('Shape: ',tfidf_matrix.shape)

# Lấy danh sách các từ từ vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()
print('Các từ:\n',feature_names)

# In ma trận TF-IDF
print(tfidf_matrix.toarray())

# Sử dụng cosine similarity để tính độ tương tự giữa các sản phẩm dựa trên vector TF-IDF
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('Cosine Similarity:')
print(cosine_sim)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        product_id = data.get('product_id')
        if product_id is None:
            return jsonify({'error': 'Thiếu thông tin sản phẩm'}), 400

        product_index = products_df[products_df['_id'] == product_id].index[0]
        similar_scores = list(enumerate(cosine_sim[product_index]))
        similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
        similar_scores = similar_scores[1:7]
        product_indices = [i[0] for i in similar_scores]
        recommended_products = products_df['_id'].iloc[product_indices]

        return jsonify({'recommended_products': recommended_products.tolist()})
    except Exception as e:
        return jsonify({'error': 'Có lỗi xảy ra: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)