from flask import Flask, request, jsonify
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('./model')

# Load FAISS index
index = faiss.read_index('faiss_index.bin')

# Load answers list
with open('answers.pkl', 'rb') as f:
    answers = pickle.load(f)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Embed query
    query_vec = model.encode([query], convert_to_numpy=True)
    
    # Search
    D, I = index.search(query_vec, 1)

    return jsonify({"answer": answers[I[0][0]]})

if __name__ == '__main__':
    app.run(debug=True)
