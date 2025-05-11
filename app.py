from flask import Flask, render_template, request, jsonify
from summarizer import Summarizer

app = Flask(__name__)
summarizer = Summarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        try:
            summary = summarizer.summarize(text)
            return jsonify({'summary': summary})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 