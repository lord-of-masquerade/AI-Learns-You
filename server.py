# server.py — Python proxy for AI That Learns You (alternative to server.js)
# pip install flask flask-cors requests
# python server.py
# Open http://localhost:3000/demo.html

import os, json, requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

@app.route('/api/claude', methods=['POST','OPTIONS'])
def claude_proxy():
    if request.method == 'OPTIONS':
        return '', 204
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return jsonify({'error': 'ANTHROPIC_API_KEY not set'}), 500
    resp = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'Content-Type':      'application/json',
            'x-api-key':         api_key,
            'anthropic-version': '2023-06-01',
        },
        json=request.get_json(),
        timeout=60,
    )
    return jsonify(resp.json()), resp.status_code

@app.route('/')
@app.route('/demo.html')
def serve_demo():
    return send_from_directory('.', 'demo.html')

if __name__ == '__main__':
    print('\n✅ Server running at http://localhost:3000')
    print('   Open http://localhost:3000/demo.html\n')
    app.run(port=3000, debug=False)
