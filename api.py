# proxy.py - Bybit API Proxy Server
# Setup:
# 1. pip install flask flask-cors requests
# 2. python proxy.py
# 3. Keep "Use Proxy Server" checked in the HTML interface

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route('/proxy', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def proxy():
    # Handle OPTIONS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    # Get target info from headers
    endpoint = request.headers.get('X-Target-Endpoint')
    env = request.headers.get('X-Target-Env')
    
    # Check if endpoint exists
    if not endpoint:
        return jsonify({'error': 'Missing X-Target-Endpoint header'}), 400
    
    # Select base URL
    base_url = 'https://api-testnet.bybit.com' if env == 'testnet' else 'https://api.bybit.com'
    
    # Build target URL
    url = base_url + endpoint
    
    # Forward headers
    headers = {
        'X-BAPI-API-KEY': request.headers.get('X-BAPI-API-KEY'),
        'X-BAPI-TIMESTAMP': request.headers.get('X-BAPI-TIMESTAMP'),
        'X-BAPI-SIGN': request.headers.get('X-BAPI-SIGN'),
        'X-BAPI-RECV-WINDOW': request.headers.get('X-BAPI-RECV-WINDOW'),
        'Content-Type': 'application/json'
    }
    
    # Remove None values
    headers = {k: v for k, v in headers.items() if v}
    
    try:
        # Make request
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            params=request.args,
            json=request.get_json() if request.method in ['POST', 'PUT', 'PATCH'] else None
        )
        
        # Return response
        return response.json(), response.status_code
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('✅ Proxy running on http://localhost:8080')
    app.run(host='0.0.0.0', port=8080, debug=False)