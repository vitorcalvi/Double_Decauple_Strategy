#!/usr/bin/env python3
"""
proxy.py - Bybit API Proxy Server with Server-Side Authentication
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import hmac
import hashlib
import time
import json
import os
from dotenv import load_dotenv
from urllib.parse import urlencode

load_dotenv()

app = Flask(__name__)
CORS(app)

# API credentials
CREDS = {
    'testnet': {
        'url': 'https://api-testnet.bybit.com',
        'key': os.getenv('TESTNET_BYBIT_API_KEY', ''),
        'secret': os.getenv('TESTNET_BYBIT_API_SECRET', '')
    },
    'mainnet': {
        'url': 'https://api.bybit.com',
        'key': os.getenv('BYBIT_API_KEY', ''),
        'secret': os.getenv('BYBIT_API_SECRET', '')
    }
}

PUBLIC_ENDPOINTS = ['/v5/market/', '/v5/announcements/', '/derivatives/']

def sign(secret, params_str):
    return hmac.new(secret.encode(), params_str.encode(), hashlib.sha256).hexdigest()

@app.route('/proxy', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def proxy():
    if request.method == 'OPTIONS':
        return '', 200
    
    endpoint = request.headers.get('X-Target-Endpoint')
    if not endpoint:
        return jsonify({'error': 'Missing X-Target-Endpoint'}), 400
    
    env = request.headers.get('X-Target-Env', 'testnet')
    cred = CREDS[env]
    
    # Validate API credentials
    if len(cred['key']) < 30:
        return jsonify({
            'error': f'Invalid API key length ({len(cred["key"])} chars). Bybit API keys are typically 36+ chars.',
            'hint': f'Check your .env file for {env.upper()}_BYBIT_API_KEY',
            'current_key': cred['key'][:8] + '...' if cred['key'] else 'NOT SET'
        }), 500
    
    url = cred['url'] + endpoint
    headers = {}
    
    # Add auth if needed
    needs_auth = not any(endpoint.startswith(pub) for pub in PUBLIC_ENDPOINTS)
    if needs_auth and cred['key'] and cred['secret']:
        ts = str(int(time.time() * 1000))
        recv = '5000'
        
        if request.method in ['POST', 'PUT', 'PATCH']:
            headers['Content-Type'] = 'application/json'
            body = request.get_json() or {}
            body_str = json.dumps(body, separators=(',', ':'))
            sign_str = f"{ts}{cred['key']}{recv}{body_str}"
            
            headers.update({
                'X-BAPI-API-KEY': cred['key'],
                'X-BAPI-TIMESTAMP': ts,
                'X-BAPI-SIGN': sign(cred['secret'], sign_str),
                'X-BAPI-RECV-WINDOW': recv
            })
            
            response = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                json=body
            )
        else:
            # GET/DELETE requests
            params = request.args.to_dict()
            
            if params:
                # Sort and encode parameters
                sorted_params = sorted(params.items())
                query_string = urlencode(sorted_params)
                sign_str = f"{ts}{cred['key']}{recv}{query_string}"
            else:
                query_string = ""
                sign_str = f"{ts}{cred['key']}{recv}"
            
            signature = sign(cred['secret'], sign_str)
            headers.update({
                'X-BAPI-API-KEY': cred['key'],
                'X-BAPI-TIMESTAMP': ts,
                'X-BAPI-SIGN': signature,
                'X-BAPI-RECV-WINDOW': recv
            })
            
            # Debug
            print(f"DEBUG: endpoint={endpoint}")
            print(f"DEBUG: params={params}")
            print(f"DEBUG: query_string={query_string}")
            print(f"DEBUG: API_KEY_LENGTH={len(cred['key'])}")
            print(f"DEBUG: sign_str={sign_str[:50]}...")  # Only show first 50 chars
            print(f"DEBUG: signature={signature}")
            
            response = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                params=params
            )
    else:
        # Public endpoints
        if request.method in ['POST', 'PUT', 'PATCH']:
            headers['Content-Type'] = 'application/json'
            response = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                json=request.get_json()
            )
        else:
            params = request.args.to_dict()
            response = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                params=params
            )
    
    try:
        resp_data = response.json()
        if response.status_code != 200:
            print(f"ERROR: {response.status_code} - {resp_data}")
            if response.status_code == 401 and 'Invalid api_key' in str(resp_data):
                return jsonify({
                    'error': 'Authentication failed',
                    'message': resp_data.get('retMsg', 'Invalid API key'),
                    'hint': 'Make sure you created API keys at https://testnet.bybit.com/ (NOT the main site)',
                    'api_key_length': len(cred['key']),
                    'expected_length': '36+ characters'
                }), 401
        return resp_data, response.status_code
    except:
        return response.text, response.status_code

@app.route('/status', methods=['GET'])
def status():
    testnet_valid = len(CREDS['testnet']['key']) >= 30
    mainnet_valid = len(CREDS['mainnet']['key']) >= 30
    
    return jsonify({
        'status': 'running',
        'testnet': {
            'configured': testnet_valid,
            'api_key': CREDS['testnet']['key'][:8] + '...' if CREDS['testnet']['key'] else 'Not configured',
            'key_length': len(CREDS['testnet']['key']),
            'valid': testnet_valid,
            'error': None if testnet_valid else f'API key too short ({len(CREDS["testnet"]["key"])} chars, need 36+)'
        },
        'mainnet': {
            'configured': mainnet_valid,
            'api_key': CREDS['mainnet']['key'][:8] + '...' if CREDS['mainnet']['key'] else 'Not configured',
            'key_length': len(CREDS['mainnet']['key']),
            'valid': mainnet_valid,
            'error': None if mainnet_valid else f'API key too short ({len(CREDS["mainnet"]["key"])} chars, need 36+)'
        }
    })

@app.route('/test-auth', methods=['GET'])
def test_auth():
    """Test endpoint to verify API credentials"""
    env = request.args.get('env', 'testnet')
    cred = CREDS[env]
    
    if len(cred['key']) < 30:
        return jsonify({
            'error': f'API key too short ({len(cred["key"])} chars)',
            'hint': 'Bybit API keys are typically 36+ characters. Check your .env file.',
            'instructions': [
                f'1. Go to https://{"testnet" if env == "testnet" else "www"}.bybit.com/app/user/api-management',
                '2. Create a new API key',
                '3. Copy the FULL API key and secret',
                f'4. Add to .env file:',
                f'   {env.upper()}_BYBIT_API_KEY=your_full_api_key_here',
                f'   {env.upper()}_BYBIT_API_SECRET=your_full_secret_here'
            ]
        }), 400
    
    # Test with server time first (public endpoint)
    try:
        url = cred['url'] + '/v5/market/time'
        response = requests.get(url)
        server_time = response.json()
    except Exception as e:
        return jsonify({'error': f'Cannot reach Bybit: {str(e)}'}), 500
    
    # Try authenticated request
    ts = str(int(time.time() * 1000))
    recv = '5000'
    query_string = 'accountType=UNIFIED'
    sign_str = f"{ts}{cred['key']}{recv}{query_string}"
    signature = sign(cred['secret'], sign_str)
    
    headers = {
        'X-BAPI-API-KEY': cred['key'],
        'X-BAPI-TIMESTAMP': ts,
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': recv
    }
    
    wallet_url = cred['url'] + f'/v5/account/wallet-balance?{query_string}'
    wallet_response = requests.get(wallet_url, headers=headers)
    
    return jsonify({
        'server_time': server_time,
        'api_key_length': len(cred['key']),
        'secret_length': len(cred['secret']),
        'wallet_response': wallet_response.json(),
        'auth_success': wallet_response.status_code == 200
    })

if __name__ == '__main__':
    print('=' * 60)
    print('Bybit API Proxy Server')
    print('=' * 60)
    
    # Check API key validity
    testnet_valid = len(CREDS['testnet']['key']) >= 30
    mainnet_valid = len(CREDS['mainnet']['key']) >= 30
    
    print(f'✅ Server: http://localhost:8080')
    
    if testnet_valid:
        print(f'📊 Testnet: Configured ✓ (Key: {CREDS["testnet"]["key"][:8]}...)')
    else:
        print(f'❌ Testnet: INVALID KEY! ({len(CREDS["testnet"]["key"])} chars, need 36+)')
        print(f'   Current key: {CREDS["testnet"]["key"]}')
    
    if mainnet_valid:
        print(f'💰 Mainnet: Configured ✓ (Key: {CREDS["mainnet"]["key"][:8]}...)')
    else:
        print(f'⚠️  Mainnet: Not configured or invalid')
    
    if not testnet_valid:
        print('\n🚨 FIX YOUR API KEYS:')
        print('   1. Go to https://testnet.bybit.com/app/user/api-management')
        print('   2. Create API key (NOT from main bybit.com!)')
        print('   3. Copy the FULL key (36+ characters)')
        print('   4. Update .env file:')
        print('      TESTNET_BYBIT_API_KEY=<your-full-36-char-key>')
        print('      TESTNET_BYBIT_API_SECRET=<your-secret>')
    
    print('\n📌 Test auth: http://localhost:8080/test-auth')
    print('📌 Status: http://localhost:8080/status')
    print('=' * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)