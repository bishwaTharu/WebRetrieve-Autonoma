"""
Netlify serverless function for WebRetrieve Autonoma API.
"""

import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the FastAPI app
from WebRetrieve_Autonoma.api.main import app

# Netlify serverless function handler
def handler(event, context):
    """
    Netlify serverless function handler.
    """
    try:
        # Convert Netlify event to ASGI format
        path = event.get('path', '/')
        http_method = event.get('httpMethod', 'GET')
        headers = event.get('headers', {})
        query_parameters = event.get('queryStringParameters', {}) or {}
        body = event.get('body', '')
        
        # Create ASGI scope
        scope = {
            'type': 'http',
            'path': path,
            'method': http_method,
            'headers': [[k.lower().encode(), v.encode()] for k, v in headers.items()],
            'query_string': '&'.join([f"{k}={v}" for k, v in query_parameters.items()]).encode(),
            'server': ('netlify', 1),
            'client': ('0.0.0.0', 0),
            'asgi': {'version': '3.0'},
            'state': {},
        }
        
        # Create receive and send channels
        async def receive():
            if body:
                return {'type': 'http.request', 'body': body.encode(), 'more_body': False}
            else:
                return {'type': 'http.request', 'body': b'', 'more_body': False}
        
        # Store response
        response = {}
        
        async def send(message):
            nonlocal response
            if message['type'] == 'http.response.start':
                response['status'] = message['status']
                response['headers'] = dict(message.get('headers', []))
            elif message['type'] == 'http.response.body':
                response['body'] = message.get('body', b'').decode('utf-8')
        
        # Run the app
        import asyncio
        asyncio.run(app(scope, receive, send))
        
        # Format response for Netlify
        return {
            'statusCode': response.get('status', 200),
            'headers': response.get('headers', {}),
            'body': response.get('body', '')
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/plain'},
            'body': f'Internal Server Error: {str(e)}'
        }
