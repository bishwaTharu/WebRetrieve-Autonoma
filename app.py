"""
Vercel entrypoint for WebRetrieve Autonoma API.
This file exports the FastAPI app for Vercel deployment.
"""

from WebRetrieve_Autonoma.api.main import app

# Export the FastAPI app for Vercel
app = app

# Vercel serverless function handler
def handler(request):
    """Vercel serverless function handler."""
    return app(request)
