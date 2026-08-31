import os
import sys

# Ensure root workspace is in sys.path for Vercel Serverless Function imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app

# Export app for Vercel Serverless Function
app = app
