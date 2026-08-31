import os
import sys

# Ensure root workspace is at top of sys.path for Vercel Serverless Function imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.main import app

# Export app for Vercel Serverless Function
app = app
