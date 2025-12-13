#!/usr/bin/env python3
"""
Backend server launcher script.

This script properly sets up the Python path and starts the uvicorn server
to handle relative imports correctly.
"""

import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Now we can use absolute imports
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[backend_dir],
    )



