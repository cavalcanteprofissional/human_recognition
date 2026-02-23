#!/usr/bin/env python3
"""
Ponto de entrada para Hugging Face Spaces.
"""

from dashboard import build_interface

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
