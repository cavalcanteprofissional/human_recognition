#!/usr/bin/env python3
"""
Ponto de entrada para Streamlit Dashboard.
"""

import streamlit as st
from streamlit import main as dashboard_main, detector

if __name__ == "__main__":
    st.set_page_config(
        page_title="Human Recognition Dashboard",
        page_icon="👤",
        layout="wide"
    )
    dashboard_main()
