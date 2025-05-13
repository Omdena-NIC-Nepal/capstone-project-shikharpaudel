import streamlit as st

def show_footer():
    st.markdown("""
    <hr style='margin-top: 50px;'>
    <div style='text-align: center; color: grey; font-size: 0.9em;'>
        Made with ❤️ by Shikhar Paudel | Omdena Batch II <br>
        Climate Change Impact Dashboard • 2025 <br>
                <a href="https://www.linkedin.com/in/shikharpaudel/" target="_blank" style="margin-right: 15px;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="20" style="vertical-align: middle;">
            LinkedIn
        </a>
        <a href="https://github.com/shikharpaudel" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="20" style="vertical-align: middle;">
            GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
