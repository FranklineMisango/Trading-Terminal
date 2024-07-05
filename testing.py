import streamlit as st
import subprocess
import time
import yfinance as yf

st.title('Frankline & Associates LLP. Comprehensive Lite Algorithmic Trading Terminal')
st.success("Our intelligent Terminal will run instructions here, the UX app shold open in a few seconds...")


# Start the ttyd server to show terminal output
ttyd_cmd = "ttyd -p 7681 streamlit run app.py"
try:
    ttydprocess = subprocess.Popen(ttyd_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    st.text("Starting Terminal server...")
except Exception as e:
    st.text(f"Error starting ttyd server: {e}")
    st.stop()

# Give some time for the ttyd server to start
time.sleep(5)  # Increase the sleep time

# Check if the ttyd server is running
if ttydprocess.poll() is None:  # If ttydprocess is still running
    st.text("ttyd server is running.")
    port = 7681
else:
    # Capture and display the error from ttyd
    stdout, stderr = ttydprocess.communicate()
    st.text(f"Failed to start ttyd server.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")
    st.stop()

# Embed the terminal in the Streamlit app
terminal_url = f"http://localhost:{port}"
st.markdown(f"""
    <iframe src="{terminal_url}" width="100%" height="500px"></iframe>
""", unsafe_allow_html=True)
    
# Option to stop the ttyd server manually (optional)
if st.button("Stop Terminal"):
    ttydprocess.terminate()
    st.text("ttyd server has been stopped.")

