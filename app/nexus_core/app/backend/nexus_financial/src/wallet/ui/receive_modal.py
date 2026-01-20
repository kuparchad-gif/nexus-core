import qrcode
from io import BytesIO
import streamlit as st

def show_receive(address: str):
    st.write("Receive Address")
    st.code(address)
    img = qrcode.make(address)
    buf = BytesIO()
    img.save(buf, format='PNG')
    st.image(buf.getvalue(), caption="Scan to receive")
