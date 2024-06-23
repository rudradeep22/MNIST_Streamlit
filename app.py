import streamlit as st
from PIL import Image
import numpy as np
import os
from model import generate_image, generator

st.title("GAN Image Generator")

if st.button('Generate'):
    generated_image = generate_image(generator)
    image = Image.fromarray((generated_image * 255).astype(np.uint8))
    image.save('static/generated_image.png')
    st.image(image, caption='Generated Image', use_column_width=True)