import streamlit as st
from PIL import Image
import functools
import pickle
import torch
from PIL import Image
import PIL
from io import BytesIO
import base64


st.title('Generate New Sneaker Designs')
model= './saved_model/network-snapshot-004000.pkl'


def load_model(model_path):

    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema']  # torch.nn.Module
        z = torch.randn([1, G.z_dim])  # latent codes
        c = None  # class labels (not used in this example)
        G.forward = functools.partial(G.forward, force_fp32=True)
        return G,z,c


def generate_sneaker(generator, latent_space, class_label):
    out_img = generator(latent_space, class_label)  # NCHW, float32, dynamic range [-1, +1]
    out_img = (out_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    out_img = PIL.Image.fromarray(out_img[0].cpu().numpy(), 'RGB')

    return out_img

def get_image_download_link(img,filename='AI-Generated-Image',text='Download Image'):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/PNG;base64,{img_str}" download="{filename}">{text}</a>'
    return href

txt1, txt2, txt3 = st.columns([1, 1, 1])
text_box = txt2.empty()


btn1, btn2, btn3 = st.columns([1, 1, 1])
generate = txt1.button('    Generate    ')


if generate:
    text_box.write('##### Loading Model ...')
    G,z,c = load_model(model)

    text_box.write('##### Generating New Design ...')
    generated_img = generate_sneaker(G,z,c)

    text_box.write('')

    text_box.markdown(get_image_download_link(generated_img), unsafe_allow_html=True)

    st.image(generated_img)




