import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import io 
import matplotlib.pyplot as plt
import pickle

import struct

import os
import re
from typing import List, Optional

import click

import sys
sys.path.insert(0, '../')
import dnnlib
import numpy as np
import torch
import urllib

import librosa
import librosa.display
import soundfile as sf

from utils import util, google_analytics
from tifresi.utils import load_signal
from tifresi.utils import preprocess_signal
from tifresi.stft import GaussTF, GaussTruncTF
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram

import time
import pyloudnorm as pyln

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import uuid
from argparse import Namespace

# st.markdown("<h1 style='text-align: center;'>Semantic Factorization (From Computer Vision)</h1>", unsafe_allow_html=True)
# st.title('Semantic Factorization (From Computer Vision)')

config = util.get_config('../config/config.json')
config = Namespace(**dict(**config))

meter = pyln.Meter(16000)

def pghi_istft(x):
    model = st.session_state['model_picked']
    use_truncated_window = True
    if use_truncated_window:
        stft_system = GaussTruncTF(hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)
    else:
        stft_system = GaussTF(hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)

    x = np.squeeze(x,axis=0)
    new_Y = inv_log_spectrogram(x)
    new_y = stft_system.invert_spectrogram(new_Y)
    return np.array(new_y)

# From - https://stackoverflow.com/questions/67317366/how-to-add-header-info-to-a-wav-file-to-get-a-same-result-as-ffmpeg
def pcm2wav(sample_rate, pcm_voice):
    # if pcm_voice.startswith("RIFF".encode()):
    #     return pcm_voice
    # else:
    sampleNum = len(pcm_voice)
    # print(sampleNum)
    rHeaderInfo = "RIFF".encode()
    rHeaderInfo += struct.pack('i', sampleNum + 44)
    rHeaderInfo += 'WAVEfmt '.encode()
    rHeaderInfo += struct.pack('i', 16)
    rHeaderInfo += struct.pack('h', 3)
    rHeaderInfo += struct.pack('h', 1)
    rHeaderInfo += struct.pack('i', sample_rate)
    rHeaderInfo += struct.pack('i', sample_rate * int(32 / 8))
    rHeaderInfo += struct.pack("h", int(32 / 8))
    rHeaderInfo += struct.pack("h", 32)
    rHeaderInfo += "data".encode()
    rHeaderInfo += struct.pack('i', sampleNum)
    rHeaderInfo += pcm_voice.tobytes()
    return rHeaderInfo

@st.cache_data
def factorize_weights_hitsscratches(_generator):
    layers = ['b4','b8','b16','b32','b64','b128','b256'] #layernames in Synthesis network
    layers.extend(layers) ## THIS IS SUPER IMPORTANT. Remember, the dimensionality of y is twice the number of feature maps (see first Style GAN paper)
    layers.sort(key=lambda x: int(x.replace('b','')))

    weights = []
    layer_ids = []
    for layer_id, layer_name in enumerate(layers):
        # weight = _generator.synthesis.__getattr__(layer_name).__getattr__('torgb').affine.weight.T
        weight = _generator.synthesis.__getattr__(layer_name).__getattr__('conv1').affine.weight.T
        weights.append(weight.cpu().detach().numpy())
        layer_ids.append(layer_id)
        
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    boundaries, values = eigen_vectors.T, eigen_values

    #Sorting values
    values_ind = np.array([a for a in range(len(values))])
    temp = np.array(sorted(zip(values, values_ind), key=lambda x: x[0], reverse=True))
    values, values_ind = temp[:, 0], temp[:, 1]
    # print(values, values_ind)
    return boundaries, values, layer_ids, values_ind


@st.cache_data
def factorize_weights_envsounds(_generator):
    layers = ['b4','b8','b16','b32','b64','b128','b256'] #layernames in Synthesis network
    layers.extend(layers) ## THIS IS SUPER IMPORTANT. Remember, the dimensionality of y is twice the number of feature maps (see first Style GAN paper)
    layers.sort(key=lambda x: int(x.replace('b','')))

    weights = []
    layer_ids = []
    for layer_id, layer_name in enumerate(layers):
        # weight = _generator.synthesis.__getattr__(layer_name).__getattr__('torgb').affine.weight.T
        weight = _generator.synthesis.__getattr__(layer_name).__getattr__('conv1').affine.weight.T
        weights.append(weight.cpu().detach().numpy())
        layer_ids.append(layer_id)
        
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    boundaries, values = eigen_vectors.T, eigen_values

    #Sorting values
    values_ind = np.array([a for a in range(len(values))])
    temp = np.array(sorted(zip(values, values_ind), key=lambda x: x[0], reverse=True))
    values, values_ind = temp[:, 0], temp[:, 1]
    # print(values, values_ind)
    return boundaries, values, layer_ids, values_ind

@st.cache_data
def get_hitscratch_sefa_model(model):
    print('getting model', model)
    try:
        stylegan_pkl = config.model_list[model]['ckpt_stylegan2_path']
        stylegan_pkl_url = config.model_list[model]['stylegan_pkl_url']
    except:
        print("Unknown Model!")
        return None, None

    if not os.path.isfile(stylegan_pkl):
        os.makedirs(config.model_list[model]['ckpt_download_stylegan2_path'], exist_ok=True)
        urllib.request.urlretrieve(stylegan_pkl_url, stylegan_pkl)

    with dnnlib.util.open_url(stylegan_pkl) as f:
        network = pickle.load(f)
        G = network['G'].eval().cuda()
        # G = legacy.load_network_pkl(f)['G']
        st.session_state['hitscratch_sefa_G'] = G
    return st.session_state['hitscratch_sefa_G']


@st.cache_data
def get_envsounds_sefa_model(model):
    print('getting model', model)
    try:
        stylegan_pkl = config.model_list[model]['ckpt_stylegan2_path']
        stylegan_pkl_url = config.model_list[model]['stylegan_pkl_url']
    except:
        print("Unknown Model!")
        return None, None

    if not os.path.isfile(stylegan_pkl):
        os.makedirs(config.model_list[model]['ckpt_download_stylegan2_path'], exist_ok=True)
        urllib.request.urlretrieve(stylegan_pkl_url, stylegan_pkl)

    with dnnlib.util.open_url(stylegan_pkl) as f:
        network = pickle.load(f)
        G = network['G'].eval().cuda()
        # G = legacy.load_network_pkl(f)['G']
        st.session_state['envsounds_sefa_G'] = G
    return st.session_state['envsounds_sefa_G']

def sample(pos, session_uuid=''):
    model = st.session_state['model_picked']
    device = torch.device('cuda')
    # G = get_sefa_model(model).to(device).eval()

    if model == 'hits_scratches':
        G = get_hitscratch_sefa_model(model)
        boundaries, values, layer_ids, values_ind = factorize_weights_hitsscratches(G)
    elif model == 'environmental_sounds':
        G = get_envsounds_sefa_model(model)
        boundaries, values, layer_ids, values_ind = factorize_weights_envsounds(G)

    # print(values_ind[0], values_ind, boundaries)
    boundary_1 = boundaries[int(values_ind[0])] #Looking at the first semantic only
    boundary_2 = boundaries[int(values_ind[1])] #Looking at the second semantic only
    boundary_3 = boundaries[int(values_ind[2])] #Looking at the first semantic only
    boundary_4 = boundaries[int(values_ind[3])] #Looking at the second semantic only
    boundary_5 = boundaries[int(values_ind[4])] #Looking at the first semantic only
    boundary_6 = boundaries[int(values_ind[5])] #Looking at the second semantic only
    boundary_7 = boundaries[int(values_ind[6])] #Looking at the first semantic only
    boundary_8 = boundaries[int(values_ind[7])] #Looking at the second semantic only
    boundary_9 = boundaries[int(values_ind[8])] #Looking at the first semantic only
    boundary_10 = boundaries[int(values_ind[9])] #Looking at the second semantic only


    if f'%s_sefa_initial_sample'%model not in st.session_state:
        st.session_state[f'%s_sefa_initial_sample'%model] = torch.from_numpy(np.random.randn(1, G.z_dim))
        #np.savez(f'{config.sefa_tmp_audio_loc_path}{session_uuid}z_tensor.npz',z=st.session_state[f'%s_sefa_initial_sample'%model].numpy())

    z = st.session_state[f'%s_sefa_initial_sample'%model].to(device)
    # label = torch.zeros([1, G.z_dim], device=device)

    
    start_time = time.time()

    if 'Random' not in st.session_state['selected_preset_option']: ## Random bug fix. When example is selected, we receive the W vector. 
        code = torch.stack([z] * 14, dim=1)
    else:
        code = G.mapping(z, None)
    # w_avg = G.mapping.w_avg
    # code = w_avg + (code - w_avg) * truncation_psi # Truncation not needed?
    code = code.detach().cpu().numpy()
    temp_z = code.copy()
    temp_z[:, layer_ids, :] += boundary_1 * pos[0]
    temp_z[:, layer_ids, :] += boundary_2 * pos[1]
    temp_z[:, layer_ids, :] += boundary_3 * pos[2]
    temp_z[:, layer_ids, :] += boundary_4 * pos[3]
    temp_z[:, layer_ids, :] += boundary_5 * pos[4]
    temp_z[:, layer_ids, :] += boundary_6 * pos[5]
    temp_z[:, layer_ids, :] += boundary_7 * pos[6]
    temp_z[:, layer_ids, :] += boundary_8 * pos[7]
    temp_z[:, layer_ids, :] += boundary_9 * pos[8]
    temp_z[:, layer_ids, :] += boundary_10 * pos[9]


    # print('generating')
    img = G.synthesis(torch.from_numpy(temp_z).cuda())

    # print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    img = (img  * 127.5+ 128).clamp(0, 255).to(torch.uint8)
    img = img.detach().cpu().numpy()[0]
    filler = np.full((1, 1, img[0][0].shape[0]), np.min(img))
    img_1 = np.append(img, filler, axis=1) # UNDOING THAT CODE!
    img_1 = img_1/255
    img_1 = -50+img_1*50

    audio = pghi_istft(img_1)

    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio to -12 dB LUFS
    audio = pyln.normalize.loudness(audio, loudness, -14.0)


    #Uncomment for final study
    # fig, ax = plt.subplots(nrows=2, figsize=(7,8))
    # a=librosa.display.specshow(img_1[0],x_axis='time', y_axis='linear',sr=config.sample_rate, hop_length=config.model_list[model]['hop_size'],ax=ax[1])
    # b=librosa.display.waveshow(audio, sr=config.sample_rate, axis='time', ax=ax[0])
    # ax[0].set_xlim(0, config.model_list[model]['total_time'])
    # ax[0].set_xlabel('')

    fig, ax = plt.subplots(nrows=1, figsize=(7,5))
    a=librosa.display.specshow(img_1[0],x_axis='time', y_axis='linear',sr=config.sample_rate, hop_length=config.model_list[model]['hop_size'], ax=ax)
    ax.set_xlim(0, config.model_list[model]['total_time'])
    # ax.set_xlabel('')


    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()


    os.makedirs(config.sefa_tmp_audio_loc_path, exist_ok=True)
    sf.write(f'{config.sefa_tmp_audio_loc_path}{session_uuid}_sefa_interface_temp_audio_loc.wav', audio.astype(float), config.sample_rate)
    # print('--------------------------------------------------')


    audio_file = open(f'{config.sefa_tmp_audio_loc_path}{session_uuid}_sefa_interface_temp_audio_loc.wav', 'rb')
    audio_bytes = audio_file.read()

    # print(audio_bytes)
    # components.html('test<script>\
    #     alert(window.parent);\
    #     window.parent.onload = function() {\
    #         window.parent.document.getElementsByClassName("stAudio")[0].play();\
    #     }\
    #     </script>\
    #     ')
    return img_arr, audio_bytes


def draw_audio():
    components.html('<script>\
        window.onload = function() {\
            window.parent.document.getElementsByClassName("stAudio")[0].play();\
        }\
        </script>\
        ')


#     audio_placeholder = st.empty()
#     audio_str = '''
#     <audio id="audio" controls="" autoplay src="http://localhost:8000/tmp/audio-design-toolkit/sefa/{session_uuid}_sefa_interface_temp_audio_loc.wav" class="stAudio" style="width: 704px;">
#     </audio>
#     '''
#     audio_placeholder.markdown(audio_str, unsafe_allow_html=True)

# def change_z(a):
#     print(a)
#     selected_option = st.session_state['sefa_selected_preset_option']
#     if selected_option == 'Random':
#         if f'%s_sefa_initial_sample'%model in st.session_state:
#             del st.session_state[f'%s_sefa_initial_sample'%model] 
#     else:
#         with np.load(selected_option) as data:
#             print(data)
#             new_z = data['z']
#         st.session_state[f'%s_sefa_initial_sample'%model] = torch.from_numpy(new_z)
#     st.session_state['sefa_slider_1_position'] = 0
#     st.session_state['sefa_slider_2_position'] = 0

def map_dropdown_name(input):
    return config.model_list[input]['name']

def main():
    google_analytics.set_google_analytics()
    st.markdown("<h1 style='text-align: center;'>Exploring Environmental Sound Spaces - 2</h1>", unsafe_allow_html=True)


    if 'session_uuid' not in st.session_state:
        st.session_state['session_uuid'] = str(uuid.uuid4())
    session_uuid = st.session_state['session_uuid']

    model_names = []
    for key in config.model_list:
        model_names.append(key)
    model_names = tuple(model_names)
    model_picked =  st.sidebar.selectbox('Select Model', model_names, format_func=map_dropdown_name, key='model_picked')

    try:
        example_arr = os.listdir(f'../config/resources/sefa-examples/{model_picked}')
    except:
        example_arr = []
    example_arr_extensionless = [os.path.splitext(file_name)[0] for file_name in example_arr]
    example_arr_extensionless = sorted(example_arr_extensionless)

    example_arr_extensionless.insert(0, 'Random (Refresh Page)')
    sefa_selected_preset_option = st.sidebar.selectbox(
    'Select Example',
    example_arr_extensionless, key='selected_preset_option')


    if sefa_selected_preset_option == 'Random (Refresh Page)':
        if f'%s_sefa_initial_sample'%model_picked in st.session_state:
            del st.session_state[f'%s_sefa_initial_sample'%model_picked] 
    else:
        try:
            config_from_example = np.load(f'../config/resources/sefa-examples/{model_picked}/{sefa_selected_preset_option}.npy')
            st.session_state[f'%s_sefa_initial_sample'%model_picked] = torch.from_numpy(config_from_example)
        except:
            config_from_example = None
        # with np.load(f'../config/resources/sefa-examples/{sefa_selected_preset_option}.npy') as data:
        #     print(data)
        # new_z = data['z']
        # st.session_state[f'%s_sefa_initial_sample'%model_picked] = torch.from_numpy(new_z)

    slider_1_position=st.sidebar.slider('Dimension 1', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_1_position', help=None, args=None, kwargs=None, disabled=False)
    slider_2_position=st.sidebar.slider('Dimension 2', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_2_position', help=None, args=None, kwargs=None, disabled=False)
    slider_3_position=st.sidebar.slider('Dimension 3', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_3_position', help=None, args=None, kwargs=None, disabled=False)
    slider_4_position=st.sidebar.slider('Dimension 4', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_4_position', help=None, args=None, kwargs=None, disabled=False)
    slider_5_position=st.sidebar.slider('Dimension 5', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_5_position', help=None, args=None, kwargs=None, disabled=False)
    slider_6_position=st.sidebar.slider('Dimension 6', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_6_position', help=None, args=None, kwargs=None, disabled=False)
    slider_7_position=st.sidebar.slider('Dimension 7', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_7_position', help=None, args=None, kwargs=None, disabled=False)
    slider_8_position=st.sidebar.slider('Dimension 8', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_8_position', help=None, args=None, kwargs=None, disabled=False)
    slider_9_position=st.sidebar.slider('Dimension 9', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_9_position', help=None, args=None, kwargs=None, disabled=False)
    slider_10_position=st.sidebar.slider('Dimension 10', min_value=-5.0, max_value=5.0, value=0.0, step=0.01,  format=None, key='slider_10_position', help=None, args=None, kwargs=None, disabled=False)


    #on_change=draw_audio(),

    sefa_col1, sefa_col2, sefa_col3 = st.columns((1,2,1))
    spectrogram_placeholder = sefa_col2.empty()
    audio_placeholder = sefa_col2.empty()

    # spectrogram_placeholder = st.empty()
    # audio_placeholder = st.empty()



    s = sample([slider_1_position, slider_2_position,slider_3_position, slider_4_position,slider_5_position,\
        slider_6_position, slider_7_position,slider_8_position, slider_9_position,slider_10_position], session_uuid)
    spectrogram_placeholder.image(s[0])
    audio_element = audio_placeholder.audio(s[1], format="audio/wav", start_time=0)
    # print(audio_element)
    # draw_audio() # Unfortunately audio is not redrawable
    

if __name__ == '__main__':
    main()