import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import io 
import matplotlib.pyplot as plt
import urllib

import os

import sys
sys.path.insert(0, '../')
from networks import stylegan_encoder
from utils import util, gaver_sounds, google_analytics
import numpy as np
import torch

import librosa
import librosa.display
import pickle

from tifresi.utils import preprocess_signal
from tifresi.stft import GaussTruncTF
from tifresi.transforms import log_spectrogram
from tifresi.transforms import inv_log_spectrogram

from scipy.signal import butter, lfilter

import pyloudnorm as pyln

import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import uuid
from argparse import Namespace

# st.title('Analysis-Synthesis')
somehtml = '<h1 style="text-align:center">Analysis-Synthesis In The Latent Space</h1>'
# st.markdown(somehtml, unsafe_allow_html=True)
# st.title("Analysis-Synthesis In The Latent Space")

config = util.get_config('../config/config.json')
config = Namespace(**dict(**config))

impulse_dict = {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}

meter = pyln.Meter(16000)

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend/build")

def my_component(key,
                value, 
                step,
                label,
                min_value, 
                max_value,
                track_color,
                example,
                thumb_color):
    _component_func = components.declare_component("my_component", path=build_dir)
    component_value = _component_func(key=key, value=value, step=step, label=label, min_value=min_value, max_value=max_value, track_color=track_color, example=example, thumb_color=thumb_color)
    return component_value

def pghi_stft(x):
    model = st.session_state['model_picked']
    stft_system = GaussTruncTF(hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)
    Y = stft_system.spectrogram(x)
    log_Y= log_spectrogram(Y)
    return np.expand_dims(log_Y, axis=0)

def pghi_istft(x):
    model = st.session_state['model_picked']
    stft_system = GaussTruncTF(hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)
    x = np.squeeze(x,axis=0)
    new_Y = inv_log_spectrogram(x)
    new_y = stft_system.invert_spectrogram(new_Y)
    return np.array(new_y)

def zeropad(signal, audio_length):
    if len(signal) < audio_length:
        return np.append(
            signal, 
            np.zeros(audio_length - len(signal))
        )
    else:
        signal = signal[0:audio_length]
        return signal

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def get_vector(x):
    return torch.from_numpy(x).float().cuda()

def get_spectrogram(audio):
    model = st.session_state['model_picked']
    audio_pghi = preprocess_signal(audio)
    audio_pghi = zeropad(audio_pghi, config.n_frames * config.model_list[model]['hop_size'] )
    audio_pghi = pghi_stft(audio_pghi)
    return audio_pghi

def applyFBFadeFilter(forward_fadetime,backward_fadetime,signal,fs,expo=1):
    forward_num_fad_samp = int(forward_fadetime*fs) 
    backward_num_fad_samp = int(backward_fadetime*fs) 
    signal_length = len(signal) 
    fadefilter = np.ones(signal_length)
    if forward_num_fad_samp>0:
        fadefilter[0:forward_num_fad_samp]=np.linspace(0,1,forward_num_fad_samp)**expo
    if backward_num_fad_samp>0:
        fadefilter[signal_length-backward_num_fad_samp:signal_length]=np.linspace(1,0,backward_num_fad_samp)**expo
    return fadefilter*signal

def butter_bandpass(lowcut, highcut, fs, order=5,btype='bandpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a

def butter_lowhighpass(cut, fs, order=5, btype='lowpass'):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype=btype)
    return b, a

def butter_bandpass_filter(data, highcut, fs,lowcut=None,  order=5, btype='bandpass'):
    if btype=='bandpass':
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    else:
        b, a = butter_lowhighpass(highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y

def get_gaver_sounds(initial_amplitude, impulse_time, filters, total_time, locs=None, \
                             sample_rate=config.sample_rate, hittype='hit', 
                             backward_damping_mult=None, forward_damping_mult=None, damping_fade_expo=None, 
                             filter_order=None,
                             session_uuid=''):
    model = st.session_state['model_picked']

    signal = gaver_sounds.get_synthetic_sounds(initial_amplitude=initial_amplitude, 
                                                impulse_time=impulse_time, 
                                                filters=filters, 
                                                total_time=total_time, 
                                                locs=locs, 
                                                sample_rate=sample_rate,
                                                backward_damping_mult=backward_damping_mult, 
                                                forward_damping_mult=forward_damping_mult, 
                                                damping_fade_expo=damping_fade_expo, 
                                                filter_order=filter_order)
                        
    signal = signal/np.max(signal)
    fig =plt.figure(figsize=(7, 5))
    a=librosa.display.specshow(get_spectrogram(signal)[0],x_axis='time', y_axis='linear',sr=config.sample_rate, hop_length=config.model_list[model]['hop_size'])
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    st.session_state['gaver_audio_loc'] = f'{config.query_gan_tmp_audio_loc_path}{session_uuid}_temp_signal_loc.wav'
    return img_arr, signal.astype(float)

@st.cache_data
def get_model(model):
    print('getting model', model)
    try:
        stylegan_pkl = config.model_list[model]['ckpt_stylegan2_path']
        encoder_pkl = config.model_list[model]['ckpt_encoder_path']
        stylegan_pkl_url = config.model_list[model]['stylegan_pkl_url']
        encoder_pkl_url = config.model_list[model]['encoder_pkl_url']
    except:
        print("Unknown Model!")
        return None, None

    if not os.path.isfile(stylegan_pkl):
        os.makedirs(config.model_list[model]['ckpt_download_stylegan2_path'], exist_ok=True)
        urllib.request.urlretrieve(stylegan_pkl_url, stylegan_pkl)

    if not os.path.isfile(encoder_pkl):
        os.makedirs(config.model_list[model]['ckpt_download_encoder_path'], exist_ok=True)
        urllib.request.urlretrieve(encoder_pkl_url, encoder_pkl)

    with open(stylegan_pkl, 'rb') as pklfile:
        network = pickle.load(pklfile)
        G = network['G'].eval().cuda()

    netE = stylegan_encoder.load_stylegan_encoder(domain=None, nz=G.z_dim,
                                                outdim=G.z_dim,
                                                use_RGBM=True,
                                                use_VAE=False,
                                                resnet_depth=34,
                                                ckpt_path=encoder_pkl).eval().cuda()
    return G, netE


def encode_and_reconstruct(audio):
    model = st.session_state['model_picked']
    audio_pghi = preprocess_signal(audio)
    G, netE = get_model(model)

    im_min = config.im_min
    im_max = config.im_max
    pghi_min = config.pghi_min
    pghi_max = config.pghi_max
    
    audio_pghi = util.zeropad(audio_pghi, config.n_frames * config.model_list[model]['hop_size'] )
    audio_pghi = util.pghi_stft(audio_pghi, hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)

    if model == 'environmental_sounds':
        audio_pghi = np.clip(audio_pghi + -50*(audio_pghi<(-1*25)), -50, 0)

    audio_pghi = util.renormalize(audio_pghi, (np.min(audio_pghi), np.max(audio_pghi)), (im_min, im_max))

    audio_pghi = torch.from_numpy(audio_pghi.astype(np.float32)).cuda().unsqueeze(dim=0)
    mask = torch.ones_like(audio_pghi)[:, :1, :, :]
    net_input = torch.cat([audio_pghi, mask], dim=1).cuda()
    
    encoded_start = time.time()
    with torch.no_grad():
        encoded = netE(net_input)
    print("--- Time taken for [encoded = netE(net_input)] = %s seconds ---" % (time.time() - encoded_start))


    reconstructed_audio_start = time.time()
    reconstructed_audio = G.synthesis(torch.stack([encoded] * 14, dim=1))
    print("--- Time taken for [reconstructed_audio] = %s seconds ---" % (time.time() - reconstructed_audio_start))
    filler = torch.full((1, 1, 1, reconstructed_audio[0].shape[1]), torch.min(reconstructed_audio)).cuda()
    reconstructed_audio = torch.cat([reconstructed_audio, filler], dim=2)
    reconstructed_audio = util.renormalize(reconstructed_audio, (torch.min(reconstructed_audio), torch.max(reconstructed_audio)), (pghi_min, pghi_max))
    reconstructed_audio = reconstructed_audio.detach().cpu().numpy()[0]
    reconstructed_audio_wav = util.pghi_istft(reconstructed_audio, hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)



    loudness_eq_start = time.time()
    loudness = meter.integrated_loudness(reconstructed_audio_wav)
    # loudness normalize audio to -12 dB LUFS
    reconstructed_audio_wav = pyln.normalize.loudness(reconstructed_audio_wav, loudness, -14.0)
    print("--- Time taken to equalize loudness (without soft prior) = %s seconds ---" % (time.time() - loudness_eq_start))
    return encoded, reconstructed_audio_wav


def sample(audio):
    model = st.session_state['model_picked']
    soft_prior_picked = st.session_state['soft_prior_picked']
    try:
        prior_centriod_embedding = np.load(f'../config/resources/prototype/{model}/{soft_prior_picked}.npy')
    except:
        prior_centriod_embedding = None
    
    if prior_centriod_embedding is not None:
        encoded, reconstructed_audio_wav = encode_and_reconstruct_with_soft_prior(audio, prior_centriod_embedding)
    else:
        encoded, reconstructed_audio_wav = encode_and_reconstruct(audio)

    img_arr_time_start = time.time()
    fig, ax = plt.subplots(nrows=1, figsize=(7,5))
    a=librosa.display.specshow(get_spectrogram(reconstructed_audio_wav)[0], x_axis='time', y_axis='linear',sr=config.sample_rate, hop_length=config.model_list[model]['hop_size'], ax=ax)
    # b=librosa.display.waveshow(reconstructed_audio_wav, sr=config.sample_rate, ax=ax)
    ax.set_xlim(0, config.model_list[model]['total_time'])
    ax.set_xlabel('')


    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    print("--- Time taken to read img_arr = %s seconds ---" % (time.time() - img_arr_time_start))
    return img_arr, reconstructed_audio_wav

def map_dropdown_name(input):
    return config.model_list[input]['name']

def get_locs(rate, burst=False, total_time=2.0):
    if rate == 'Very Low':
        return np.linspace(0.0, total_time, 1)
    if rate == 'Low':
        return np.linspace(0.05, total_time-0.3, 3)
    if rate == 'Medium' and not burst:
        if total_time == 2.0:
            return np.linspace(0.05, total_time, 7)
        else:
            return np.linspace(0.0, total_time, 9)
    if rate == 'Medium' and burst:
        burst_width = total_time/3
        return [*np.linspace(0.05, burst_width, 4), *np.linspace(burst_width*2, total_time, 4)]
    if rate == 'High' and not burst:
        return np.linspace(0.05, total_time-0.1, 15)
    if rate == 'High' and burst:
        burst_width = total_time/5
        return [*np.linspace(0.05, burst_width, 6), *np.linspace(burst_width*2, burst_width*3, 6), *np.linspace(burst_width*4, total_time, 6)]
    if rate == 'Very High' and not burst:
        return np.linspace(0.05, total_time-0.1, 20)
    if rate == 'Very High' and burst:
        burst_width = total_time/5
        return [*np.linspace(0.05, burst_width, 12), *np.linspace(burst_width*2, burst_width*3, 8), *np.linspace(burst_width*4, total_time, 8)]

def encode_and_reconstruct_with_soft_prior(audio, prior_centriod_embedding):
    model = st.session_state['model_picked']
    audio_pghi = preprocess_signal(audio)
    G, netE = get_model(model)
    
    im_min = config.im_min
    im_max = config.im_max
    pghi_min = config.pghi_min
    pghi_max = config.pghi_max
    
    audio_pghi = util.zeropad(audio_pghi, config.n_frames * config.model_list[model]['hop_size'] )
    audio_pghi = util.pghi_stft(audio_pghi, hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)
    audio_pghi = np.clip(audio_pghi + -50*(audio_pghi<(-1*25)), -50, 0)
    audio_pghi = util.renormalize(audio_pghi, (np.min(audio_pghi), np.max(audio_pghi)), (im_min, im_max))

    audio_pghi = torch.from_numpy(audio_pghi.astype(np.float32)).cuda().unsqueeze(dim=0)
    mask = torch.ones_like(audio_pghi)[:, :1, :, :]
    net_input = torch.cat([audio_pghi, mask], dim=1).cuda()
    
    encoded_start = time.time()
    with torch.no_grad():
        encoded = netE(net_input)
    print("--- Time taken for [encoded = netE(net_input)] = %s seconds ---" % (time.time() - encoded_start))

    if prior_centriod_embedding is not None:
        prior_centriod_embedding = torch.from_numpy(prior_centriod_embedding).cuda().float()
        encoded = encoded + 0.7*(prior_centriod_embedding - encoded) #Soft prior changed line

    reconstructed_audio_start = time.time()
    reconstructed_audio = G.synthesis(torch.stack([encoded] * 14, dim=1))
    print("--- Time taken for [reconstructed_audio] = %s seconds ---" % (time.time() - reconstructed_audio_start))
    filler = torch.full((1, 1, 1, reconstructed_audio[0].shape[1]), torch.min(reconstructed_audio)).cuda()
    reconstructed_audio = torch.cat([reconstructed_audio, filler], dim=2)
    reconstructed_audio = util.renormalize(reconstructed_audio, (torch.min(reconstructed_audio), torch.max(reconstructed_audio)), (pghi_min, pghi_max))
    reconstructed_audio = reconstructed_audio.detach().cpu().numpy()[0]
    reconstructed_audio_wav = util.pghi_istft(reconstructed_audio, hop_size=config.model_list[model]['hop_size'], stft_channels=config.stft_channels)

    loudness_eq_soft_prior_start = time.time()
    loudness = meter.integrated_loudness(reconstructed_audio_wav)
    # loudness normalize audio to -12 dB LUFS
    reconstructed_audio_wav = pyln.normalize.loudness(reconstructed_audio_wav, loudness, -14.0)
    print("--- Time taken to equalize loudness (with soft prior) = %s seconds ---" % (time.time() - loudness_eq_soft_prior_start))
    return encoded, reconstructed_audio_wav

def on_select_example_change():
    try:
        config_from_example = util.get_config(f'../config/resources/examples/{st.session_state.model_picked}/{st.session_state.example_picked}.json')
    except:
        config_from_example = None
    add_irregularity_value = config_from_example['locs_burst'] if config_from_example is not None else False
    st.session_state.add_irregularity = add_irregularity_value

def main():
    keeper_list = ['model_picked']
    for key in keeper_list:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    if config.allow_analytics:
        google_analytics.set_google_analytics()
    st.markdown("<div style='display: flex;justify-content: center;'><h1 style='text-align: center; width: 500px;'>Exploring Environmental Sound Spaces - 1</h1></div>", unsafe_allow_html=True)

    st.markdown(f'''
        <style>
            section[data-testid="stSidebar"] {{
                min-width: 30%;
                max-width: 30%;
            }}
            .stDownloadButton {{text-align: center;}}
            .stDownloadButton > button {{background-color: #fafafa; color: rgb(19, 23, 32);}}
        </style>
    ''',unsafe_allow_html=True)
    if 'session_uuid' not in st.session_state:
        st.session_state['session_uuid'] = str(uuid.uuid4())
    session_uuid = st.session_state['session_uuid']

    model_names = []
    for key in config.model_list:
        model_names.append(key)
    model_names = tuple(model_names)
    model_picked =  st.sidebar.selectbox('Select Model', model_names, format_func=map_dropdown_name, key='model_picked')

    try:
        example_arr = os.listdir(f'../config/resources/examples/{model_picked}')
    except:
        example_arr = []
    example_arr_extensionless = [os.path.splitext(file_name)[0] for file_name in example_arr]
    example_arr_extensionless.insert(0, '-None-')
    example_arr_extensionless = sorted(example_arr_extensionless)
    example_picked =  st.sidebar.selectbox('Select Example', example_arr_extensionless, on_change=on_select_example_change, key='example_picked')
    
    try:
        config_from_example = util.get_config(f'../config/resources/examples/{model_picked}/{example_picked}.json')
    except:
        config_from_example = None


    horizontal_line = '<div style="border: solid #404040 2px; margin-bottom:10%"></div>'
    st.sidebar.markdown(horizontal_line, unsafe_allow_html=True)

    impact_type = 'hit'
    impulse_time_value = float(config_from_example['impulse_time'] if config_from_example is not None else 0.05)

    rate_value = config_from_example['locs'] if config_from_example is not None else 'Medium'
    impulse_rate_config = []
    for dic in config.impulse_rate:
        impulse_rate_config.append(dic['label'])
    rate =  st.sidebar.selectbox('Number of Impulses (Rate)', impulse_rate_config, index=impulse_dict[rate_value], key='rate_position',)

    if 'add_irregularity' not in st.session_state:
        st.session_state['add_irregularity'] = config_from_example['locs_burst'] if config_from_example is not None else False
    
    if model_picked=='environmental_sounds':
        add_irregularity = st.sidebar.checkbox('Add Irregularity to Rate', key='add_irregularity')
    else:
        add_irregularity = st.session_state['add_irregularity']

    locs_value = get_locs(rate, add_irregularity, config.model_list[model_picked]['total_time'])

    trail_lf_value = config_from_example['trail_lf'] if config_from_example is not None else 1
    trail_hf_value = config_from_example['trail_hf'] if config_from_example is not None else 700
    lf, hf = st.sidebar.select_slider(
                            'Frequency Band',
                            options=[1, *np.arange(10,7999,10)],
                            value=(trail_lf_value, trail_hf_value))
    
    filter_order_value = float(config_from_example['filter_order'] if config_from_example is not None else 0.0)
    
    forward_damping_mult_value = float(config_from_example['forward_damping_mult'] if config_from_example is not None else 0.0)
    backward_damping_mult_value = float(config_from_example['backward_damping_mult'] if config_from_example is not None else 0.0)

    damping_fade_expo_value = float(config_from_example['damping_fade_expo'] if config_from_example is not None else 1.0)

    with st.sidebar:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            impulse_time = my_component(key="impulse_width_position", 
                    value=impulse_time_value, 
                    step=0.01,
                    min_value=0, 
                    max_value=config.model_list[model_picked]['total_time'],
                    track_color="gray",
                    thumb_color="black",
                    example=example_picked,
                    label="Impulse Width"
                    )
            if impulse_time == None:
                impulse_time = impulse_time_value

        with col2:
            filter_order = my_component(key="filter_order_position", 
                    value=filter_order_value, 
                    step=1,
                    min_value=0, 
                    max_value=5,
                    track_color="gray",
                    thumb_color="black",
                    example=example_picked,
                    label="Filter Order"
                    )
            if filter_order == None:
                filter_order = filter_order_value

        with col3:
            damping_fade_expo = my_component(key="damping_fade_expo_position", 
                    value=damping_fade_expo_value, 
                    step=1,
                    min_value=1, 
                    max_value=5,
                    track_color="gray",
                    thumb_color="black",
                    example=example_picked,
                    label="Fade Exponent"
                    )
            if damping_fade_expo == None:
                damping_fade_expo = damping_fade_expo_value

        with col4:
            forward_damping_mult = my_component(key="fdamping_mult_position", 
                    value=forward_damping_mult_value, 
                    step=0.1,
                    min_value=0, 
                    max_value=1,
                    track_color="gray",
                    thumb_color="black",
                    example=example_picked,
                    label="Fade In"
                    )
            if forward_damping_mult == None:
                forward_damping_mult = forward_damping_mult_value

        with col5:
            backward_damping_mult = my_component(key="bdamping_mult_position", 
                    value=backward_damping_mult_value, 
                    step=0.1,
                    min_value=0, 
                    max_value=1,
                    track_color="gray",
                    thumb_color="black",
                    example=example_picked,
                    label="Fade Out"
                    )
            if backward_damping_mult == None:
                backward_damping_mult = backward_damping_mult_value
    st.sidebar.markdown(horizontal_line, unsafe_allow_html=True)


    try:
        soft_prior_list = os.listdir(f'../config/resources/prototype/{model_picked}')
    except:
        soft_prior_list = []
    soft_prior_list_extensionless = [os.path.splitext(file_name)[0] for file_name in soft_prior_list]
    soft_prior_list_extensionless = sorted(soft_prior_list_extensionless)
    soft_prior_list_extensionless.insert(0, 'None')
    

    if len(soft_prior_list_extensionless) == 1:
        is_soft_prior_picked_disabled = True
    else:
        is_soft_prior_picked_disabled = False
    soft_prior_picked =  st.sidebar.selectbox('Constraint', soft_prior_list_extensionless, key='soft_prior_picked', disabled=is_soft_prior_picked_disabled)

    col1, col2, col3 = st.columns((4,1,4))

    s_pghi, s_audio = get_gaver_sounds(initial_amplitude=1.0, hittype=impact_type, total_time=config.model_list[model_picked]['total_time'],\
                                    impulse_time=impulse_time, sample_rate=config.sample_rate,\
                                    filters=[lf, hf], locs=locs_value,\
                                    filter_order=filter_order, \
                                    forward_damping_mult=forward_damping_mult, \
                                    backward_damping_mult=backward_damping_mult, \
                                    damping_fade_expo=damping_fade_expo,\
                                    session_uuid=session_uuid)
    
    if 'gaver_audio_bytes' not in st.session_state:
        st.session_state['gaver_audio_bytes'] = byte_array = bytes([])
        st.session_state['gaver_img_arr'] = np.zeros((500,700,4))
    
    s_recon_pghi, s_recon_wav = sample(s_audio)#, session_uuid=session_uuid) 

    with col1:
        colname = '<div style="text-align:center"><h3><b><i>Synthetic Reference<br/><span>&nbsp;</span></i></b></h3></div>'
        st.markdown(colname, unsafe_allow_html=True)
        st.image(s_pghi)
        st.audio(s_audio, format="audio/wav", start_time=0, sample_rate=config.sample_rate)
    with col2:
        colname = '&nbsp;'
        st.markdown(colname, unsafe_allow_html=True)
    with col3:
        colname = '<div style="padding-left: 0%;text-align: center;"><h3><b><i>AI Generated Sound Matching <br/>Synthetic Reference</i></b></h3></div>'
        st.markdown(colname, unsafe_allow_html=True)
        st.image(s_recon_pghi)
        st.audio(s_recon_wav, format="audio/wav", start_time=0, sample_rate=config.sample_rate)
        # st.download_button(
        #     label="Download Sound",
        #     data=s_recon_wav,
        #     file_name='Reconstructed_Audio.wav',
        #     mime='audio/wav',
        # )

if __name__ == '__main__':
    main()
