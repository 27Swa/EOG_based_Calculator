import streamlit as st
import main as mn
import handlingfunctions as hd
import joblib
from scipy.signal import butter, filtfilt, find_peaks
import os
import signal
# initializing text to display


button_labels = [
    ['', '', '', '4', '', '', ''],
    ['', '', '5', '6', '7', '', ''],
    ['', '1', '', '', '', '9', ''],
    ['0', '2', '', '.', '', 'E', '8'],
    ['', '3', '', '', '', 'C', ''],
    ['', '', '/', '+', '-', '', ''],
    ['', '', '', '*', '', '', '']
]

arrows = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "blink": "✅",  # Optional: for 'blink' or selection
    "C": "🔄"   # Optional: for reset or clear
}

def movement(m,pos):
    r, c = pos
    val = 2 if r == 3 and c == 3 else 1

    if m == "up":
        r -= val
    elif m == "down":
        r += val
    elif m == "left":
        c -= val
    elif m == "right":
        c += val
    else:
        r = c = 3
    if (r >= 0 )and (r < 7) and (c >= 0) and (c < 7):
        return (r, c)
    else:
        return pos

def define_calculator():
    oper_labels = {
        '*': '&#42;',
        '+': '&#43;',
        '-': '&#45;',
        '/': '&#47;'
    }
    st.markdown("""
               <style>
               .calculator_button {
                   display: inline-block;
                   width: 50px;
                   height: 50px;
                   line-height: 50px;
                   background-color: #AACCF9;
                   color: white;
                   text-align: center;
                   border-radius: 8px;
                   margin: 5px;
                   font-weight: bold;
                   font-size: 20px;
                   user-select: none;
               }

             .active {
           background-color: #00BFFF !important;
           color: white !important;
       }


               </style>
           """, unsafe_allow_html=True)
    for i,row in enumerate(button_labels):
        cols = st.columns(len(row))
        for j, label in enumerate(row):
            if label:
                if label in oper_labels:
                    l = oper_labels[label]
                else:
                    l = label
                class_name = "calculator_button"
                if (i, j) == st.session_state.current_pos:
                    class_name += " active"

                cols[j].markdown(f"<div class = '{class_name}'>{l}</div>", unsafe_allow_html=True)
            else:
                cols[j].write(" ")

def calculator(op):
    n1,o,n2 = int(op[0]),op[1],int(op[2])
    if o == '+':
        return str(n1 + n2)
    elif o =='*':
        return str(n1 * n2)
    elif o == '/':
        if n2==0:
            return str(-1)
        else:
            return str(n1/n2)
    else:
        return str(n1-n2)

def deploy():
    if 'oper' not in st.session_state:
        st.session_state.oper = []
    if 'mov' not in st.session_state:
        st.session_state.mov =[]
    col1, col2= st.columns([1,5])

    with col1:
        st.image("EOG.png",width=150)

    with col2:
        st.title("EOG based calculator")
        st.markdown("Turning thoughts into actions — one signal at a time.")

    st.markdown("---")

    # Getting files from user
    col1, col2,col3= st.columns([5,3,3])
    with col1:
        st.header("📁 Upload EOG Signal Files")
    with col2:
        hor_file = st.file_uploader("Upload horizontal signal",type = ['csv','txt','xlsx'],key = 1)
    with col3:
        ver_file = st.file_uploader("Upload vertical signal",type = ['csv','txt','xlsx'],key = 2)

    # Preprocessing and prediction
    if hor_file and ver_file:
        if st.button("Run Preprocessing and Predict"):
            # reading files
            hfile = hd.uploaded_file(hor_file)
            vfile = hd.uploaded_file(ver_file)

            h_signal = hd.validate_signal_file(hfile)
            v_signal = hd.validate_signal_file(vfile)

            # apply band bass filter on it
            h_filtered = hd.butter_bandpass_filter(h_signal, hd.LOW_CUTOFF, hd.HIGH_CUTOFF, hd.SAMPLE_RATE, hd.ORDER)
            v_filtered = hd.butter_bandpass_filter(v_signal, hd.LOW_CUTOFF, hd.HIGH_CUTOFF, hd.SAMPLE_RATE, hd.ORDER)

            # extracting features
            h_features = hd.extract_morphological_features(h_filtered.reshape(1, -1))
            v_features = hd.extract_morphological_features(v_filtered.reshape(1, -1))

            # apply prediction
            selected_features = hd.features_selection(h_features, v_features)
            label = hd.prediction(selected_features)

            if len(st.session_state.oper) == 5:
                st.session_state.oper = []
                st.session_state.mov = []

            st.success(f"Prediction result: {label}")
            st.session_state.mov.append(arrows[label])

            pre_pos =  st.session_state.current_pos
            st.session_state.current_pos = movement(label,st.session_state.current_pos)
            st.write("Current Position:", st.session_state.current_pos)            # making  the calculator
            define_calculator()

            if label == 'blink' and button_labels[pre_pos[0]][pre_pos[1]] == 'C':
                st.session_state.oper = []
                st.session_state.mov.append(arrows['C'])

            elif label == 'blink' and button_labels[pre_pos[0]][pre_pos[1]] == 'E':
                st.markdown("### 🔚 Exiting application...")
                st.stop()

            elif label == 'blink':
                st.session_state.oper.append(button_labels[pre_pos[0]][pre_pos[1]])
                if len(st.session_state.oper) == 3:
                    res = calculator(st.session_state.oper)
                    st.session_state.oper.append('=')
                    st.session_state.oper.append(res)

            st.markdown("---")
            st.markdown(f"### **Input Sequence:** `{''.join(st.session_state.oper)}`")
            st.markdown("---")
            st.markdown(f"### **Movements Sequence:** `{','.join(st.session_state.mov)}`")


            # delete files from its temporary file after using it
            hd.delete_file_safely(hfile)
            hd.delete_file_safely(vfile)
        else:
            if 'current_pos' not in st.session_state:
                st.session_state.current_pos = (3, 3)

            # making  the calculator
            define_calculator()

            # printing the results
            st.markdown(f"### **Input Sequence:** `{''.join(st.session_state.oper)}`")
            st.markdown("---")
            st.markdown(f"### **Movements Sequence:** `{','.join(st.session_state.mov)}`")

    st.markdown("---")
    st.markdown("""
    ### ℹ️ About

    This application is an **EOG-based virtual calculator** designed to assist users with limited motor abilities by enabling them to perform calculations through **eye movement signals**.

    Using *Electroculography (EOG) signals* — specifically horizontal and vertical eye movements — this tool captures, filters, and processes real-time gaze data to determine movement directions (e.g., left, right, up, down, blink). These signals are then mapped to navigate a virtual calculator interface.
    """)
deploy()