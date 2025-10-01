import json
from hashlib import sha256
import numpy as np
import math
import time

def array_lerp(arr_a, arr_b, x):
    return arr_a+(arr_b - arr_a)*x
    
def list_lerp(list_a, list_b, x):
    list_result = [None]*len(list_a)
    for i in range(len(list_a)):
        list_result[i] = list_a[i] + (list_b[i] - list_a[i]) * x
    return list_result
    
def get_unit(r):
    _list = [0.000001,0.000002,0.000005,0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000]
    choice = 0
    while _list[choice] < r * 0.2:
        choice += 1
    return _list[choice]
    
def hue_to_rgb(hue):
    h = (hue*6)%1.0
    r = (0,0,0)
    if hue < 1/6:
        r = (1,h,0)
    elif hue < 2/6:
        r = (1-h,1,0)
    elif hue < 3/6:
        r = (0,1,h)
    elif hue < 4/6:
        r = (h*0.4,1-h*0.6,1)
    elif hue < 5/6:
        r = (0.4+0.6*h,0.4,1)
    else:
        r = (1,0.4-0.4*h,1-h)
    return 255*r[0], 200*r[1], 255*r[2]
    
def species_to_name(s, ui):
    salted = str(s)+ui.salt
    _hex = sha256(salted.encode('utf-8')).hexdigest()
    result = int(_hex, 16)
    length_choices = [5,5,6,6,7]
    length_choice = result%5
    result = result//5
    
    letters = ["bcdfghjklmnprstvwxz","aeiouy"]
    name_len = length_choices[length_choice]
    name = ""
    for n in range(name_len):
        letter_type = n%len(letters)
        option_count = len(letters[letter_type])
        choice = result%option_count
        letter = letters[letter_type][choice]
        if n >= 2 and letter == "g" and name[n-2].lower() == "n":
            letter = "m"
        if n == 0:
            letter = letter.upper()
        name += letter
        result = result//option_count

    return name
    
def brighten(color, b):
    if b >= 1:
        fac = b-1
        return lerp(color[0],255,fac),lerp(color[1],255,fac),lerp(color[2],255,fac)
    else:
        return color[0]*b, color[1]*b, color[2]*b

def species_to_color(s, ui):
    # Convert species to a simple integer (deterministic)
    if isinstance(s, int):
        val = s
    else:
        val = sum(ord(c) for c in str(s) + ui.salt)

    # Simple fast pseudo-randomness using bit shifts
    rand = ((val << 5) ^ (val >> 3)) & 0xFF

    # Generate hue [0,1) with slight randomness
    hue = ((val & 0xFF) + rand) % 256 / 256.0

    # Generate brighter brightness [0.7,1.0) with randomness
    brightness = 0.7 + (((val >> 8) + rand) & 0x3F) / 100.0
    if brightness > 1.0:
        brightness = 1.0

    # Convert to RGB
    color = hue_to_rgb(hue)

    # Brighten
    new_color = brighten(color, brightness)

    return new_color

    
def bound(x):
    return min(max(x,0),1)
    
def lerp(a,b,x):
    return a+(b-a)*x
    
def dist_to_text(dist, sigfigs, u):
    if sigfigs:
        return f"{dist/u:.2f}cm"
    else:
        return str(int(dist/u))+"cm"
        
def get_distance_array(a, b):
    x_dist = a[:,:,:,0]-b[:,:,:,0]
    y_dist = a[:,:,:,1]-b[:,:,:,1]
    return np.sqrt(np.square(x_dist)+np.square(y_dist))

def apply_muscles(n, m, muscle_coef):
    B, H, W, _ = n.shape
    Hm, Wm = m.shape[1:3]

    # --- Compute neighbor distances ---
    x_dist = np.hypot(n[:, 1:, :, 0] - n[:, :-1, :, 0],
                      n[:, 1:, :, 1] - n[:, :-1, :, 1])
    y_dist = np.hypot(n[:, :, 1:, 0] - n[:, :, :-1, 0],
                      n[:, :, 1:, 1] - n[:, :, :-1, 1])
    pos_diag_dist = np.hypot(n[:, 1:, 1:, 0] - n[:, :-1, :-1, 0],
                             n[:, 1:, 1:, 1] - n[:, :-1, :-1, 1])
    neg_diag_dist = np.hypot(n[:, 1:, :-1, 0] - n[:, :-1, 1:, 0],
                             n[:, 1:, :-1, 1] - n[:, :-1, 1:, 1])

    # --- Slice neighbor distances to match muscle grid ---
    x_dist0 = x_dist[:, :Hm, :Wm]
    x_dist1 = x_dist[:, :Hm, 1:Wm+1]
    y_dist0 = y_dist[:, :Hm, :Wm]
    y_dist1 = y_dist[:, 1:Hm+1, :Wm]
    pos_diag_dist = pos_diag_dist[:, :Hm, :Wm]
    neg_diag_dist = neg_diag_dist[:, :Hm, :Wm]

    # --- Muscle adjustments ---
    m_as = [
        (m[:, :, :, 0] - x_dist0) * muscle_coef,
        (m[:, :, :, 0] - x_dist1) * muscle_coef,
        (m[:, :, :, 1] - y_dist0) * muscle_coef,
        (m[:, :, :, 1] - y_dist1) * muscle_coef,
        (m[:, :, :, 3] - pos_diag_dist) * muscle_coef,
        (m[:, :, :, 3] - neg_diag_dist) * muscle_coef,
    ]

    segments = [
        [0, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ]

    # --- Apply muscle updates ---
    for dire, (h1, w1, h2, w2) in enumerate(segments):
        h_len = H-1
        w_len = W-1
        sli1 = n[:, h1:h1+h_len, w1:w1+w_len, :2]
        sli2 = n[:, h2:h2+h_len, w2:w2+w_len, :2]

        delta = sli1 - sli2
        delta_norm = delta / (np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-12)
        update = delta_norm * m_as[dire][..., np.newaxis]

        n[:, h1:h1+h_len, w1:w1+w_len, 2:4] += update
        n[:, h2:h2+h_len, w2:w2+w_len, 2:4] -= update


def get_muscle_attraction(dists, m, muscle_coef):
    return (m-dists) * muscle_coef
    
def get_dist(x1, y1, x2, y2):
    return np.linalg.norm(np.array([x2-x1,y2-y1]))
    
def array_int_multiply(arr, factor):

    result = [None] * len(arr)

    for i in range(len(arr)):
        result[i] = int(arr[i] * factor)

    return result

def read_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        result: dict = json.load(f)

        return result