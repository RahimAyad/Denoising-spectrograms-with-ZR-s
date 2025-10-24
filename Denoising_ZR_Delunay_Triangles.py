'''
The goal of this program is to do denoise spectrogram tooken from
the paper "Unsupervised Classification of the Spectrogram Zeros with 
an Application to Signal Detection and Denoising"
Juan M. Miramont,al,2024

This code will be composed of : 
- generating a synthetic noisy signal,
- compute the spectrogram with superlets
- detect the zeros (ZR's, or the local minimas)
- compute the Delaunay triangulation based on ZR's
- estimate l_max by hand to select the right triangles
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
from sklearn.mixture import GaussianMixture
from typing import Tuple, List
from superlet import superlets
from scipy.ndimage import minimum_filter
from matplotlib.widgets import Slider

def generate_synthetic_siganl(N:int = 500 ,fs:int = 1000, snr_db:float = 5.0 ):
    """
    This function generate a signal, a sum of sin with a gaussian white nise

        Parameters:
        -----------
        N : number of points
        fs : sampling frequency (Hz)
        snr_db : signal/noise in dB

        Return:
        ---------
        t : time (secondes)
        y : noisy signal 
        x : clean original signal
    """

    print("\n" + "="*70)
    print("Step 1 : Synthetic signal generation")
    print("="*70)

    t = np.arange(N)/fs # time vector
    print(f' Duration of the signal : {t[-1]:.3f} seconds') 

    # denoised signal 
    x = (  np.sin(2*np.pi* 50 *t)      # 50 Hz
         + np.sin(2*np.pi* 150 *t)     # 150 Hz
         + np.sin(2*np.pi* 300 *t))    # 300 Hz

    # computation of the noisy signal
    signal_power = np.mean(x**2)
    noise_power = signal_power / (10**(snr_db /10)) # SNR formula from the paper section 2.2
    noise = np.sqrt(noise_power) * np.random.randn(N)

    y = x + noise

    print(f"  → SNR : {snr_db} dB")
    print(f"  → Power of the signal : {signal_power:.4f}")
    print(f"  → Power of the noise : {noise_power:.4f}")
    print(f"  → square root of the noise : {np.sqrt(noise_power):.4f}")

    return t, y, x 

def spectrogram_superlets ( signal:np.ndarray,
                            sampling_frequency:int,
                            down_sampling:int = 1, 
                            frequencies:tuple=(int,int,int),
                            first_cycle:int = 5,
                            order_range:tuple=(int,int) ):

    freq = np.linspace(frequencies[0],frequencies[1],frequencies[2])
    c1 = first_cycle 
    ord_range = (order_range[0],order_range[1])
    fs = sampling_frequency

    spec = superlets(signal, fs=(fs/down_sampling), foi=freq, c1=c1, ord=ord_range)

    return spec, freq

def Gaussian_white_noise(size_of_the_spec:float, variance:float=0.01):
    noise  = np.random.normal(0,variance,size_of_the_spec)
    return noise

def detection_of_the_zeros (spectrogram:np.ndarray, size:int, threshold: float= 1e-6) -> np.ndarray:
    """
    Détecte les minima locaux (zéros approchés)

    Parameters:
    -----------
    spectrogram : magnitude du spectrogramme
    size : taille du filtre (neighbourhood)

    Returns:
    --------
    local_minima : masque booléen des minima
    """
    local_minima = (spectrogram == minimum_filter(spectrogram, size=size))
    return local_minima

def Delaunay_tessellation(zeros_mask: np.ndarray, t: np.ndarray, freqs: np.ndarray):
    """
    Calcule la triangulation de Delaunay sur les zéros détectés

    Parameters:
    -----------
    zeros_mask : masque booléen des zéros (shape: freq × time)
    t : vecteur temps
    freqs : vecteur fréquences

    Returns:
    --------
    tri : objet Delaunay
    points : coordonnées (temps, freq) des zéros
    """
    print("\n" + "="*70)
    print("Step 2 : Delaunay Triangulation")
    print("="*70)

    # Extraction des coordonnées (indices)
    zeros_y, zeros_x = np.where(zeros_mask)

    # Conversion en coordonnées physiques (temps, fréquence)
    points_time = t[zeros_x]
    points_freq = freqs[zeros_y]
    points = np.column_stack([points_time, points_freq])

    print(f"  → Nombre de zéros : {len(points)}")

    # Triangulation
    if len(points) < 3:
        print("  ⚠ Pas assez de points pour la triangulation (minimum 3)")
        return None, points

    tri = Delaunay(points)
    print(f"  → Nombre de triangles : {len(tri.simplices)}")

    return tri, points

def choose_triangles_to_keep(tri: Delaunay, points: np.ndarray, l_max: float):
    """
    Garde les triangles ayant au moins UNE arête > l_max
    
    Parameters:
    -----------
    tri : objet Delaunay
    points : coordonnées (N, 2) des points [temps, freq]
    l_max : seuil de longueur maximale
    
    Returns:
    --------
    mask : masque booléen (True = garder le triangle)
    """
    if tri is None:
        return np.array([])
    
    mask = np.zeros(len(tri.simplices), dtype=bool)
    
    for i, simplex in enumerate(tri.simplices):
        # Récupère les 3 sommets du triangle
        p0, p1, p2 = points[simplex]
        
        # Calcule les 3 longueurs d'arêtes
        edge1 = np.linalg.norm(p1 - p0)
        edge2 = np.linalg.norm(p2 - p1)
        edge3 = np.linalg.norm(p0 - p2)
        
        # Garde si AU MOINS une arête > l_max
        if max(edge1, edge2, edge3) > l_max:
            mask[i] = True
    
    return mask

# ============================================================================
# EXÉCUTION ET CALCULS
# ============================================================================

t, y, x = generate_synthetic_siganl(30, 1000, 5)
spec, freqs = spectrogram_superlets(y, sampling_frequency=1000, 
                                    frequencies=(1, 350, 200), 
                                    first_cycle=5, 
                                    order_range=(3, 20))
noise = Gaussian_white_noise(size_of_the_spec=spec.shape, variance=0.01)
zeros = detection_of_the_zeros(np.abs(spec+noise), size=10, threshold=1e-6)

# Triangulation de Delaunay
tri, points = Delaunay_tessellation(zeros, t, freqs)

# Positions des zéros pour overlay
zeros_y, zeros_x = np.where(zeros)

# Calcul de l_max initial et range
if tri is not None:
    all_edges = []
    for simplex in tri.simplices:
        p0, p1, p2 = points[simplex]
        all_edges.extend([
            np.linalg.norm(p1 - p0),
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p0 - p2)
        ])
    l_max_min = np.min(all_edges)
    l_max_max = np.max(all_edges)
    l_max_init = (l_max_min + l_max_max) / 2
else:
    l_max_min, l_max_max, l_max_init = 0, 1, 0.5

# ============================================================================
# FIGURE AVEC WIDGET INTERACTIF
# ============================================================================

fig = plt.figure(figsize=(16, 10))

# Grid layout: 3 petits en haut, 1 grand en bas avec slider
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 4], hspace=0.3, wspace=0.3)

# --- PETITS SUBPLOTS EN HAUT ---
ax1 = fig.add_subplot(gs[0, :])  # Signaux temporels
ax2 = fig.add_subplot(gs[1, :])  # Spectrogramme
ax3 = fig.add_subplot(gs[2, :])  # Zéros

# SUBPLOT 1 : Signaux temporels
ax1.plot(t, x, 'b-', linewidth=1.2, label='Signal propre', alpha=0.7)
ax1.plot(t, y, 'r-', linewidth=0.6, label='Signal bruité', alpha=0.5)
ax1.set_ylabel('Amplitude', fontsize=9)
ax1.set_title('Signaux temporels', fontsize=10, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=8)

# SUBPLOT 2 : Spectrogramme
im = ax2.imshow(np.abs(spec), aspect='auto', origin='lower', 
                extent=[t[0], t[-1], freqs[0], freqs[-1]], 
                cmap='viridis', interpolation='bilinear')
ax2.set_ylabel('Fréquence (Hz)', fontsize=9)
ax2.set_title('Spectrogramme Superlet', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Magnitude', fraction=0.046, pad=0.04)
ax2.tick_params(labelsize=8)

# SUBPLOT 3 : Zéros
ax3.imshow(np.abs(spec), aspect='auto', origin='lower', 
           extent=[t[0], t[-1], freqs[0], freqs[-1]], 
           cmap='gray', alpha=0.4, interpolation='bilinear')
ax3.scatter(t[zeros_x], freqs[zeros_y], c='red', s=0.8, alpha=0.7)
ax3.set_ylabel('Fréquence (Hz)', fontsize=9)
ax3.set_title(f'Zéros détectés ({zeros.sum()} points)', fontsize=10, fontweight='bold')
ax3.tick_params(labelsize=8)

# --- GRAND SUBPLOT PRINCIPAL ---
ax_main = fig.add_subplot(gs[3, :])

# --- CRÉATION DU SLIDER D'ABORD ---
ax_slider = plt.axes([0.15, 0.02, 0.7, 0.02], facecolor='lightgray')
slider = Slider(
    ax_slider, 
    'l_max', 
    l_max_min, 
    l_max_max, 
    valinit=l_max_init,
    valstep=(l_max_max - l_max_min) / 10000
)

# --- FONCTION DE MISE À JOUR (APRÈS LA CRÉATION DU SLIDER) ---
def update(val):
    ax_main.clear()
    
    l_max = slider.val  # Maintenant slider existe !
    
    # Background spectrogramme
    ax_main.imshow(np.abs(spec), aspect='auto', origin='lower', 
                   extent=[t[0], t[-1], freqs[0], freqs[-1]], 
                   cmap='gray', alpha=0.25, interpolation='bilinear')
    
    if tri is not None:
        # Filtrage des triangles
        mask = choose_triangles_to_keep(tri, points, l_max)
        selected_simplices = tri.simplices[mask]
        
        # Dessiner UNIQUEMENT les triangles sélectionnés
        if len(selected_simplices) > 0:
            ax_main.triplot(points[:, 0], points[:, 1], selected_simplices, 
                           'c-', linewidth=0.5, alpha=0.8)
        
        # Points des zéros
        ax_main.plot(points[:, 0], points[:, 1], 'r.', markersize=1.5, alpha=0.6)
        
        ax_main.set_title(
            f'Triangulation filtrée: {mask.sum()}/{len(tri.simplices)} triangles '
            f'(l_max = {l_max:.4f})', 
            fontsize=12, fontweight='bold'
        )
    
    ax_main.set_xlabel('Temps (s)', fontsize=11)
    ax_main.set_ylabel('Fréquence (Hz)', fontsize=11)
    ax_main.set_xlim([t[0], t[-1]])
    ax_main.set_ylim([freqs[0], freqs[-1]])
    fig.canvas.draw_idle()

# Initialisation du plot principal
update(l_max_init)

# Connexion du callback
slider.on_changed(update)

plt.show()

