'''
DENOISING SPECTROGRAM
Based on: "Unsupervised Classification of the Spectrogram Zeros with 
an Application to Signal Detection and Denoising"
Juan M. Miramont et al., 2024

This version will be done in a different way that the one proposed in the paper
looking for the faster way to treat our spectrograms, because i use KDTree 
and the square decomposition unstead of the voronoi tessellation is more 
efficient.

PIPELINE:
1. Generate synthetic noisy signal with a known signal to noise ration (SNR)
2. Compute spectrogram with superlets
3. Add multiple Gaussian White Noise (J realizations) to get multiple ZR's
4. Creat a grid and Assign each Z_n to each square of the grid
5. Compute convex hull area for zeros in each square
6. Compare square areas with convex hull areas
7. Classify square cell as NN, SN, or SS (N=Noise, S=Signal)
8. Generate mask to remove NN cells
 

Adventages: we dont have to compute the area of the voronoi cells because we
generate a square with a know and general width and lenght.
no nead to compute the areas using "polygon" 
the compatibility between KDTree and the squares is better than with the 
vornoi cells.
we get an easier control on the squares 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial import Delaunay, ConvexHull
from typing import Tuple, Optional, List, Dict
from superlet import superlets
from scipy.ndimage import minimum_filter
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree

import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_synthetic_signal(N: int = 500,
                              fs: int = 10000,
                               snr_db : float = 2.0,
                                times :int = 1 ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Generate a synthetic signal with N points, a sampling frequency of fs, and a sound ot noise ration in db of snr_db
    """
    print("\n" + "="*70)
    print("Synthetic Signal Generation")
    print("="*70)

    t = np.arange(N)/fs
    print(f"    Signal duration : {t[-1]:.3f} s")

    x = ((np.sin(2*np.pi*25*t))+   # a signal of 25 Hz, 250 Hz and 300 Hz
         (np.sin(2*np.pi*250*t))+  # same frequencies we could find in a bacteria
         (np.sin(2*np.pi*300*t)))

    # add noise
    signal_power = np.mean(x**2)
    noise_power = signal_power / 10**(snr_db/10)
    noise = np.sqrt(noise_power) * np.random.randn(N)

    y = x + (noise*times)

    print (f'   SNR : {snr_db} dB')
    print (f'   Signal_power : {signal_power:.4f}')
    print (f'   Noise_power : {noise_power:.4f} ')

    return t, y, x

# ============================================================================
# Noise
# ============================================================================

def add_gaussian_white_noise(spec_shape: Tuple[int, int], 
                            variance: float = 0.01) -> np.ndarray:
    """Generate complex Gaussian white noise"""
    return np.random.normal(0, variance, spec_shape)

# ============================================================================
# Spectrogram computation using superlets
# ============================================================================

def compute_spectrogram_superlets(signal: np.ndarray,
                                  sampling_frequency: int,
                                  frequencies: Tuple[int, int, int] = (1, 350, 200),
                                  first_cycle: int = 5,
                                  order_range: Tuple[int, int] = (3, 20)) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectrogram using Superlets transform,
        Note : 
        The first cycle is the the number of the cycles in the morlet wavelet so, if the first cycle is small (2-4)
        the temporal resolution is good and the frequency resolution not that much and if high (7-12) it's the opposite,
        good frequency resolution and bad temporal one.

        The order is the number of wavelets that you combine at each frequency
    """
    print("\n" + "="*70)
    print("Spectrogram Computation")
    print("="*70)

    freq = np.linspace(frequencies[0], frequencies[1], frequencies[2])
    spec = superlets(signal, fs=sampling_frequency, foi=freq, 
                    c1=first_cycle, ord=order_range)

    print(f"  Frequency range: {freq[0]:.1f} - {freq[-1]:.1f} Hz")
    print(f"  Spectrogram shape: {spec.shape}")

    return spec, freq

# ============================================================================
# Zeros (Z_0) detection
# ============================================================================

def detect_zeros(spectrograms: np.ndarray, 
                       threshold_percentile: float = 5,
                       verbose: bool = False,
                       size :int = 5) -> np.ndarray:
    """
    Détecte les minima locaux sur un batch de spectrogrammes.
    
    Args:
        spectrograms: Shape (J, F, T) ou (F, T)
    Returns:
        zeros_masks: Shape (J, F, T) ou (F, T)
    """
    
    # Gestion input 2D ou 3D
    is_single = (spectrograms.ndim == 2)
    if is_single:
        spectrograms = spectrograms[np.newaxis, :, :]  # (1, F, T)
    
    J = spectrograms.shape[0]
    
    # 1️⃣ Magnitude + normalisation vectorisée
    magnitudes = np.abs(spectrograms)  # (J, F, T)
    
    # Normalisation par réalisation (axis=(1,2))
    mag_min = magnitudes.min(axis=(1, 2), keepdims=True)  # (J, 1, 1)
    mag_max = magnitudes.max(axis=(1, 2), keepdims=True)
    magnitude_norm = (magnitudes - mag_min) / (mag_max - mag_min + 1e-10)  # Évite div/0
    
    # 2️⃣ Seuils par réalisation
    thresholds = np.percentile(magnitude_norm, threshold_percentile, axis=(1, 2))  # (J,)
    thresholds = thresholds[:, np.newaxis, np.newaxis]  # (J, 1, 1) pour broadcasting
    
    # 3️⃣ Minima multi-échelles VECTORISÉ
    zeros_mask = np.zeros_like(magnitude_norm, dtype=bool)
    '''
    for size in [5, 7, 10]:
        #  Applique le filtre sur tout le batch d'un coup
        # axis=0 préservé, filtre sur (F, T) pour chaque J
        local_min = np.array([
            minimum_filter(magnitude_norm[j], size=size) 
            for j in range(J)
        ])  # Reste une mini-boucle, mais 3x moins d'appels qu'avant
        
        is_minimum = (magnitude_norm == local_min) & (magnitude_norm < thresholds)
        zeros_mask |= is_minimum
    '''
    local_min = np.array([minimum_filter(magnitude_norm[j], size=size)
                        for j in range(J)])

    # 4️⃣ Suppression des bords (vectorisé)
    zeros_mask[:, 0, :] = False
    zeros_mask[:, -1, :] = False
    zeros_mask[:, :, 0] = False
    zeros_mask[:, :, -1] = False
    
    # 5️⃣ Fallback si trop peu de zeros
    n_zeros = np.sum(zeros_mask, axis=(1, 2))  # (J,) zeros par réalisation
    
    for j in range(J):
        if n_zeros[j] < 10:
            if verbose:
                print(f"  ⚠ Réalisation {j}: seulement {n_zeros[j]} zeros, ajustement...")
            
            new_threshold = np.percentile(magnitude_norm[j], threshold_percentile + 5)
            zeros_mask[j] = (magnitude_norm[j] < new_threshold)
            
            # Re-supprime les bords
            zeros_mask[j, 0, :] = False
            zeros_mask[j, -1, :] = False
            zeros_mask[j, :, 0] = False
            zeros_mask[j, :, -1] = False
            
            if verbose:
                print(f"     → {np.sum(zeros_mask[j])} zeros après ajustement")
    
    if verbose:
        print(f"   Seuils: {thresholds.flatten()}")
        print(f"   Zeros détectés: {n_zeros}")
    
    # Retour au format original si input 2D
    if is_single:
        return zeros_mask[0]
    
    return zeros_mask
  
# ============================================================================
# Generation of all the ZR's with J realizations
# ============================================================================

 
def generate_multiple_noise_realization_optimized(
    spec: np.ndarray,  # (1000, 60000)
    J: int = 50,
    beta: float = 1.0,
    batch_size: int = None,  # Auto-détection si None
    verbose: bool = True
) -> tuple:
    """
    Version optimisée pour grands spectrogrammes.
    Auto-détecte le batch_size optimal si non fourni.
    """
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZED NOISE REALIZATION GENERATOR")
        print("="*70)
    
    # Auto-détection du batch_size optimal
    if batch_size is None:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        bytes_per_real = spec.size * 16  # complex128
        mb_per_real = bytes_per_real / (1024**2)
        
        # Utiliser 35% de la RAM disponible pour être safe
        safe_ram_gb = available_ram_gb * 0.3
        batch_size = max(1, int((safe_ram_gb * 1024) / mb_per_real))
        
        if verbose:
            print(f"  Available RAM: {available_ram_gb:.1f} GB")
            print(f"  Memory per realization: {mb_per_real:.1f} MB")
            print(f"  Auto-selected batch_size: {batch_size}")
    
    # Estimation gamma
    magnitude = np.abs(spec)
    sorted_magnitude = np.sort(magnitude.flatten())
    gamma_0 = np.median(sorted_magnitude[:len(sorted_magnitude)//10])
    gamma_j = beta * gamma_0
    
    if verbose:
        print(f"\n  Spectrogram shape: {spec.shape}")
        print(f"  Total realizations (J): {J}")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of batches: {int(np.ceil(J / batch_size))}")
        print(f"  gamma_j: {gamma_j:.6f}")
    
    # Pré-allocation du résultat final (seulement les masques booléens = léger)
    all_zeros_masks = np.zeros((J, *spec.shape), dtype=bool)
    
    # Traitement par batch
    n_batches = int(np.ceil(J / batch_size))
    
    import time
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        batch_start_time = time.time()
        
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, J)
        current_batch_size = end_idx - start_idx
        
        if verbose:
            print(f"\n  Batch {batch_idx+1}/{n_batches} "
                  f"[realizations {start_idx}→{end_idx-1}]", end='')
        
        # 1. Générer le bruit (seulement pour ce batch)
        noise_real = np.random.normal(0, gamma_j, (current_batch_size, *spec.shape))
        noise_imag = np.random.normal(0, gamma_j, (current_batch_size, *spec.shape))
        noise = noise_real + 1j * noise_imag
        
        # Libération immédiate
        del noise_real, noise_imag
        
        # 2. Ajouter le bruit au spectrogramme original
        noisy_specs_batch = spec[np.newaxis, :, :] + noise
        del noise
        
        # 3. Détecter les zeros pour ce batch
        zeros_batch = detect_zeros(
            noisy_specs_batch,
            threshold_percentile=5,
            verbose=False
        )
        del noisy_specs_batch
        
        # 4. Stocker dans le résultat final
        all_zeros_masks[start_idx:end_idx] = zeros_batch
        del zeros_batch
        
        if verbose:
            batch_time = time.time() - batch_start_time
            print(f" ✓ ({batch_time:.2f}s)")
    
    total_time = time.time() - start_time
    
    if verbose:
        # Statistiques finales
        total_zeros = np.sum(all_zeros_masks)
        print(f"\n  {'─'*66}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average time per batch: {total_time/n_batches:.2f}s")
        print(f"  Total zeros detected: {total_zeros:,}")
        print(f"  Average zeros per realization: {total_zeros/J:.1f}")
        print(f"  Zeros density: {100*total_zeros/all_zeros_masks.size:.3f}%")
        print("="*70 + "\n")
    
    return all_zeros_masks, gamma_j

# ============================================================================
# 2D histogram of all the zeros (ZR's)
# ============================================================================

def build_2d_histogram(all_zeros: np.ndarray,
                       t: np.ndarray,
                       freqs: np.ndarray) -> np.ndarray:
    """Histogramme 2D G[n,m] des zeros sur toutes les réalisations."""
    
    print("\n" + "="*70)
    print("Histogram construction G[n,m]")
    print("="*70)
    
    G_histogram = np.sum(all_zeros, axis=0, dtype=np.int32)
    
    print(f'Max count: {G_histogram.max()} | '
          f'Mean: {G_histogram.mean():.1f} | '
          f'Non-zero bins: {np.count_nonzero(G_histogram)}')
    
    return G_histogram

# ============================================================================
# Assignment of the zeros to each cell of the grid generated 
# ============================================================================

def assign_zeros_to_grid(all_zeros: np.ndarray,
                         t_bounds: Tuple[float, float],
                         f_bounds: Tuple[float, float],
                         n_bins: Tuple[int, int],
                         return_indices: bool = False) -> np.ndarray:
    """
    Assigne chaque zero à une cellule de grille régulière.
    
    Args:
        all_zeros: Shape (J, F, T) masques booléens
        t_bounds: (t_min, t_max)
        f_bounds: (f_min, f_max)
        n_bins: (n_bins_t, n_bins_f)
        return_indices: Si True, retourne aussi les indices bruts
    
    Returns:
        grid_counts: Shape (n_bins_f, n_bins_t) - nombre de zeros par bin
        [optionnel] bin_indices: (t_idx, f_idx) de chaque zero
    """
    
    print("\n" + "="*70)
    print("Grid Assignment")
    print("="*70)
    
    # 1️⃣ Extraction des coordonnées de TOUS les zeros
    # all_zeros shape: (J, F, T) → où sont les True ?
    J, n_freqs, n_times = all_zeros.shape
    
    # np.where retourne (j_idx, f_idx, t_idx)
    j_coords, f_coords, t_coords = np.where(all_zeros)
    
    n_total_zeros = len(j_coords)
    print(f"Total zeros to assign: {n_total_zeros}")
    
    # 2️⃣ Conversion indices → valeurs physiques (temps/fréquence)
    t_min, t_max = t_bounds
    f_min, f_max = f_bounds
    n_bins_t, n_bins_f = n_bins
    
    # Mapping linéaire : idx → valeur réelle
    t_values = t_min + (t_max - t_min) * t_coords / (n_times - 1)
    f_values = f_min + (f_max - f_min) * f_coords / (n_freqs - 1)
    
    # 3️⃣ Bords de la grille d'assignment
    t_edges = np.linspace(t_min, t_max, n_bins_t + 1)
    f_edges = np.linspace(f_min, f_max, n_bins_f + 1)
    
    # 4️⃣ Assignment vectorisé
    t_bin_indices = np.digitize(t_values, t_edges) - 1
    f_bin_indices = np.digitize(f_values, f_edges) - 1
    
    # Clipping (au cas où des points tombent exactement sur les bords)
    t_bin_indices = np.clip(t_bin_indices, 0, n_bins_t - 1)
    f_bin_indices = np.clip(f_bin_indices, 0, n_bins_f - 1)
    
    # 5️⃣ Comptage par bin (histogramme 2D)
    grid_counts = np.zeros((n_bins_f, n_bins_t), dtype=np.int32)
    
    # 🚀 Méthode ultra-rapide : np.add.at
    np.add.at(grid_counts, (f_bin_indices, t_bin_indices), 1)
    
    print(f"Grid shape: {grid_counts.shape}")
    print(f"Max zeros in a bin: {grid_counts.max()}")
    print(f"Mean zeros per bin: {grid_counts.mean():.2f}")
    print(f"Empty bins: {np.sum(grid_counts == 0)}")
    
    if return_indices:
        return grid_counts, (t_bin_indices, f_bin_indices)
    
    return grid_counts


# ============================================================================
# Caculate the area of the zeros occupying each cell of the grid
# ============================================================================
def compute_bin_areas_optimized(
    all_zeros: np.ndarray,
    t_bounds: Tuple[float, float],
    f_bounds: Tuple[float, float],
    n_bins: Tuple[int, int],
    method: str = 'convex_hull',
    batch_size_j: int = 5,  # Traiter J réalisations par batch
    max_points_per_hull: int = 10000,  # Limite pour ConvexHull
    verbose: bool = True
) -> np.ndarray:
    """
    Version optimisée qui traite les réalisations par batch.
    
    Args:
        all_zeros: Shape (J, F, T) masques booléens
        t_bounds: (t_min, t_max)
        f_bounds: (f_min, f_max)
        n_bins: (n_bins_t, n_bins_f)
        method: 'convex_hull' ou 'pixel_count'
        batch_size_j: Nombre de réalisations traitées simultanément
        max_points_per_hull: Limite de points pour ConvexHull (sous-échantillonnage au-delà)
        verbose: Afficher les progrès
    
    Returns:
        areas: Shape (n_bins_f, n_bins_t) - aire normalisée [0,1] par bin
    """
    
    if verbose:
        print("\n" + "="*70)
        print(f"Computing bin areas (method: {method})")
        print("="*70)
    
    # Paramètres
    J, n_freqs, n_times = all_zeros.shape
    t_min, t_max = t_bounds
    f_min, f_max = f_bounds
    n_bins_t, n_bins_f = n_bins
    
    if verbose:
        print(f"  Input shape: {all_zeros.shape}")
        print(f"  Grid: {n_bins_f} × {n_bins_t} bins")
        print(f"  Processing in batches of {batch_size_j} realizations")
    
    # Pré-calcul des edges (une seule fois)
    t_edges = np.linspace(t_min, t_max, n_bins_t + 1)
    f_edges = np.linspace(f_min, f_max, n_bins_f + 1)
    
    bin_width_t = (t_max - t_min) / n_bins_t
    bin_width_f = (f_max - f_min) / n_bins_f
    bin_area_max = bin_width_t * bin_width_f
    
    # Accumulateurs pour les aires de tous les batchs
    areas_accumulator = np.zeros((n_bins_f, n_bins_t), dtype=np.float32)
    counts_accumulator = np.zeros((n_bins_f, n_bins_t), dtype=np.int32)
    
    # Traitement par batch de J
    n_batches = int(np.ceil(J / batch_size_j))
    
    import time
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        batch_start = time.time()
        
        start_j = batch_idx * batch_size_j
        end_j = min((batch_idx + 1) * batch_size_j, J)
        
        if verbose:
            print(f"\n  Batch {batch_idx+1}/{n_batches} [J={start_j}→{end_j-1}]", end='')
        
        # Extraire seulement ce batch
        batch_zeros = all_zeros[start_j:end_j]
        
        # 1. Extraction des coordonnées (seulement pour ce batch)
        j_coords, f_coords, t_coords = np.where(batch_zeros)
        n_zeros = len(j_coords)
        
        if n_zeros == 0:
            if verbose:
                print(" → No zeros, skipped")
            continue
        
        # 2. Conversion en coordonnées physiques
        t_values = t_min + (t_max - t_min) * t_coords / (n_times - 1)
        f_values = f_min + (f_max - f_min) * f_coords / (n_freqs - 1)
        
        # 3. Assignment aux bins (vectorisé)
        t_idx = np.clip(np.searchsorted(t_edges[1:], t_values), 0, n_bins_t - 1)
        f_idx = np.clip(np.searchsorted(f_edges[1:], f_values), 0, n_bins_f - 1)
        
        # 4. Regroupement efficace avec numpy
        # Créer des clés uniques pour chaque bin
        bin_keys = f_idx * n_bins_t + t_idx
        
        # Trier pour regrouper les points du même bin
        sort_indices = np.argsort(bin_keys)
        sorted_keys = bin_keys[sort_indices]
        sorted_t = t_values[sort_indices]
        sorted_f = f_values[sort_indices]
        
        # Trouver les frontières de chaque bin
        unique_keys, split_indices = np.unique(sorted_keys, return_index=True)
        split_indices = np.append(split_indices, len(sorted_keys))
        
        # 5. Calculer l'aire pour chaque bin non-vide
        n_bins_with_data = len(unique_keys)
        
        for i in range(n_bins_with_data):
            # Récupérer les indices du bin actuel
            start = split_indices[i]
            end = split_indices[i + 1]
            
            # Points de ce bin
            bin_t = sorted_t[start:end]
            bin_f = sorted_f[start:end]
            points = np.column_stack([bin_t, bin_f])
            
            # Retrouver les indices (f_bin, t_bin)
            bin_key = unique_keys[i]
            f_bin = int(bin_key // n_bins_t)
            t_bin = int(bin_key % n_bins_t)
            
            if len(points) < 3:
                continue
            
            # Calcul de l'aire selon la méthode
            if method == 'convex_hull':
                # Sous-échantillonnage si trop de points
                if len(points) > max_points_per_hull:
                    indices = np.random.choice(len(points), max_points_per_hull, replace=False)
                    points = points[indices]
                
                try:
                    hull = ConvexHull(points)
                    area = hull.volume  # En 2D = aire
                except:
                    area = 0.0
            
            elif method == 'pixel_count':
                # Estimation par densité
                unique_points = np.unique(points, axis=0)
                area = len(unique_points) * (bin_area_max / 100)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Normalisation
            normalized_area = min(area / bin_area_max, 1.0)
            
            # Accumulation (moyenne progressive)
            areas_accumulator[f_bin, t_bin] += normalized_area
            counts_accumulator[f_bin, t_bin] += 1
        
        if verbose:
            batch_time = time.time() - batch_start
            print(f" → {n_zeros:,} zeros, {n_bins_with_data} bins ({batch_time:.2f}s)")
    
    # 6. Moyenner les aires sur tous les batchs
    mask = counts_accumulator > 0
    areas = np.zeros_like(areas_accumulator)
    areas[mask] = areas_accumulator[mask] / counts_accumulator[mask]
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n  {'─'*66}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Max area ratio: {areas.max():.3f}")
        print(f"  Mean area ratio: {areas.mean():.3f}")
        print(f"  Bins with area > 0: {np.sum(areas > 0)}/{n_bins_f * n_bins_t}")
        print("="*70 + "\n")
    
    return areas



# ============================================================================
# Caculate the area of the zeros occupying each cell of the grid
# ============================================================================


def classify_bins(areas: np.ndarray,
                  threshold_noise: float = 0.7,
                  threshold_signal: float = 0.3) -> np.ndarray:
    """
    Classify each bin as a pure signal, pure noise or a mix between them
    
    Args:
        areas: Normalized grid [0, 1]
        threshold_noise: if area > seuil → "NN" (pur bruit)
        threshold_signal: if area < seuil → "SS" (pur signal)
        Entre les deux → "SN" (mixte)
    
    Returns:
        labels: String grid ('NN', 'SS', 'SN')
    """
    
    print("\n" + "="*70)
    print("Bin Classification")
    print(f"Thresholds: Signal < {threshold_signal}, Noise > {threshold_noise}")
    print("="*70)
    
    n_bins_f, n_bins_t = areas.shape
    
    # Initialisation avec dtype='U2' (Unicode strings de 2 chars)
    labels = np.empty((n_bins_f, n_bins_t), dtype='U2')
    
    # Classification vectorisée
    labels[areas >= threshold_noise] = 'NN'  # Beaucoup de zeros → bruit
    labels[areas <= threshold_signal] = 'SS'  # Peu de zeros → signal
    
    # Cas intermédiaires (sera écrasé si déjà assigné)
    mask_mixed = (areas > threshold_signal) & (areas < threshold_noise)
    labels[mask_mixed] = 'SN'
    
    # Statistiques
    n_nn = np.sum(labels == 'NN')
    n_ss = np.sum(labels == 'SS')
    n_sn = np.sum(labels == 'SN')
    total = n_bins_f * n_bins_t
    
    print(f"NN (noise):  {n_nn:5d} bins ({100*n_nn/total:5.1f}%)")
    print(f"SS (signal): {n_ss:5d} bins ({100*n_ss/total:5.1f}%)")
    print(f"SN (mixed):  {n_sn:5d} bins ({100*n_sn/total:5.1f}%)")
    print("="*70)
    
    return labels


def generate_grid_mask(labels: np.ndarray,
                       keep_classes: list = ['SS'],
                       output_shape: tuple = None) -> np.ndarray:
    """
    Génère un masque binaire à partir des labels de classification.
    
    Args:
        labels: Grille de classification (n_bins_f, n_bins_t) avec 'SS', 'SN', 'NN'
        keep_classes: Liste des classes à garder (ex: ['SS'] ou ['SS', 'SN'])
        output_shape: Si fourni (n_freqs, n_times), upscale le masque à cette résolution
                     Sinon, retourne le masque à la résolution de la grille
    
    Returns:
        mask: Booléen True = garder, False = rejeter
              Shape = output_shape si fourni, sinon labels.shape
    
    Examples:
        >>> # Masque grille (basse résolution)
        >>> mask = generate_grid_mask(labels, keep_classes=['SS'])
        
        >>> # Masque TF complet (haute résolution)
        >>> mask_tf = generate_grid_mask(labels, ['SS', 'SN'], 
        ...                               output_shape=(1025, 431))
    """
    
    print("\n" + "="*70)
    print("Grid Mask Generation")
    print(f"Keep classes: {keep_classes}")
    print("="*70)
    
    # 1️⃣ Création du masque binaire sur la grille
    mask_grid = np.zeros_like(labels, dtype=bool)
    
    for class_name in keep_classes:
        mask_grid |= (labels == class_name)  # OR logique
    
    n_bins_f, n_bins_t = labels.shape
    n_kept = np.sum(mask_grid)
    n_total = n_bins_f * n_bins_t
    
    print(f"Grid level: {n_kept}/{n_total} bins kept ({100*n_kept/n_total:.1f}%)")
    
    # 2️⃣ Upscaling si nécessaire (pour appliquer sur spectrogrammes)
    if output_shape is not None:
        n_freqs, n_times = output_shape
        
        # Répétition de chaque bin selon les dimensions du spectre
        freq_repeats = n_freqs // n_bins_f
        time_repeats = n_times // n_bins_t
        
        # Gestion des résidus (si division non exacte)
        freq_remainder = n_freqs % n_bins_f
        time_remainder = n_times % n_bins_t
        
        # Upscaling avec np.repeat
        mask_upscaled = np.repeat(mask_grid, freq_repeats, axis=0)
        mask_upscaled = np.repeat(mask_upscaled, time_repeats, axis=1)
        
        # Ajustement des dimensions finales si nécessaire
        if mask_upscaled.shape[0] < n_freqs:
            # Ajouter des lignes en répétant la dernière
            extra_rows = np.repeat(mask_grid[-1:, :], 
                                   n_freqs - mask_upscaled.shape[0], axis=0)
            extra_rows = np.repeat(extra_rows, time_repeats, axis=1)
            mask_upscaled = np.vstack([mask_upscaled, extra_rows])
        
        if mask_upscaled.shape[1] < n_times:
            # Ajouter des colonnes en répétant la dernière
            extra_cols = np.repeat(mask_upscaled[:, -1:], 
                                   n_times - mask_upscaled.shape[1], axis=1)
            mask_upscaled = np.hstack([mask_upscaled, extra_cols])
        
        # Crop si dépassement (sécurité)
        mask_upscaled = mask_upscaled[:n_freqs, :n_times]
        
        n_kept_pixels = np.sum(mask_upscaled)
        n_total_pixels = n_freqs * n_times
        print(f"TF level:   {n_kept_pixels}/{n_total_pixels} pixels kept "
              f"({100*n_kept_pixels/n_total_pixels:.1f}%)")
        print("="*70)
        
        return mask_upscaled
    
    print("="*70)
    return mask_grid


# Variantes spécialisées (shortcuts)
def generate_signal_mask(labels: np.ndarray, output_shape: tuple = None) -> np.ndarray:
    """Garde uniquement les bins de pur signal (SS)."""
    return generate_grid_mask(labels, keep_classes=['SS'], output_shape=output_shape)


def generate_mixed_mask(labels: np.ndarray, output_shape: tuple = None) -> np.ndarray:
    """Garde signal pur + mixte (SS + SN)."""
    return generate_grid_mask(labels, keep_classes=['SS', 'SN'], output_shape=output_shape)


def generate_all_signal_mask(labels: np.ndarray, output_shape: tuple = None) -> np.ndarray:
    """Garde tout sauf le pur bruit (SS + SN, exclut NN)."""
    return generate_grid_mask(labels, keep_classes=['SS', 'SN'], output_shape=output_shape)


# ============================================================================
# MAIN FUNCTION - Complete Pipeline Test
# ============================================================================

def main():
    """
    Complete pipeline for spectrogram denoising using grid-based zero classification.
    
    Pipeline:
    1. Generate synthetic noisy signal
    2. Compute spectrogram with superlets
    3. Add multiple Gaussian White Noise realizations
    4. Create grid and assign zeros to squares
    5. Compute convex hull areas in each square
    6. Classify squares as NN, SN, or SS
    7. Generate masks to remove NN cells
    8. Visualize results
    """
    
    print("\n" + "="*70)
    print("SPECTROGRAM DENOISING PIPELINE")
    print("Grid-based Zero Classification Method")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Signal Generation
    # ========================================================================
    N = 5000              # Number of samples
    fs = 10000            # Sampling frequency (Hz)
    snr_db = 5.0          # Signal-to-Noise Ratio (dB)
    
    t, noisy_signal, clean_signal = generate_synthetic_signal(N, fs, snr_db, times=2)
    
    # ========================================================================
    # STEP 2: Spectrogram Computation
    # ========================================================================
    frequencies = (1, 350, 200)      # (min, max, n_points)
    first_cycle = 5                  # Morlet wavelet cycles
    order_range = (3, 20)            # Superlet order range
    
    spec, freqs = compute_spectrogram_superlets(
        noisy_signal,
        fs,
        frequencies=frequencies,
        first_cycle=first_cycle,
        order_range=order_range
    )
    
    spec_amplitude = np.abs(spec)
    
    print(f"\n  Time axis: {len(t)} points ({t[-1]:.3f} s)")
    print(f"  Frequency axis: {len(freqs)} points ({freqs[0]:.1f}-{freqs[-1]:.1f} Hz)")
    print(f"  Spectrogram shape: {spec_amplitude.shape}")
    
    # ========================================================================
    # STEP 3: Multiple Noise Realizations
    # ========================================================================
    J = 100                          # Number of noise realizations
    noise_variance = 0.01            # Gaussian noise variance
    
    print("\n" + "="*70)
    print(f"Generating {J} Noise Realizations")
    print("="*70)
    
    # Stack original spectrogram J times
    noisy_specs = np.tile(spec_amplitude[None, :, :], (J, 1, 1))  # (J, F, T)
    
    # Add different noise to each realization
    for j in range(J):
        noise = add_gaussian_white_noise(spec_amplitude.shape, noise_variance)
        noisy_specs[j] += noise
    
    print(f"  Noisy spectrograms shape: {noisy_specs.shape}")
    print(f"  Noise variance: {noise_variance}")
    
    # ========================================================================
    # STEP 4: Zero Detection
    # ========================================================================
    threshold_percentile = 5  # Bottom 5% are considered zeros
    
    print("\n" + "="*70)
    print("Zero Detection (Batch Processing)")
    print("="*70)
    
    all_zeros = detect_zeros(
        noisy_specs,
        threshold_percentile=threshold_percentile,
    )
    
    print(f"  Total zeros detected: {np.sum(all_zeros)}")
    print(f"  Average zeros per realization: {np.sum(all_zeros) / J:.0f}")
    print(f"  Zeros density: {100 * np.sum(all_zeros) / all_zeros.size:.2f}%")
    
    # ========================================================================
    # STEP 5: Grid Creation & Area Computation
    # ========================================================================
    n_bins = (50, 100)  # (n_bins_frequency, n_bins_time)
    t_bounds = (t[0], t[-1])
    f_bounds = (freqs[0], freqs[-1])
    
    print("\n" + "="*70)
    print(f"Grid Definition: {n_bins[0]}×{n_bins[1]} bins")
    print("="*70)
    
    areas = compute_bin_areas_optimized(
        all_zeros=all_zeros,
        t_bounds=t_bounds,
        f_bounds=f_bounds,
        n_bins=n_bins,
        method='convex_hull'
    )
    
    # ========================================================================
    # STEP 6: Square Classification
    # ========================================================================
    threshold_noise = 0.7    # area > 0.7 → NN (pure noise)
    threshold_signal = 0.3   # area < 0.3 → SS (pure signal)
    
    labels = classify_bins(
        areas=areas,
        threshold_noise=threshold_noise,
        threshold_signal=threshold_signal
    )
    
    # ========================================================================
    # STEP 7: Mask Generation
    # ========================================================================
    print("\n" + "="*70)
    print("Mask Generation")
    print("="*70)
    
    # Mask 1: Keep only pure signal (SS)
    mask_ss = generate_signal_mask(
        labels,
        output_shape=spec_amplitude.shape
    )
    
    # Mask 2: Keep signal + mixed (SS + SN)
    mask_mixed = generate_mixed_mask(
        labels,
        output_shape=spec_amplitude.shape
    )
    
    # Apply masks
    spec_ss_only = spec_amplitude * mask_ss
    spec_mixed = spec_amplitude * mask_mixed
    
    # ========================================================================
    # STEP 8: Visualization
    # ========================================================================
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.3)
    
    # Row 1: Original signals
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, clean_signal, 'b-', linewidth=0.8, label='Clean signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Clean Signal (Ground Truth)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, noisy_signal, 'r-', linewidth=0.8, alpha=0.7, label=f'SNR = {snr_db} dB')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Noisy Signal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.7, f'Pipeline Parameters:', ha='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.55, f'Noise realizations: J = {J}', ha='center', fontsize=11)
    ax3.text(0.5, 0.45, f'Grid: {n_bins[0]}×{n_bins[1]} bins', ha='center', fontsize=11)
    ax3.text(0.5, 0.35, f'Threshold noise: {threshold_noise}', ha='center', fontsize=11)
    ax3.text(0.5, 0.25, f'Threshold signal: {threshold_signal}', ha='center', fontsize=11)
    ax3.text(0.5, 0.10, f'Zero percentile: {threshold_percentile}%', ha='center', fontsize=11)
    ax3.axis('off')
    
    # Row 2: Spectrograms
    extent = [t[0], t[-1], freqs[0], freqs[-1]]
    
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.imshow(spec_amplitude, aspect='auto', origin='lower', 
                     extent=extent, cmap='viridis', interpolation='nearest')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Original Noisy Spectrogram')
    plt.colorbar(im1, ax=ax4, label='Amplitude')
    
    ax5 = fig.add_subplot(gs[1, 1])
    # Visualize zeros from first realization
    zeros_viz = all_zeros[0].astype(float)
    zeros_viz[zeros_viz == 0] = np.nan  # Transparent non-zeros
    im2 = ax5.imshow(spec_amplitude, aspect='auto', origin='lower',
                     extent=extent, cmap='gray', alpha=0.5)
    im2b = ax5.imshow(zeros_viz, aspect='auto', origin='lower',
                      extent=extent, cmap='Reds', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_title(f'Detected Zeros (Realization 1/{J})')
    
    ax6 = fig.add_subplot(gs[1, 2])
    im3 = ax6.imshow(areas, aspect='auto', origin='lower', 
                     extent=extent, cmap='coolwarm', vmin=0, vmax=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_title('Normalized Convex Hull Areas')
    plt.colorbar(im3, ax=ax6, label='Area ratio')
    
    # Row 3: Classification & Masks
    ax7 = fig.add_subplot(gs[2, 0])
    # Create RGB image for labels
    label_rgb = np.zeros((*labels.shape, 3))
    label_rgb[labels == 'NN'] = [1, 0, 0]      # Red = Noise
    label_rgb[labels == 'SN'] = [1, 1, 0]      # Yellow = Mixed
    label_rgb[labels == 'SS'] = [0, 1, 0]      # Green = Signal
    
    im4 = ax7.imshow(label_rgb, aspect='auto', origin='lower', extent=extent)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Frequency (Hz)')
    ax7.set_title('Grid Classification (NN/SN/SS)')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='NN (Noise)'),
                      Patch(facecolor='yellow', label='SN (Mixed)'),
                      Patch(facecolor='green', label='SS (Signal)')]
    ax7.legend(handles=legend_elements, loc='upper right')
    
    ax8 = fig.add_subplot(gs[2, 1])
    im5 = ax8.imshow(mask_ss, aspect='auto', origin='lower',
                     extent=extent, cmap='Greens', vmin=0, vmax=1)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Frequency (Hz)')
    ax8.set_title('Mask SS Only (Pure Signal)')
    plt.colorbar(im5, ax=ax8, label='Keep (1) / Reject (0)')
    
    ax9 = fig.add_subplot(gs[2, 2])
    im6 = ax9.imshow(mask_mixed, aspect='auto', origin='lower',
                     extent=extent, cmap='Blues', vmin=0, vmax=1)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Frequency (Hz)')
    ax9.set_title('Mask SS+SN (Signal + Mixed)')
    plt.colorbar(im6, ax=ax9, label='Keep (1) / Reject (0)')
    
    # Row 4: Denoised spectrograms
    ax10 = fig.add_subplot(gs[3, 0])
    im7 = ax10.imshow(spec_amplitude, aspect='auto', origin='lower',
                      extent=extent, cmap='viridis', interpolation='nearest')
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Frequency (Hz)')
    ax10.set_title('Reference: Original Spectrogram')
    plt.colorbar(im7, ax=ax10, label='Amplitude')
    
    ax11 = fig.add_subplot(gs[3, 1])
    im8 = ax11.imshow(spec_ss_only, aspect='auto', origin='lower',
                      extent=extent, cmap='viridis', interpolation='nearest')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Frequency (Hz)')
    ax11.set_title('Denoised: SS Only (Conservative)')
    plt.colorbar(im8, ax=ax11, label='Amplitude')
    
    ax12 = fig.add_subplot(gs[3, 2])
    im9 = ax12.imshow(spec_mixed, aspect='auto', origin='lower',
                      extent=extent, cmap='viridis', interpolation='nearest')
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Frequency (Hz)')
    ax12.set_title('Denoised: SS+SN (Aggressive)')
    plt.colorbar(im9, ax=ax12, label='Amplitude')
    
    fig.suptitle('Grid-Based Spectrogram Denoising Pipeline', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('denoising_pipeline_results.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved: denoising_pipeline_results.png")
    
    plt.show()
    
    # ========================================================================
    # Performance Metrics
    # ========================================================================
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print(f"\nClassification Statistics:")
    print(f"  NN bins (pure noise):  {np.sum(labels == 'NN'):5d} "
          f"({100*np.mean(labels == 'NN'):5.1f}%)")
    print(f"  SN bins (mixed):       {np.sum(labels == 'SN'):5d} "
          f"({100*np.mean(labels == 'SN'):5.1f}%)")
    print(f"  SS bins (pure signal): {np.sum(labels == 'SS'):5d} "
          f"({100*np.mean(labels == 'SS'):5.1f}%)")
    
    print(f"\nMask Statistics:")
    print(f"  Pixels kept (SS only):    {100*np.mean(mask_ss):5.1f}%")
    print(f"  Pixels kept (SS+SN):      {100*np.mean(mask_mixed):5.1f}%")
    print(f"  Pixels removed (NN):      {100*(1-np.mean(mask_mixed)):5.1f}%")
    
    print(f"\nEnergy Preservation:")
    original_energy = np.sum(spec_amplitude**2)
    ss_energy = np.sum(spec_ss_only**2)
    mixed_energy = np.sum(spec_mixed**2)
    
    print(f"  Original energy:          {original_energy:.2e}")
    print(f"  SS only energy:           {ss_energy:.2e} "
          f"({100*ss_energy/original_energy:.1f}%)")
    print(f"  SS+SN energy:             {mixed_energy:.2e} "
          f"({100*mixed_energy/original_energy:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return {
        'noisy_signal': noisy_signal,
        'clean_signal': clean_signal,
        'spectrogram': spec_amplitude,
        'zeros': all_zeros,
        'areas': areas,
        'labels': labels,
        'mask_ss': mask_ss,
        'mask_mixed': mask_mixed,
        'denoised_ss': spec_ss_only,
        'denoised_mixed': spec_mixed,
        't': t,
        'freqs': freqs
    }


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = main()
    
    # Optional: Save results
    # np.savez('denoising_results.npz', **results)
    # print("Results saved to 'denoising_results.npz'")
