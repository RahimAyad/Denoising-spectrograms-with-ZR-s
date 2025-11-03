'''
DENOISING SPECTROGRAM
Based on: "Unsupervised Classification of the Spectrogram Zeros with 
an Application to Signal Detection and Denoising"
Juan M. Miramont et al., 2024

PIPELINE:
1. Generate synthetic noisy signal with a known signal to noise ration (SNR)
2. Compute spectrogram with superlets
3. Detect initial zeros (Z_0 or local minimas)
4. Construct Voronoi cells from Z_0
5. Add multiple Gaussian White Noise (J realizations) to get multiple ZR's
6. Assign each Z_n to its Voronoi cell
7. Compute convex hull area for zeros in each cell
8. Compare Voronoi cell areas with convex hull areas
9. Classify each cell as NN, SN, or SS (N=Noise, S=Signal)
10. Generate mask to remove NN cells
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull
from typing import Tuple, Optional, List, Dict
from superlet import superlets
from scipy.ndimage import minimum_filter
from matplotlib.patches import Polygon as MplPolygon  
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree

import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_synthetic_signal(N: int = 500,
                              fs: int = 10000,
                               snr_db : float = 2.0 ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
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

    y = x + noise

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


def detect_zeros(spectrogram: np.ndarray, 
                threshold_percentile:float = 5,
                verbose:bool = False) -> np.ndarray:
    """Detect local minima (approximate zeros) in spectrogram"""


    # calcul of the magnitude 
    # (cause even in the negative values there is informations)
    magnitude  = np.abs(spectrogram)

    # normalization of the magnitude
    magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    
    threshold = np.percentile(magnitude_norm, threshold_percentile)

    zeros_mask = np.zeros_like(magnitude, dtype=bool)

    for size in [5, 7 ,10]:
        local_min = minimum_filter(magnitude_norm, size=size)
        is_minimum = (magnitude_norm == local_min) & (magnitude_norm < threshold)
        zeros_mask |= is_minimum

    # Remove border zeros
    zeros_mask[0, :] = False
    zeros_mask[-1, :] = False
    zeros_mask[: ,0] = False
    zeros_mask[: ,-1] = False

    n_zeros = np.sum(zeros_mask)
    if verbose is not False:
        print(f"   Threshold: {threshold:.4f}, ({threshold_percentile}th percentile)")
        print(f"   Detected zeros (Z_0): {n_zeros}")
        
    if n_zeros < 10:
        print("  ⚠ WARNING: Very few zeros detected! ")
        print(f"Adjusting threshold from the {threshold_percentile}th percentile to the {threshold_percentile+5}th percentile")
        threshold = np.percentile(magnitude_norm, threshold_percentile+5)
        zeros_mask = (magnitude_norm < threshold)
        zeros_mask[0, :] = False
        zeros_mask[-1, :] = False
        zeros_mask[:, 0] = False
        zeros_mask[:, -1] = False
        print(f"   New zeros count: {np.sum(zeros_mask)}")
    
    return zeros_mask    
 
# ============================================================================
# Voronoi tesselation
# ============================================================================

def generate_Voronoi_tesselation(points:np.ndarray,
                                 t: np.ndarray,
                                 freqs: np.ndarray)-> Tuple[Optional[Voronoi], dict]:
    
    """Compute Voronoi diagram with coordinate normalization"""
    print("\n" + "="*70)
    print("Voronoi tesselation")
    print("="*70)
  
    if len(points) < 4:
        print("⚠ Insufficient points for Voronoi")
        return None, points, np.array([]), {}
    
    # Normalize coordinates
    t_min, t_max = t.min(), t.max()
    f_min, f_max = freqs.min(), freqs.max()

    points_norm = points.copy()
    points_norm[:, 0] = (points[:, 0] - t_min) / (t_max - t_min)
    points_norm[:, 1] = (points[:, 1] - f_min) / (f_max - f_min)

    vor = Voronoi(points_norm)

    # Normalization parameters
    scale_params = {
        't_min': t_min,
        't_max': t_max,
        'f_min': f_min,
        'f_max': f_max
    }
    print(f"   Voronoi cells: {len(vor.point_region)}")

    return  vor, scale_params


def plot_voronoi_on_spectrogram_simple(vor, scale_params, ax, **kwargs):
    """Version simplifiée : seulement les arêtes finies"""
    if vor is None:
        return
    
    line_colors = kwargs.get('line_colors', 'red')
    line_width = kwargs.get('line_width', 1.5)
    
    t_min = scale_params['t_min']
    t_max = scale_params['t_max']
    f_min = scale_params['f_min']
    f_max = scale_params['f_max']
    
    # Dénormaliser vertices
    vertices_denorm = vor.vertices.copy()
    vertices_denorm[:, 0] = vertices_denorm[:, 0] * (t_max - t_min) + t_min
    vertices_denorm[:, 1] = vertices_denorm[:, 1] * (f_max - f_min) + f_min
    
    # Plot arêtes finies uniquement
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vertices_denorm[simplex, 0], 
                   vertices_denorm[simplex, 1],
                   color=line_colors, 
                   linewidth=line_width,
                   zorder=10)
    
    # ✅ FORCER LES LIMITES
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(f_min, f_max)

# ============================================================================
# Noise realizations 
# ============================================================================

def generate_noise_realizations(spectrogram: np.ndarray,
                                J:int = 256,
                                beta: float = 1.0) -> Tuple[List[np.ndarray], float]:
    '''
    Generating J reallization of noise, 
    gamma_j = beta * gamma_0
    and gamma_0 is the bottom 10% of the power of the signal
    '''
    print("\n" + "="*70)
    print(f"{J} Noise Realizations Generation")
    print("="*70)

    # Estimation of the noise level gamma_0 from the spectrogram
    magnitude = np.abs(spectrogram)
    # Estimation for low magnitude region
    sorted_magnitude = np.sort(magnitude.flatten())
    gamma_0 = np.median(sorted_magnitude[:len(sorted_magnitude)//10]) # Bottom 10%

    gamma_j = beta * gamma_0
    
    print(f"   Estimated gamma_0: {gamma_0:.6f}")
    print(f"   Beta: {beta}")
    print(f"   gamma_j (noise std): {gamma_j:.6f}")
    print(f"   Number of realizations (J): {J}")
    
    all_zeros_masks = []

    for j in range (J):
        noise_real = np.random.normal(0, gamma_j, spectrogram.shape)
        noise_img = np.random.normal(0, gamma_j, spectrogram.shape)
        noise = noise_real + (1j)*noise_img

        noisy_spec = spectrogram + noise
        zeros_j = detect_zeros(noisy_spec,threshold_percentile=5)
        all_zeros_masks.append(zeros_j)

    total_zeros = sum(np.sum(mask) for mask in all_zeros_masks)
    print(f"Total zeros across all realization: {total_zeros}")
    print(f"Average zeros per realization:  {total_zeros/J:.1f}")
    
    return all_zeros_masks, gamma_j

# ============================================================================
# 2D histogram of all the zeros (ZR's)
# ============================================================================

def build_2d_histogram (all_zeros: List[np.ndarray],
                        t: np.ndarray,
                        freqs: np.ndarray) -> np.ndarray:
    
    """
    Build 2d histogram G[n,m]
    Count the zeros falling in each TF bin
    """
    print("\n", "="*70)
    print("Histogram construction G[n,m]")
    print("="*70)

    # Initializing a histogram with the same shape than the spectrogram
    G_histogram = np.zeros(((len(freqs)),len(t)),dtype=int)

    # Count zeros from all realization 
    for zeros in all_zeros:
        G_histogram += zeros.astype(int)

    print (f'Max count in a bin: {G_histogram.max()} ')
   
    return G_histogram

# ============================================================================
# Assignment of the zeros to each Voronoi cell with KDTREE
# ============================================================================

def assign_zeros_to_voronoi_cells(Z_0_points: np.ndarray,
                                  all_zrs: np.ndarray,
                                  scale_params: dict) -> Tuple[dict, dict]:
    
    """
    Assign all ZR's to their nearest Vornoi cell using KDTree, it's a technique 
    of subdivising space to find the points that are the closest to a specifique point, 
    it's not perfect because it's can have some divergences with the voronoi, but it 
    should not be too disturbing.

      Parameters:
    -----------
    Z_0_points : np.ndarray
        Original Z_0 points (sites of Voronoi)
    all_zrs : np.ndarray
        All zeros from J realizations (shape: [N, 2])
    scale_params : dict
        Normalization parameters from compute_voronoi_diagram
        
    Returns:
    --------
    cell_assignments : dict
        {cell_index: array of ZR points in this cell}
    convex_hulls : dict
        {cell_index: ConvexHull object or None}
    """
    print("\n" + "="*70)
    print("Assignment of ZRs to Voronoi Cells (KDTree)")
    print("="*70)

    # Normalization of the ZR's with the parameter of normalization of the Z_0

    t_min = scale_params['t_min']
    t_max = scale_params['t_max']
    f_min = scale_params['f_min']
    f_max = scale_params['f_max']

    all_zrs_norm = all_zrs.copy()
    all_zrs_norm[:, 0] = (all_zrs[:, 0] - t_min)/(t_max-t_min)
    all_zrs_norm[:, 1] = (all_zrs[:, 1] - f_min)/(f_max-f_min)

    # Normalization of Z_0 
    Z_0_norm = Z_0_points.copy()
    Z_0_norm[:, 0] = (Z_0_points[:, 0] - t_min)/(t_max-t_min)
    Z_0_norm[:, 1] = (Z_0_points[:, 1] - f_min)/(f_max-f_min)


    print(f"   Total ZRs to assign: {len(all_zrs)}")
    print(f"   Voronoi cells (Z_0 sites): {len(Z_0_norm)}")

    # Construction of a KDTree
    tree = cKDTree(Z_0_norm)
    
    # Find the closest cell to each ZR's
    # distances : distance to the closest cell
    # cell_indices : index of the Voronoi cell
    distances , cell_indices = tree.query(all_zrs_norm)

    # Group the ZR's by the corresponding cell
    cell_assignement = {}

    for cell_index in range(len(Z_0_points)):
        mask = (cell_indices == cell_index)
        zrs_in_cell = all_zrs[mask]
        cell_assignement[cell_index] = zrs_in_cell

    return cell_assignement, cell_indices


# ============================================================================
# Voronoi areas whith shoelace formula
# ============================================================================

def compute_voronoi_areas_shoelace_formula(vor: Voronoi, 
                               scale_params: dict) -> np.ndarray:
    """
    Compute area of each Voronoi cell (denormalized coordinates)
    with shoelace formula (beautifull formula)
    Returns:
    --------
    areas : np.ndarray
        Area of each Voronoi cell (same length as Z_0_points)
    """
    print("\n" + "="*70)
    print(" Voronoi cell areas computation")
    print("="*70)
    
    n_cells = len(vor.point_region)
    voronoi_areas = np.zeros(n_cells)

    # Denormalization
    t_scale = scale_params['t_max'] - scale_params['t_min'] 
    f_scale = scale_params['f_max'] - scale_params['f_min'] 
    area_scale = t_scale*f_scale # You can basically imagine the TF space as a square
    # if you multiply the two edges you get the factor by what the area will scale 
    
    n_valid = 0
    n_infinite = 0
    
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]

        # First deal with infinite regions
        if -1 in region or len(region) == 0 :
            voronoi_areas[i] = np.inf
            n_infinite+=1
            continue

        vertices = vor.vertices[region]

        # Deal also with voronois cells that dont even form a triangle
        if len(vertices) < 3:
            voronoi_areas[i] = 0
            continue
    
        try: 
            # Compute areas using Shoelace formula (faster than Convexhull)
            x = vertices[:, 0]
            y = vertices[:, 1]
            area_norm = 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

            # Denormalization of the area
            voronoi_areas[i] = area_norm * area_scale
            n_valid+=1

        except Exception as e:
            voronoi_areas[i] = 0

    print(f"   Valid finite cells: {n_valid}")
    print(f"   Infinite cells: {n_infinite}")

    return voronoi_areas

# ============================================================================
# convex hull computation
# ============================================================================

def convex_hull(cell_assignments: dict) -> Tuple[dict, np.ndarray]:
    """
    Compute the convex hull for the points present in each Voronoi cell

    This function is made for the ZR's, in this case we dont use the shoelace formula
    because we dont have a minimal number of points forming a polygon but a number of 
    points that we need to look for the minimal polygon regrouping them.
    
    Returns:
    --------
    convex_hulls : dict
        {cell_index: ConvexHull object or None}
    hull_areas : np.ndarray
        Array of convex hull areas for each cell

    """
    print("\n" + "="*70)
    print("*Convex Hull computation of the ZR's in each cell")
    print("="*70)
    
    convex_hulls = {}
    hull_areas = np.zeros(len(cell_assignments))

    successful_hulls = 0
    failed_hulls = 0
    not_enouth_points_hulls = 0
    for cell_idx, zrs_in_cell in cell_assignments.items():
        if len(zrs_in_cell)< 3:
            # Need's at least 3 points for a convex hull
            convex_hulls[cell_idx] = None
            hull_areas[cell_idx] = 0
            not_enouth_points_hulls+=1
            continue

        try:
            hull =  ConvexHull(zrs_in_cell)
            convex_hulls[cell_idx] = hull
            hull_areas[cell_idx] = hull.volume # in 2D the volume is the area
            successful_hulls+=1
        
        except Exception as e:
            convex_hulls[cell_idx] = None
            hull_areas[cell_idx] = 0
            failed_hulls+=1  

    print(f"  Successful hulls: {successful_hulls}/{len(cell_assignments)}")
    print(f"  failed hulls: {failed_hulls}/{len(cell_assignments)}")
    print(f"  not enough points hulls: {not_enouth_points_hulls}/{len(cell_assignments)}")

    return convex_hulls, hull_areas

# ============================================================================
# Classification of the  NN / SN / SS
# ============================================================================
# NN for (Noise-Noise), SN (Signal-Noise), SS (Signal-Signal)

def classification_Vornoi_cells(voronoi_areas: np.ndarray,
                                hull_areas: np.ndarray,
                                threshold_SS_ratio:float = 0.2,
                                threshold_NN_ratio:float= 0.8) -> np.ndarray:
    
    """
    Classify each Voronoi cell as:
    - SS (Signal-Signal): hull_area / voronoi_area < threshold_SS
    - SN (Signal-Noise): intermediate
    - NN (Noise-Noise): hull_area / voronoi_area ≥ threshold_NN 
    
    Parameters:
    -----------
    voronoi_areas : np.ndarray
        Area of each Voronoi cell
    hull_areas : np.ndarray
        Area of convex hull in each cell
    threshold_SS_ratio : float 
        Threshold of the ration to consider that the cell is pure signal
    threshold_NN_ratio : float 
        Threshold of the ration to consider that the cell is noise signal
    Returns:
    --------
    labels : np.ndarray
        Array of labels: 0=NN, 1=SN, 2=SS
    """
    print("\n" + "="*70)
    print("Cell classification (NN / SN / SS)")
    print("="*70)
    
    n_cells = len(voronoi_areas)
    labels = np.zeros(n_cells, dtype=int)  # 0=NN, 1=SN, 2=SS
    
    # Compute ratios
    ratios = np.zeros(n_cells)
    
    for i in range(n_cells):
        # Skip infinite Voronoi cells  classify as NN
        if not np.isfinite(voronoi_areas[i]) or voronoi_areas[i] == 0:
            labels[i] = 0  # NN
            ratios[i] = 1
            continue
        
        # Skip cells without convex hull
        if hull_areas[i] == 0:
            labels[i] = 2  # SS
            ratios[i] = 0
            continue
        
        # Compute ratio
        ratio = hull_areas[i] / (voronoi_areas[i])
        ratios[i] = ratio

        # Classification logic
        if ratio < threshold_SS_ratio:
            labels[i] = 2  # SS
        elif ratio > threshold_NN_ratio:
            labels[i] = 0  # NN
        else:
            labels[i] = 1  # SN (intermediate)
    
    # Statistics
    n_NN = np.sum(labels == 0)
    n_SN = np.sum(labels == 1)
    n_SS = np.sum(labels == 2)
    
    print(f"  Threshold ratio SS, NN: {threshold_SS_ratio,threshold_NN_ratio}")
    print(f"\n Classification Results:")
    print(f"  NN (Noise-Noise):    {n_NN:4d} ({n_NN/n_cells*100:.1f}%)")
    print(f"  SN (Signal-Noise):   {n_SN:4d} ({n_SN/n_cells*100:.1f}%)")
    print(f"  SS (Signal-Signal):  {n_SS:4d} ({n_SS/n_cells*100:.1f}%)")
    

    return labels

# ============================================================================
# Visualization of the Voronoi cells classification (IA generated cause i didn't manage to do it)
# ============================================================================

def plot_voronoi_classification(vor, Z_0_points, labels, scale_params, ax, 
                                title="Voronoi Classification"):
    """
    Plot Voronoi cells colored by their classification (NN/SN/SS)
    """

    t_min = scale_params['t_min']
    t_max = scale_params['t_max']
    f_min = scale_params['f_min']
    f_max = scale_params['f_max']
    
    # Dénormaliser les vertices
    vertices_denorm = vor.vertices.copy()
    vertices_denorm[:, 0] = vertices_denorm[:, 0] * (t_max - t_min) + t_min
    vertices_denorm[:, 1] = vertices_denorm[:, 1] * (f_max - f_min) + f_min
    
    # Couleurs pour chaque classe
    colors_map = {
        0: '#FF4444',  # NN - Rouge
        1: '#FFA500',  # SN - Orange
        2: '#44FF44'   # SS - Vert
    }
    
    labels_map = {
        0: 'NN (Noise-Noise)',
        1: 'SN (Signal-Noise)',
        2: 'SS (Signal-Signal)'
    }
    
    patches = []
    colors = []
    
    # Pour chaque cellule de Voronoi
    for point_idx, region_idx in enumerate(vor.point_region):
        if point_idx >= len(labels):
            continue
            
        region = vor.regions[region_idx]
        
        # Skip infinite regions ou régions vides
        if not region or -1 in region:
            continue
        
        # Créer le polygone de la cellule
        polygon_vertices = vertices_denorm[region]
        polygon = MplPolygon(polygon_vertices, closed=True)
        patches.append(polygon)
        colors.append(colors_map[labels[point_idx]])
    
    # Créer la collection de patches
    pc = PatchCollection(patches, facecolors=colors, alpha=0.6, 
                        edgecolors='black', linewidths=0.5)
    ax.add_collection(pc)
    
    # Plot Z_0 sites
    ax.scatter(Z_0_points[:, 0], Z_0_points[:, 1], 
              c='white', s=4, marker='o', 
              edgecolors='black', linewidths=1.5,
              zorder=10, label='Z₀ sites')
    
    # Légende personnalisée
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_map[0], alpha=0.6, edgecolor='black', label=labels_map[0]),
        Patch(facecolor=colors_map[1], alpha=0.6, edgecolor='black', label=labels_map[1]),
        Patch(facecolor=colors_map[2], alpha=0.6, edgecolor='black', label=labels_map[2])
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(f_min, f_max)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
