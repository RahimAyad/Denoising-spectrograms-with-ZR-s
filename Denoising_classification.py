from Denoising_ZR_GMM_copy import generate_synthetic_signal, add_gaussian_white_noise, compute_spectrogram_superlets, detect_zeros, generate_Voronoi_tesselation, plot_voronoi_on_spectrogram_simple, generate_noise_realizations, build_2d_histogram, assign_zeros_to_voronoi_cells, compute_voronoi_areas_shoelace_formula, convex_hull, classification_Vornoi_cells, plot_voronoi_classification
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon  


if __name__=="__main__":

    fs = 10000

    t, noisy_signal, x = generate_synthetic_signal(N=500, fs=fs, snr_db=0.5)

    spec, freq = compute_spectrogram_superlets(
        signal=noisy_signal,
        sampling_frequency=fs,
        frequencies=(1, 350, 300),
        first_cycle=5,
        order_range=(3, 20)
    )

    noise = add_gaussian_white_noise(spec_shape=spec.shape, variance=0.1)
    spec_with_Z_0 = spec + noise

    # 1. Détection Z_0
    Z_0 = detect_zeros(spec_with_Z_0, threshold_percentile=5)
    y_Z_0, x_Z_0 = np.where(Z_0)
    Z_0_points = np.column_stack([t[x_Z_0], freq[y_Z_0]])

    # 2. Voronoi sur Z_0
    vor, scale_params = generate_Voronoi_tesselation(
        Z_0_points, t, freq
    )

    # 3. Générer J réalisations
    J = 256
    all_zeros, gamma_j = generate_noise_realizations(spec, J=J, beta=50)

    # 4. Histogramme
    G = build_2d_histogram(all_zeros, t, freq)

    # ✅ 5. FUSIONNER TOUS LES MASQUES
    print("\n" + "="*70)
    print("Extraction of all ZR coordinates")
    print("="*70)
    
    all_ZR_mask = np.zeros_like(all_zeros[0], dtype=bool)
    for mask in all_zeros:
        all_ZR_mask |= mask
    
    y_ZR, x_ZR = np.where(all_ZR_mask)
    # ✅ 5. FUSIONNER TOUS LES MASQUES
    print("\n" + "="*70)
    print("Extraction of all ZR coordinates")
    print("="*70)

    all_ZR_mask = np.zeros_like(all_zeros[0], dtype=bool)
    for mask in all_zeros:
        all_ZR_mask |= mask

    y_ZR, x_ZR = np.where(all_ZR_mask)
    all_zrs = np.column_stack([t[x_ZR], freq[y_ZR]])

    print(f"  → Total unique zero positions: {len(all_zrs)}")
    print(f"  → Mean per realization: {len(all_zrs)/J:.1f}")

    # ✅ 6. ASSIGNER LES ZRs AUX CELLULES VORONOI (KDTREE)
    cell_assignments, cell_indices = assign_zeros_to_voronoi_cells( Z_0_points=Z_0_points,
                                                                    all_zrs=all_zrs,
                                                                    scale_params=scale_params)

    # ✅ 7. CALCULER LES CONVEX HULLS
    convex_hulls, hull_areas = convex_hull(cell_assignments)
    voronoi_areas_fast = compute_voronoi_areas_shoelace_formula(vor, 
                                                                scale_params)

    lables = classification_Vornoi_cells(voronoi_areas_fast,
                                         hull_areas,
                                         threshold_SS_ratio = 0.2,
                                         threshold_NN_ratio = 0.7)

# ============================================================================
# PLOTS
# ============================================================================
    
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(5, 3, hspace=0.8, wspace=0.4)

# ROW 0: Signal
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, x, 'b-', label="Original signal", linewidth=1.5)
ax1.plot(t, noisy_signal, 'r-', label="Noisy signal", alpha=0.7)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Input Signals')

# ROW 1: Spectrograms
ax2 = fig.add_subplot(gs[1, 0])
im2 = ax2.imshow(np.abs(spec), aspect='auto', origin='lower',
                 extent=[t[0], t[-1], freq[0], freq[-1]],
                 cmap='jet')
ax2.set_title('Superlet Transform')
ax2.set_ylabel('Frequency (Hz)')
plt.colorbar(im2, ax=ax2)

ax3 = fig.add_subplot(gs[1, 1])
im3 = ax3.imshow(np.abs(spec), aspect='auto', origin='lower', 
                 extent=[t[0], t[-1], freq[0], freq[-1]], 
                 cmap='jet', alpha=1, interpolation='bilinear')
ax3.scatter(t[x_Z_0], freq[y_Z_0], c='white', s=0.4, alpha=1)
ax3.set_ylabel('Frequency (Hz)')
ax3.set_title(fr'Initial Zeros $Z_0$ ({len(Z_0_points)} points)')
plt.colorbar(im3, ax=ax3)

ax4 = fig.add_subplot(gs[1, 2])
im4 = ax4.imshow(G, aspect='auto', origin='lower',
                 extent=[t[0], t[-1], freq[0], freq[-1]],
                 cmap='grey')
ax4.set_title(fr'ZRs Histogram ($\gamma$ = {gamma_j:.4f}, J={J})')
ax4.set_ylabel('Frequency (Hz)')
plt.colorbar(im4, ax=ax4)

# ✅ ROW 2: CLASSIFICATION DES CELLULES VORONOI
ax5 = fig.add_subplot(gs[2:, 0])
plot_voronoi_classification(vor, Z_0_points, lables, scale_params, ax5,
                            title="Voronoi Cells Classification (NN/SN/SS)")

# ROW 3-4: Detailed Voronoi + Convex Hulls
ax6 = fig.add_subplot(gs[2:, 1:])

# Background: histogram G
im6 = ax6.imshow(G, aspect='auto', origin='lower',
                 extent=[t[0], t[-1], freq[0], freq[-1]],
                 cmap='grey', alpha=0.5)

# Plot Voronoi edges
plot_voronoi_on_spectrogram_simple(vor, scale_params, ax6, 
                                   line_colors='cyan', line_width=1)

# Plot Z_0 sites with classification colors
colors_for_scatter = ['red' if l == 0 else 'orange' if l == 1 else 'lime' 
                      for l in lables]
ax6.scatter(Z_0_points[:, 0], Z_0_points[:, 1], 
           c=colors_for_scatter, s=4, marker='o', 
           label=f'Z₀ sites ({len(Z_0_points)})', 
           zorder=5, edgecolors='white', linewidths=1)

# Plot ALL ZRs
ax6.scatter(all_zrs[:, 0], all_zrs[:, 1],
           c='white', s=1, alpha=0.3, 
           label=f'All ZRs ({len(all_zrs)})')

# Plot convex hulls
n_hulls_plotted = 0
for cell_idx, hull in convex_hulls.items():
    if hull is not None:
        points = cell_assignments[cell_idx]
        hull_points = points[hull.vertices]
        
        # Color based on classification
        edge_color = 'red' if lables[cell_idx] == 0 else \
                     'orange' if lables[cell_idx] == 1 else 'lime'
        
        # ✅ UTILISER MplPolygon au lieu de Polygon
        polygon = MplPolygon(hull_points, 
                        fill=False, 
                        edgecolor=edge_color, 
                        linewidth=2,
                        linestyle='--',
                        alpha=0.8,
                        zorder=10)
        ax6.add_patch(polygon)
        n_hulls_plotted += 1

ax6.set_title(f'Voronoi Diagram + Convex Hulls ({n_hulls_plotted} hulls plotted)')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Frequency (Hz)')
ax6.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.2)
plt.colorbar(im6, ax=ax6)

plt.suptitle('Spectrogram Zeros Classification Pipeline', 
             fontsize=16, fontweight='bold', y=0.995)

plt.show()