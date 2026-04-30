"""
Mini-Projet : Sécurisation des images 2D par tatouage numérique basé sur QIM
Auteur   : L2-IRS 25/26
Description :
    Implémentation d'un système de tatouage numérique robuste utilisant :
    - Transformation DCT 2D par blocs
    - Insertion/Extraction par Quantification d'Indice (QIM)
    - Simulation d'attaques (bruit gaussien, compression JPEG)
    - Métriques : PSNR, BER
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage


# ─────────────────────────────────────────────
#  PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────
BLOCK_SIZE     = 8        # Taille des blocs DCT (8×8 comme JPEG)
STEP           = 30       # Pas de quantification QIM (Δ)
SECRET_SEED    = 42       # Clé secrète pour la sélection pseudo-aléatoire
WATERMARK_BITS = 64       # Longueur du watermark binaire (bits)


# ─────────────────────────────────────────────
#  1. UTILITAIRES IMAGE
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Charge une image en niveaux de gris, normalisée en float64."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    return img.astype(np.float64)


def save_image(img: np.ndarray, path: str) -> None:
    """Sauvegarde une image float64 [0,255] sous forme uint8."""
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Calcule le PSNR entre l'image originale et l'image tatouée."""
    o = np.clip(original, 0, 255).astype(np.uint8)
    m = np.clip(modified, 0, 255).astype(np.uint8)
    return psnr_skimage(o, m, data_range=255)


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """Calcule le Bit Error Rate (BER) entre deux séquences binaires."""
    errors = np.sum(original_bits != extracted_bits)
    return errors / len(original_bits)


# ─────────────────────────────────────────────
#  2. DCT PAR BLOCS
# ─────────────────────────────────────────────

def apply_dct_blocks(img: np.ndarray) -> np.ndarray:
    """Applique la DCT 2D sur des blocs de taille BLOCK_SIZE×BLOCK_SIZE."""
    H, W = img.shape
    dct_img = np.zeros_like(img)
    for r in range(0, H - BLOCK_SIZE + 1, BLOCK_SIZE):
        for c in range(0, W - BLOCK_SIZE + 1, BLOCK_SIZE):
            block = img[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
            dct_img[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    return dct_img


def apply_idct_blocks(dct_img: np.ndarray) -> np.ndarray:
    """Applique l'IDCT 2D par blocs pour reconstruire l'image."""
    H, W = dct_img.shape
    img = np.zeros_like(dct_img)
    for r in range(0, H - BLOCK_SIZE + 1, BLOCK_SIZE):
        for c in range(0, W - BLOCK_SIZE + 1, BLOCK_SIZE):
            block = dct_img[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
            img[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = idct(idct(block, axis=1, norm='ortho'), axis=0, norm='ortho')
    return img


# ─────────────────────────────────────────────
#  3. SÉLECTION DES COEFFICIENTS (clé secrète)
# ─────────────────────────────────────────────

def select_coefficients(dct_img: np.ndarray, n_bits: int, seed: int = SECRET_SEED):
    """
    Sélectionne n_bits coefficients DCT de fréquence moyenne de manière
    pseudo-aléatoire via une clé secrète.
    Fréquences moyennes : évite DC (0,0) et les hautes fréquences (dernière ligne/colonne).
    """
    H, W = dct_img.shape
    n_blocks_r = H // BLOCK_SIZE
    n_blocks_c = W // BLOCK_SIZE

    # Indices de fréquence moyenne dans un bloc 8×8 (diagonale 2 à 5)
    freq_candidates = [(r, c) for r in range(2, 6) for c in range(2, 6)]

    rng = np.random.default_rng(seed)
    all_positions = []
    for br in range(n_blocks_r):
        for bc in range(n_blocks_c):
            freq = freq_candidates[rng.integers(0, len(freq_candidates))]
            row = br * BLOCK_SIZE + freq[0]
            col = bc * BLOCK_SIZE + freq[1]
            all_positions.append((row, col))

    # Sélection pseudo-aléatoire de n_bits positions parmi toutes
    rng2 = np.random.default_rng(seed + 1)
    chosen = rng2.choice(len(all_positions), size=n_bits, replace=False)
    return [all_positions[i] for i in sorted(chosen)]


# ─────────────────────────────────────────────
#  4. GÉNÉRATION DU WATERMARK
# ─────────────────────────────────────────────

def generate_watermark(n_bits: int, seed: int = SECRET_SEED) -> np.ndarray:
    """Génère un watermark binaire pseudo-aléatoire."""
    rng = np.random.default_rng(seed + 999)
    return rng.integers(0, 2, size=n_bits).astype(int)


# ─────────────────────────────────────────────
#  5. INSERTION PAR QIM
# ─────────────────────────────────────────────

def qim_embed(value: float, bit: int, step: float = STEP) -> float:
    """
    QIM (Quantization Index Modulation) :
    Quantifie 'value' selon deux treillis décalés (pair/impair).
        bit=0 → treillis 0 : arrondi le plus proche de (2k)*Δ/2
        bit=1 → treillis 1 : arrondi le plus proche de (2k+1)*Δ/2
    """
    half = step / 2.0
    if bit == 0:
        return step * np.round(value / step)
    else:
        return step * np.round((value - half) / step) + half


def embed_watermark(img: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    """
    Insère le watermark dans les coefficients DCT sélectionnés via QIM.
    Retourne l'image tatouée.
    """
    dct_img   = apply_dct_blocks(img)
    positions = select_coefficients(dct_img, len(watermark))

    for idx, (r, c) in enumerate(positions):
        dct_img[r, c] = qim_embed(dct_img[r, c], watermark[idx])

    watermarked = apply_idct_blocks(dct_img)
    return watermarked


# ─────────────────────────────────────────────
#  6. EXTRACTION PAR QIM
# ─────────────────────────────────────────────

def qim_detect(value: float, step: float = STEP) -> int:
    """
    Décode le bit caché dans 'value' en comparant la distance aux deux treillis.
    """
    half = step / 2.0
    q0 = step * np.round(value / step)            # treillis bit=0
    q1 = step * np.round((value - half) / step) + half  # treillis bit=1
    return 0 if abs(value - q0) <= abs(value - q1) else 1


def extract_watermark(watermarked_img: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Extrait le watermark depuis une image (potentiellement attaquée).
    """
    dct_img   = apply_dct_blocks(watermarked_img)
    positions = select_coefficients(dct_img, n_bits)
    bits = np.array([qim_detect(dct_img[r, c]) for r, c in positions], dtype=int)
    return bits


# ─────────────────────────────────────────────
#  7. ATTAQUES
# ─────────────────────────────────────────────

def attack_gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Ajoute un bruit gaussien centré (mean=0, std=sigma)."""
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)


def attack_jpeg_compression(img: np.ndarray, quality: int = 50) -> np.ndarray:
    """Simule une compression JPEG puis décompression (en mémoire, compatible Windows)."""
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8), encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded.astype(np.float64)


# ─────────────────────────────────────────────
#  8. AFFICHAGE & RAPPORT
# ─────────────────────────────────────────────

def show_results(original, watermarked, attacked_gauss, attacked_jpeg,
                 wm_original, wm_extracted_clean, wm_extracted_gauss, wm_extracted_jpeg,
                 save_path="rapport_watermarking.png"):
    """Génère une figure synthétique de tous les résultats."""

    psnr_wm    = compute_psnr(original, watermarked)
    psnr_gauss = compute_psnr(original, attacked_gauss)
    psnr_jpeg  = compute_psnr(original, attacked_jpeg)

    ber_clean = compute_ber(wm_original, wm_extracted_clean)
    ber_gauss = compute_ber(wm_original, wm_extracted_gauss)
    ber_jpeg  = compute_ber(wm_original, wm_extracted_jpeg)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0D1117')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    def imshow(ax, image, title, subtitle="", cmap='gray'):
        ax.imshow(np.clip(image, 0, 255).astype(np.uint8), cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)
        if subtitle:
            ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                    ha='center', color='#AAAAAA', fontsize=9)
        ax.axis('off')
        ax.set_facecolor('#0D1117')

    # Row 0 — Images
    ax0 = fig.add_subplot(gs[0, 0]); imshow(ax0, original,       "Image originale")
    ax1 = fig.add_subplot(gs[0, 1]); imshow(ax1, watermarked,    "Image tatouée", f"PSNR = {psnr_wm:.2f} dB")
    ax2 = fig.add_subplot(gs[0, 2]); imshow(ax2, attacked_gauss, "Attaque bruit gaussien", f"PSNR = {psnr_gauss:.2f} dB")
    ax3 = fig.add_subplot(gs[0, 3]); imshow(ax3, attacked_jpeg,  "Attaque compression JPEG", f"PSNR = {psnr_jpeg:.2f} dB")

    # Row 1 — Difference maps
    diff_wm   = np.abs(watermarked - original) * 5
    diff_gauss = np.abs(attacked_gauss - original) * 2
    diff_jpeg  = np.abs(attacked_jpeg - original) * 2

    ax4 = fig.add_subplot(gs[1, 0]); imshow(ax4, np.zeros_like(original), "Différence originale", "(référence nulle)", cmap='hot')
    ax5 = fig.add_subplot(gs[1, 1]); imshow(ax5, diff_wm,    "Différence tatouage ×5",   "", cmap='hot')
    ax6 = fig.add_subplot(gs[1, 2]); imshow(ax6, diff_gauss, "Différence bruit ×2",      "", cmap='hot')
    ax7 = fig.add_subplot(gs[1, 3]); imshow(ax7, diff_jpeg,  "Différence JPEG ×2",       "", cmap='hot')

    # Row 2 — BER bars + watermark bits comparison
    ax8 = fig.add_subplot(gs[2, 0:2])
    labels  = ['Sans attaque', 'Bruit gaussien', 'JPEG Q50']
    ber_vals = [ber_clean, ber_gauss, ber_jpeg]
    colors  = ['#2ECC71', '#E74C3C', '#E67E22']
    bars = ax8.bar(labels, ber_vals, color=colors, edgecolor='white', linewidth=0.8)
    ax8.set_ylim(0, 1)
    ax8.set_ylabel('BER', color='white', fontsize=11)
    ax8.set_title('Bit Error Rate (BER) par scénario', color='white', fontsize=12, fontweight='bold')
    ax8.set_facecolor('#161B22')
    ax8.tick_params(colors='white')
    ax8.spines[:].set_color('#333333')
    for bar, v in zip(bars, ber_vals):
        ax8.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha='center', color='white', fontsize=10, fontweight='bold')

    # Watermark comparison (first 32 bits)
    ax9 = fig.add_subplot(gs[2, 2:4])
    show_bits = 32
    x = np.arange(show_bits)
    ax9.step(x, wm_original[:show_bits],        where='mid', color='#3498DB', lw=2, label='Watermark original')
    ax9.step(x, wm_extracted_clean[:show_bits],  where='mid', color='#2ECC71', lw=1.5, linestyle='--', label='Extrait (sans attaque)')
    ax9.step(x, wm_extracted_jpeg[:show_bits],   where='mid', color='#E67E22', lw=1.5, linestyle=':', label='Extrait (JPEG)')
    ax9.set_facecolor('#161B22')
    ax9.set_ylim(-0.2, 1.4)
    ax9.set_xlabel('Index du bit', color='white', fontsize=10)
    ax9.set_title('Comparaison watermark (32 premiers bits)', color='white', fontsize=12, fontweight='bold')
    ax9.tick_params(colors='white')
    ax9.spines[:].set_color('#333333')
    ax9.legend(facecolor='#0D1117', labelcolor='white', fontsize=9)

    # Global title
    fig.suptitle("Système de Tatouage Numérique QIM–DCT  |  L2-IRS 25/26",
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[✓] Rapport enregistré : {save_path}")
    return psnr_wm, psnr_gauss, psnr_jpeg, ber_clean, ber_gauss, ber_jpeg


# ─────────────────────────────────────────────
#  9. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_pipeline(image_path: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Système de Tatouage Numérique QIM–DCT")
    print("  L2-IRS — Mini-Projet 25/26")
    print("=" * 60)

    # 1. Chargement image
    print(f"\n[1] Chargement de l'image : {image_path}")
    host = load_image(image_path)
    H, W = host.shape
    print(f"    Dimensions : {W}×{H} px")

    # 2. Génération watermark
    print(f"\n[2] Génération du watermark ({WATERMARK_BITS} bits, seed={SECRET_SEED})")
    watermark = generate_watermark(WATERMARK_BITS)
    print(f"    Watermark : {watermark[:16]}... (affichage 16 premiers bits)")

    # 3. Tatouage
    print(f"\n[3] Insertion par QIM (Δ={STEP}, DCT 8×8, fréquences moyennes)")
    watermarked = embed_watermark(host, watermark)
    save_image(watermarked, os.path.join(output_dir, "image_tatouee.png"))
    psnr_wm = compute_psnr(host, watermarked)
    print(f"    PSNR image tatouée : {psnr_wm:.2f} dB")

    # 4. Extraction sans attaque
    print(f"\n[4] Extraction (sans attaque)")
    extracted_clean = extract_watermark(watermarked, WATERMARK_BITS)
    ber_clean = compute_ber(watermark, extracted_clean)
    print(f"    BER : {ber_clean:.4f}  ({'✓ Parfait' if ber_clean == 0 else '✗ Erreurs'})")

    # 5. Attaque bruit gaussien
    print(f"\n[5] Attaque : Bruit gaussien (σ=10)")
    attacked_gauss = attack_gaussian_noise(watermarked, sigma=10)
    save_image(attacked_gauss, os.path.join(output_dir, "attaque_bruit.png"))
    extracted_gauss = extract_watermark(attacked_gauss, WATERMARK_BITS)
    ber_gauss = compute_ber(watermark, extracted_gauss)
    psnr_gauss = compute_psnr(host, attacked_gauss)
    print(f"    PSNR : {psnr_gauss:.2f} dB  |  BER : {ber_gauss:.4f}")

    # 6. Attaque compression JPEG
    print(f"\n[6] Attaque : Compression JPEG (qualité=50)")
    attacked_jpeg = attack_jpeg_compression(watermarked, quality=50)
    save_image(attacked_jpeg, os.path.join(output_dir, "attaque_jpeg.png"))
    extracted_jpeg = extract_watermark(attacked_jpeg, WATERMARK_BITS)
    ber_jpeg = compute_ber(watermark, extracted_jpeg)
    psnr_jpeg = compute_psnr(host, attacked_jpeg)
    print(f"    PSNR : {psnr_jpeg:.2f} dB  |  BER : {ber_jpeg:.4f}")

    # 7. Rapport visuel
    print(f"\n[7] Génération du rapport visuel...")
    rapport_path = os.path.join(output_dir, "rapport_watermarking.png")
    show_results(host, watermarked, attacked_gauss, attacked_jpeg,
                 watermark, extracted_clean, extracted_gauss, extracted_jpeg,
                 save_path=rapport_path)

    # 8. Résumé
    print("\n" + "=" * 60)
    print("  RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)
    print(f"  {'Scénario':<25} {'PSNR (dB)':>10}  {'BER':>8}")
    print(f"  {'-'*45}")
    print(f"  {'Image tatouée':<25} {psnr_wm:>10.2f}  {ber_clean:>8.4f}")
    print(f"  {'Bruit gaussien (σ=10)':<25} {psnr_gauss:>10.2f}  {ber_gauss:>8.4f}")
    print(f"  {'JPEG Q=50':<25} {psnr_jpeg:>10.2f}  {ber_jpeg:>8.4f}")
    print("=" * 60)
    print(f"\n  Fichiers générés dans : {os.path.abspath(output_dir)}/")
    print("    - image_tatouee.png")
    print("    - attaque_bruit.png")
    print("    - attaque_jpeg.png")
    print("    - rapport_watermarking.png")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Usage : python watermark_qim.py <image_path> [output_dir]
    if len(sys.argv) < 2:
        print("Usage : python watermark_qim.py <chemin_image> [dossier_sortie]")
        print("Exemple : python watermark_qim.py lena.png resultats/")
        sys.exit(1)

    image_path  = sys.argv[1]
    output_dir  = sys.argv[2] if len(sys.argv) > 2 else "resultats"
    run_pipeline(image_path, output_dir)