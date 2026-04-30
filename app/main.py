#!/usr/bin/env python3
"""
Point d'entrée principal pour l'application Dockerisée
"""

import sys
import argparse
import cv2
import numpy as np
from watermark_qim import WatermarkQIM, Attacks, Evaluator
import os

def print_banner():
    """Affiche la bannière du projet"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║     🎓 Tatouage Numérique par QIM - Version Docker       ║
    ║     Sécurisation des images 2D                           ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_demo():
    """Exécute la démonstration complète"""
    print("\n🚀 Lancement de la démonstration complète...")
    
    # Créer une image de test
    print("📝 Création d'une image de test...")
    img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    # Initialiser le système
    print("🔧 Initialisation du système...")
    system = WatermarkQIM(delta=40.0, seed=42)
    
    # Générer watermark
    watermark = system.generate_watermark(1024)
    print(f"✅ Watermark généré: {len(watermark)} bits")
    
    # Insérer
    print("💉 Insertion du watermark...")
    watermarked, positions, _ = system.embed_watermark(img, watermark)
    
    # Extraire
    extracted = system.extract_watermark(watermarked, len(watermark), positions)
    ber = Evaluator.calculate_ber(watermark, extracted)
    
    print(f"\n📊 Résultats:")
    print(f"   ✅ BER: {ber:.4f} ({ber*100:.2f}%)")
    print(f"   ✅ Précision: {(1-ber)*100:.2f}%")
    
    psnr = Evaluator.calculate_psnr(img, watermarked)
    print(f"   📈 PSNR: {psnr:.2f} dB")
    
    # Test JPEG
    print(f"\n🛡️ Test de robustesse JPEG...")
    attacked = Attacks.jpeg_compression(watermarked, 50)
    extracted_jpeg = system.extract_watermark(attacked, len(watermark), positions)
    ber_jpeg = Evaluator.calculate_ber(watermark, extracted_jpeg)
    print(f"   JPEG Q=50 → BER: {ber_jpeg:.4f} ({(1-ber_jpeg)*100:.1f}% précision)")
    
    print("\n✨ Démonstration terminée avec succès!")

def process_image(input_path, output_path, delta=40.0):
    """Traite une image spécifique"""
    print(f"\n📁 Chargement de l'image: {input_path}")
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Erreur: Impossible de charger {input_path}")
        return False
    
    system = WatermarkQIM(delta=delta, seed=42)
    watermark = system.generate_watermark(min(1024, (img.shape[0]//8)*(img.shape[1]//8)))
    
    print(f"💉 Insertion du watermark ({len(watermark)} bits)...")
    watermarked, positions, _ = system.embed_watermark(img, watermark)
    
    if output_path:
        cv2.imwrite(output_path, watermarked)
        print(f"💾 Image tatouée sauvegardée: {output_path}")
    
    # Vérification
    extracted = system.extract_watermark(watermarked, len(watermark), positions)
    ber = Evaluator.calculate_ber(watermark, extracted)
    psnr = Evaluator.calculate_psnr(img, watermarked)
    
    print(f"\n📊 Résultats:")
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   BER: {ber:.4f}")
    print(f"   Précision: {(1-ber)*100:.2f}%")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Système de tatouage numérique basé sur QIM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Exécute la démonstration')
    parser.add_argument('--input', '-i', type=str, 
                       help='Chemin de l\'image d\'entrée')
    parser.add_argument('--output', '-o', type=str, 
                       help='Chemin de l\'image de sortie')
    parser.add_argument('--delta', type=float, default=40.0,
                       help='Pas de quantification (défaut: 40.0)')
    parser.add_argument('--extract', action='store_true',
                       help='Mode extraction uniquement')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.demo or (not args.input and not args.extract):
        run_demo()
    elif args.input:
        process_image(args.input, args.output, args.delta)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()