"""
Tests avancés pour analyser l'influence des paramètres
Version compatible Docker
"""

import numpy as np
import cv2
import matplotlib
# Forcer le mode non-interactif pour Docker
matplotlib.use('Agg')  # ⚠️ Doit être AVANT l'import de pyplot
import matplotlib.pyplot as plt
from watermark_qim import WatermarkQIM, Attacks, Evaluator
import os

def test_delta_parameter():
    """Test l'influence du paramètre delta"""
    print("\n" + "="*60)
    print("📊 Test d'influence du paramètre Δ (pas de quantification)")
    print("="*60)
    
    # Créer une image de test
    img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    delta_values = [10, 20, 30, 40, 50, 60, 80, 100]
    results = []
    
    for delta in delta_values:
        system = WatermarkQIM(delta=delta, seed=42)
        watermark = system.generate_watermark(1024)
        
        # Insertion
        watermarked, positions, _ = system.embed_watermark(img, watermark)
        
        # Qualité
        psnr_val = Evaluator.calculate_psnr(img, watermarked)
        
        # Robustesse JPEG
        attacked = Attacks.jpeg_compression(watermarked, 50)
        extracted = system.extract_watermark(attacked, len(watermark), positions)
        ber = Evaluator.calculate_ber(watermark, extracted)
        
        results.append({
            'delta': delta,
            'psnr': psnr_val,
            'ber': ber,
            'accuracy': 1 - ber
        })
        
        print(f"Δ = {delta:3d} → PSNR: {psnr_val:.2f} dB, Robustesse: {(1-ber)*100:.1f}%")
    
    # Sauvegarder les graphiques au lieu de les afficher
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    deltas = [r['delta'] for r in results]
    psnrs = [r['psnr'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    ax1.plot(deltas, psnrs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Δ (pas de quantification)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Qualité de l\'image tatouée')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(deltas, accuracies, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Δ (pas de quantification)')
    ax2.set_ylabel('Précision après JPEG')
    ax2.set_title('Robustesse face à JPEG (Q=50)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde dans un dossier accessible
    os.makedirs('/app/results', exist_ok=True)
    plt.savefig('/app/results/delta_analysis.png', dpi=150)
    print("📈 Graphique sauvegardé: /app/results/delta_analysis.png")
    plt.close()  # Important pour libérer la mémoire
    
    return results

def test_capacity():
    """Test la capacité maximale d'insertion"""
    print("\n" + "="*60)
    print("📊 Test de capacité d'insertion")
    print("="*60)
    
    img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    system = WatermarkQIM(delta=40, seed=42)
    
    capacities = [256, 512, 1024, 2048, 4096]
    results = []
    
    for cap in capacities:
        watermark = system.generate_watermark(cap)
        watermarked, positions, _ = system.embed_watermark(img, watermark)
        
        extracted = system.extract_watermark(watermarked, cap, positions)
        ber = Evaluator.calculate_ber(watermark, extracted)
        psnr_val = Evaluator.calculate_psnr(img, watermarked)
        
        results.append({'capacity': cap, 'psnr': psnr_val, 'ber': ber})
        print(f"Capacité: {cap:4d} bits → PSNR: {psnr_val:.2f} dB, BER: {ber:.4f}")
    
    # Sauvegarder le graphique
    plt.figure(figsize=(10, 4))
    caps = [r['capacity'] for r in results]
    psnrs = [r['psnr'] for r in results]
    
    plt.plot(caps, psnrs, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Capacité (bits)')
    plt.ylabel('PSNR (dB)')
    plt.title('Qualité en fonction de la capacité d\'insertion')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('/app/results/capacity_analysis.png', dpi=150)
    print("📈 Graphique sauvegardé: /app/results/capacity_analysis.png")
    plt.close()
    
    return results

def test_robustness_comparison():
    """Compare la robustesse pour différentes attaques"""
    print("\n" + "="*60)
    print("📊 Comparaison complète de robustesse")
    print("="*60)
    
    # Créer une image de test simple
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (412, 412), 255, -1)
    cv2.circle(img, (256, 256), 150, 128, -1)
    
    system = WatermarkQIM(delta=45, seed=42)
    watermark = system.generate_watermark(1024)
    
    # Tests
    attacks = {
        'JPEG 90': (Attacks.jpeg_compression, {'quality': 90}),
        'JPEG 70': (Attacks.jpeg_compression, {'quality': 70}),
        'JPEG 50': (Attacks.jpeg_compression, {'quality': 50}),
        'JPEG 30': (Attacks.jpeg_compression, {'quality': 30}),
        'Bruit σ=5': (Attacks.gaussian_noise, {'sigma': 5}),
        'Bruit σ=10': (Attacks.gaussian_noise, {'sigma': 10}),
        'Bruit σ=15': (Attacks.gaussian_noise, {'sigma': 15}),
    }
    
    results = []
    for attack_name, (attack_func, kwargs) in attacks.items():
        result = Evaluator.evaluate_robustness(
            system, img, watermark, attack_name, attack_func, **kwargs
        )
        results.append(result)
        print(f"{attack_name:15} → Précision: {result['accuracy']*100:.1f}%")
    
    # Graphique
    names = [r['attack'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accuracies, color='steelblue', alpha=0.7)
    plt.ylabel('Précision')
    plt.title('Robustesse face aux différentes attaques')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc*100:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/app/results/robustness_comparison.png', dpi=150)
    print("📈 Graphique sauvegardé: /app/results/robustness_comparison.png")
    plt.close()
    
    return results

def run_all_tests():
    """Exécute tous les tests"""
    print("🧪 DÉBUT DES TESTS - Version Docker")
    print("="*60)
    
    delta_results = test_delta_parameter()
    capacity_results = test_capacity()
    robustness_results = test_robustness_comparison()
    
    print("\n" + "="*60)
    print("✅ RÉSUMÉ DES TESTS")
    print("="*60)
    print(f"📊 Delta optimal: Δ = {delta_results[3]['delta']} (PSNR: {delta_results[3]['psnr']:.1f} dB)")
    print(f"📊 Capacité recommandée: 1024 bits")
    print(f"📊 Robustesse moyenne: {np.mean([r['accuracy'] for r in robustness_results])*100:.1f}%")
    print("\n💾 Tous les graphiques sont sauvegardés dans /app/results/")
    
if __name__ == "__main__":
    run_all_tests()