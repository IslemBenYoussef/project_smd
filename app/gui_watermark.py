"""
Interface graphique simple pour le système de tatouage
Version compatible Docker (détection et fallback)
"""

import sys
import os

def is_running_in_docker():
    """Détecte si on est dans un conteneur Docker"""
    return os.path.exists('/.dockerenv')

def check_gui_availability():
    """Vérifie si l'interface graphique est disponible"""
    if sys.platform == 'linux':
        # Vérifier DISPLAY
        if not os.environ.get('DISPLAY'):
            return False, "Variable DISPLAY non définie"
        
        # Vérifier /tmp/.X11-unix
        if not os.path.exists('/tmp/.X11-unix'):
            return False, "Dossier /tmp/.X11-unix non accessible"
    
    return True, "OK"

def main_with_gui():
    """Lance l'interface graphique"""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import cv2
        import numpy as np
        from watermark_qim import WatermarkQIM, Attacks, Evaluator
        
        # ... Le reste du code GUI ...
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Pour l'interface graphique, installez tkinter:")
        print("  Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  Windows/Mac: Réinstallez Python avec l'option Tk")
        return False
    
    return True

def main():
    """Point d'entrée principal"""
    print("="*50)
    print("🎨 Interface Graphique - Tatouage QIM")
    print("="*50)
    
    # Détection Docker
    if is_running_in_docker():
        print("⚠️  Détection: Exécution dans Docker")
        gui_available, msg = check_gui_availability()
        
        if not gui_available:
            print(f"❌ GUI non disponible: {msg}")
            print("\nSolutions:")
            print("  1. Utilisez la version CLI: docker run watermark-qim --demo")
            print("  2. Exécutez en local sans Docker: python gui_watermark.py")
            print("  3. Configurez X11 forwarding (Linux):")
            print("     xhost +local:docker")
            print("     docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix watermark-qim python gui_watermark.py")
            sys.exit(1)
    
    # Lancer l'interface
    print("✅ Lancement de l'interface graphique...")
    success = main_with_gui()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()