# capture_from_gui.py  (save as UTF-8)
import sys, time, os, importlib.util
from PIL import Image
from PyQt5.QtWidgets import QApplication

# === EDIT ONLY if your GUI path differs ===
gui_path = r"C:\Users\pedro\Downloads\Nucleo Atomic Vis.py"

# load module from file (handles spaces)
spec = importlib.util.spec_from_file_location('visualizer', gui_path)
visualizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualizer_module)
QuantumVisualizer = visualizer_module.QuantumVisualizer

def save_frame(visualizer, outpath, wait=0.25):
    QApplication.processEvents()
    time.sleep(wait)
    try:
        img = visualizer.view.grabFrameBuffer()
        img.save(outpath)
        print("Saved:", outpath)
        return True
    except Exception as e:
        print("Grab failed:", e)
        return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = QuantumVisualizer()

    # ======== TOP: quarks + nucleus (orbitals hidden) ========
    # hide orbitals
    for k, ms in list(getattr(v,'mesh_cache',{}).items()):
        for m in ms:
            try: m.setVisible(False)
            except: pass
    v.show_nucleus = True
    v.show_quarks = True
    v.draw_nucleus()
    save_frame(v, os.path.join(os.path.dirname(gui_path), "fig2_quarks.png"), wait=0.25)

    # ======== MIDDLE: nuclear label and small view ========
    # optionally hide quark labels for clarity
    for lbl in getattr(v,'quark_labels',[]): 
        try: lbl.setVisible(False)
        except: pass
    v.draw_nucleus()
    save_frame(v, os.path.join(os.path.dirname(gui_path), "fig2_nuclear.png"), wait=0.25)

    # ======== BOTTOM: orbitals (nucleus hidden) ========
    # show orbital meshes
    for k, ms in getattr(v,'mesh_cache',{}).items():
        for m in ms:
            try: m.setVisible(True)
            except: pass
    v.show_nucleus = False
    v.draw_nucleus()
    save_frame(v, os.path.join(os.path.dirname(gui_path), "fig2_orbitals.png"), wait=0.5)

    print("All screenshots saved in folder:", os.path.dirname(gui_path))
    # Keep app running briefly so GL can stay alive if needed
    time.sleep(0.5)
    # exit cleanly
    sys.exit(0)
