"""
Manual Tracking Annotator
=========================
Permet d'annoter manuellement les track IDs sur les frames du test set RADIal.

Contrôles :
  [0-9]      : saisir un ID numérique
  [Entrée]   : confirmer l'ID saisi pour la box courante
  [s]        : skip (auto-ID, incrémental)
  [b]        : revenir à la box précédente
  [n]        : passer au frame suivant sans annoter les boxes restantes
  [q]        : quitter et sauvegarder

Couleurs :
  Bleu       : box en attente d'annotation
  Vert       : box déjà annotée dans ce frame
  Rouge      : box courante (à annoter)

Usage :
  python manual_tracking_annotator.py \
      --annotations /path/to/annotations.csv \
      --images_root /path/to/RADIal/raw_sequences \
      --output     /path/to/gt_tracks.csv \
      --split_file /path/to/test_split.txt     # optionnel, filtre les séquences

Format CSV d'entrée attendu (RADIal) :
  numSample, frame_id, x1_pix, y1_pix, x2_pix, y2_pix, [autres colonnes...]

Format CSV de sortie :
  numSample, frame_id, x1_pix, y1_pix, x2_pix, y2_pix, track_id
"""

import os
import sys
import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
#  Couleurs BGR
# ─────────────────────────────────────────────
COL_PENDING  = (200, 200,   0)   # bleu-vert : en attente
COL_ACTIVE   = (  0,   0, 255)   # rouge     : box active
COL_DONE     = (  0, 200,   0)   # vert      : déjà annotée
COL_TEXT_BG  = (  0,   0,   0)
COL_TEXT_FG  = (255, 255, 255)


# Cache pour éviter de réouvrir le fichier à chaque appel
db_cache: dict = {}
sys.path.append('/home/skouff/')
sys.path.append('/home/skouff/master_thesis/')
from RADIal.DBReader import SyncReader


def load_image(images_root: str, seq_id: str, frame_idx: int) -> np.ndarray | None:
    if seq_id not in db_cache:
        db_cache[seq_id] = SyncReader(os.path.join(images_root, seq_id), tolerance=20000, silent=True)
        
    data_reader = db_cache[seq_id].GetSensorData(frame_idx)
    image = data_reader['camera']['data']
    return image

def RA_to_cartesian_box(data):
    L = 4 # length of the bounding box
    W = 1.8 # width of the bounding box 

    boxes = []
    for i in range(len(data)):

        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L])

    return boxes

def draw_frame(img: np.ndarray, boxes: list[dict], active_idx: int, typed: str) -> np.ndarray:
    """
    boxes : [{'x1','y1','x2','y2', 'track_id': int|None}, ...]
    active_idx : index de la box en cours d'annotation
    typed : chaîne en cours de saisie au clavier
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

        if i == active_idx:
            color = COL_ACTIVE
            thickness = 3
        elif box['track_id'] is not None:
            color = COL_DONE
            thickness = 2
        else:
            color = COL_PENDING
            thickness = 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label au-dessus de la box
        if box['track_id'] is not None:
            label = f"ID {box['track_id']}"
        elif i == active_idx:
            label = f"? {typed}_"
        else:
            label = "?"

        lx, ly = x1, max(y1 - 6, 14)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (lx, ly - th - 2), (lx + tw + 2, ly + 2), COL_TEXT_BG, -1)
        cv2.putText(vis, label, (lx + 1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT_FG, 1, cv2.LINE_AA)

    # HUD en bas
    n_done   = sum(1 for b in boxes if b['track_id'] is not None)
    hud_text = (f"Box {active_idx+1}/{len(boxes)}  |  "
                f"Annotees: {n_done}/{len(boxes)}  |  "
                f"Saisie: [{typed}]  |  "
                f"Entree=OK  s=skip  b=retour  n=frame suivant  q=quitter")
    cv2.rectangle(vis, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(vis, hud_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)

    return vis

SEQUENCES = {'Test':[
                'RECORD@2020-11-22_12.45.05',
                'RECORD@2020-11-22_12.25.47',
                'RECORD@2020-11-22_12.03.47',
                'RECORD@2020-11-22_12.54.38']}

def annotate(annotations_path: str,
             images_root: str,
             output_path: str,
             display_scale: float = 0.6):

    # ── Charger les annotations ──────────────────────────────────────────
    df = pd.read_csv(annotations_path)
    required = {'dataset', 'index', 'x1_pix', 'y1_pix', 'x2_pix', 'y2_pix', 'radar_R_m','radar_A_deg'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    # ── Filtrer sur le test set si fourni ───────────────────────────────
    df = df[df['dataset'].astype(str).isin(SEQUENCES['Test'])]
    print(f"[split] {len(SEQUENCES['Test'])} séquences test → {len(df)} annotations")

    # ── Charger résultats existants si reprise ───────────────────────────
    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path)
        done_keys = set(zip(done_df['dataset'].astype(str),
                            done_df['index'].astype(str)))
        print(f"[reprise] {len(done_df)} annotations déjà enregistrées")
    else:
        done_df   = pd.DataFrame()
        done_keys = set()

    results = done_df.to_dict('records') if len(done_df) > 0 else []

    # ── Itérer séquence → frame ──────────────────────────────────────────
    cv2.namedWindow("Annotateur", cv2.WINDOW_NORMAL)

    seq_groups = list(df.groupby('dataset'))
    total_seqs = len(seq_groups)

    for seq_i, (seq_id, seq_group) in enumerate(seq_groups):
        seq_id_str = str(seq_id)
        frame_groups = list(seq_group.groupby('index'))
        total_frames = len(frame_groups)

        print(f"\n── Séquence {seq_i+1}/{total_seqs} : {seq_id_str} "
              f"({total_frames} frames annotés) ──")

        frame_i = 0
        quit_all = False

        while frame_i < total_frames:
            frame_idx, group = frame_groups[frame_i]
            key_str = (seq_id_str, str(frame_idx))

            # Déjà annoté → skip silencieux
            if key_str in done_keys:
                frame_i += 1
                continue

            # Charger l'image
            img = load_image(images_root, seq_id_str, int(frame_idx))
            if img is None:
                print(f"  [warn] image introuvable : seq={seq_id_str} frame={frame_idx}")
                frame_i += 1
                continue

            # Redimensionner pour l'affichage
            dh = int(img.shape[0] * display_scale)
            dw = int(img.shape[1] * display_scale)

            # Construire la liste des boxes de ce frame
            boxes = []
            for _, row in group.iterrows():
                boxes.append({
                    'x1': int(row['x1_pix'] * display_scale),
                    'y1': int(row['y1_pix'] * display_scale),
                    'x2': int(row['x2_pix'] * display_scale),
                    'y2': int(row['y2_pix'] * display_scale),
                    'x1_orig': int(row['x1_pix']),
                    'y1_orig': int(row['y1_pix']),
                    'x2_orig': int(row['x2_pix']),
                    'y2_orig': int(row['y2_pix']),
                    'track_id': None,
                    '_row': row,
                })

            img_display = cv2.resize(img, (dw, dh))
            active_idx = 0
            typed = ""
            next_frame = False

            # ── Boucle d'annotation des boxes du frame ───────────────
            while active_idx < len(boxes):
                vis = draw_frame(img_display, boxes, active_idx, typed)
                title = f"Seq {seq_id_str} | Frame {frame_idx} | Box {active_idx+1}/{len(boxes)}"
                cv2.setWindowTitle("Annotateur", title)
                cv2.imshow("Annotateur", vis)

                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    quit_all = True
                    break

                elif key == ord('n'):
                    # Frame suivant sans annoter les boxes restantes
                    next_frame = True
                    break

                elif key == ord('b'):
                    # Revenir à la box précédente
                    if active_idx > 0:
                        active_idx -= 1
                        boxes[active_idx]['track_id'] = None
                        typed = ""
                    else:
                        # Revenir au frame précédent
                        frame_i = max(0, frame_i - 1)
                        # Supprimer les résultats de ce frame si déjà sauvés
                        results = [r for r in results
                                   if not (str(r['dataset']) == seq_id_str
                                           and str(r['index']) == str(frame_idx))]
                        break

                elif key == ord('s'):
                    # Skip → auto-ID (None dans le résultat, sera comblé par assign_gt_track_ids)
                    boxes[active_idx]['track_id'] = -1   # -1 = skipped
                    typed = ""
                    active_idx += 1

                elif key in (13, 10):   # Entrée
                    if typed:
                        boxes[active_idx]['track_id'] = int(typed)
                        typed = ""
                        active_idx += 1
                    # Si typed vide, rien

                elif key in range(48, 58):   # touches 0-9
                    typed += chr(key)

                elif key == 8:   # Backspace
                    typed = typed[:-1]

            if quit_all:
                break

            if next_frame:
                frame_i += 1
                continue

            # ── Sauvegarder les boxes annotées de ce frame ──────────
            all_annotated = all(b['track_id'] is not None for b in boxes)
            if all_annotated:
                for box in boxes:
                    row = box['_row']
                    record = row.to_dict()
                    record['track_id'] = box['track_id']

                    cart = RA_to_cartesian_box([[row['radar_R_m'], row['radar_A_deg']]])[0]
                    # cart = [x1,y1, x2,y2, x3,y3, x4,y4] — 4 coins
                    corners = np.array(cart).reshape(4, 2)
                    cx, cy = corners.mean(axis=0)
                    record['cx'] = round(float(cx), 3)
                    record['cy'] = round(float(cy), 3)

                    results.append(record)

                # Sauvegarde incrémentale
                out_df = pd.DataFrame(results)
                out_df.to_csv(output_path, index=False)
                done_keys.add(key_str)

                print(f"  frame {frame_idx} : {len(boxes)} boxes annotées → sauvegardé")
                frame_i += 1

        if quit_all:
            print("\n[quit] Sauvegarde finale...")
            break
    
    cv2.destroyAllWindows()

    # Sauvegarde finale
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        print(f"\n✓ Annotations sauvegardées : {output_path} ({len(out_df)} lignes)")
    else:
        print("\n[info] Aucune annotation à sauvegarder.")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotateur tracking manuel RADIal')
    parser.add_argument('--annotations', required=True,
                        help='CSV des annotations RADIal (avec x1_pix, y1_pix...)')
    parser.add_argument('--images_root', required=True,
                        help='Dossier racine des séquences RADIal')
    parser.add_argument('--output', default='gt_tracks_manual.csv',
                        help='CSV de sortie avec track_id (défaut: gt_tracks_manual.csv)')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='Facteur de redimensionnement affichage (défaut: 0.6)')
    args = parser.parse_args()

    annotate(
        annotations_path=args.annotations,
        images_root=args.images_root,
        output_path=args.output,
        display_scale=args.scale,
    )