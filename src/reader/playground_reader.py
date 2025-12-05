import os
import pickle
import yaml
import logging
import numpy as np
from tqdm import tqdm
import re
from src.dataset.augment import PlaygroundAugmentor


class Playground_Reader:
    """
    Reader para generar los archivos procesados de skeleton data (train/val)
    del dataset Playground, compatible con el pipeline MP-GCN/EfficientGCN.
    """

    def __init__(self, dataset_root_folder, out_folder, **kwargs):
        self.max_channel = 2      # (x, y)
        self.max_joint = 17
        self.max_person = 6
        self.max_frame = 300
        self.n_obj_max = 7

        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder

        self.pose_dir = os.path.join(dataset_root_folder, 'yolo_outputs')
        self.objects_path = os.path.join(dataset_root_folder, 'objects.yaml')
        self.annotations_path = os.path.join(dataset_root_folder, 'annotations.pkl')

        
        self.transform = kwargs.get("transform", False)
        if self.transform:
            logging.info("Data augmentation ACTIVADA en Reader.")
            self.augmentor = PlaygroundAugmentor()
        else:
            self.augmentor = None

        # Cargar objetos y anotaciones
        if not os.path.exists(self.objects_path):
            raise FileNotFoundError(f"No se encontró {self.objects_path}")
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"No se encontró {self.annotations_path}")

        with open(self.objects_path, 'r') as f:
            self.objects_by_cam = yaml.safe_load(f) or {}


        with open(self.annotations_path, 'rb') as f:
            self.annotations = pickle.load(f)

        # --- class2idx opcional ---
        self.class2idx = kwargs.get('class2idx', None)
        if not self.class2idx:
            # NEW: generar automáticamente a partir de las anotaciones
            all_labels = sorted(
                list(set(a['label'] for a in self.annotations.get('annotations', [])))
            )
            self.class2idx = {lbl: i for i, lbl in enumerate(all_labels)}
            logging.info(f"Generado class2idx automáticamente: {self.class2idx}")


        # Compila patrón para IDs de cámara
        self._digits_re = re.compile(r'(\d+)$')

    # ------------------------------------------------------------------

    def _resolve_camera_key(self, cam_id: str):
        """Devuelve la llave de 'objects_by_cam' que mejor coincide con cam_id."""
        if not isinstance(self.objects_by_cam, dict) or not self.objects_by_cam:
            return None

        keys = list(self.objects_by_cam.keys())
        low_map = {k.lower(): k for k in keys}

        if cam_id in self.objects_by_cam:
            return cam_id
        if cam_id.lower() in low_map:
            return low_map[cam_id.lower()]

        m = self._digits_re.search(cam_id or '')
        if m:
            num = int(m.group(1))
            for candidate in (
                f'camera_{num:02d}', f'camera_{num}', f'cam_{num:02d}', f'cam_{num}'
            ):
                if candidate in self.objects_by_cam:
                    return candidate
                if candidate.lower() in low_map:
                    return low_map[candidate.lower()]

        if len(keys) == 1:
            k = keys[0]
            logging.warning(f"[objects.yaml] No match para cam_id='{cam_id}'. Usando '{k}'.")
            return k

        logging.warning(
            f"[objects.yaml] No se encontró bloque para cam_id='{cam_id}'. Claves: {keys}"
        )
        return None

    # ------------------------------------------------------------------

    def read_pose(self, npy_path):
        """Lee el archivo .npy generado por YOLO pose y devuelve [T, M, V, C]."""
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim == 3:
            data = np.expand_dims(data, axis=1)  # [T,V,C] → [T,1,V,C]
        return data

    # ------------------------------------------------------------------

    def integrate_objects(self, skeleton_data, cam_objects: dict):
        """Agrega objetos (x,y) del bloque correspondiente a la cámara."""
        T, M, V, C = skeleton_data.shape
        if not isinstance(cam_objects, dict) or not cam_objects:
            return skeleton_data

        obj_list = []
        for name, val in cam_objects.items():
            try:
                if isinstance(val, dict) and 'x' in val and 'y' in val:
                    obj_list.append([float(val['x']), float(val['y'])])
                elif isinstance(val, (list, tuple)) and len(val) >= 2:
                    obj_list.append([float(val[0]), float(val[1])])
            except Exception as e:
                logging.warning(f"Error leyendo objeto '{name}': {e}")

        if not obj_list:
            return skeleton_data

        obj_coords = np.array(obj_list, dtype=np.float32)
        new_V = V + len(obj_coords)
        skeleton_extended = np.zeros((T, M, new_V, C), dtype=np.float32)
        skeleton_extended[:, :, :V, :] = skeleton_data

        # Repetir objetos en todos los frames y personas
        for t in range(T):
            for m in range(M):
                skeleton_extended[t, m, V:, :] = obj_coords
        return skeleton_extended

    # ------------------------------------------------------------------

    def normalize_objects(self, skeleton_data, n_obj_current, n_obj_max):
        """
        Garantiza que todos los videos tengan V = 17 + n_obj_max nodos.
        """
        T, M, V, C = skeleton_data.shape
        expected_V = 17 + n_obj_max

        # Si no hay objetos, agrega todos en cero
        if V < expected_V:
            pad = np.zeros((T, M, expected_V - V, C), dtype=np.float32)
            skeleton_data = np.concatenate([skeleton_data, pad], axis=2)

        # Si por alguna razón excede, recorta
        elif V > expected_V:
            skeleton_data = skeleton_data[:, :, :expected_V, :]

        return skeleton_data


    def clean_and_interpolate_people(self, data):
        """
        data: shape (T, M_detected, 17, 2)
        Devuelve (T, 6, 17, 2) con:
        - personas inexistentes → todo 0
        - personas reales → interpoladas sin NaNs
        - máximo 6 personas
        """
        T, M, V, C = data.shape
        M_max = self.max_person  # = 6

        # 1) Identificar personas reales (tienen al menos un joint válido)
        person_valid = []
        for m in range(M):
            person_m = data[:, m, :, :]        # (T, 17, 2)
            nan_count = np.isnan(person_m).sum()
            if nan_count < T * V * C:          # NO es todo NaN
                person_valid.append(m)

        # 2) Reconstruir lista de personas reales
        cleaned = []

        for m in person_valid:
            pm = data[:, m, :, :]  # (T,17,2)

            # Interpolar cada joint/coord por separado
            for v in range(V):
                for c in range(C):
                    seq = pm[:, v, c]                 # (T,)
                    mask = ~np.isnan(seq)

                    if mask.sum() == 0:
                        # Persona real pero este joint NUNCA fue detectado
                        # → dejamos en 0
                        pm[:, v, c] = 0
                    else:
                        # interpolación temporal lineal
                        valid_x = np.where(mask)[0]
                        valid_y = seq[mask]
                        pm[:, v, c] = np.interp(
                            np.arange(T), valid_x, valid_y
                        )

            cleaned.append(pm)

        # 3) Limitar a máximo 6 personas
        cleaned = cleaned[:M_max]

        # 4) Si no hay suficientes personas, hacer padding
        while len(cleaned) < M_max:
            cleaned.append(np.zeros((T, V, C), dtype=np.float32))

        # 5) Apilar en nuevo arreglo
        out = np.stack(cleaned, axis=1)  # (T,6,17,2)

        return out.astype(np.float32)


    # ------------------------------------------------------------------

    def gendata(self, phase):
        """Genera los archivos *_data.npy y *_label.pkl."""
        videos = self.annotations['split'].get(phase, [])
        res_skeleton = []
        labels = []

        if not videos:
            logging.warning(f"No hay videos para phase='{phase}'")
            return

        for vid in tqdm(videos, desc=f'Processing {phase}', dynamic_ncols=True):
            try:
                anno = next(a for a in self.annotations['annotations'] if a['video_id'] == vid)
            except StopIteration:
                logging.warning(f"No hay anotación para video_id='{vid}'.")
                continue

            video_name = anno['video_name']
            cam_id = anno.get('cam_id', '')
            label_name = anno['label']

            npy_path = os.path.join(self.pose_dir, f"{video_name}.npy")
            if not os.path.exists(npy_path):
                logging.warning(f"Falta .npy: {npy_path}")
                continue

            # 1️⃣ Leer y extender con objetos
            skeleton_data = self.read_pose(npy_path)

            # interpolar, reemplazar NaNs con 0s
            skeleton_data = self.clean_and_interpolate_people(skeleton_data)


            if self.transform:
                skeleton_data = self.augmentor(skeleton_data)

            cam_key = self._resolve_camera_key(cam_id)
            cam_objects = self.objects_by_cam.get(cam_key, {}) if cam_key else {}
            skeleton_data = self.integrate_objects(skeleton_data, cam_objects)

            # 2️⃣ Homogeneizar tamaño
            T, M, V, C = skeleton_data.shape
            T_max, M_max = self.max_frame, self.max_person
            if T > T_max:
                skeleton_data = skeleton_data[:T_max]
            elif T < T_max:
                pad = np.zeros((T_max - T, M, V, C), dtype=np.float32)
                skeleton_data = np.concatenate([skeleton_data, pad], axis=0)

            if M > M_max:
                skeleton_data = skeleton_data[:, :M_max]
            elif M < M_max:
                pad = np.zeros((T_max, M_max - M, V, C), dtype=np.float32)
                skeleton_data = np.concatenate([skeleton_data, pad], axis=1)

            # 3️⃣ Normalizar objetos
            n_obj_current = len(cam_objects)
            skeleton_data = self.normalize_objects(skeleton_data, n_obj_current, self.n_obj_max)

            res_skeleton.append(skeleton_data)

            # 4️⃣ Etiqueta numérica
            # print("label_name:", repr(label_name))
            # print("class2idx:", self.class2idx)
            # print("lookup:", self.class2idx.get(label_name, None))
            label_idx = self.class2idx.get(label_name, -1)
            labels.append([label_idx, video_name])

        # Guardar resultados
        res_skeleton = np.array(res_skeleton, dtype=np.float32)
        os.makedirs(self.out_folder, exist_ok=True)

        np.save(os.path.join(self.out_folder, f"{phase}_data.npy"), res_skeleton)
        with open(os.path.join(self.out_folder, f"{phase}_label.pkl"), 'wb') as f:
            pickle.dump(labels, f)

        logging.info(f"{phase} set: {len(labels)} samples procesados correctamente.")

    # ------------------------------------------------------------------

    def start(self):
        """Genera train y val sets."""
        for phase in ['train', 'eval']:
            logging.info(f"=== Generating {phase} data ===")
            self.gendata(phase)
