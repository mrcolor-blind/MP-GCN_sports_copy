import os
import pickle
import logging
import numpy as np
from torch.utils.data import Dataset
from .utils import graph_processing, multi_input  # usa mismas utils que NBA_Feeder
from src.dataset.graphs import Graph

class Playground_Feeder(Dataset):
    """
    Feeder para el dataset Playground.
    Diseñado para clasificación de actividades grupales por video completo.
    Usa los tensores generados por Playground_Reader (N, T, M, V, C).
    """

    def __init__(self, phase, graph, root_folder, person_id=[0], inputs='J', debug=False,
                 window=[0, 50], input_dims=2, processing='default', **kwargs):
        self.phase = phase
        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.inputs = inputs
        self.input_dims = input_dims
        self.window = window
        self.processing = processing
        self.debug = debug

        self.num_person = graph.num_person #From Nba

        # --- rutas ---
        data_path = os.path.join(root_folder, f"{phase}_data.npy")
        label_path = os.path.join(root_folder, f"{phase}_label.pkl")

        # --- carga de datos ---
        try:
            logging.info(f"Cargando datos {phase} desde {data_path}")
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.label = pickle.load(f)
        except Exception as e:
            logging.error(f"Error al cargar datos {phase}: {e}")
            raise e

        # --- debug ---
        if self.debug:
            self.data = self.data[:30]
            self.label = self.label[:30]

        # --- cortar frames según ventana ---
        self.data = self.data[:, range(*self.window), :, :, :]

        # --- reorganizar ---
        # (N, T, M, V, C) → (N, C, T, V, M)
        self.data = self.data.transpose(0, 4, 1, 3, 2)

        self.M = len(person_id)
        self.datashape = self.get_datashape()
       
    # ------------------------------------------------------

    def __len__(self):
        return len(self.label)

    # ------------------------------------------------------

    def __getitem__(self, idx):
        """
        Retorna:
            data_new: np.array [I, C, T, V, M]
            label: int
            name: str
        """
        
        data = np.array(self.data[idx])  # (C, T, V, M)
        label, name = self.label[idx]
        
        # --- procesamiento de grafo (normalización, centrado, etc.) ---
        #print("PLAYGROUND DATA SHAPE PRE GRAPH PROCESSING" + str(data.shape))
        data = graph_processing(data, self.graph, self.processing)
        
        # --- generar modalidades ---
        #print(f"[DEBUG] idx={idx}, data.shape={data.shape}, len(conn)={len(self.conn)}, len(center)={len(self.center)}")
        #print("PLAYGROUND DATA SHAPE PRE MULTI INPUT" + str(data.shape))
        data_new = multi_input(data, self.conn, self.inputs, self.center)
        #print("PLAYGROUND DATA NEW SHAPE POST MULTI INPUT" + str(data_new.shape))
        # --- validación ---
        #print("SELF.DATASHAPE POST MULTI INPUT: "+ str(self.datashape))
        try:
            assert list(data_new.shape) == self.datashape
        except AssertionError:
            logging.error(f"Forma inesperada en {name}: {data_new.shape}, esperado {self.datashape}")
            raise ValueError()

        return data_new, label, name

    # ------------------------------------------------------

    def get_datashape(self):
        """
        Define la forma esperada de salida:
        [I, C, T, V, M]
        """
        I = len(self.inputs) if self.inputs.isupper() else 1
        C = self.input_dims * 2  # duplicado por multi_input
        T = len(range(*self.window))
        V = self.num_node
        # En playground, el reader homogeniza 6 personas
        #M = Graph.num_person if hasattr(self, 'graph') and hasattr(self.graph, 'num_person') else 6
        M = self.M // self.num_person
        return [I, C, T, V, M]
