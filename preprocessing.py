from food_statistics import Statistics
from typing import Dict, List, Set, Any


class MissingValueProcessor:
    """
    Processa valores ausentes (representados como None) no dataset.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        """Retorna as colunas a serem processadas. Se 'columns' for vazio, retorna todas as colunas."""
        return list(columns) if columns else list(self.dataset.keys())

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Retorna um novo dataset contendo apenas as linhas que possuem
        pelo menos um valor nulo (None) em uma das colunas especificadas.

        Args:
            columns (Set[str]): Um conjunto de nomes de colunas a serem verificadas.
                               Se vazio, todas as colunas são consideradas.

        Returns:
            Dict[str, List[Any]]: Um dicionário representando as linhas com valores nulos.
        """
        pass

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Retorna um novo dataset contendo apenas as linhas que não possuem
        valores nulos (None) em nenhuma das colunas especificadas.

        Args:
            columns (Set[str]): Um conjunto de nomes de colunas a serem verificadas.
                               Se vazio, todas as colunas são consideradas.

        Returns:
            Dict[str, List[Any]]: Um dicionário representando as linhas sem valores nulos.
        """
        pass

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        """
        Preenche valores nulos (None) nas colunas especificadas usando um método.
        Modifica o dataset da classe.

        Args:
            columns (Set[str]): Colunas onde o preenchimento será aplicado. Se vazio, aplica a todas.
            method (str): 'mean', 'median', 'mode', ou 'default_value'.
            default_value (Any): Valor para usar com o método 'default_value'.
        """
        pass

    def dropna(self, columns: Set[str] = None):
        """
        Remove as linhas que contêm valores nulos (None) nas colunas especificadas.
        Modifica o dataset da classe.

        Args:
            columns (Set[str]): Colunas a serem verificadas para valores nulos. Se vazio, todas as colunas são verificadas.
        """
        pass

class Scaler:
    """
    Aplica transformações de escala em colunas numéricas,
    reaproveitando validações e cálculos da classe Statistics.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.stats = Statistics(dataset)

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.stats.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None):
        """
        Normaliza cada coluna para [0,1]:
            X_norm = (X - X_min) / (X_max - X_min)
        """
        for col in self._get_target_columns(columns):
            values = self.stats._get_column(col)               # [uso de _get_column]
            self.stats._assert_numeric(values, 'minMax_scaler')# [uso de _assert_numeric]

            if not values:
                continue

            v_min, v_max = min(values), max(values)
            diff = v_max - v_min

            if diff == 0:
                scaled = [0.0] * len(values)
            else:
                scaled = [(v - v_min) / diff for v in values]

            self.stats.dataset[col] = scaled

    def standard_scaler(self, columns: Set[str] = None):
        """
        Padroniza cada coluna para média 0 e desvio padrão 1:
            X_std = (X - μ) / σ
        """
        for col in self._get_target_columns(columns):
            values = self.stats._get_column(col)                 # [uso de _get_column]
            self.stats._assert_numeric(values, 'standard_scaler')# [uso de _assert_numeric]

            if not values:
                continue

            mu = self.stats.mean(col)      # [uso de mean]
            sigma = self.stats.stdev(col)  # [uso de stdev]

            if sigma == 0:
                scaled = [0.0] * len(values)
            else:
                scaled = [(v - mu) / sigma for v in values]

            self.stats.dataset[col] = scaled


class Encoder:
    """
    Aplica codificação em colunas categóricas,
    reaproveitando extração de categorias da classe Statistics.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.stats = Statistics(dataset)

    def label_encode(self, columns: Set[str]):
        """
        Transforma cada categoria em inteiro:
        {'azul','verde'} → {0,1}
        """
        for col in columns:
            values = self.stats._get_column(col)                   # [uso de _get_column]
            unique = list(self.stats.absolute_frequency(col).keys())# [uso de absolute_frequency]
            mapping = {cat: idx for idx, cat in enumerate(unique)}
            self.stats.dataset[col] = [mapping[v] for v in values]

    def oneHot_encode(self, columns: Set[str]):
        """
        Cria colunas binárias para cada categoria:
        col_azul, col_verde, etc.
        """
        for col in columns:
            values = self.stats._get_column(col)                   # [uso de _get_column]
            unique = list(self.stats.absolute_frequency(col).keys())# [uso de absolute_frequency]
            n = len(values)

            for cat in unique:
                new_col = f"{col}_{cat}"
                self.stats.dataset[new_col] = [1 if v == cat else 0 for v in values]

            del self.stats.dataset[col]

class Preprocessing:
    """
    Classe principal que orquestra as operações de pré-processamento de dados.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        
        # Atributos compostos para cada tipo de tarefa
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        """
        Valida se todas as listas (colunas) no dicionário do dataset
        têm o mesmo comprimento.
        """
        pass

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.isna(). Retorna as linhas com valores nulos.
        """
        return self.missing_values.isna(columns)

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.notna(). Retorna as linhas sem valores nulos.
        """
        return self.missing_values.notna(columns)

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        """
        Atalho para missing_values.fillna(). Preenche valores nulos.
        Retorna 'self' para permitir encadeamento de métodos.
        """
        self.missing_values.fillna(columns, method, default_value)
        return self

    def dropna(self, columns: Set[str] = None):
        """
        Atalho para missing_values.dropna(). Remove linhas com valores nulos.
        Retorna 'self' para permitir encadeamento de métodos.
        """
        self.missing_values.dropna(columns)
        return self

    def scale(self, columns: Set[str] = None, method: str = 'minMax'):
        """
        Aplica escalonamento nas colunas especificadas.

        Args:
            columns (Set[str]): Colunas para aplicar o escalonamento.
            method (str): O método a ser usado: 'minMax' ou 'standard'.

        Retorna 'self' para permitir encadeamento de métodos.
        """
        if method == 'minMax':
            self.scaler.minMax_scaler(columns)
        elif method == 'standard':
            self.scaler.standard_scaler(columns)
        else:
            raise ValueError(f"Método de escalonamento '{method}' não suportado. Use 'minMax' ou 'standard'.")
        return self

    def encode(self, columns: Set[str], method: str = 'label'):
        """
        Aplica codificação nas colunas especificadas.

        Args:
            columns (Set[str]): Colunas para aplicar a codificação.
            method (str): O método a ser usado: 'label' ou 'oneHot'.
        
        Retorna 'self' para permitir encadeamento de métodos.
        """
        if not columns:
            print("Aviso: Nenhuma coluna especificada para codificação. Nenhuma ação foi tomada.")
            return self

        if method == 'label':
            self.encoder.label_encode(columns)
        elif method == 'oneHot':
            self.encoder.oneHot_encode(columns)
        else:
            raise ValueError(f"Método de codificação '{method}' não suportado. Use 'label' ou 'oneHot'.")
        return self            
