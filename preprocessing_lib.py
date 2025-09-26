from food_statistics import Statistics
from typing import Dict, List, Set, Any, Callable


class MissingValueProcessor:
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self.statistics = Statistics(dataset)

    def _get_target_columns(self, columns: Set[str] = None) -> List[str]:
        """Retorna as colunas alvo ou todas se None."""
        target_columns = list(columns) if columns else list(self.dataset.keys())
        self._validate_columns(target_columns)
        return target_columns
    
    def _validate_columns(self, columns: List[str]) -> None:
        """Valida se as colunas existem no dataset."""
        for column in columns:
            if column not in self.dataset:
                raise KeyError(f"A coluna '{column}' não existe no dataset.")
    
    def _filter_rows(
        self, 
        columns: Set[str], 
        condition: Callable[[Dict[str, Any]], bool]
    ) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)
        
        lengths = [len(vals) for vals in self.dataset.values()]
        if lengths and len(set(lengths)) != 1:
            raise ValueError("Todas as colunas no dataset devem ter o mesmo tamanho.")

        num_rows = lengths[0] if lengths else 0
        indices = [
            i for i in range(num_rows)
            if condition({col: self.dataset[col][i] for col in target_columns})
        ]

        return {col: [vals[i] for i in indices] for col, vals in self.dataset.items()}

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        if columns is None:
            has_null = any(None in col_vals for col_vals in self.dataset.values())
            return self.dataset if has_null else {}
        else:
            target_columns = self._get_target_columns(columns)
            return {col: [v for v in self.dataset[col] if v is None] for col in target_columns}

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        return self._filter_rows(columns, lambda row: all(v is not None for v in row.values()))

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        for column in self._get_target_columns(columns):
            values = self.dataset[column]
            if all(v is not None for v in values):
                continue

            non_null = self.statistics._non_null(column)

            if method == 'mean':
                fill_value = sum(non_null) / len(non_null) if non_null else default_value
            elif method == 'median':
                if non_null:
                    sorted_vals = sorted(non_null)
                    n, mid = len(sorted_vals), len(sorted_vals) // 2
                    fill_value = sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
                else:
                    fill_value = default_value
            elif method == 'mode':
                if non_null:
                    freq = {}
                    for v in non_null:
                        freq[v] = freq.get(v, 0) + 1
                    max_freq = max(freq.values())
                    fill_value = next(k for k, v in freq.items() if v == max_freq)
                else:
                    fill_value = default_value
            else:
                fill_value = default_value

            self.dataset[column] = [fill_value if v is None else v for v in values]

        return self
    
    def dropna(self, columns: Set[str] = None) -> None:
        self.dataset = self._filter_rows(columns, lambda row: all(v is not None for v in row.values()))


class Scaler:
    """Aplica transformações de escala em colunas numéricas."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self.statistics = Statistics(dataset)

    def _get_target_columns(self, columns: Set[str] = None) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None):
        for column in self._get_target_columns(columns):
            values = self.statistics._non_null(column)
            self.statistics._assert_numeric(values, 'minMax_scaler')
            if not values:
                continue

            v_min, v_max = min(values), max(values)
            diff = v_max - v_min
            self.dataset[column] = [
                0.0 if diff == 0 and v is not None else ((v - v_min) / diff if v is not None else None)
                for v in self.dataset[column]
            ]

    def standard_scaler(self, columns: Set[str] = None):
        for column in self._get_target_columns(columns):
            values = self.statistics._non_null(column)
            self.statistics._assert_numeric(values, 'standard_scaler')
            if not values:
                continue

            mu, sigma = self.statistics.mean(column), self.statistics.stdev(column)
            self.dataset[column] = [
                None if v is None else (0.0 if sigma == 0 else (v - mu) / sigma)
                for v in self.dataset[column]
            ]


class Encoder:
    """Aplica codificação em colunas categóricas."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self.statistics = Statistics(dataset)

    def _validate_columns(self, columns: Set[str]):
        for col in columns:
            if col not in self.dataset:
                raise KeyError(f"A coluna '{col}' não existe no dataset.")

    def label_encode(self, columns: Set[str]):
        self._validate_columns(columns)
        for column in columns:
            values = self.statistics._non_null(column)
            mapping = {cat: idx for idx, cat in enumerate(sorted(set(values)))}
            self.dataset[column] = [mapping[v] if v is not None else None for v in self.dataset[column]]

    def oneHot_encode(self, columns: Set[str]):
        self._validate_columns(columns)
        for column in columns:
            values = self.statistics._non_null(column)
            for cat in sorted(set(values)):
                self.dataset[f"{column}_{cat}"] = [1 if v == cat else 0 for v in self.dataset[column]]
            del self.dataset[column]


class Preprocessing:
    """Classe principal que orquestra as operações de pré-processamento de dados."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(dataset)
        self.scaler = Scaler(dataset)
        self.encoder = Encoder(dataset)

    def _validate_dataset_shape(self):
        lengths = [len(values) for values in self.dataset.values()]
        if lengths and len(set(lengths)) != 1:
            raise ValueError("Todas as colunas devem ter o mesmo número de linhas.")

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        return self.missing_values.isna(columns=columns)

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        return self.missing_values.notna(columns=columns)

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        self.missing_values.fillna(columns=columns, method=method, default_value=default_value)
        return self

    def dropna(self, columns: Set[str] = None):
        self.missing_values.dropna(columns=columns)
        return self

    def scale(self, columns: Set[str] = None, method: str = 'minMax'):
        if method == 'minMax':
            self.scaler.minMax_scaler(columns=columns)
        elif method == 'standard':
            self.scaler.standard_scaler(columns=columns)
        else:
            raise ValueError("Método de escalonamento inválido. Use 'minMax' ou 'standard'.")
        return self

    def encode(self, columns: Set[str], method: str = 'label'):
        if not columns:
            print("Aviso: Nenhuma coluna especificada para codificação. Nenhuma ação foi tomada.")
            return self
        if method == 'label':
            self.encoder.label_encode(columns=columns)
        elif method == 'oneHot':
            self.encoder.oneHot_encode(columns=columns)
        else:
            raise ValueError("Método de codificação inválido. Use 'label' ou 'oneHot'.")
        return self