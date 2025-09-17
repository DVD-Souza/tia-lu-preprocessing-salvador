from food_statistics import Statistics
from typing import Dict, List, Set, Any

class Scaler:
    """
    Aplica transformações de escala em colunas numéricas do dataset.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())
    
    
    def minMax_scaler(self, columns: Set[str] = None):
        """
        Aplica a normalização Min-Max ($X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$)
        nas colunas especificadas. Modifica o dataset.

        Args:
            columns (Set[str]): Colunas para aplicar o scaler. Se vazio, tenta aplicar a todas.
        """
        
        target_colums = self._get_target_columns(columns)
        for col in target_colums:
            values = Statistics._validate_column(col)
            if not values:
                continue
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                # todos os valores que são iguais irão virar 0.0
                self.dataset[col] = [0.0] * len(values)
            else:
                self.dataset[col] = [(x - min_val) / (max_val - min_val) for x in values]


    def standard_scaler(self, columns: Set[str] = None):
       """
        Aplica a padronização (Z-score) nas colunas especificadas.
        Modifica o dataset original.
        """
       target_columns = self._get_target_columns(columns)

       for col in target_columns:
           values = Statistics._validate_column(col)
           if not values:
               continue
           #mean_val = sum(values) / len(values)
           mean_val = Statistics.mean(values)
           stdev_val = Statistics.stdev(values)
           if stdev_val == 0:
               self.dataset[col] = [0.0] * len(values)
           else:
               self.dataset[col] = [(x - mean_val) / stdev_val for x in values]    


class Encoder:
    """
    Aplica codificação em colunas categóricas.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        """
        Retorna a lista de colunas a serem processadas.
        Se 'columns' for None ou vazio, retorna todas as colunas do dataset.
        """
        return list(columns) if columns else list(self.dataset.keys())

    def _validate_categorical_column(self, column: str) -> List[Any]:
        """
        Garante que a coluna existe e contém valores categóricos (não numéricos).
        """
        if column not in self.dataset:
            raise KeyError(f"A coluna '{column}' não existe no dataset.")
        values = self.dataset[column]
        for v in values:
            if isinstance(v, (list, dict)):
                raise TypeError(f"A coluna '{column}' contém valores não codificáveis.")
        return values

    def label_encode(self, columns: Set[str]):
        """
        Converte cada categoria em um número inteiro único (ordem alfabética).
        Modifica o dataset in-place.
        """
        target_columns = self._get_target_columns(columns)

        for col in target_columns:
            values = self._validate_categorical_column(col)
            if not values:
                continue  # coluna vazia, nada a fazer

            # Categorias únicas em ordem alfabética
            categorias_unicas = sorted(set(values))

            # Criar mapeamento categoria -> inteiro
            mapping = {cat: idx for idx, cat in enumerate(categorias_unicas)}

            # Substituir valores pela codificação
            self.dataset[col] = [mapping[v] for v in values]

    def oneHot_encode(self, columns: Set[str]):
        """
        Cria novas colunas binárias (0/1) para cada categoria.
        Remove a coluna original.
        """
        target_columns = self._get_target_columns(columns)

        for col in target_columns:
            values = self._validate_categorical_column(col)
            if not values:
                continue  # coluna vazia, nada a fazer

            # Categorias únicas em ordem alfabética
            categorias_unicas = sorted(set(values))

            # Criar colunas binárias com underscore simples
            for cat in categorias_unicas:
                col_name = f"{col}_{str(cat)}"
                self.dataset[col_name] = [1 if v == cat else 0 for v in values]

            # Remover coluna original
            del self.dataset[col]

                       


        
        

