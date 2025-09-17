class Statistics:
    """
    Classe para realizar cálculos estatísticos em um conjunto de dados.
    """

    def __init__(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError("O dataset deve ser um dicionário.")

        for column, values in dataset.items():
            if not isinstance(values, list):
                raise TypeError("Todos os valores no dicionário do dataset devem ser listas.")

        if len({len(v) for v in dataset.values()}) != 1:
            raise ValueError("Todas as colunas no dataset devem ter o mesmo tamanho.")

        self.dataset = dataset

    def _validate_column(self, column, numeric_required=False):
        """
        Valida se a coluna existe e, opcionalmente, se contém apenas valores numéricos.

        Parâmetros
        ----------
        column : str
            Nome da coluna no dataset.
        numeric_required : bool
            Se True, verifica se todos os valores são int ou float.

        Retorno
        -------
        list
            Lista de valores da coluna.
        """
        if column not in self.dataset:
            raise KeyError(f"A coluna '{column}' não existe no dataset.")

        values = self.dataset[column]

        if numeric_required:
            for v in values:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"A coluna '{column}' contém valores não numéricos.")

        return values

    def mean(self, column):
        dados = self._validate_column(column, numeric_required=True)
        if not dados:
            return 0.0
        return sum(dados) / len(dados)

    def median(self, column):
        dados = self._validate_column(column, numeric_required=True)
        n = len(dados)
        if n == 0:
            return 0.0
        sorted_data = sorted(dados)
        mid = n // 2
        if n % 2 == 1:
            return sorted_data[mid]
        else:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2

    def mode(self, column):
        dados = self._validate_column(column)
        if not dados:
            return []
        freq = self.absolute_frequency(column)
        max_freq = max(freq.values())
        return [item for item, count in freq.items() if count == max_freq]

    def variance(self, column):
        dados = self._validate_column(column, numeric_required=True)
        if not dados:
            return 0.0
        media = self.mean(column)
        return sum((x - media) ** 2 for x in dados) / len(dados)

    def stdev(self, column):
        dados = self._validate_column(column, numeric_required=True)
        if not dados:
            return 0.0
        return self.variance(column) ** 0.5

    def covariance(self, column_a, column_b):
        dados_a = self._validate_column(column_a, numeric_required=True)
        dados_b = self._validate_column(column_b, numeric_required=True)
        if not dados_a or not dados_b:
            return 0.0
        media_a = self.mean(column_a)
        media_b = self.mean(column_b)
        return sum((x - media_a) * (y - media_b) for x, y in zip(dados_a, dados_b)) / len(dados_a)

    def itemset(self, column):
        dados = self._validate_column(column)
        return set(dados)

    def absolute_frequency(self, column):
        dados = self._validate_column(column)
        freq = {}
        for v in dados:
            freq[v] = freq.get(v, 0) + 1
        return freq

    def relative_frequency(self, column):
        dados = self._validate_column(column)
        if not dados:
            return {}
        abs_freq = self.absolute_frequency(column)
        total = sum(abs_freq.values())
        return {k: v / total for k, v in abs_freq.items()}

    def cumulative_frequency(self, column, frequency_method='absolute'):
        if frequency_method == 'absolute':
            freq = self.absolute_frequency(column)
        elif frequency_method == 'relative':
            freq = self.relative_frequency(column)
        else:
            raise ValueError("O 'frequency_method' deve ser 'absolute' ou 'relative'.")
        sorted_values = sorted(freq.keys())
        cumulative_freq = {}
        cumulative = 0
        for value in sorted_values:
            cumulative += freq[value]
            cumulative_freq[value] = cumulative
        return cumulative_freq

    def conditional_probability(self, column, value1, value2):
        dados = self._validate_column(column)
        if len(dados) < 2:
            return 0.0
        count_b = 0
        count_ba = 0
        event = dados[0]
        for i in dados:
            if event == value2:
                count_b += 1
                if i == value1:
                    count_ba += 1
            event = i
        if count_b > 0:
            return round(count_ba / count_b, 7)
        else:
            return 0.0
