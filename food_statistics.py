class Statistics:
    def __init__(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError("O dataset deve ser um dicionário.")
        for _, column_values in dataset.items():
            if not isinstance(column_values, list):
                raise TypeError("Todos os valores no dicionário do dataset devem ser listas.")
        lengths = [len(vals) for vals in dataset.values()]
        if len(set(lengths)) != 1:
            raise ValueError("Todas as colunas no dataset devem ter o mesmo tamanho.")
        self.dataset = dataset

    def _get_column(self, column):
        if column not in self.dataset:
            raise KeyError(f"A coluna '{column}' não existe no dataset.")
        return self.dataset[column]

    def _non_null(self, column):
        return [v for v in self._get_column(column) if v is not None]

    def _assert_numeric(self, values, method_name):
        if any(not isinstance(v, (int, float)) for v in values):
            raise TypeError(f"Os dados da coluna não são numéricos para o método '{method_name}'.")

    # --- Estatísticas básicas ---
    def mean(self, column):
        data = self._non_null(column)
        self._assert_numeric(data, 'média')
        return sum(data) / len(data) if data else 0.0

    def median(self, column):
        data = self._non_null(column)
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        return sorted_data[mid] if n % 2 == 1 else (sorted_data[mid - 1] + sorted_data[mid]) / 2

    def mode(self, column):
        freq = self.absolute_frequency(column) 
        if not freq:
            return []
        max_freq = max(freq.values())
        return [item for item, count in freq.items() if count == max_freq]

    def variance(self, column):
        data = self._non_null(column)
        self._assert_numeric(data, 'variância')
        if not data or len(data) < 2:
            return 0.0
        media = self.mean(column)  
        return sum((v - media) ** 2 for v in data) / len(data)

    def stdev(self, column):
        return self.variance(column) ** 0.5  

    def covariance(self, column_a, column_b):
        raw_a = self._get_column(column_a)
        raw_b = self._get_column(column_b)
        paired = [(x, y) for x, y in zip(raw_a, raw_b) if x is not None and y is not None]
        if not paired:
            return 0.0
        dados_a, dados_b = zip(*paired)
        self._assert_numeric(dados_a, 'covariância')
        self._assert_numeric(dados_b, 'covariância')

        media_a = self.mean(column_a)
        media_b = self.mean(column_b)

        return sum((x - media_a) * (y - media_b) for x, y in paired) / len(paired)

    # --- Frequências ---
    def itemset(self, column):
        return set(self._non_null(column))

    def absolute_frequency(self, column):
        values = self._non_null(column)
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        return freq

    def relative_frequency(self, column):
        abs_freq = self.absolute_frequency(column)
        total = sum(abs_freq.values())
        return {item: count / total for item, count in abs_freq.items()} if total > 0 else {}

    def cumulative_frequency(self, column, frequency_method='absolute'):
        if frequency_method == 'absolute':
            frequencies = self.absolute_frequency(column)
        elif frequency_method == 'relative':
            frequencies = self.relative_frequency(column)
        else:
            raise ValueError("O 'frequency_method' deve ser 'absolute' ou 'relative'.")
        if not frequencies:
            return {}
        sorted_values = sorted(frequencies.keys())
        cumulative, cumulative_freq = 0, {}
        for value in sorted_values:
            cumulative += frequencies[value]
            cumulative_freq[value] = cumulative
        if frequency_method == 'relative':
            cumulative_freq[sorted_values[-1]] = 1.0
        return cumulative_freq

    # --- Probabilidade ---
    def conditional_probability(self, column, value1, value2):
        data = self._non_null(column)
        if len(data) < 2:
            return 0.0
        count_b = data.count(value2)
        count_ba = sum(1 for prev, curr in zip(data, data[1:]) if prev == value2 and curr == value1)
        return count_ba / count_b if count_b > 0 else 0.0