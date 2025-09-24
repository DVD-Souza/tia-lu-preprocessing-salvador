class Statistics:
    """
    Uma classe para realizar cálculos estatísticos em um conjunto de dados.
    """

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

    def _assert_numeric(self, values, method_name):
        # [C1] Garante que os métodos numéricos sejam usados apenas com dados numéricos.
        if any(not isinstance(v, (int, float)) for v in values):
            raise TypeError(f"Os dados da coluna não são numéricos para o método '{method_name}'.")

    def mean(self, column):
        data = self._get_column(column)
        self._assert_numeric(data, 'mean')  # [C1]
        if len(data) == 0:
            return 0.0
        return sum(data) / len(data)

    def median(self, column):
        data = self._get_column(column)
        if len(data) == 0:
            return 0.0
        n = len(data)
        sorted_data = sorted(data)
        if n % 2 == 1:
            return sorted_data[n // 2]
        else:
            # [C2] Para listas não numéricas com tamanho par, isto deve levantar TypeError explicitamente
            if not isinstance(sorted_data[0], (int, float)) or not isinstance(sorted_data[1], (int, float)):
                raise TypeError("Dados não numéricos em mediana com quantidade par.")
            middle1 = sorted_data[n // 2 - 1]
            middle2 = sorted_data[n // 2]
            return (middle1 + middle2) / 2

    def mode(self, column):
        data = self._get_column(column)
        if len(data) == 0:
            return []
        frequencies = self.absolute_frequency(column)
        max_freq = max(frequencies.values())
        return [item for item, freq in frequencies.items() if freq == max_freq]

    def stdev(self, column):
        data = self._get_column(column)
        self._assert_numeric(data, 'stdev')  # [C1]
        if len(data) == 0:
            return 0.0
        variance = self.variance(column)
        return float(variance ** 0.5)

    def variance(self, column):
        data = self._get_column(column)
        self._assert_numeric(data, 'variance')  # [C1]
        if len(data) == 0:
            return 0.0
        media = self.mean(column)
        soma_quadrado_diferencas = 0
        for valor in data:
            soma_quadrado_diferencas += (valor - media) ** 2
        return soma_quadrado_diferencas / len(data)

    def covariance(self, column_a, column_b):
        dados_a = self._get_column(column_a)
        dados_b = self._get_column(column_b)

        # [C3] Covariância só faz sentido com dados numéricos
        self._assert_numeric(dados_a, 'covariance')  # [C3]
        self._assert_numeric(dados_b, 'covariance')  # [C3]

        if len(dados_a) != len(dados_b):
            raise ValueError("As colunas devem ter o mesmo número de elementos.")

        if len(dados_a) == 0:
            return 0.0

        media_a = self.mean(column_a)
        media_b = self.mean(column_b)

        soma_covar = 0
        for x, y in zip(dados_a, dados_b):
            soma_covar += (x - media_a) * (y - media_b)

        return soma_covar / len(dados_a)

    def itemset(self, column):
        values = self._get_column(column)
        return set(values)

    def absolute_frequency(self, column):
        values = self._get_column(column)
        frequencies = {}
        for value in values:
            frequencies[value] = frequencies.get(value, 0) + 1
        return frequencies

    def relative_frequency(self, column):
        stats = self._get_column(column)
        if len(stats) == 0:
            return {}

        abs_freq = self.absolute_frequency(column)

        total = 0
        for count in abs_freq.values():
            total += count

        relative_freq = {}
        for item, count in abs_freq.items():
            relative_freq[item] = count / total

        return relative_freq

    def cumulative_frequency(self, column, frequency_method='absolute'):
        data = self._get_column(column)
        if len(data) == 0:
            return {}

        if frequency_method == 'absolute':
            frequencies = self.absolute_frequency(column)
        elif frequency_method == 'relative':
            frequencies = self.relative_frequency(column)
        else:
            raise ValueError("O 'frequency_method' deve ser 'absolute' ou 'relative'.")

        sorted_values = sorted(frequencies.keys())
        cumulative_freq = {}
        cumulative = 0
        for value in sorted_values:
            cumulative += frequencies[value]
            cumulative_freq[value] = cumulative

        return cumulative_freq

    def conditional_probability(self, column, value1, value2):
        data = self._get_column(column)
        if len(data) < 2:
            return 0.0

        # [C4] Denominador: todas as ocorrências de value2 na coluna
        count_b = data.count(value2)

        # [C5] Numerador: transições onde value2 é seguido por value1
        count_ba = 0
        for prev, curr in zip(data, data[1:]):
            if prev == value2 and curr == value1:
                count_ba += 1

        if count_b > 0:
            # [C6] Retorno sem arredondamento para manter precisão nos testes
            return count_ba / count_b
        else:
            return 0.0  # [C7] Caso sem ocorrências de value2, retorna 0.0
