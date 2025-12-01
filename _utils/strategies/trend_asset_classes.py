import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy


from _utils.core_functions import *

from _utils.normalized_price import calculate_normalised_price_dict

"""
This is a code based on book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate, and the author is not responsible for any losses that may result from it's use.

Results may not match the book exactly as different data may be used
Results may be different from the corresponding spreadsheet as methods may be slightly different
"""



def calculate_asset_class_price_dict(
    normalised_price_dict: dict,
    asset_class_groupings: dict
) -> dict:
    """
    Build a dictionary mapping each instrument to the *normalised price
    of its asset class*.

    Para cada instrumento, determina sua classe de ativo e retorna a 
    série de preço normalizado representativa daquela classe.

    Parameters
    ----------
    normalised_price_dict : dict[str, pd.Series]
        Dicionário com preços normalizados por instrumento.
    asset_class_groupings : dict[str, list[str]]
        Dicionário onde as chaves são nomes de classes de ativos e os 
        valores são listas de instrumentos pertencentes a cada classe.

    Returns
    -------
    dict[str, pd.Series]
        Dicionário onde cada chave é um instrumento e o valor é a série 
        de preço normalizado correspondente à classe desse instrumento.
    """
    list_of_instruments = list(normalised_price_dict.keys())

    asset_class_price_dict = {
        instrument_code: calculate_asset_prices_for_instrument(
            instrument_code=instrument_code,
            normalised_price_dict=normalised_price_dict,
            asset_class_groupings=asset_class_groupings,
        )
        for instrument_code in list_of_instruments
    }

    return asset_class_price_dict


def calculate_asset_prices_for_instrument(
    instrument_code: str,
    normalised_price_dict: dict,
    asset_class_groupings: dict
) -> pd.Series:
    """
    Return the normalised price series for the *asset class* corresponding 
    to a given instrument.

    Esse wrapper:
    1. Identifica a classe do instrumento.
    2. Retorna o preço normalizado agregado dessa classe.

    Parameters
    ----------
    instrument_code : str
        Código do instrumento (e.g. "SP500", "CL", "BUND").
    normalised_price_dict : dict[str, pd.Series]
        Preços normalizados individuais.
    asset_class_groupings : dict[str, list[str]]
        Classes de ativos e seus instrumentos.

    Returns
    -------
    pd.Series
        Série representando o preço normalizado da classe de ativo.
    """
    asset_class = get_asset_class_for_instrument(
        instrument_code=instrument_code,
        asset_class_groupings=asset_class_groupings
    )

    return get_normalised_price_for_asset_class(
        asset_class=asset_class,
        asset_class_groupings=asset_class_groupings,
        normalised_price_dict=normalised_price_dict,
    )


def get_asset_class_for_instrument(
    instrument_code: str,
    asset_class_groupings: dict
) -> str:
    """
    Determine the asset class (string) to which an instrument belongs.

    Procura o instrumento dentro das listas de cada asset class. 
    Assume-se que cada instrumento pertence a exatamente uma classe.

    Parameters
    ----------
    instrument_code : str
        Código do instrumento.
    asset_class_groupings : dict[str, list[str]]
        Dicionário com asset classes como chaves e instrumentos como valores.

    Returns
    -------
    str
        Nome da classe de ativo à qual o instrumento pertence.

    Raises
    ------
    ValueError
        Se o instrumento não pertencer a nenhuma classe.
    """
    possible_asset_classes = list(asset_class_groupings.keys())

    matches = [
        asset_class
        for asset_class in possible_asset_classes
        if instrument_code in asset_class_groupings[asset_class]
    ]

    if len(matches) == 0:
        raise ValueError(f"Instrument '{instrument_code}' not found in any asset class.")
    if len(matches) > 1:
        raise ValueError(f"Instrument '{instrument_code}' found in multiple asset classes: {matches}")

    return matches[0]

def get_normalised_price_for_asset_class(
    asset_class: str,
    normalised_price_dict: dict,
    asset_class_groupings: dict
) -> pd.Series:
    """
    Compute the *asset-class normalised price* by averaging the 
    normalised returns of all instruments belonging to the class.

    Lógica (modelo Carver):
    1. Seleciona todos os instrumentos da classe.
    2. Concatena suas séries de preços normalizados.
    3. Calcula o retorno normalizado diário de cada instrumento (diff).
    4. Faz a média cross-sectional diária desses retornos.
    5. Acumula a média para formar o “asset class normalised price”.

    Isso cria uma proxy agregada da classe com risco homogêneo e
    comparável entre classes.

    Parameters
    ----------
    asset_class : str
        Nome da classe (ex.: "Equities", "Bonds", "Commodities").
    normalised_price_dict : dict[str, pd.Series]
        Dicionário contendo preços normalizados por instrumento.
    asset_class_groupings : dict[str, list[str]]
        Mapeamento de cada classe para sua lista de instrumentos.

    Returns
    -------
    pd.Series
        Série acumulada de normalised price da classe de ativo.

    Notes
    -----
    - Code é “wasteful” propositalmente: recalcula o agregado mesmo se
      múltiplos instrumentos da mesma classe forem chamados repetidamente.
      É a abordagem mais simples e clara.
    - Todos os instrumentos são igualmente ponderados.
    """
    instruments_in_asset_class = asset_class_groupings[asset_class]

    list_of_normalised_prices_over_asset_class = [
        normalised_price_dict[instrument_code]
        for instrument_code in instruments_in_asset_class
    ]

    # Concatena todas as séries e forward-fill
    normalised_prices_over_asset_class = pd.concat(
        list_of_normalised_prices_over_asset_class, axis=1
    ).ffill()

    # Retorno normalizado diário de cada instrumento
    normalised_returns_over_asset_class = normalised_prices_over_asset_class.diff()

    # Média cross-sectional
    average_normalised_return_over_asset_class = (
        normalised_returns_over_asset_class.mean(axis=1)
    )

    # Preço normalizado da classe
    asset_class_price = average_normalised_return_over_asset_class.cumsum()

    return asset_class_price
#%% =====================
# Relative price strategies
def calculate_relative_price_dict(
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    asset_class_groupings: dict
) -> dict:
    """
    Compute the *relative price* series for all instruments.

    A lógica segue o framework do Carver:
    1. Normaliza preços individuais em unidades de risco (normalised_price).
    2. Calcula o preço normalizado da classe do ativo.
    3. Relative price = normalised_price_instrument - normalised_price_asset_class.
    
    A diferença captura a “riqueza relativa” do ativo versus sua classe,
    removendo movimentos sistêmicos e deixando apenas comportamento idiossincrático.

    Parameters
    ----------
    adjusted_prices_dict : dict[str, pd.Series]
        Preços ajustados por instrumento.
    std_dev_dict : dict[str, standardDeviation]
        Objetos contendo volatilidade diária (daily risk).
    asset_class_groupings : dict[str, list[str]]
        Classes de ativos e seus instrumentos.

    Returns
    -------
    dict[str, pd.Series]
        Dicionário contendo a série de preço relativo para cada instrumento.
    """
    # 1 — Preços normalizados
    normalised_price_dict = calculate_normalised_price_dict(
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
    )

    # 2 — Preço de cada asset class
    asset_class_price_dict = calculate_asset_class_price_dict(
        normalised_price_dict=normalised_price_dict,
        asset_class_groupings=asset_class_groupings,
    )

    # 3 — Relative price por instrumento
    list_of_instruments = list(normalised_price_dict.keys())

    relative_price_dict = {
        instrument_code: relative_price_for_instrument(
            instrument_code=instrument_code,
            normalised_price_dict=normalised_price_dict,
            asset_class_price_dict=asset_class_price_dict,
        )
        for instrument_code in list_of_instruments
    }

    return relative_price_dict


def relative_price_for_instrument(
    instrument_code: str,
    normalised_price_dict: dict,
    asset_class_price_dict: dict
) -> pd.Series:
    """
    Compute the *relative price* of one instrument vs. its asset class.

    Fórmula:
        relative_price[t] = normalised_price[t] - asset_class_normalised_price[t]

    Esse cálculo remove o beta do ativo dentro da classe, deixando apenas
    movimentos idiossincráticos — útil para modelos de cross-sectional,
    replacement overlays e análise de divergência dentro da classe.

    Regras adicionais:
    - O índice da classe é alinhado ao do ativo via reindex + ffill.
    - Valores exatamente zero são substituídos por NaN (Carver-style),
      indicando uma ausência de desvio relativo.

    Parameters
    ----------
    instrument_code : str
        Código do instrumento.
    normalised_price_dict : dict[str, pd.Series]
        Preços normalizados individuais.
    asset_class_price_dict : dict[str, pd.Series]
        Preços normalizados das asset classes.
    
    Returns
    -------
    pd.Series
        Série de preços relativos do instrumento.
    """
    # Série do ativo
    normalised_price = normalised_price_dict[instrument_code]

    # Série da classe referente ao ativo
    asset_class_price = asset_class_price_dict[instrument_code]

    # Alinhamento obrigatório
    asset_class_price_matched = asset_class_price.reindex(
        normalised_price.index
    ).ffill()

    # Diferença
    relative_price = normalised_price - asset_class_price_matched

    # Conforme o livro: valores 0 → NaN
    relative_price[relative_price == 0] = np.nan

    return relative_price
