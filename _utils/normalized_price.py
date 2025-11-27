import pandas as pd

from _utils.core_functions import standardDeviation


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

def calculate_normalised_price_dict(
    adjusted_prices_dict: dict, 
    std_dev_dict: dict
) -> dict:
    """
    Build a dictionary of *normalised price series* for multiple instruments.

    Normalisation here follows Robert Carver’s convention:
    - Converte variações de preço para “unidades de risco diário”.
    - Multiplica por 100 para deixar a escala comparável entre ativos.
    - Acumula os retornos normalizados para gerar um “normalised price”.

    Parameters
    ----------
    adjusted_prices_dict : dict[str, pd.Series]
        Dicionário onde cada chave é um código de instrumento e o valor é 
        a série de preços ajustados (pd.Series).
    std_dev_dict : dict[str, standardDeviation]
        Dicionário contendo o objeto de volatilidade (Carver-style) 
        para cada instrumento. O objeto deve possuir o método 
        `.daily_risk_price_terms()`.

    Returns
    -------
    dict[str, pd.Series]
        Dicionário com o normalised price para cada instrumento.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())

    normalised_price_dict = {
        instrument_code: calculate_normalised_price(
            adjusted_price=adjusted_prices_dict[instrument_code],
            instrument_risk=std_dev_dict[instrument_code],
        )
        for instrument_code in list_of_instruments
    }

    return normalised_price_dict


def calculate_normalised_price(
    adjusted_price: pd.Series,
    instrument_risk: standardDeviation,
) -> pd.Series:
    """
    Compute a *normalised price* series based on daily price changes scaled 
    by daily risk.

    Fórmula:
        normalised_return[t] = 100 * ( ΔPreço[t] / daily_risk[t] )
        normalised_price[t] = soma cumulativa dos normalised_return

    Isso fornece uma série comparável entre ativos com volatilidades distintas
    (mais intuitiva para comparar movimentos relativos).

    Parameters
    ----------
    adjusted_price : pd.Series
        Série de preços ajustados do ativo.
    instrument_risk : standardDeviation
        Objeto contendo o método `.daily_risk_price_terms()`, que retorna
        a série de “daily risk” em termos de preço.

    Returns
    -------
    pd.Series
        Série acumulada de preços normalizados.
    """
    daily_price_instrument_risk = instrument_risk.daily_risk_price_terms()

    normalised_returns = 100 * (adjusted_price.diff() / daily_price_instrument_risk)
    normalised_returns[normalised_returns.isna()] = 0

    normalised_price = normalised_returns.cumsum()

    return normalised_price
