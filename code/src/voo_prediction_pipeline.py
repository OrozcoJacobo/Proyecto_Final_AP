"""
Pipeline de limpieza, preparación de variables y recomendación para el ETF VOO.

Este módulo contiene la lógica reutilizable por la aplicación Streamlit. La idea
es mantener separada la interfaz gráfica del procesamiento de datos y de la
generación de recomendaciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COLUMNAS_REQUERIDAS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]


COLUMNAS_PREDICTORAS = [
    "cambio_porcentual",
    "retorno_diario",
    "retorno_logaritmico",
    "retorno_3d",
    "retorno_5d",
    "retorno_10d",
    "retorno_20d",
    "rango_diario",
    "cambio_apertura_cierre",
    "posicion_cierre_rango",
    "distancia_media_movil_20d",
    "distancia_media_movil_50d",
    "distancia_media_movil_100d",
    "volatilidad_5d",
    "volatilidad_10d",
    "volatilidad_20d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "distancia_maximo_20d",
    "distancia_minimo_20d",
    "distancia_maximo_50d",
    "distancia_minimo_50d",
    "distancia_maximo_100d",
    "distancia_minimo_100d",
    "cambio_volumen",
    "ratio_volumen_5d",
    "ratio_volumen_10d",
    "ratio_volumen_20d",
]


@dataclass
class ResultadoRecomendacion:
    """
    Representa la salida interpretable generada por la capa de decisión.

    Attributes:
        nivel_senal: Categoría general de la señal generada.
        recomendacion: Recomendación resumida para el usuario.
        interpretacion: Explicación detallada del resultado.
    """

    nivel_senal: str
    recomendacion: str
    interpretacion: str


def convertir_precio_a_float(valor_original):
    """
    Convierte un valor de precio del CSV a número decimal.

    Args:
        valor_original: Valor original de una columna de precio.

    Returns:
        Valor convertido a `float`. Si el valor es nulo, retorna `np.nan`.
    """
    if pd.isna(valor_original):
        return np.nan

    valor_limpio = str(valor_original).replace(",", "").strip()

    return float(valor_limpio)


def convertir_porcentaje_a_decimal(valor_original):
    """
    Convierte un porcentaje textual a proporción decimal.

    Args:
        valor_original: Valor original de la columna porcentual.

    Returns:
        Valor convertido a proporción decimal.
    """
    if pd.isna(valor_original):
        return np.nan

    valor_limpio = str(valor_original).replace("%", "").replace(",", "").strip()

    return float(valor_limpio) / 100


def convertir_volumen_a_float(valor_original):
    """
    Convierte el volumen transado a valor numérico.

    Args:
        valor_original: Valor original de la columna de volumen.

    Returns:
        Volumen convertido a `float`. Si el valor no contiene información
        válida, retorna `np.nan`.
    """
    if pd.isna(valor_original):
        return np.nan

    valor_texto = str(valor_original).replace(",", "").strip().upper()

    if valor_texto in {"", "-", "NAN"}:
        return np.nan

    multiplicador = 1.0

    if valor_texto.endswith("K"):
        multiplicador = 1_000
        valor_texto = valor_texto[:-1]
    elif valor_texto.endswith("M"):
        multiplicador = 1_000_000
        valor_texto = valor_texto[:-1]
    elif valor_texto.endswith("B"):
        multiplicador = 1_000_000_000
        valor_texto = valor_texto[:-1]

    return float(valor_texto) * multiplicador


def validar_columnas_requeridas(datos_crudos):
    """
    Valida que el CSV tenga las columnas mínimas necesarias.

    Args:
        datos_crudos: DataFrame cargado desde el archivo CSV.

    Raises:
        ValueError: Si el dataset no contiene una o más columnas requeridas.
    """
    columnas_faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in datos_crudos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "El archivo CSV no contiene las columnas requeridas: "
            f"{columnas_faltantes}."
        )


def cargar_y_limpiar_datos(archivo_csv):
    """
    Carga, limpia y ordena cronológicamente el dataset del ETF VOO.

    Args:
        archivo_csv: Archivo CSV cargado desde Streamlit o ruta local.

    Returns:
        DataFrame limpio con columnas estandarizadas.
    """
    datos_crudos = pd.read_csv(archivo_csv)

    validar_columnas_requeridas(datos_crudos)

    datos_voo = datos_crudos.copy()

    datos_voo["fecha"] = pd.to_datetime(datos_voo["Date"])
    datos_voo["precio_cierre"] = datos_voo["Price"].apply(convertir_precio_a_float)
    datos_voo["precio_apertura"] = datos_voo["Open"].apply(convertir_precio_a_float)
    datos_voo["precio_maximo"] = datos_voo["High"].apply(convertir_precio_a_float)
    datos_voo["precio_minimo"] = datos_voo["Low"].apply(convertir_precio_a_float)
    datos_voo["volumen"] = datos_voo["Vol."].apply(convertir_volumen_a_float)
    datos_voo["cambio_porcentual"] = datos_voo["Change %"].apply(
        convertir_porcentaje_a_decimal
    )

    columnas_limpias = [
        "fecha",
        "precio_cierre",
        "precio_apertura",
        "precio_maximo",
        "precio_minimo",
        "volumen",
        "cambio_porcentual",
    ]

    datos_voo = datos_voo[columnas_limpias]
    datos_voo = datos_voo.sort_values("fecha").reset_index(drop=True)

    registros_sin_operacion = (
        datos_voo["volumen"].isna()
        & (datos_voo["precio_cierre"] == datos_voo["precio_apertura"])
        & (datos_voo["precio_cierre"] == datos_voo["precio_maximo"])
        & (datos_voo["precio_cierre"] == datos_voo["precio_minimo"])
        & (datos_voo["cambio_porcentual"] == 0)
    )

    datos_voo = datos_voo.loc[~registros_sin_operacion].copy()
    datos_voo = datos_voo.dropna().reset_index(drop=True)

    return datos_voo


def agregar_variables_de_retorno(datos):
    """
    Agrega variables basadas en retornos históricos del precio.

    Args:
        datos: DataFrame con la columna `precio_cierre`.

    Returns:
        DataFrame con variables de retorno.
    """
    datos_features = datos.copy()

    datos_features["retorno_diario"] = datos_features["precio_cierre"].pct_change()
    datos_features["retorno_logaritmico"] = np.log(
        datos_features["precio_cierre"] / datos_features["precio_cierre"].shift(1)
    )

    datos_features["retorno_3d"] = datos_features["precio_cierre"].pct_change(3)
    datos_features["retorno_5d"] = datos_features["precio_cierre"].pct_change(5)
    datos_features["retorno_10d"] = datos_features["precio_cierre"].pct_change(10)
    datos_features["retorno_20d"] = datos_features["precio_cierre"].pct_change(20)

    return datos_features


def agregar_variables_de_rango(datos):
    """
    Agrega variables relacionadas con el rango diario de negociación.

    Args:
        datos: DataFrame con columnas OHLC.

    Returns:
        DataFrame con variables de rango.
    """
    datos_features = datos.copy()

    datos_features["rango_diario"] = (
        datos_features["precio_maximo"] - datos_features["precio_minimo"]
    ) / datos_features["precio_cierre"]

    datos_features["cambio_apertura_cierre"] = (
        datos_features["precio_cierre"] - datos_features["precio_apertura"]
    ) / datos_features["precio_apertura"]

    denominador_rango = (
        datos_features["precio_maximo"] - datos_features["precio_minimo"]
    )

    datos_features["posicion_cierre_rango"] = (
        datos_features["precio_cierre"] - datos_features["precio_minimo"]
    ) / denominador_rango

    datos_features["posicion_cierre_rango"] = datos_features[
        "posicion_cierre_rango"
    ].replace([np.inf, -np.inf], np.nan)

    return datos_features


def agregar_medias_moviles(datos, ventanas):
    """
    Agrega medias móviles y distancias relativas frente a esas medias.

    Args:
        datos: DataFrame con la columna `precio_cierre`.
        ventanas: Lista de ventanas temporales.

    Returns:
        DataFrame con medias móviles y distancias relativas.
    """
    datos_features = datos.copy()

    for ventana in ventanas:
        columna_media = f"media_movil_{ventana}d"
        columna_distancia = f"distancia_media_movil_{ventana}d"

        datos_features[columna_media] = (
            datos_features["precio_cierre"]
            .rolling(window=ventana)
            .mean()
        )

        datos_features[columna_distancia] = (
            datos_features["precio_cierre"] - datos_features[columna_media]
        ) / datos_features[columna_media]

    return datos_features


def agregar_variables_de_volatilidad(datos, ventanas):
    """
    Agrega variables de volatilidad histórica.

    Args:
        datos: DataFrame con la columna `retorno_diario`.
        ventanas: Lista de ventanas temporales.

    Returns:
        DataFrame con variables de volatilidad.
    """
    datos_features = datos.copy()

    for ventana in ventanas:
        datos_features[f"volatilidad_{ventana}d"] = (
            datos_features["retorno_diario"]
            .rolling(window=ventana)
            .std()
        )

    return datos_features


def agregar_variables_de_momentum(datos, ventanas):
    """
    Agrega variables de momentum del precio.

    Args:
        datos: DataFrame con la columna `precio_cierre`.
        ventanas: Lista de horizontes pasados.

    Returns:
        DataFrame con variables de momentum.
    """
    datos_features = datos.copy()

    for ventana in ventanas:
        datos_features[f"momentum_{ventana}d"] = (
            datos_features["precio_cierre"]
            / datos_features["precio_cierre"].shift(ventana)
        ) - 1

    return datos_features


def agregar_distancia_a_extremos(datos, ventanas):
    """
    Agrega distancias relativas frente a máximos y mínimos recientes.

    Args:
        datos: DataFrame con la columna `precio_cierre`.
        ventanas: Lista de ventanas temporales.

    Returns:
        DataFrame con distancias a máximos y mínimos recientes.
    """
    datos_features = datos.copy()

    for ventana in ventanas:
        maximo_reciente = datos_features["precio_cierre"].rolling(window=ventana).max()
        minimo_reciente = datos_features["precio_cierre"].rolling(window=ventana).min()

        datos_features[f"distancia_maximo_{ventana}d"] = (
            datos_features["precio_cierre"] - maximo_reciente
        ) / maximo_reciente

        datos_features[f"distancia_minimo_{ventana}d"] = (
            datos_features["precio_cierre"] - minimo_reciente
        ) / minimo_reciente

    return datos_features


def agregar_variables_de_volumen(datos, ventanas):
    """
    Agrega variables relativas basadas en volumen.

    Args:
        datos: DataFrame con la columna `volumen`.
        ventanas: Lista de ventanas temporales.

    Returns:
        DataFrame con variables derivadas del volumen.
    """
    datos_features = datos.copy()

    datos_features["cambio_volumen"] = datos_features["volumen"].pct_change()
    datos_features["cambio_volumen"] = datos_features["cambio_volumen"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for ventana in ventanas:
        columna_media_volumen = f"media_volumen_{ventana}d"
        columna_ratio_volumen = f"ratio_volumen_{ventana}d"

        datos_features[columna_media_volumen] = (
            datos_features["volumen"]
            .rolling(window=ventana)
            .mean()
        )

        datos_features[columna_ratio_volumen] = (
            datos_features["volumen"] / datos_features[columna_media_volumen]
        )

    return datos_features


def construir_variables_predictoras(datos):
    """
    Construye las variables predictoras usadas por los modelos finales.

    Args:
        datos: DataFrame limpio del ETF VOO.

    Returns:
        DataFrame enriquecido con variables predictoras.
    """
    ventanas_cortas = [5, 10, 20]
    ventanas_largas = [20, 50, 100]

    datos_features = datos.copy()

    datos_features = agregar_variables_de_retorno(datos_features)
    datos_features = agregar_variables_de_rango(datos_features)
    datos_features = agregar_medias_moviles(datos_features, ventanas_largas)
    datos_features = agregar_variables_de_volatilidad(datos_features, ventanas_cortas)
    datos_features = agregar_variables_de_momentum(datos_features, ventanas_cortas)
    datos_features = agregar_distancia_a_extremos(datos_features, ventanas_largas)
    datos_features = agregar_variables_de_volumen(datos_features, ventanas_cortas)

    datos_features = datos_features.replace([np.inf, -np.inf], np.nan)

    return datos_features


def obtener_ultima_observacion_modelo(datos_features):
    """
    Obtiene la última observación válida para predicción.

    Args:
        datos_features: DataFrame con variables predictoras calculadas.

    Returns:
        Tupla con la última fila completa y el DataFrame X para predicción.

    Raises:
        ValueError: Si no existen suficientes datos para calcular las variables.
    """
    datos_validos = datos_features.dropna(subset=COLUMNAS_PREDICTORAS).copy()

    if datos_validos.empty:
        raise ValueError(
            "El archivo no contiene suficientes datos históricos para calcular "
            "las variables predictoras. Se requieren al menos 100 registros válidos."
        )

    ultima_observacion = datos_validos.tail(1).copy()
    x_ultima_observacion = ultima_observacion[COLUMNAS_PREDICTORAS]

    return ultima_observacion, x_ultima_observacion


def obtener_probabilidad_clase_positiva(modelo, x_datos):
    """
    Obtiene la probabilidad estimada para la clase positiva.

    Args:
        modelo: Modelo de clasificación entrenado.
        x_datos: Variables predictoras.

    Returns:
        Probabilidad de pertenencia a la clase positiva.
    """
    probabilidades = modelo.predict_proba(x_datos)
    clases_modelo = list(modelo.classes_)

    if 1 not in clases_modelo:
        return 0.0

    indice_clase_positiva = clases_modelo.index(1)

    return float(probabilidades[:, indice_clase_positiva][0])


def generar_recomendacion_inversion(
    probabilidad_15d,
    probabilidad_20d,
    umbral_15d,
    umbral_20d,
):
    """
    Genera una recomendación interpretable a partir de probabilidades.

    Args:
        probabilidad_15d: Probabilidad estimada de oportunidad a 15 días.
        probabilidad_20d: Probabilidad estimada de oportunidad a 20 días.
        umbral_15d: Umbral de decisión del modelo de 15 días.
        umbral_20d: Umbral de decisión del modelo de 20 días.

    Returns:
        Objeto `ResultadoRecomendacion` con nivel, recomendación e interpretación.
    """
    zona_monitoreo_15d = umbral_15d * 0.85
    zona_monitoreo_20d = umbral_20d * 0.85

    if probabilidad_20d >= umbral_20d and probabilidad_15d >= zona_monitoreo_15d:
        return ResultadoRecomendacion(
            nivel_senal="Oportunidad fuerte",
            recomendacion="El sistema identifica una señal conservadora favorable.",
            interpretacion=(
                "El modelo principal de 20 días supera el umbral de decisión y "
                "el modelo de 15 días acompaña con una probabilidad relevante. "
                "Esto sugiere una posible oportunidad de inversión bajo los "
                "criterios definidos en el proyecto."
            ),
        )

    if probabilidad_20d >= umbral_20d:
        return ResultadoRecomendacion(
            nivel_senal="Oportunidad potencial a 20 días",
            recomendacion="Podría existir una oportunidad en el horizonte mensual.",
            interpretacion=(
                "El modelo principal supera el umbral conservador de 20 días, "
                "pero la señal de 15 días no acompaña con suficiente fuerza. "
                "Se recomienda interpretar el resultado con moderación."
            ),
        )

    if probabilidad_15d >= umbral_15d:
        return ResultadoRecomendacion(
            nivel_senal="Oportunidad potencial a 15 días",
            recomendacion=(
                "Existe una señal de corto plazo, pero no es confirmada por "
                "el horizonte principal."
            ),
            interpretacion=(
                "El modelo de 15 días supera su umbral, pero el modelo principal "
                "de 20 días no confirma una señal fuerte. El sistema sugiere "
                "monitorear antes de tomar una decisión."
            ),
        )

    if probabilidad_20d >= zona_monitoreo_20d or probabilidad_15d >= zona_monitoreo_15d:
        return ResultadoRecomendacion(
            nivel_senal="Monitorear",
            recomendacion=(
                "No hay señal fuerte, pero conviene revisar datos actualizados "
                "próximamente."
            ),
            interpretacion=(
                "Las probabilidades estimadas se encuentran en una zona intermedia. "
                "El sistema no emite una recomendación positiva fuerte, pero sugiere "
                "monitorear la evolución del ETF con datos más recientes."
            ),
        )

    return ResultadoRecomendacion(
        nivel_senal="Sin oportunidad clara",
        recomendacion=(
            "El sistema no identifica una oportunidad conservadora en este momento."
        ),
        interpretacion=(
            "Las probabilidades estimadas no superan los umbrales definidos. "
            "Bajo el enfoque conservador del proyecto, es preferible no emitir "
            "una señal positiva de inversión."
        ),
    )