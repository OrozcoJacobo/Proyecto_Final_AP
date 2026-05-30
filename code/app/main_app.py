"""
Aplicación Streamlit para apoyar decisiones de inversión sobre el ETF VOO.

La aplicación permite cargar un CSV histórico del ETF, procesarlo con la misma
lógica usada durante el entrenamiento y generar una recomendación conservadora
basada en modelos de aprendizaje de máquina.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


RUTA_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_SRC = RUTA_PROYECTO / "code" / "src"
RUTA_MODELOS = RUTA_PROYECTO / "code" / "models"

sys.path.append(str(RUTA_SRC))

from voo_prediction_pipeline import (  # noqa: E402
    cargar_y_limpiar_datos,
    construir_variables_predictoras,
    generar_recomendacion_inversion,
    obtener_probabilidad_clase_positiva,
    obtener_ultima_observacion_modelo,
)


st.set_page_config(
    page_title="Sistema de apoyo para inversión en VOO",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def cargar_artefactos_modelo():
    """
    Carga modelos y metadatos desde la carpeta `code/models`.

    Returns:
        Diccionario con modelos y metadatos necesarios para la predicción.
    """
    ruta_metadata = RUTA_MODELOS / "metadata_modelos_finales.json"

    with open(ruta_metadata, "r", encoding="utf-8") as archivo_metadata:
        metadata = json.load(archivo_metadata)

    ruta_modelo_principal = RUTA_MODELOS / metadata["modelo_principal"]["archivo"]
    ruta_modelo_apoyo = RUTA_MODELOS / metadata["modelo_apoyo"]["archivo"]

    modelo_principal = joblib.load(ruta_modelo_principal)
    modelo_apoyo = joblib.load(ruta_modelo_apoyo)

    return {
        "metadata": metadata,
        "modelo_principal": modelo_principal,
        "modelo_apoyo": modelo_apoyo,
    }


def construir_grafica_precio(datos_features):
    """
    Construye la gráfica histórica del precio de cierre.

    Args:
        datos_features: DataFrame con precios y medias móviles.

    Returns:
        Figura de Plotly.
    """
    figura = go.Figure()

    figura.add_trace(
        go.Scatter(
            x=datos_features["fecha"],
            y=datos_features["precio_cierre"],
            mode="lines",
            name="Precio de cierre",
        )
    )

    if "media_movil_20d" in datos_features.columns:
        figura.add_trace(
            go.Scatter(
                x=datos_features["fecha"],
                y=datos_features["media_movil_20d"],
                mode="lines",
                name="Media móvil 20 días",
            )
        )

    if "media_movil_50d" in datos_features.columns:
        figura.add_trace(
            go.Scatter(
                x=datos_features["fecha"],
                y=datos_features["media_movil_50d"],
                mode="lines",
                name="Media móvil 50 días",
            )
        )

    ultima_fila = datos_features.dropna(subset=["precio_cierre"]).tail(1)

    figura.add_trace(
        go.Scatter(
            x=ultima_fila["fecha"],
            y=ultima_fila["precio_cierre"],
            mode="markers",
            marker={"size": 10},
            name="Último dato disponible",
        )
    )

    figura.update_layout(
        title="Histórico del precio de cierre del ETF VOO",
        xaxis_title="Fecha",
        yaxis_title="Precio de cierre",
        hovermode="x unified",
        height=520,
    )

    return figura


def construir_grafica_probabilidades(
    probabilidad_15d,
    probabilidad_20d,
    umbral_15d,
    umbral_20d,
):
    """
    Construye una gráfica comparativa de probabilidades y umbrales.

    Args:
        probabilidad_15d: Probabilidad estimada a 15 días.
        probabilidad_20d: Probabilidad estimada a 20 días.
        umbral_15d: Umbral de decisión a 15 días.
        umbral_20d: Umbral de decisión a 20 días.

    Returns:
        Figura de Plotly.
    """
    datos_probabilidades = pd.DataFrame(
        {
            "horizonte": ["15 días", "20 días"],
            "probabilidad": [probabilidad_15d, probabilidad_20d],
            "umbral": [umbral_15d, umbral_20d],
        }
    )

    figura = go.Figure()

    figura.add_trace(
        go.Bar(
            x=datos_probabilidades["horizonte"],
            y=datos_probabilidades["probabilidad"],
            name="Probabilidad estimada",
        )
    )

    figura.add_trace(
        go.Scatter(
            x=datos_probabilidades["horizonte"],
            y=datos_probabilidades["umbral"],
            mode="markers+lines",
            name="Umbral conservador",
        )
    )

    figura.update_layout(
        title="Probabilidad estimada vs. umbral de decisión",
        xaxis_title="Horizonte",
        yaxis_title="Probabilidad",
        yaxis={"range": [0, 1]},
        height=420,
    )

    return figura


def mostrar_recomendacion(resultado_recomendacion):
    """
    Muestra visualmente la recomendación generada.

    Args:
        resultado_recomendacion: Objeto con nivel, recomendación e interpretación.
    """
    nivel_senal = resultado_recomendacion.nivel_senal

    if nivel_senal == "Oportunidad fuerte":
        st.success(f"### {nivel_senal}")
    elif "Oportunidad potencial" in nivel_senal:
        st.warning(f"### {nivel_senal}")
    elif nivel_senal == "Monitorear":
        st.info(f"### {nivel_senal}")
    else:
        st.error(f"### {nivel_senal}")

    st.write(f"**Recomendación:** {resultado_recomendacion.recomendacion}")
    st.write(resultado_recomendacion.interpretacion)


def mostrar_alerta_actualizacion(ultima_fecha):
    """
    Muestra advertencia si el CSV no parece estar actualizado.

    Args:
        ultima_fecha: Fecha más reciente disponible en el archivo cargado.
    """
    fecha_actual = pd.Timestamp.today().normalize()
    dias_desde_ultimo_dato = (fecha_actual - ultima_fecha.normalize()).days

    if dias_desde_ultimo_dato > 7:
        st.warning(
            "El último dato del archivo tiene más de 7 días de antigüedad. "
            f"Última fecha disponible: {ultima_fecha.date()}. "
            "Para una decisión real, se recomienda cargar datos más recientes."
        )
    else:
        st.caption(
            f"Última fecha disponible en el archivo: {ultima_fecha.date()}."
        )


def main():
    """
    Ejecuta la aplicación Streamlit.
    """
    st.title("📈 Sistema de apoyo para inversión en ETF VOO")

    st.markdown(
        """
        Esta aplicación carga un archivo histórico del ETF VOO, calcula variables
        financieras derivadas y utiliza modelos de aprendizaje de máquina para
        emitir una señal conservadora de apoyo a la decisión.

        El sistema no predice el precio exacto futuro. Su salida corresponde a
        una probabilidad estimada de oportunidad bajo los criterios definidos en
        el proyecto académico.
        """
    )

    with st.sidebar:
        st.header("Carga de datos")
        archivo_csv = st.file_uploader(
            "Sube el archivo CSV histórico del ETF",
            type=["csv"],
        )

        st.divider()

        st.caption(
            "Formato esperado: columnas Date, Price, Open, High, Low, Vol. y Change %."
        )

    try:
        artefactos = cargar_artefactos_modelo()
    except FileNotFoundError:
        st.error(
            "No se encontraron los modelos finales. Verifica que la carpeta "
            "`code/models` contenga los archivos `.joblib` y la metadata."
        )
        st.stop()

    if archivo_csv is None:
        st.info("Carga un archivo CSV desde el menú lateral para generar la recomendación.")
        st.stop()

    try:
        datos_voo = cargar_y_limpiar_datos(archivo_csv)
        datos_features = construir_variables_predictoras(datos_voo)
        ultima_observacion, x_ultima_observacion = obtener_ultima_observacion_modelo(
            datos_features
        )
    except Exception as error:
        st.error("No fue posible procesar el archivo cargado.")
        st.exception(error)
        st.stop()

    metadata = artefactos["metadata"]

    modelo_principal = artefactos["modelo_principal"]
    modelo_apoyo = artefactos["modelo_apoyo"]

    umbral_20d = metadata["modelo_principal"]["umbral_decision"]
    umbral_15d = metadata["modelo_apoyo"]["umbral_decision"]

    probabilidad_20d = obtener_probabilidad_clase_positiva(
        modelo=modelo_principal,
        x_datos=x_ultima_observacion,
    )

    probabilidad_15d = obtener_probabilidad_clase_positiva(
        modelo=modelo_apoyo,
        x_datos=x_ultima_observacion,
    )

    resultado_recomendacion = generar_recomendacion_inversion(
        probabilidad_15d=probabilidad_15d,
        probabilidad_20d=probabilidad_20d,
        umbral_15d=umbral_15d,
        umbral_20d=umbral_20d,
    )

    ultima_fecha = ultima_observacion["fecha"].iloc[0]
    ultimo_precio = ultima_observacion["precio_cierre"].iloc[0]

    mostrar_alerta_actualizacion(ultima_fecha)

    col_metrica_1, col_metrica_2, col_metrica_3 = st.columns(3)

    col_metrica_1.metric(
        label="Último precio de cierre",
        value=f"{ultimo_precio:,.2f} USD",
    )

    col_metrica_2.metric(
        label="Probabilidad oportunidad 15 días",
        value=f"{probabilidad_15d:.2%}",
        help=f"Umbral conservador: {umbral_15d:.2%}",
    )

    col_metrica_3.metric(
        label="Probabilidad oportunidad 20 días",
        value=f"{probabilidad_20d:.2%}",
        help=f"Umbral conservador: {umbral_20d:.2%}",
    )

    st.divider()

    col_grafica, col_recomendacion = st.columns([2, 1])

    with col_grafica:
        figura_precio = construir_grafica_precio(datos_features)
        st.plotly_chart(figura_precio, use_container_width=True)

    with col_recomendacion:
        mostrar_recomendacion(resultado_recomendacion)

        st.markdown(
            """
            **Lectura correcta:**  
            Una señal positiva no significa certeza de ganancia. Significa que,
            según el modelo, el escenario supera el nivel mínimo de confianza
            definido durante el proyecto.
            """
        )

    st.divider()

    figura_probabilidades = construir_grafica_probabilidades(
        probabilidad_15d=probabilidad_15d,
        probabilidad_20d=probabilidad_20d,
        umbral_15d=umbral_15d,
        umbral_20d=umbral_20d,
    )

    st.plotly_chart(figura_probabilidades, use_container_width=True)

    with st.expander("Ver detalle técnico"):
        st.write("### Configuración del modelo principal")
        st.json(metadata["modelo_principal"])

        st.write("### Configuración del modelo de apoyo")
        st.json(metadata["modelo_apoyo"])

        st.write("### Última observación usada para predicción")
        st.dataframe(
            ultima_observacion[
                [
                    "fecha",
                    "precio_cierre",
                    "precio_apertura",
                    "precio_maximo",
                    "precio_minimo",
                    "volumen",
                    "cambio_porcentual",
                ]
            ],
            use_container_width=True,
        )

    st.caption(
        "Este sistema fue desarrollado con fines académicos. No constituye "
        "asesoría financiera ni una recomendación profesional de inversión."
    )


if __name__ == "__main__":
    main()