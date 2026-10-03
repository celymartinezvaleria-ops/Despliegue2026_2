import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# 1. Cargar artefactos necesarios de forma segura
@st.cache_resource
def load_artifacts():
    try:
        columnas_one_hot = joblib.load('/content/one_hot_columns.joblib')
        scaler = joblib.load('/content/min_max_scaler.joblib')
        model = joblib.load('/content/bagging_optimizado.joblib')
        return columnas_one_hot, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None

columnas_one_hot, scaler, model = load_artifacts()

if columnas_one_hot and scaler and model:
    st.header("Datos de Entrada")

    # Opción para permitir que el usuario suba su propio archivo Excel de pruebas
    uploaded_file = st.file_uploader("Sube un archivo de Excel (.xlsx) con datos de prueba (opcional)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Leer el archivo Excel cargado
            df_excel = pd.read_excel(uploaded_file)
            st.subheader("Datos del Archivo Cargado")
            st.dataframe(df_excel)

            if 'Felder' in df_excel.columns and 'Examen_admisión' in df_excel.columns:
                # Copia de trabajo para procesamiento
                df_procesado = df_excel.copy()

                # Aplicar codificación One-Hot manual
                for col in columnas_one_hot:
                    if col.startswith('Felder_'):
                        categoria = col.replace('Felder_', '')
                        df_procesado[col] = df_procesado['Felder'].apply(lambda x: 1.0 if x == categoria else 0.0)

                # Normalizar la variable Examen_admisión
                df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])

                # Reordenar las columnas conforme lo espera el modelo
                columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
                df_procesado_final = df_procesado[columnas_finales]

                # Realizar predicciones para todo el dataset
                predicciones = model.predict(df_procesado_final)

                # Añadir predicciones al DataFrame original
                df_excel['Nota_final_estimada'] = predicciones
                df_excel['Prediccion_aprobo'] = df_excel['Nota_final_estimada'].apply(lambda x: 'si' if x >= 3.0 else 'no')

                st.subheader("Resultados de las Predicciones")
                st.dataframe(df_excel)
            else:
                st.error("El archivo Excel debe contener las columnas 'Felder' y 'Examen_admisión'.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

    else:
        st.info("No se ha subido ningún archivo. Utiliza el formulario manual a continuación:")

        # Opciones para la variable Felder obtenidas de los datos reales
        categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]
        if not categorias_felder:
            categorias_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']

        # Entradas del formulario
        felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", options=categorias_felder)
        examen_input = st.slider("Nota de Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)

        if st.button("Realizar Predicción"):
            try:
                df_input = pd.DataFrame([{'Felder': felder_input, 'Examen_admisión': examen_input}])
                df_procesado = df_input.copy()

                # Aplicar codificación One-Hot manual
                for col in columnas_one_hot:
                    if col.startswith('Felder_'):
                        categoria = col.replace('Felder_', '')
                        df_procesado[col] = 1.0 if felder_input == categoria else 0.0

                # Aplicar el Min-Max Scaler cargado
                df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])[0][0]

                # Seleccionar y ordenar las columnas según las que espera el modelo
                columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
                df_procesado_final = df_procesado[columnas_finales]

                # Realizar la predicción con el modelo
                prediccion = model.predict(df_procesado_final)[0]

                st.success(f"### Nota Final Estimada: {prediccion:.3f}")
                if prediccion >= 3.0:
                    st.balloons()
                    st.success("¡El modelo estima que el estudiante APROBARÁ el curso!")
                else:
                    st.warning("El modelo estima que el estudiante no alcanzará la nota de aprobación.")

            except Exception as e:
                st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")
else:
    st.warning("Por favor, asegúrate de que los archivos 'one_hot_columns.joblib', 'min_max_scaler.joblib' y 'bagging_optimizado.joblib' se encuentren en la ruta /content/.")

# --- Código complementario para ejecutar localtunnel desde Colab ---
import urllib
import subprocess

# Ejecutar Streamlit en segundo plano
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

# Obtener IP pública de este entorno para la contraseña de localtunnel
print("Tu IP pública para usar en el campo 'Endpoint IP' de localtunnel es:")
pub_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
print(pub_ip)
print("\nHaz clic en el enlace de abajo cuando se genere, escribe la IP de arriba y presiona Submit:\n")

# Lanzar localtunnel en el puerto 8501
!npx localtunnel --port 8501
