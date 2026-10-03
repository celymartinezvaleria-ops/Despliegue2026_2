import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="wide")
st.title("Predicción de Aprobación de Curso 🎓")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones utilizando un modelo optimizado de Bagging Regressor.")

# 1. Cargar artefactos necesarios de forma segura
@st.cache_resource
def load_artifacts():
    try:
        columnas_one_hot = joblib.load('/content/one_hot_columns.joblib')
        scaler = joblib.load('/content/min_max_scaler.joblib')
        model = joblib.load('/content/bagging_optimizado.joblib')
        return columnas_one_hot, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos serializados (.joblib): {e}")
        return None, None, None

columnas_one_hot, scaler, model = load_artifacts()

if columnas_one_hot and scaler and model:
    st.header("Método de Entrada de Datos")
    opcion = st.radio("Selecciona cómo deseas ingresar los datos:", ["Subir archivo de Excel (.xlsx)", "Ingresar un estudiante manualmente"])

    if opcion == "Subir archivo de Excel (.xlsx)":
        st.subheader("Predicción en Lote")
        uploaded_file = st.file_uploader("Carga tu archivo Excel (.xlsx)", type=["xlsx"])
        
        if uploaded_file is not None:
            try:
                # Leer el archivo excel
                df_excel = pd.read_excel(uploaded_file)
                st.write("**Vista previa de los datos cargados:**")
                st.dataframe(df_excel.head())
                
                # Validar columnas requeridas
                if 'Felder' in df_excel.columns and 'Examen_admisión' in df_excel.columns:
                    df_procesado = df_excel.copy()
                    
                    # Aplicar codificación One-Hot manual de acuerdo a las columnas del modelo
                    for col in columnas_one_hot:
                        if col.startswith('Felder_'):
                            categoria = col.replace('Felder_', '')
                            df_procesado[col] = df_procesado['Felder'].apply(lambda x: 1.0 if str(x).strip().lower() == str(categoria).strip().lower() else 0.0)
                    
                    # Escalar variable Examen_admisión
                    df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
                    
                    # Seleccionar únicamente las columnas en el orden estricto que requiere el modelo
                    columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
                    df_procesado_final = df_procesado[columnas_finales]
                    
                    # Predicciones
                    predicciones = model.predict(df_procesado_final)
                    
                    # Adjuntar resultados
                    df_excel['Nota_final_estimada'] = predicciones
                    df_excel['Aprobo_estimado'] = df_excel['Nota_final_estimada'].apply(lambda x: 'si' if x >= 3.0 else 'no')
                    
                    st.success("¡Predicciones en lote completadas con éxito!")
                    st.subheader("Resultados de Predicción")
                    st.dataframe(df_excel)
                    
                else:
                    st.error("El archivo cargado debe contener al menos las columnas 'Felder' y 'Examen_admisión'.")
                    
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo Excel: {e}")
                
    else:
        st.subheader("Predicción Manual de Estudiante")
        
        # Extraer categorías para el selector basándonos en las columnas One-Hot guardadas
        categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]
        if not categorias_felder:
            categorias_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
            
        felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", categorias_felder)
        examen_input = st.slider("Nota obtenida en el Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)
        
        if st.button("Calcular Predicción"):
            try:
                df_input = pd.DataFrame([{'Felder': felder_input, 'Examen_admisión': examen_input}])
                df_procesado = df_input.copy()
                
                # One-Hot codificación manual
                for col in columnas_one_hot:
                    if col.startswith('Felder_'):
                        categoria = col.replace('Felder_', '')
                        df_procesado[col] = 1.0 if felder_input == categoria else 0.0
                        
                # Escalamiento de la admisión
                df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])[0][0]
                
                # Ordenar columnas
                columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
                df_procesado_final = df_procesado[columnas_finales]
                
                # Realizar inferencia
                prediccion = model.predict(df_procesado_final)[0]
                
                st.subheader("Resultado de Inferencia Individual")
                st.metric(label="Nota Final Estimada", value=f"{prediccion:.3f}")
                if prediccion >= 3.0:
                    st.success("¡El modelo estima que el estudiante APROBARÁ la materia!")
                else:
                    st.warning("El modelo estima que el estudiante no alcanzará la nota aprobatoria.")
                    
            except Exception as e:
                st.error(f"Ocurrió un error durante el cálculo de la predicción: {e}")
else:
    st.warning("Los artefactos necesarios no se encuentran disponibles para ejecutar la aplicación.")
