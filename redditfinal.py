import requests
import pandas as pd
from datetime import datetime, timezone
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# --- Lista de URLs de Reddit en formato .json ---
urls = [
    "https://www.reddit.com/r/crashbandicoot/comments/j383r0/crash_bandicoot_4_its_about_time_review_thread/.json",
    "https://www.reddit.com/r/Games/comments/11tjv4i/resident_evil_4_2023_remake_review_thread/.json",
    "https://www.reddit.com/r/patientgamers/comments/rjf9mz/alan_wake_is_a_masterpiece/.json",
    "https://www.reddit.com/r/Games/comments/1lq0ld/outlast_review_thread/.json",
    "https://www.reddit.com/r/patientgamers/comments/1f6bev4/alien_isolation_the_good_the_bad_and_that/.json",
    "https://www.reddit.com/r/patientgamers/comments/17g54k8/amnesia_rebirth_is_a_remarkable_but_deeply_flawed/.json",
    "https://www.reddit.com/r/Games/comments/10lv6tw/dead_space_remake_review_thread/.json",

]

# Diccionario de URL a nombre amigable
url_a_nombre = {
    urls[0]: "Crash Bandicoot",
    urls[1]: "Resident Evil 4",
    urls[2]: "Alan Wake",
    urls[3]: "Outlast",
    urls[4]: "Alien Isolation",
    urls[5]: "Amnesia Rebirth",
    urls[6]: "Dead Space Remake",

}

headers = {"User-Agent": "Mozilla/5.0"}
comentarios = []

# Función recursiva para extraer comentarios
def extraer_comentarios(lista_comentarios, nivel, url_base):
    for item in lista_comentarios:
        if item["kind"] == "t1":
            datos = item["data"]
            cuerpo = datos.get("body", "")
            autor = datos.get("author", "[deleted]")
            timestamp = datos.get("created_utc", 0)
            fecha = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            score = datos.get("score", 0)
            id_comentario = datos.get("id", "")
            permalink = f"https://www.reddit.com{datos.get('permalink', '')}"

            comentarios.append({
                "Post URL": url_base.replace(".json", ""),
                "Fecha UTC": fecha,
                "Autor": autor,
                "Comentario": cuerpo,
                "Puntaje": score,
                "Nivel (anidamiento)": nivel,
                "ID Comentario": id_comentario,
                "Enlace Comentario": permalink
            })

            respuestas = datos.get("replies")
            if isinstance(respuestas, dict):
                hijos = respuestas["data"]["children"]
                extraer_comentarios(hijos, nivel+1, url_base)

# Procesar URLs
for url in urls:
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        comentarios_principales = data[1]["data"]["children"]
        extraer_comentarios(comentarios_principales, nivel=1, url_base=url)
        print(f"✅ Procesado: {url}")
    except Exception as e:
        print(f"❌ Error con {url}: {e}")

# Crear y limpiar DataFrame
df = pd.DataFrame(comentarios)
df['Juego'] = df['Post URL'].map(lambda url: url_a_nombre.get(url + ".json", "Otro"))
df['Fecha UTC'] = pd.to_datetime(df['Fecha UTC'], errors='coerce')
df['Puntaje'] = pd.to_numeric(df['Puntaje'], errors='coerce').fillna(0).astype(int)
df['Comentario'] = df['Comentario'].str.strip().str.lower().str.replace(r'[^\w\s]', '', regex=True)
df['Autor'] = df['Autor'].str.strip()
df = df[df['Comentario'].notnull() & (df['Comentario'] != "")]
df = df.drop_duplicates(subset='ID Comentario')
df['Fecha'] = df['Fecha UTC'].dt.date
df['Longitud Comentario'] = df['Comentario'].apply(len)
df['Cantidad Palabras'] = df['Comentario'].apply(lambda texto: len(texto.split()))
df['Polaridad'] = df['Comentario'].apply(lambda texto: TextBlob(texto).sentiment.polarity)

# Clasificación de sentimiento
def obtener_sentimiento(p):
    if p > 0.1:
        return "Positivo"
    elif p < -0.1:
        return "Negativo"
    else:
        return "Neutro"

df['Sentimiento'] = df['Polaridad'].apply(obtener_sentimiento)

# Guardar archivo Excel
ruta_salida = r"C:/Users/aaram/Documents/5to semestre/webscraping/reddit_comentarios_consolidados2.xlsx"
df.to_excel(ruta_salida, index=False)
print(f"✅ Archivo guardado: {ruta_salida}")

# Gráfico: Cantidad de Reseñas por Post
plt.figure(figsize=(10,6))
df['Juego'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Cantidad de Reseñas por Juego")
plt.xlabel("Juego")
plt.ylabel("Cantidad de Comentarios")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico: Distribución de Calificaciones
plt.figure(figsize=(8,5))
plt.hist(df['Puntaje'], bins=30, color='orange', edgecolor='black')
plt.title("Distribución de Calificaciones")
plt.xlabel("Puntaje")
plt.ylabel("Cantidad de Comentarios")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico: Polaridad Promedio por Juego
polaridad_promedio = df.groupby('Juego')['Polaridad'].mean()
plt.figure(figsize=(10,6))
polaridad_promedio.plot(kind='bar', color='seagreen')
plt.title("Polaridad Promedio por Juego")
plt.xlabel("Juego")
plt.ylabel("Polaridad Promedio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Violin plot: Distribución de Polaridad por Juego
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x='Juego', y='Polaridad', palette='Set2')
plt.title('Distribución de Polaridad por Juego')
plt.xlabel('Juego')
plt.ylabel('Polaridad del Comentario')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot: Polaridad por Post URL
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Juego', y='Polaridad', palette='coolwarm')
plt.title('Distribución de Polaridad de Comentarios por Juego')
plt.xlabel('Juego')
plt.ylabel('Polaridad del Comentario')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Contar número de comentarios por Juego y Sentimiento
conteo_sentimientos = df.groupby(['Juego', 'Sentimiento']).size().reset_index(name='Cantidad Comentarios')

# Crear gráfico de barras agrupadas
plt.figure(figsize=(12,6))
sns.barplot(data=conteo_sentimientos, x='Juego', y='Cantidad Comentarios', hue='Sentimiento', palette='Set1')

plt.title('Cantidad de Comentarios por Juego y Sentimiento')
plt.xlabel('Juego')
plt.ylabel('Número de Comentarios')
plt.xticks(rotation=45)
plt.legend(title='Sentimiento')
plt.tight_layout()
plt.show()

#correlacion entre variables
import matplotlib.pyplot as plt

# Seleccionar solo columnas numéricas relevantes
variables_numericas = df[['Puntaje', 'Longitud Comentario', 'Cantidad Palabras', 'Polaridad']]

# Calcular la matriz de correlación
correlaciones = variables_numericas.corr()

# Graficar mapa de calor
plt.figure(figsize=(8,6))
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', center=0)
plt.title('Mapa de Correlación entre Variables Numéricas')
plt.show()

#diagrama de dispersión

# Scatterplot: Cantidad de Palabras vs Polaridad
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Cantidad Palabras', y='Polaridad', hue='Juego', alpha=0.6)
plt.title('Relación entre Cantidad de Palabras y Polaridad')
plt.xlabel('Cantidad de Palabras')
plt.ylabel('Polaridad del Comentario')
plt.legend(title='Juego', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Crear una matriz de correlación con solo dos columnas
correlacion_polaridad_palabras = df[['Polaridad', 'Cantidad Palabras']].corr()

# Graficar el heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlacion_polaridad_palabras, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlación entre Polaridad y Cantidad de Palabras")
plt.tight_layout()
plt.show()

#dispersion 
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Cantidad Palabras', y='Polaridad', alpha=0.5)
plt.title("Relación entre Cantidad de Palabras y Polaridad")
plt.xlabel("Cantidad de Palabras")
plt.ylabel("Polaridad del Comentario")
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np



# Asegúrate de que no haya valores nulos
df = df.dropna(subset=['Cantidad Palabras', 'Polaridad'])

# Variables para la regresión
X = df[['Cantidad Palabras']]  # Necesita estar en formato 2D
y = df['Polaridad']

# Crear y ajustar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para la línea de regresión
x_vals = np.linspace(df['Cantidad Palabras'].min(), df['Cantidad Palabras'].max(), 100).reshape(-1, 1)
y_pred = modelo.predict(x_vals)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(df['Cantidad Palabras'], df['Polaridad'], alpha=0.5, label='Datos reales')
plt.plot(x_vals, y_pred, color='red', label='Línea de regresión')
plt.xlabel('Cantidad de Palabras')
plt.ylabel('Polaridad del Comentario')
plt.title('Regresión Lineal: Cantidad de Palabras vs Polaridad')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar coeficiente y R²
print(f"Coeficiente (pendiente): {modelo.coef_[0]:.4f}")
print(f"Intercepto: {modelo.intercept_:.4f}")
print(f"R² (coeficiente de determinación): {modelo.score(X, y):.4f}")

#¿Cuál es el producto mejor valorado según la polaridad de sus comentarios?

#¿Existe alguna relación entre el número de palabras promedio de los comentarios y la calificación del producto?

#¿Cuál de los productos ha tenido más ventas en la última semana?

# --------------------- Posibles análisis adicionales ---------------------

# ¿Cuál es el producto mejor valorado según polaridad?
juego_mejor_valorado = df.groupby("Juego")["Polaridad"].mean().sort_values(ascending=False).head(1)
print("\n🏆 Juego mejor valorado según polaridad promedio:")
print(juego_mejor_valorado)

# ¿Existe relación entre número de palabras y puntaje?
correlacion_palabras_puntaje = df['Cantidad Palabras'].corr(df['Puntaje'])
print(f"\n🧮 Correlación entre Cantidad de Palabras y Puntaje: {correlacion_palabras_puntaje:.4f}")

# Diagrama de dispersión de palabras vs puntaje
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Cantidad Palabras', y='Puntaje', hue='Juego', alpha=0.6)
plt.title("Relación entre Cantidad de Palabras y Puntaje")
plt.xlabel("Cantidad de Palabras")
plt.ylabel("Puntaje del Comentario")
plt.grid(True)
plt.tight_layout()
plt.show()


#nivel de Nivel (anidamiento) con puntaje 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------- SCATTERPLOT con regresión global ----------
sns.lmplot(df, x='Nivel (anidamiento)', y='Puntaje', palette='Set2', height=6, aspect=1.5)
plt.title('Relación entre Nivel (anidamiento) y Puntaje')
plt.xlabel('Nivel (anidamiento)')
plt.ylabel('Puntaje')
plt.tight_layout()
plt.show()

# ---------- SCATTERPLOT con regresión por JUEGO ----------
sns.lmplot(df, x='Nivel (anidamiento)', y='Puntaje', hue='Juego', palette='muted', height=6, aspect=1.5)
plt.title('Regresión por Juego: Anidamiento vs Puntaje')
plt.xlabel('Anidamiento')
plt.ylabel('Puntaje')
plt.tight_layout()
plt.show()

# ---------- BOXPLOT del Puntaje según Anidamiento ----------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Nivel (anidamiento)', y='Puntaje', palette='coolwarm')
plt.title('Distribución de Puntaje por Nivel de Anidamiento')
plt.tight_layout()
plt.show()

# ---------- VIOLINPLOT del Puntaje según Anidamiento ----------
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Nivel (anidamiento)', y='Puntaje', palette='Set2')
plt.title('Distribución de Puntaje por Nivel de Anidamiento')
plt.tight_layout()
plt.show()

# ---------- REGRESIÓN GLOBAL ----------
X = df[['Nivel (anidamiento)']]
y = df['Puntaje']
modelo = LinearRegression()
modelo.fit(X, y)
r2 = r2_score(y, modelo.predict(X))

print("\n--- Regresión lineal global (Anidamiento vs Puntaje) ---")
print(f"Coeficiente (pendiente): {round(modelo.coef_[0], 4)}")
print(f"Intercepto: {round(modelo.intercept_, 4)}")
print(f"R² (coeficiente de determinación): {round(r2, 4)}")

# ---------- REGRESIÓN POR JUEGO ----------
print("\n--- Regresión lineal por juego ---")
for juego in df['Juego'].unique():
    sub_df = df[df['Juego'] == juego]
    X_sub = sub_df[['Nivel (anidamiento)']]
    y_sub = sub_df['Puntaje']
    
    modelo_juego = LinearRegression()
    modelo_juego.fit(X_sub, y_sub)
    r2_sub = r2_score(y_sub, modelo_juego.predict(X_sub))

    print(f"\nJuego: {juego}")
    print(f"  Coeficiente: {round(modelo_juego.coef_[0], 4)}")
    print(f"  Intercepto: {round(modelo_juego.intercept_, 4)}")
    print(f"  R²: {round(r2_sub, 4)}")


# analis de fechas y puntaje 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# --- Asegúrate de tener el DataFrame cargado como df y con la columna 'created_utc' ---
# Convertir la columna de timestamp a datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Asegúrate de que la columna está en formato datetime
df['Fecha UTC'] = pd.to_datetime(df['Fecha UTC'])

# Extraer componentes de fecha
df['Año'] = df['Fecha UTC'].dt.year
df['Mes'] = df['Fecha UTC'].dt.month
df['Día'] = df['Fecha UTC'].dt.day
df['Hora'] = df['Fecha UTC'].dt.hour

# Análisis de correlación
print("Correlación entre Año y Puntaje:", df['Año'].corr(df['Puntaje']).round(4))
print("Correlación entre Mes y Puntaje:", df['Mes'].corr(df['Puntaje']).round(4))
print("Correlación entre Día y Puntaje:", df['Día'].corr(df['Puntaje']).round(4))
print("Correlación entre Hora y Puntaje:", df['Hora'].corr(df['Puntaje']).round(4))

# Gráficos
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(data=df, x='Año', y='Puntaje', ax=axs[0,0])
sns.boxplot(data=df, x='Mes', y='Puntaje', ax=axs[0,1])
sns.boxplot(data=df, x='Día', y='Puntaje', ax=axs[1,0])
sns.boxplot(data=df, x='Hora', y='Puntaje', ax=axs[1,1])
fig.suptitle("Distribución del Puntaje según componentes de Fecha UTC", fontsize=16)
plt.tight_layout()
plt.show()
