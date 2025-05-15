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
    "hhttps://www.reddit.com/r/silenthill/comments/1g8cyaq/silent_hill_2_remake_honest_critical_review_as_a/.json",
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
    urls[7]: "Silent Hill 2 Remake"
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


#¿Cuál es el producto mejor valorado según la polaridad de sus comentarios?

#¿Existe alguna relación entre el número de palabras promedio de los comentarios y la calificación del producto?

#¿Cuál de los productos ha tenido más ventas en la última semana?

#¿Cuál es el lugar de donde más compran cada producto?