import requests
import pandas as pd
from datetime import datetime, timezone

# --- Lista de URLs de Reddit en formato .json ---
urls = [
    "https://www.reddit.com/r/crashbandicoot/comments/j383r0/crash_bandicoot_4_its_about_time_review_thread/.json",
    "https://www.reddit.com/r/Games/comments/11tjv4i/resident_evil_4_2023_remake_review_thread/.json",
    "https://www.reddit.com/r/patientgamers/comments/rjf9mz/alan_wake_is_a_masterpiece/.json",
    "https://www.reddit.com/r/Games/comments/1lq0ld/outlast_review_thread/.json",
    "https://www.reddit.com/r/patientgamers/comments/1f6bev4/alien_isolation_the_good_the_bad_and_that/.json",
    "https://www.reddit.com/r/patientgamers/comments/17g54k8/amnesia_rebirth_is_a_remarkable_but_deeply_flawed/.json",
    "https://www.reddit.com/r/gaming/comments/gyc32/my_honest_review_of_my_first_few_hours_of_amnesia/.json",
    "https://www.reddit.com/r/HorrorGaming/comments/kv8g4z/visage_any_good/.json",
    #""https://www.reddit.com/r/Games/comments/1fvt8b9/silent_hill_2_review_thread/#:~:text=Faithful%20and%20very%20di
    #"""https://www.reddit.com/r/Games/comments/1fvt8b9/silent_hill_2_review_thread/#:~:text=Faithful%20and%20very%20different%2C%20this,great%20hope%20for%20the%20future."  # <- Agrega más si deseas
]

headers = {"User-Agent": "Mozilla/5.0"}
comentarios = []

# --- Función recursiva para extraer comentarios ---
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

# --- Procesar cada URL ---
for url in urls:
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        comentarios_principales = data[1]["data"]["children"]
        extraer_comentarios(comentarios_principales, nivel=1, url_base=url)
        print(f"✅ Procesado: {url}")
    except Exception as e:
        print(f"❌ Error con {url}: {e}")

# --- Exportar todo a un solo Excel ---
df = pd.DataFrame(comentarios)
ruta_salida = r"C:/Users/aaram/Documents/5to semestre/webscraping/reddit_comentarios_consolidados2.xlsx"
df.to_excel(ruta_salida, index=False)
print(f"✅ Archivo consolidado guardado en: {ruta_salida}")

#definir data frame Una vez extraída, almacénela en un DataFrame de Pandas.


# Crear el DataFrame desde la lista de diccionarios
df = pd.DataFrame(comentarios)

# Mostrar las primeras filas para verificar
print(df.head())



# Convertir columna de fecha a tipo datetime
df['Fecha UTC'] = pd.to_datetime(df['Fecha UTC'], errors='coerce')

# Asegurar que el puntaje es tipo entero
df['Puntaje'] = pd.to_numeric(df['Puntaje'], errors='coerce').fillna(0).astype(int)

# Limpiar espacios en strings de columnas de texto
df['Comentario'] = df['Comentario'].str.strip()
df['Autor'] = df['Autor'].str.strip()

# Eliminar filas sin comentarios (pueden ser eliminadas o revisadas)
df = df[df['Comentario'].notnull() & (df['Comentario'] != "")]

# Opción: eliminar duplicados por ID Comentario
df = df.drop_duplicates(subset='ID Comentario')

# Convertir a minúsculas, quitar caracteres especiales simples
df['Comentario'] = df['Comentario'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Día del comentario (puede servir para gráficos)
df['Fecha'] = df['Fecha UTC'].dt.date

# Longitud del comentario (útil para histogramas)
df['Longitud Comentario'] = df['Comentario'].apply(len)

print(df.head())
print(df.dtypes)

#4.Cree una columna adicional para guardar la polaridad o el sentimiento asociado a cada uno de los comentarios usando la librería TextBlob (Revise el ejemplo de RottenTomatoes).
   #pip install textblob
     # python -m textblob.download_corpora
from textblob import TextBlob

# sentimiento
def obtener_sentimiento(texto):
    analisis = TextBlob(texto)
    polaridad = analisis.sentiment.polarity  # entre -1 (negativo) y 1 (positivo)
    if polaridad > 0.1:
        return "Positivo"
    elif polaridad < -0.1:
        return "Negativo"
    else:
        return "Neutro"

# Aplicar la función a cada comentario
df['Sentimiento'] = df['Comentario'].apply(obtener_sentimiento)

#5. Cree una columna adicional para contabilizar el número de palabras por cada comentario.

# Contar la cantidad de palabras de cada comentario
df['Cantidad Palabras'] = df['Comentario'].apply(lambda texto: len(texto.split()))


#6. Finalmente, use Matplotlib para comparar la cantidad de calificaciones y reseñas de cada referencia, la distribución de las calificaciones y la polaridad promedio de los comentarios de cada referencia.
#pip install matplotlib

import matplotlib.pyplot as plt
from textblob import TextBlob

# Crear columna 'Polaridad' 
df['Polaridad'] = df['Comentario'].apply(lambda texto: TextBlob(texto).sentiment.polarity)


# Agrupar por Post URL y contar comentarios
cantidad_resenas = df['Post URL'].value_counts()

# Graficar

#rese;as y refernecias
plt.figure(figsize=(10,6))
cantidad_resenas.plot(kind='bar', color='skyblue')
plt.title("Cantidad de Reseñas por Post")
plt.xlabel("Referencia (Post URL)")
plt.ylabel("Cantidad de Comentarios")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#distribucion de puntajes
plt.figure(figsize=(8,5))
plt.hist(df['Puntaje'], bins=30, color='orange', edgecolor='black')
plt.title("Distribución de Calificaciones (Puntajes)")
plt.xlabel("Puntaje del Comentario")
plt.ylabel("Cantidad de Comentarios")
plt.grid(True)
plt.tight_layout()
plt.show()



#polaridad promedio  > 0.1: Sentimiento positivo. < -0.1: Sentimiento negativo. Entre -0.1 y 0.1: Neutro.


# Agrupar por Post URL y obtener polaridad promedio


polaridad_promedio = df.groupby('Post URL')['Polaridad'].mean()

# Graficar
plt.figure(figsize=(10,6))
polaridad_promedio.plot(kind='bar', color='seagreen')
plt.title("Polaridad Promedio por Post")
plt.xlabel("Referencia (Post URL)")
plt.ylabel("Polaridad Promedio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#otros graficos de interes
#pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Post URL', y='Polaridad', palette='coolwarm')

plt.title('Distribución de Polaridad de Comentarios por Post')
plt.xlabel('Referencia (Post URL)')
plt.ylabel('Polaridad del Comentario')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Post URL', y='Polaridad', palette='Set2')

plt.title('Distribución de Sentimiento (Polaridad) por Post')
plt.xlabel('Referencia (Post URL)')
plt.ylabel('Polaridad del Comentario')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()





#7. Analice el problema y defina qué tipo de gráfico y operaciones son más útiles para analizar este tipo de información.

#preguntas a responder>
#1. ¿Cuál es el producto mejor valorado según la polaridad de sus comentarios?

#¿Existe alguna relación entre el número de palabras promedio de los comentarios y la calificación del producto?

# ¿Cuál de los productos ha tenido más ventas en la última semana?

#¿Cuál es el lugar de donde más compran cada producto?