# Importar bibliotecas necesarias
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import wikipedia

import streamlit as st

# Función para obtener atracciones reales del destino
def obtener_atracciones(destino):
    try:
        resumen = wikipedia.summary(destino, sentences=5)
        return resumen
    except:
        return "No real data found for this destination."

# Función para limpiar el itinerario
def limpiar_itinerario(texto):
    inicio = texto.rfind("Day 1")
    if inicio == -1:
        inicio = texto.rfind("Día 1")
    if inicio != -1:
        texto = texto[inicio:]
    lineas_limpias = []
    for linea in texto.split("\n"):
        linea_strip = linea.strip()
        if not linea_strip:
            continue
        if "include activities" in linea_strip.lower():
            continue
        if "organize the response" in linea_strip.lower():
            continue
        if "do not repeat" in linea_strip.lower():
            continue
        lineas_limpias.append(linea_strip)
    return "\n".join(lineas_limpias)

# Cargar modelo Falcon
@st.cache_resource
def cargar_modelo():
    modelo_nombre = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_nombre,
        device_map="auto",
        torch_dtype=torch.float16
    )
    generator = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizer,
        max_new_tokens=400,
        temperature=0.7
    )
    return generator

generator = cargar_modelo()

# Interfaz de usuario
st.title("Travel Planner")

destino = st.text_input("Enter your travel destination:")
dias = st.number_input("Enter the trip duration (in days):", min_value=1, step=1)
presupuesto = st.text_input("Enter your approximate budget in euros:")
intereses = st.text_input("Enter your interests (e.g., culture, gastronomy, nature):")

if st.button("Generate Itinerary"):
    if destino and dias and presupuesto and intereses:
        st.write("Getting real destination information...")
        info_real = obtener_atracciones(destino)

        # Prompt completamente en inglés para mejor eficacia
        prompt = f"""
        You are a travel planner.

        Real context about {destino}:
        {info_real}

        Example of format:
        Day 1: Explore the main square, visit the cathedral, and enjoy a traditional lunch.
        Day 2: Visit local markets, explore a museum, and relax in a park.

        Now, create a {dias}-day travel itinerary for {destino} with a budget of {presupuesto} euros.
        Include activities related to: {intereses}.
        Organize the response day by day with short and specific activities.
        Do not repeat the instruction, just provide the itinerary.
        """

        # Generar y limpiar itinerario
        resultado = generator(prompt)[0]["generated_text"]
        itinerario_limpio = limpiar_itinerario(resultado)

        st.subheader("📅 Suggested Travel Itinerary")
        st.text(itinerario_limpio)
    else:
        st.warning("Please fill in all fields before generating the itinerary.")
