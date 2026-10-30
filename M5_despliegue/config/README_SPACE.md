---
title: Tutor de Matemáticas
emoji: 📐
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Tutor de matemáticas · SI4006

Tutor que explica procedimientos de matemáticas escolares y, dada la foto de una
hoja, intenta señalar en qué paso está el error.

**Estado de la evaluación.** La revisión de fotos está en desarrollo: acierta
3 de 9 casos de desarrollo. Los 12 casos visuales de M4 **ya se gastaron** —tres
corridas exploratorias, 4/12, 1/12 y 2/12— y se conservan como **conjunto de
regresión**, no como evaluación ciega. La configuración actual **no se ha medido
con fotografías nuevas**. La interfaz lo indica en pantalla. No emite diagnósticos
médicos.

Construido sobre Qwen2.5-1.5B-Instruct + LoRA, con RAG sobre documentos del
colegio, herramientas de aritmética exacta y una vía visual con Qwen2-VL-2B.

## Variables de entorno

Ver `.env.example`. La única que es **secreto** es `WANDB_API_KEY`, y es
opcional: sin ella la aplicación funciona y la telemetría queda apagada.

`TUTOR_GRADIO_SHARE` debe quedar vacío en un Space: Hugging Face ya entrega la
URL pública. Esa variable se reserva para obtener un enlace temporal al ejecutar
la misma aplicación desde una sesión de Kaggle.

## Health check

La pestaña **Estado** expone versión, tiempo activo, dispositivo, adaptador
cargado y avisos.
