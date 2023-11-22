import gradio as gr

from transformers import pipeline

# https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
pipe = pipeline(model="mistralai/Mistral-7B-v0.1")

demo = gr.Interface.from_pipeline(pipe)
demo.launch()
