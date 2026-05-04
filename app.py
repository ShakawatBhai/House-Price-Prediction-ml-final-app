import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model.pkl")

def predict(medinc, houseage, rooms, bedrooms, population, occupancy, lat, lon):
    df = pd.DataFrame([{
        'MedInc': medinc,
        'HouseAge': houseage,
        'AveRooms': rooms,
        'AveBedrms': bedrooms,
        'Population': population,
        'AveOccup': occupancy,
        'Latitude': lat,
        'Longitude': lon
    }])
    
    return model.predict(df)[0]

with gr.Blocks() as demo:
    gr.Markdown("# House Price Prediction")

    inputs = [
        gr.Number(label="MedInc"),
        gr.Number(label="HouseAge"),
        gr.Number(label="AveRooms"),
        gr.Number(label="AveBedrms"),
        gr.Number(label="Population"),
        gr.Number(label="AveOccup"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude")
    ]

    output = gr.Textbox()

    btn = gr.Button("Predict")

    btn.click(predict, inputs=inputs, outputs=output)

demo.launch()
