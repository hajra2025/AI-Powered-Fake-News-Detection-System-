import re
import joblib
import gradio as gr

# Load the model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join(text.split())

# Prediction function
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "TRUE NEWS" if pred == 1 else "FAKE NEWS"

with gr.Blocks() as gui:
    gr.Markdown("<h1 style='color: orange; font-weight: bold;'>Fake News Detector</h1>", elem_id="heading")
    gr.Markdown("<p style='font-weight: bold;'>Paste your news article below and click Predict to see if it is TRUE or FAKE!</p>")
    news_input = gr.TextArea(label="News Text", placeholder="Paste news article here...", lines=15)
    output_box = gr.Textbox(label="Prediction Result")
    predict_button = gr.Button("Predict")
    predict_button.click(fn=predict_news, inputs=news_input, outputs=output_box)
# Launch locally
gui.launch(server_name="127.0.0.1", server_port=7860, share=False)