from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)


model = tf.keras.models.load_model("WasteClassificationModel.h5")

labels = ['Aluminium', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood']

label_to_category = {
    "Aluminium": "Non-Biodegradable",
    "Carton": "Biodegradable",
    "Other Plastics": "Non-Biodegradable",
    "Organic Waste": "Biodegradable",
    "Glass": "Non-Biodegradable",
    "Plastic": "Non-Biodegradable",
    "Paper and Cardboard": "Biodegradable",
    "Textiles": "Non-Biodegradable",
    "Wood": "Non-Biodegradable"
}

non_biodegradable_recyclable = ["Aluminium", "Glass"]
non_biodegradable_non_recyclable = ["Other Plastics", "Plastic", "Textiles", "Wood"]

def classify_waste(predicted_label):
    if predicted_label in non_biodegradable_recyclable:
        return "Non-Biodegradable (Recyclable)"
    elif predicted_label in non_biodegradable_non_recyclable:
        return "Non-Biodegradable (Non-Recyclable)"
    else:
        return "Unclassified"

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the image from the request
    image = request.files['image']
    image = Image.open(image)
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))

    predicted_label_index = np.argmax(prediction)

    predicted_label = labels[np.argmax(prediction)]
    print(predicted_label)

    waste_category = classify_waste(predicted_label)

    
    result = {
            "class": waste_category,
            "confidence": float(np.max(prediction))
        }


    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)