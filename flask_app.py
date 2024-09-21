from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io

# Initialize Flask app
flask_app = Flask(__name__)
CORS(flask_app)

# Load the TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Function to extract text from individual word images using TrOCR
def extract_text(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text

# Define a POST endpoint to receive multiple image blobs
@flask_app.route('/extract', methods=['POST'])
def api_extract_text():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []
    for file in request.files.getlist('files'):
        img_data = file.read()
        extracted_text = extract_text(img_data)
        results.append(extracted_text)

    return jsonify({'texts': results})

if __name__ == "__main__":
    flask_app.run(port=5001, host='0.0.0.0')
