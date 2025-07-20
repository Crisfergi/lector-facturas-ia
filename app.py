from flask import Flask, request, jsonify
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import fitz  # PyMuPDF
import io
import tempfile
import os

app = Flask(__name__)

# Cargar modelo DONUT
print("Cargando modelo Donut...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
print("Modelo cargado")

@app.route('/analizar', methods=['POST'])
def analizar():
    file = request.files.get('documento')
    if not file:
        return jsonify({'error': 'No se recibió ningún archivo'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        # Extraer primera página del PDF como imagen
        doc = fitz.open(tmp_path)
        pix = doc[0].get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preparar pregunta al modelo
        prompt = "<s_docvqa><s_question>¿Cuál es el NIT, número de factura, fecha y monto?</s_question><s_answer>"
        pixel_values = processor(image, return_tensors="pt").pixel_values
        input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids

        # Generar respuesta
        outputs = model.generate(pixel_values, decoder_input_ids=input_ids)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return jsonify({
            'resultado': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(tmp_path)

@app.route('/')
def home():
    return "Microservicio de análisis de documentos (Donut Model)", 200

if __name__ == '__main__':
    app.run()