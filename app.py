from flask import Flask, render_template, request
import os
from detector import predict_sentiment_analysis
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        if img.filename != '':
            filename = secure_filename(img.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            prediction = predict_sentiment_analysis(filepath)
            return render_template('index.html', filename=filename, prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
