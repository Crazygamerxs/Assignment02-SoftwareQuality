from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiate the Flask app
app = Flask(__name__)

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route("/")
def main():
    return render_template("index.html")

# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if 'file' not in request.files:
            return render_template("result.html", err="No file part.")
        file = request.files['file']
        if file.filename == '':
            return render_template("result.html", err="No selected file.")
        if not allowed_file(file.filename):
            return render_template("result.html", err="File type not supported.")

        img = preprocess_img(file.stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))

    except (FileNotFoundError, OSError):
        error = "File cannot be processed."
        return render_template("result.html", err=error)

# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
