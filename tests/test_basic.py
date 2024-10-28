import pytest
from app import app as flask_app, allowed_file
from model import preprocess_img, predict_result  # Ensure model.py is in the same directory

@pytest.fixture
def test_client():
    """Create a temporary test client."""
    with flask_app.test_client() as client:
        yield client

def test_prediction_route_no_file(test_client):
    """Test prediction route with no file."""
    response = test_client.post('/prediction')
    assert response.status_code == 200
    assert b'No file part' in response.data

def test_allowed_file():
    """Test allowed file function."""
    assert allowed_file(r'test_images\0\Sign 0 (21).jpeg') is True
    assert allowed_file(r'test_images\0\Sign 0 (189).jpeg') is True
    assert allowed_file(r'test_images\0\Sign 0 (195).jpeg') is True
    assert allowed_file(r'test_images\0\Sign 0 (177).jpeg') is True
    assert allowed_file(r'D:\empty file.txt') is False

def test_preprocess_img_valid():
    """Test image preprocessing with a valid image."""
    img_path = r'test_images\0\Sign 0 (97).jpeg'
    img = preprocess_img(img_path)
    assert img.shape == (1, 224, 224, 3)

def test_preprocess_img_invalid():
    """Test image preprocessing with an invalid image."""
    with pytest.raises(OSError):
        preprocess_img('path/to/invalid/image.txt')

def test_predict_result():
    """Test the prediction result function."""
    img_path = r'test_images\0\Sign 0 (21).jpeg'
    img = preprocess_img(img_path)
    pred = predict_result(img)  # Call the function to get the prediction
    assert isinstance(pred, int)  # Check the type of pred

# Run tests with pytest
if __name__ == '__main__':
    pytest.main()
