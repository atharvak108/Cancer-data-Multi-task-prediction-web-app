from keras.models import load_model

try:
    model = load_model('./unet_model.h5')
except OSError as e:
    print(f"Error loading model: {e}")