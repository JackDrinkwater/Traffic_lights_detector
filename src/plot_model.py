from make_model import make_model, custom_loss
import pydot
from tensorflow.keras.utils import plot_model

model = make_model()
plot_model(model, show_shapes=True)