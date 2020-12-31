from data_gen import data_gen
from make_model import make_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

TOTAL_VALIDATE = 5000
TOTAL_TRAIN = 5000
BATCH_SIZE = 50
NUM_EPOCHS = 1

earlystop = EarlyStopping(patience=10)
lear_rate_reduc = ReduceLROnPlateau(monitor='val_loss',
                                    patience=2,
                                    verbose=1,
                                    factor=0.5,
                                    min_lr=0.00001)
callbacks = [earlystop, lear_rate_reduc]

model = make_model()

def train_model(model):
	history = model.train(data_gen(),
                          epochs=NUM_EPOCHS,
                          validation_data=validation_gen(),
                          validation_steps=TOTAL_VALIDATE // (BATCH_SIZE),
                          steps_per_epoch=TOTAL_TRAIN // (BATCH_SIZE),
                          callbacks=callbacks)
	model.save('models/model.h5')