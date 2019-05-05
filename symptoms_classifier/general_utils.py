# save trained model
def save_model_json(model,output_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    return 0

# load json and create model
def load_model_json(json_file):
    from keras.models import model_from_json
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model & compile model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    print("Loaded model from disk")

    #score_saved_model = loaded_model.evaluate(x_test, y_test, verbose=0)
    return loaded_model