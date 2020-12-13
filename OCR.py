from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import cv2
import numpy as np

import GetFeatures
import ImageProcessing

def makeTrainingDataSet(train_dir, csv_name, mode):
    """
    Unused method to built training data.
    Originally, training creating a training dataset used individual images for each letters.
    However, upon testing the code, it seemed to achieve only low accuracies.
    Therefore, a new method of creating a training dataset was created,
    which is makeTrainingDataSetFromImage().
    Still, this function is kept as a form of artifact.
    """
    if mode == "w":
        header = "75,32,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,0,1,2,3,4,5,6,7,8,9,*,/,+,-,#,.,\",\",(,),:,=,@"
    with open(csv_name+".csv", mode) as f:
        if mode == "w":
            f.write(header + "\n")
        for ch_img in os.listdir(train_dir):
            img = ImageProcessing.getBinaryImage(train_dir + "/" + ch_img)
            features = GetFeatures.getFeaturesInStr(img, ch_img[:2])
            f.write(features)

def makeTrainingDataSetFromImage(img_set_path, csv_name, mode):
    """
    Function used to create dataset.
    """
    if mode == "w":
        header = "75,32,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,0,1,2,3,4,5,6,7,8,9,*,/,+,-,#,.,\",\",(,),:,=,@"
        with open(csv_name+".csv", mode) as f:
            f.write(header + "\n")    
    dex = "01,27,02,28,03,29,04,30,05,31,06,32,07,33,08,34,09,35,10,36,11,37,12,38,13,39,14,40,15,41,16,42,17,43,18,44,19,45,20,46,21,47,22,48,23,49,24,50,25,51,26,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74"
    for img_name in os.listdir(img_set_path):
        if img_name[:5] == "type1":
            dex = "01,52,02,51,03,50,04,49,05,48,06,47,07,46,08,45,09,44,10,43,11,42,12,41,13,40,14,39,15,38,16,37,17,36,18,35,19,34,20,33,21,32,22,31,23,30,24,29,25,28,26,27,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74"
        with open(csv_name+".csv", "a") as f:
            train_char = ImageProcessing.processImage(img_set_path + "/" + img_name)
            name = dex.split(",")
            i = 0
            for line in train_char:
                for char in line:
                    if np.count_nonzero(char):
                        features = GetFeatures.getFeaturesInStr(char, name[i])
                        i += 1
                        f.write(features)
            f.write("0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,00\n")
        dex = "01,27,02,28,03,29,04,30,05,31,06,32,07,33,08,34,09,35,10,36,11,37,12,38,13,39,14,40,15,41,16,42,17,43,18,44,19,45,20,46,21,47,22,48,23,49,24,50,25,51,26,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74"

def createTrainingDataSet(db, column_names, batch_size, csv_training_db):
    feature_names = column_names[:-1]
    label_name = column_names[-1]    
    train_dataset = tf.contrib.data.make_csv_dataset(csv_training_db, 
                                                          batch_size, 
                                                          column_names=column_names, 
                                                          label_name=label_name, 
                                                          num_epochs=1)
    features, labels = next(iter(train_dataset))
    return train_dataset

def createEmptyModel(nfeatures):
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(nfeatures,)),
      tf.keras.layers.Dense(200, activation=tf.nn.relu),#input_shape=(nfeatures,)),
      tf.keras.layers.Dense(200, activation=tf.nn.relu),
      # tf.keras.layers.Dense(300, activation=tf.nn.relu),
      # tf.keras.layers.Dense(300, activation=tf.nn.relu),
      tf.keras.layers.Dense(76) #, activation=tf.nn.softmax)
    ])
    return model

def createLossFunc(model, x, y):
    y_prime = model(x)
    #return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_prime)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_prime)
    
def createGradient(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = createLossFunc(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
def doTrainingLoop(num_epochs, optimizer, train_dataset, global_step, model):
    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
    
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = createGradient(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables), global_step)
    
            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
    
        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
    
        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

def performOCR(inputName):    
    # Check if training database has been made before. If it was not, make a new one.
    csv_training_db = "Training_fonts.csv"
    if not os.path.isfile(csv_training_db):
        makeTrainingDataSetFromImage("./Train_image_bulks", "Training_fonts", "w")
    
    # Set up the model and perform training loop.
    column_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14',  'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'character']
    train_dataset = createTrainingDataSet(csv_training_db, column_names, 16, csv_training_db)
    train_dataset = train_dataset.map(pack_features_vector)    
    model = createEmptyModel(32)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    global_step = tf.train.get_or_create_global_step()
    doTrainingLoop(200, optimizer, train_dataset, global_step, model)
    
    # Extract features from the input image.
    class_names = [" ", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
                   "P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g",
                   "h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
                   "0","1","2","3","4","5","6","7","8","9","*","/","+","-","#",".",",","(",")",":","=","@"]    
    processed = ImageProcessing.processImage(inputName)
    features = []
    for line in processed:
        line_f = []
        for char in line:
            if char.size != 0:
                line_f.append(GetFeatures.getFeatures(char))
        if line_f:
            features.append(line_f)

    with open("text_output.txt", "w") as f:
        for i in range(len(features)):
            predict_dataset = tf.convert_to_tensor(features[i])
            predictions = model(predict_dataset)
            line = ""
            for i, logits in enumerate(predictions):
                class_idx = tf.argmax(logits).numpy()
                p = tf.nn.softmax(logits)[class_idx]
                name = class_names[class_idx]
                line += name
            f.write(line+"\n")    

if __name__ == '__main__':  
    tf.enable_eager_execution()
    print("Enter the name of input file: ")
    inputName = input()
    performOCR(inputName)
