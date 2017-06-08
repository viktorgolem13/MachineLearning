import os
import sys 
import tensorflow as tf
from models_master.models_master.slim.nets import inception_resnet_v2
from models_master.models_master.slim.preprocessing import inception_preprocessing
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import cluster
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib
from PIL import Image


def main():

    directory = sys.argv[1] #path to the directory which contains pictures to be clusterized

    convert_to_jpg(directory)

    list_of_paths = get_paths(directory)

    extracted_features, logits_extracted_features, prepool_extracted_features = extract_features(list_of_paths)

    min_n_clusters = 5
    max_n_clusters = 20

    n_clusters = determine_n_clusters(min_n_clusters, max_n_clusters, extracted_features)

    max_n_of_prepool_training_images = 400

    trained_classifier, predictions = train_and_predict(extracted_features, n_clusters)
    logits_trained_classifier, logits_predictions = train_and_predict(logits_extracted_features, n_clusters)
    prepool_trained_classifier, prepool_predictions = train_and_predict_prepool(prepool_extracted_features, n_clusters, max_n_of_prepool_training_images) 

    create_output_file("Mozgalo_result", predictions, directory)
    create_output_file("Mozgalo_result_logits", logits_predictions, directory)
    create_output_file_prepool("Mozgalo_result_prepool", prepool_predictions, directory)

    plt.hist(predictions, np.arange(0, n_clusters + 1)) #making of a histogram which shows how many pictures each cluster contains

    plt.show()

    #following three commands save the trained classifiers
    joblib.dump(trained_classifier, 'trainedKmeans.pkl')

    joblib.dump(logits_trained_classifier, 'trainedKmeansLogits.pkl') 

    joblib.dump(prepool_trained_classifier, 'trainedKmeansPrepool.pkl') 



#Function used to create jpg copies of non-jpg images in the given directory
def convert_to_jpg(directory):

    image_names = os.listdir(directory)

    for image_name in image_names:
        
        title, extension = image_name.split(".")
        new_title = title + ".jpg"

        if image_name != new_title:

            try:
                Image.open(directory + "\\" + image_name).save(directory + "\\" + new_title)
            except IOError:
                print ("Unable to convert " + extension + " file to jpg. " + "(" + image_name + ")")
                os.remove(directory + "\\" + new_title)
    
#Function used to get all paths to the jpg image files from the given directory
def get_paths(directory):

    image_names = os.listdir(directory)

    list_of_paths = []

    for image_name in image_names:

        if "jpg" in image_name:

            full_path = directory + "\\" + image_name

            list_of_paths.append(full_path)
    
    return list_of_paths

#Function used to extract features from the jpg image files using a convolutional neural network
def extract_features(list_of_paths):

    checkpoints_dir = "inception_resnet_v2_2016_08_30" #path to the directory where pretrained weights are saved (no need to train the neural network)
    slim = tf.contrib.slim
    
    image_width = 299
    image_height = 299

    #The following 4 lines are used to initialize TensorBoard 
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs" 
    logdir = "{}/run-{}/".format(root_logdir, now)
    step = 0

    string_image_placeholder = tf.placeholder(dtype = "string")

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):

        test_image = tf.image.decode_jpeg(string_image_placeholder, channels=3)

        #Data augmentation
        test_image2 = tf.image.flip_left_right(test_image)

        test_image1_2 = tf.image.random_brightness(test_image, 0.33)
        test_image2_2 = tf.image.random_brightness(test_image2, 0.33)

        test_image1_3 = tf.image.random_contrast(test_image, 0.5, 1)
        test_image2_3 = tf.image.random_contrast(test_image2, 0.5, 1)

        #Data preprocessing
        list_of_preprocessed_images = []

        preprocessed_image = inception_preprocessing.preprocess_image(test_image, image_height, image_width, is_training=False, fast_mode = True)
        preprocessed_image2 = inception_preprocessing.preprocess_image(test_image2, image_height, image_width, is_training=False, fast_mode = True)
        preprocessed_image1_2 = inception_preprocessing.preprocess_image(test_image1_2, image_height, image_width, is_training=False, fast_mode = True)
        preprocessed_image2_2 = inception_preprocessing.preprocess_image(test_image2_2, image_height, image_width, is_training=False, fast_mode = True)
        preprocessed_image1_3 = inception_preprocessing.preprocess_image(test_image1_3, image_height, image_width, is_training=False, fast_mode = True)
        preprocessed_image2_3 = inception_preprocessing.preprocess_image(test_image2_3, image_height, image_width, is_training=False, fast_mode = True)
                        
        list_of_preprocessed_images.append(preprocessed_image)
        list_of_preprocessed_images.append(preprocessed_image2)
        list_of_preprocessed_images.append(preprocessed_image1_2)
        list_of_preprocessed_images.append(preprocessed_image2_2)
        list_of_preprocessed_images.append(preprocessed_image1_3)
        list_of_preprocessed_images.append(preprocessed_image2_3)

        preprocessed_images = tf.stack(list_of_preprocessed_images)

        logits, end_points = inception_resnet_v2.inception_resnet_v2(preprocessed_images, num_classes=1001, is_training=False) #Images are passed on to the neural network, 
                                                                                                                            #logits - features obtained from the output layer
                                                                                                                            #end_points - features obtained from the hidden
                                                                                                                            #layers

        init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt'), slim.get_model_variables('InceptionResnetV2'))

        mse_summary = tf.summary.scalar('logits', tf.reduce_max(logits)) 
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        with tf.Session() as sess:

            init_fn(sess)

            extracted_features = []
            logits_extracted_features = []
            prepool_extracted_features = []

            for path in list_of_paths:

                test_image_string = tf.gfile.FastGFile(path, 'rb').read()

                summary_str, end_points2 = sess.run([mse_summary, end_points], feed_dict = {string_image_placeholder : test_image_string})


                step += 1
                file_writer.add_summary(summary_str, step)

                #List of the extracted features contains features obtained from the last hidden layer
                for end_point in end_points2['PreLogitsFlatten']:
                    extracted_features.append(end_point)

                #List of logits extracted features contains features obtained from the last layer
                for end_point in end_points2['Logits']:
                    logits_extracted_features.append(end_point)

                #List of prepool features contains features obtained from the next to last hidden layer
                var1 = end_points2['PrePool'].shape[1] 
                var2 = end_points2['PrePool'].shape[2] 
                var3 = end_points2['PrePool'].shape[3]
                num_of_prepool_features = var1 * var2 * var3
                

                #num_of_prepool_features = 98304 
                        
                reshaped_end_point = np.reshape(end_points2['PrePool'][0], num_of_prepool_features)

                prepool_extracted_features.append(reshaped_end_point)

            file_writer.close()

            return extracted_features, logits_extracted_features, prepool_extracted_features

#Function used to determine the optimal number of clusters.
#If the dataset differs from the original Mozgalo dataset, n_clusters vs inertia graph is plotted
#and the user is expected to choose the number of clusters using elbow method.
def determine_n_clusters(min_n_clusters, max_n_clusters, extracted_features):

    while True:

        is_the_original_dataset = input("Je li program pokrenut za \"glavni\" dataset? (Mozgalo dataset) - [Da/Ne]: ")

        if is_the_original_dataset == "Da":
            return 11

        elif is_the_original_dataset == "Ne":

            n_clusters_vs_inertia_plot(extracted_features, min_n_clusters, max_n_clusters)

            while True:

                string_n_clusters = input("Molimo unesite zeljeni broj kategorija: ")

                try:
                    n_clusters = int (string_n_clusters)

                    if n_clusters >= min_n_clusters and n_clusters <= max_n_clusters:
                        return n_clusters
                except:
                    pass

def train_and_predict(extracted_features, n_clusters):

    classifier = cluster.KMeans(n_clusters = n_clusters)

    trained_classifier = classifier.fit(extracted_features)
    predictions = trained_classifier.predict(extracted_features)

    return trained_classifier, predictions

#Knowing that prepool layer has many features, we've decided to use only a subset of data to fit K-Means
def train_and_predict_prepool(extracted_features, n_clusters, max_n_of_prepool_training_images):

    prepool_extracted_features = []

    for i in range(extracted_features.__len__()):
        if i < max_n_of_prepool_training_images:
            prepool_extracted_features.append(extracted_features[i])
        else:
            break

    classifier = cluster.KMeans(n_clusters = n_clusters)

    trained_classifier = classifier.fit(prepool_extracted_features)
    predictions = trained_classifier.predict(extracted_features)

    return trained_classifier, predictions

#Function used to create output files using simple formatting
def create_output_file(filename, predictions, directory):

    output_file = open(filename + ".txt", "w")

    header = "{:^20}".format("Naziv slike") + "|" + "{:^12}".format("Kategorija") + "\n"
    output_file.write(header)

    image_names = os.listdir(directory)
    jpg_only = []

    for image_name in image_names:
        if "jpg" in image_name:
            jpg_only.append(image_name)

    image_names = jpg_only

    for i in range(predictions.shape[0]):

        if i % 6 == 0: #The division is used in order to exclude the images obtained through data augmentation.
            to_write = "{:^20}".format(image_names[i//6]) + "|" + "{:^12}".format(str(predictions[i])) + "\n"
            output_file.write(to_write)

    output_file.close()


#Function used to create prepool output file using simple formatting
def create_output_file_prepool(filename, predictions, directory):

    output_file = open(filename + ".txt", "w")

    header = "{:^20}".format("Naziv slike") + "|" + "{:^12}".format("Kategorija") + "\n"
    output_file.write(header)

    image_names = os.listdir(directory)
    jpg_only = []

    for image_name in image_names:
        if "jpg" in image_name:
            jpg_only.append(image_name)

    image_names = jpg_only

    for i in range(predictions.shape[0]):

        to_write = "{:^20}".format(image_names[i]) + "|" + "{:^12}".format(str(predictions[i])) + "\n"
        output_file.write(to_write)

    output_file.close()


#Function used to plot the total distance with respect to the number of clusters
def n_clusters_vs_inertia_plot(extracted_features, min_n_clusters, max_n_clusters):

    distances = []

    for i in range(min_n_clusters, max_n_clusters + 1):

        temp_classifier = cluster.KMeans(n_clusters = i)
        temp_trained_classifier = temp_classifier.fit(extracted_features)
        distance = temp_trained_classifier.inertia_
        distances.append(distance)

    plt.plot(np.arange(min_n_clusters, max_n_clusters + 1), distances,  '-')

    plt.show()

main() 