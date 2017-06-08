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

    n_clusters = 11

    trained_classifier = joblib.load('trainedKmeans.pkl')

    trained_classifier_logits = joblib.load('trainedKmeansLogits.pkl')

    trained_classifier_prepool = joblib.load('trainedKmeansPrepool.pkl')

    predictions = trained_classifier.predict(extracted_features)
    logits_predictions = trained_classifier_logits.predict(logits_extracted_features)
    prepool_predictions = trained_classifier_prepool.predict(prepool_extracted_features)

    create_output_file("Mozgalo_test_result", predictions, directory)
    create_output_file("Mozgalo_test_result_logits", logits_predictions, directory)
    create_output_file("Mozgalo_test_result_prepool", prepool_predictions, directory)

    plt.hist(predictions, np.arange(0, n_clusters + 1)) #making of a histogram which shows how many pictures each cluster contains
    plt.show()

 




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

    checkpoints_dir = "inception_resnet_v2_2016_08_30"
    slim = tf.contrib.slim
    
    image_width = 299
    image_height = 299

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs" 
    logdir = "{}/run-{}/".format(root_logdir, now)
    step = 0

    string_image_placeholder = tf.placeholder(dtype = "string")

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):

        test_image = tf.image.decode_jpeg(string_image_placeholder, channels=3)

        processed_image = inception_preprocessing.preprocess_image(test_image, image_height, image_width, is_training=False, fast_mode = True)

        processed_images = tf.expand_dims(processed_image, 0)

        logits, end_points = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)

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

                for end_point in end_points2['PreLogitsFlatten']:
                    extracted_features.append(end_point)

                for end_point in end_points2['Logits']:
                    logits_extracted_features.append(end_point)

                
                var1 = end_points2['PrePool'].shape[1] 
                var2 = end_points2['PrePool'].shape[2] 
                var3 = end_points2['PrePool'].shape[3]
                num_of_prepool_features = var1 * var2 * var3
                
                for end_point in end_points2['PrePool']:

                    #num_of_prepool_features = 98304 
                        
                    reshaped_end_point = np.reshape(end_point, num_of_prepool_features)

                    prepool_extracted_features.append(reshaped_end_point)

            file_writer.close()

            return extracted_features, logits_extracted_features, prepool_extracted_features



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

        to_write = "{:^20}".format(image_names[i]) + "|" + "{:^12}".format(str(predictions[i])) + "\n"
        output_file.write(to_write)

    output_file.close()



main() 