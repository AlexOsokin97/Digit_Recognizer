import numpy as np
import pandas as pd
import pickle

def data_transformation(data):
        
        x_features = data.iloc[[0], :].copy()

        ############################################################################################

        val = x_features.iloc[[0], :].copy()
        counter = 0

        for col in val.columns:

            if val[col].unique() > 0:

                counter += 1

        x_features.loc[:, 'avg_pixel'] = counter/784

        #########################################################################################

        img = x_features.iloc[[0], :-1].values.reshape(-1, 28, 28, 1) / 255.0

        sub_img1 = np.average(img[0][:14, :14, 0])
        sub_img2 = np.average(img[0][14:, :14, 0])
        sub_img3 = np.average(img[0][:14, 14:, 0])
        sub_img4 = np.average(img[0][14:, 14:, 0])

        maximum = np.max([sub_img1, sub_img2, sub_img3, sub_img4])

        x_features.loc[:, 'max_val'] = maximum

        ##########################################################################################

        images = x_features.iloc[[0], :-2].values.reshape(-1, 28, 28, 1) / 255.0


        x_features.loc[[0], 'avg'] = np.average(images[0][:, :, 0])

        ##########################################################################################

        x_features = x_features.values
        x_features[[0], :-3] = x_features[[0], :-3]/255.0

        #########################################################################################
        
        pca = pickle.load(open("../Models/pca.pkl",'rb'))
        components = pca.transform(x_features[[0], :-3])

        x_features = pd.DataFrame(x_features)

        for x in range(components.shape[1]):
            x_features['PCA' + str(x)] = components[:, [x]]

        #########################################################################################

        x_features.drop(x_features.columns[:784], axis=1, inplace=True)
        x_features = x_features.values
    
        #######################################################################################

        return x_features