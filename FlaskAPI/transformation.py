import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def data_transformation(data):

        x_features = data.iloc[[0], :]

        ############################################################################################

        for i in range(len(x_features)):

                val = x_features.iloc[[i]]
                counter = 0

                for col in val.columns:

                    if val[col].unique() > 0:

                        counter += 1

                x_features.loc[[i], 'avg_pixel'] = counter/784

        img = x_features.iloc[:, :-1].values.reshape(-1, 28, 28, 1) / 255.0

        #########################################################################################

        for i in range(len(img)):
            sub_img1 = np.average(img[i][:14, :14, 0])
            sub_img2 = np.average(img[i][14:, :14, 0])
            sub_img3 = np.average(img[i][:14, 14:, 0])
            sub_img4 = np.average(img[i][14:, 14:, 0])

            maximum = np.max([sub_img1, sub_img2, sub_img3, sub_img4])

            x_features.loc[[i], 'max_val'] = maximum

        ##########################################################################################

        images = x_features.iloc[:, :-2].values.reshape(-1, 28, 28, 1) / 255.0

        for i in range(len(data)):
            x_features.loc[[i], 'avg'] = np.average(images[i][:, :, 0])

        ##########################################################################################

        x_features = x_features.values
        x_features[:, :-3] = x_features[:, :-3]/255.0

        #########################################################################################

        pca = PCA(n_components=45)
        components = pca.fit_transform(x_features[:, :-3])

        x_features = pd.DataFrame(x_features)

        for x in range(components.shape[1]):
            x_features['PCA' + str(x)] = components[:, [x]]

        #########################################################################################

        x_features.drop(x_features.columns[:784], axis=1, inplace=True).values

        #######################################################################################

        return x_features