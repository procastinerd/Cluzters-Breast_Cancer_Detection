#importing libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('rf_model.pkl', 'rb'))    #importing model file
pca = pickle.load(open('pca.pkl', 'rb'))           #importing pca file
scaler = pickle.load(open('scaler.pkl','rb'))      #importing standard scaler file

class_names = ['Benign tumor','Malign tumor']
data= pd.read_csv('test.csv')

def predict(data): 
    
    data = data[["radius_mean",	"texture_mean",	"perimeter_mean","area_mean",	"smoothness_mean",	"compactness_mean",	"concavity_mean",
                       "concave points_mean",	"symmetry_mean",	"fractal_dimension_mean", 'radius_se','texture_se',  'perimeter_se', 'area_se',       
                        'smoothness_se', 'compactness_se', 'concavity_se','concave points_se','symmetry_se','fractal_dimension_se',  "radius_worst",	"texture_worst",
                        "perimeter_worst","area_worst",	"smoothness_worst",	"compactness_worst",	"concavity_worst",	"concave points_worst",	"symmetry_worst",	"fractal_dimension_worst"]]
    
    scaler.transform(data)
    pca_scaled_data = pca.fit_transform(data)
    predictions = model.predict_proba(pca_scaled_data)

    output = ["The patient is more likely to have benign tumor. Confidence for this classification is: {}%".format(round(predictions[i][0]*100,2)) 
                if predictions[i][0]>predictions[i][1] 
                else "The patient is more likely to have malign tumor. Confidence for this classification is: {}%".format(round(predictions[i][1]*100,2)) 
                for i in range(0,len(predictions))]
    return output

   # output = [class_names[i] for i in predictions]
   # return output

predict(data)

#data
#data= data.drop("Unnamed: 0",axis=1)
