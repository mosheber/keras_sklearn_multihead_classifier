from sklearn.base import BaseEstimator, ClassifierMixin
import keras
import inspect
import numpy as np

class KerasMultiClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self,build_fn,features,**sk_params):
    """
      build_fn: a function that returns a compiled keras model, same as in KerasClassifier
      features: list of dictionaries which describes each feature.Each dictionary contains:
          name: name of the feature, eg cat_height
          type: type of the feature, eg numerical, categorial
          indices: the zero-based indices in the X matrix that correspond to this feature. 
            eg: X is has 10 columns and columns at indices 0 to 4 are the indices of this feature.
            see example for more details.
      sk_params: params for the fit function of the model, and those remaining are for the build_fn function. 
          same as in KerasClassifier
          
      runnable example: 
      
      import keras
      from sklearn.datasets import make_classification
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from keras.layers import  Dense,Input,BatchNormalization,concatenate
      from keras import  Model


      ## create random dataset

      number_of_classes=4

      X,y=make_classification(n_classes=number_of_classes,n_informative=number_of_classes)

      ##define features in X matrix
      features=[{'name':'first_feature','type':'numberical','indices':list(range(0,10))},
                {'name':'second_feature','type':'numerical','indices':list(range(10,20))}]

      first_feature=pd.DataFrame(X[:,list(range(0,10))])
      second_feature=pd.DataFrame(X[:,list(range(0,10))])

      first_feature=first_feature.rename(columns={ column:'{name}_{col}'.format(name='first_feature',col=column)  for column in first_feature.columns})
      second_feature=second_feature.rename(columns={ column:'{name}_{col}'.format(name='second_feature',col=column)  for column in second_feature.columns})

      full_data=first_feature.join(second_feature)

      label_column='label'
      full_data[label_column]=y

      ## Split to train and test
      x_train,x_test,y_train,y_test = train_test_split(
          full_data.drop(columns=[label_column]).values, \
          full_data[label_column].values.ravel(), \
          test_size=0.2, \
          stratify = full_data[label_column])

      ##define build_fn 
      def build_fn(first_feature_shape,second_feature_shape):
        input_1=Input(shape=first_feature_shape)
        dense_1=Dense(5,activation='relu')(input_1)
        batch_1=BatchNormalization()(dense_1)

        input_2=Input(shape=second_feature_shape)
        dense_2=Dense(5,activation='relu')(input_2)
        batch_2=BatchNormalization()(dense_2)

        concat=concatenate([batch_1,batch_2])

        dense_full=Dense(5,activation='relu')(concat)
        softmax=Dense(4,activation='softmax')(dense_full)

        model=Model([input_1,input_2],softmax)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
        return model

      multiClassifier=KerasMultiClassifier(
          build_fn=build_fn, \
          features=features, \
          epochs=4, \
          validation_split=0.1, \
          first_feature_shape=(10,), \
          second_feature_shape=(10,))


      #fit
      hist=multiClassifier.fit(x_train,y_train)

      #predict_proba
      predict_proba=multiClassifier.predict_proba(x_test)

      #predict
      predict=multiClassifier.predict(x_test)

      print('predict_proba: ',str(predict_proba.shape),'predict: ',str(predict.shape))

    """
    self.build_fn=build_fn
    self.features=features
    
     # get all possible model fit params
    fit_params=list(inspect.getfullargspec(keras.Model.fit))[0]
    
    
    self.model_fit_params={key:value for key,value in sk_params.items() if key in fit_params}
    self.build_fn_params={key:value for key,value in sk_params.items() if key not in fit_params}
    
  def divide_X_to_features(self,X):
    inputs=[]
    for feature in self.features:
      indices=feature['indices']
      inputs.append(X[:,indices])
      
    return inputs
  
  def fit(self,X,y):
    inputs=self.divide_X_to_features(X)
    self.model=self.build_fn(**self.build_fn_params)  
    hist = self.model.fit(inputs,y,**self.model_fit_params)
    return hist
  
  def predict_proba(self, X):
    inputs=self.divide_X_to_features(X)
    prediction= self.model.predict(inputs)
    return prediction
  
  def predict(self, X):
    prediction= self.predict_proba(X)
    return np.argmax(prediction,axis=1)
  
    
    
    
    
    

    
