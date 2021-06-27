from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import scipy.io
import numpy as np
from sklearn.model_selection import GridSearchCV


mat=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-n1.mat')
mat1=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-n2.mat')
mat2=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-n3.mat')
mat3=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-n4.mat')
mat4=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-s1.mat')
mat5=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-s2.mat')
mat6=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-s3.mat')
mat7=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-s4.mat')
mat8=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-s5.mat')
mat9=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-s6.mat')


var_train=np.concatenate((mat['CC_N_A1'],mat1['CC_N_A2'],mat2['CC_N_A3'],mat3['CC_N_A4'],mat4['CC_S_A1'],mat5['CC_S_A2'],mat6['CC_S_A3'],mat7['CC_S_A4'],mat8['CC_S_A5'],mat9['CC_S_A6']),axis=0)
targ_train=np.concatenate((-1*np.ones((40,1)),np.ones((60,1))),axis=0)

## find The best Parameters 
svclassifier=SVC(kernel='linear',C=1)
grid_param = {  
    'C': [1000,100,10,1,0.1,0.01,0.001,0.0001]} 
clf=GridSearchCV(estimator=svclassifier,param_grid=grid_param,scoring='accuracy',cv=5,n_jobs=-1)
clf.fit(var_train,targ_train)
best_parameters = clf.best_params_  
print(best_parameters) 

##-----------------------------train error---------------------------------

svclassifier=SVC(kernel='linear',C=best_parameters['C'])
svclassifier.fit(var_train,targ_train) 
y_pred_train_li = svclassifier.predict(var_train)
print(confusion_matrix(targ_train,y_pred_train_li))  
print(classification_report(targ_train,y_pred_train_li))


#Test

mat_test=scipy.io.loadmat(r'F:\Projects\Master\Second Semester\Speech Processing\Final Project\Results\cc-a-test1.mat')
y_pred_test=svclassifier.predict(mat_test['CC_Test_A'])
