import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
class Prediction():
    model = None

    df= pd.read_excel('app/dataset_final.xlsx')
    col = ['age(y)_18-23', 'age(y)_24-29', 'age(y)_<18', 'age(y)_>29', 'career_crit', 'career_crnm', 'career_crwb', 'career_stit', 'career_stnm', 'career_stud', 'career_unem', 'exp(y)_1-5', 'exp(y)_6-10', 'exp(y)_<1', 'exp(y)_>10', 'eng', 'th', 'web_pro_lang_JavaScript', 'web_pro_lang_HTML', 
    'web_pro_lang_CSS', 'web_pro_lang_TypeScript',  'UI_Libs_libal', 'UI_Libs_libpt', 'UI_Libs_no library',  'use', 'Learning_src_expert', 'Learning_src_media', 'Learning_src_doc', 'Learning_src_course', 'factor_market_needs', 'factor_compensation', 'website_content', 'website_graphics', 'website_functions', 'duration(m)_1-6', 'duration(m)_7-12', 'duration(m)_<1', 'duration(m)_>12', 'website_function_present_attractive', 
    'website_function_present_both', 'website_function_present_usability']
    x = df.iloc[:, 0:41]
    y = df.iloc[:, -1]

    def __init__(self) :
        self.training()

    def padding(self, text, n):
        i = len(text)
        ans = ""
        while i < n:
            ans += "0"
            i += 1

        return ans+text
        
    def transform_input(self, input, n):
        sum = 0;
        ans = ""
        for i in range(len(input)):
            sum += int(input[i])

        while sum>0.9 :
            ans = str(sum%2) + ans
            sum = int(sum/2)
            #print(sum)

        return self.padding(ans,n)

    

    def predict(self, data):
        return self.model.predict(data)
        

    def training(self) :
        x = self.x
        y = self.y
        clf_svm = svm.SVC(class_weight='balanced')
        clf_svm.fit(x, y)
        y_pred = cross_val_predict(clf_svm, x, y, cv=10)


        scores = cross_val_score(clf_svm, x, y, cv=10)
        print("cross-validated scores : ", scores)

        avg_score = np.mean(scores)
        print("\ncross-validated avg score : ", avg_score)

        confusion_matrix = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=[
                                    'Predicted'], margins=True)
        print("\n", confusion_matrix)

        print("\n", classification_report(y, y_pred))
        self.model = clf_svm

#pd = Prediction()