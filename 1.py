# 本次案例使用的数据集
# 填充分类变量（基于Imputer的自定义填充器，用众数填充）
from sklearn.preprocessing import Imputer
# 填充分类变量（基于TransformerMixin的自定义填充器，用众数填充）
from sklearn.base import TransformerMixin
import pandas as pd

X = pd.DataFrame({'city':['tokyo','beijing','london','seattle','san fancisco','tokyo'],
                  'boolean':['y','n',None,'n','n','y'],
                  'ordinal_column':['somewhat like','like','somewhat like','like','somewhat like','dislike'],
                  'quantitative_column':[1,11,-.5,10,None,20]})



# 类别变量的编码（独热编码）
class CustomDummifier():
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, X):
        return pd.get_dummies(X, columns=self.cols)

# 调用自定义的填充器
cd = CustomDummifier(cols=['boolean'])
cd = cd.transform(X)
# cd =cd.fit_transform(X)

print(cd)

