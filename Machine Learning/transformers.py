import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scipy.stats as st
import feature_engine.outliers as feo
import feature_engine.transformation as fet
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

random_state=120

class CheckGaussian():
    

    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of CheckGaussian class
            Output: None
        '''
        pass


    def check_gaussian(self, X):
        '''
            Method Name: check_gaussian
            Description: This method classifies features from dataset into gaussian vs non-gaussian columns.
            Output: self
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        self.gaussian_columns = []
        self.non_gaussian_columns = []
        for column in X_.columns:
            result = st.anderson(X_[column])
            if result[0] > result[1][2]:
                self.non_gaussian_columns.append(column)
            else:
                self.gaussian_columns.append(column)
        return


class OutlierCapTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of OutlierCapTransformer class
            Output: None
            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(OutlierCapTransformer, self).__init__()
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method classifies way of handling outliers based on whether features are gaussian 
            or non-gaussian, while fitting respective class methods from feature-engine.outliers library
            Output: self
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        self.check_gaussian(X_[self.continuous])
        if self.non_gaussian_columns!=[]:
            self.non_gaussian_winsorizer = feo.Winsorizer(
                capping_method='iqr', tail='both', fold=1.5, add_indicators=False,variables=self.non_gaussian_columns)
            self.non_gaussian_winsorizer.fit(X_)
        if self.gaussian_columns!=[]:
            self.gaussian_winsorizer = feo.Winsorizer(
                capping_method='gaussian', tail='both', fold=3, add_indicators=False,variables=self.gaussian_columns)
            self.gaussian_winsorizer.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective class methods from 
            feature-engine.outliers library
            Output: Transformed features from dataset in dataframe format.
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        if self.non_gaussian_columns != []:
            X_ = self.non_gaussian_winsorizer.transform(X_)
        if self.gaussian_columns != []:
            X_ = self.gaussian_winsorizer.transform(X_)
        return X_


class GaussianTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, continuous):
        '''
            Method Name: __init__
            Description: This method initializes instance of GaussianTransformer class
            Output: None
            Parameters:
            - continuous: Continuous features from dataset
        '''
        super(GaussianTransformer, self).__init__()
        self.continuous = continuous


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method tests for various gaussian transformation techniques on non-gaussian variables. 
            Non-gaussian variables that best successfully transformed to gaussian variables based on Anderson test will be 
            used for fitting on respective gaussian transformers.
            Output: self
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        self.check_gaussian(X_[self.continuous])
        transformer_list = [
            fet.LogTransformer(), fet.ReciprocalTransformer(), fet.PowerTransformer(exp=0.5), fet.YeoJohnsonTransformer(), fet.PowerTransformer(exp=2), QuantileTransformer(output_distribution='normal')
        ]
        transformer_names = [
            'logarithmic','reciprocal','square-root','yeo-johnson','square','quantile'
        ]
        result_names, result_test_stats, result_columns, result_critical_value=[], [], [], []
        for transformer, name in zip(transformer_list, transformer_names):
            for column in self.non_gaussian_columns:
                try:
                    X_transformed = pd.DataFrame(
                        transformer.fit_transform(X_[[column]]), columns = [column])
                    result_columns.append(column)
                    result_names.append(name)
                    result_test_stats.append(
                        st.anderson(X_transformed[column])[0])
                    result_critical_value.append(
                        st.anderson(X_transformed[column])[1][2])
                except:
                    continue
        results = pd.DataFrame(
            [pd.Series(result_columns, name='Variable'), 
            pd.Series(result_names,name='Transformation_Type'),
            pd.Series(result_test_stats, name='Test-stats'), 
            pd.Series(result_critical_value, name='Critical value')]).T
        best_results = results[results['Test-stats']<results['Critical value']].groupby(by='Variable')[['Transformation_Type','Test-stats']].min()
        transformer_types = best_results['Transformation_Type'].unique()
        for type in transformer_types:
            variable_list = best_results[best_results['Transformation_Type'] == type].index.tolist()
            if type == 'logarithmic':
                self.logtransformer = fet.LogTransformer(variables=variable_list)
                self.logtransformer.fit(X_)
            elif type == 'reciprocal':
                self.reciprocaltransformer = fet.ReciprocalTransformer(variables=variable_list)
                self.reciprocaltransformer.fit(X_)
            elif type == 'square-root':
                self.sqrttransformer = fet.PowerTransformer(exp=0.5, variables=variable_list)
                self.sqrttransformer.fit(X_)
            elif type == 'yeo-johnson':
                self.yeojohnsontransformer = fet.YeoJohnsonTransformer(variables=variable_list)
                self.yeojohnsontransformer.fit(X_)
            elif type == 'square':
                self.squaretransformer = fet.PowerTransformer(exp=2, variables=variable_list)
                self.squaretransformer.fit(X_)
            elif type == 'quantile':
                self.quantiletransformer = QuantileTransformer(output_distribution='normal',random_state=random_state)
                self.quantiletransformer.fit(X_[variable_list])
                self.quantilevariables = variable_list
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs gaussian transformation on features using respective gaussian transformers.
            Output: Transformed features from dataset in dataframe format.
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.reset_index(drop=True).copy()
        if hasattr(self, 'logtransformer'):
            try:
                X_ = self.logtransformer.transform(X_)
            except:
                old_variable_list = self.logtransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]<=0).sum()>0:
                        self.logtransformer.variables_.remove(var)
                X_ = self.logtransformer.transform(X_)
        if hasattr(self, 'reciprocaltransformer'):
            try:
                X_ = self.reciprocaltransformer.transform(X_)
            except:
                old_variable_list = self.reciprocaltransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]==0).sum()>0:
                        self.reciprocaltransformer.variables_.remove(var)
                X_ = self.reciprocaltransformer.transform(X_)
        if hasattr(self, 'sqrttransformer'):
            try:
                X_ = self.sqrttransformer.transform(X_)
            except:
                old_variable_list = self.sqrttransformer.variables_.copy()
                for var in old_variable_list:
                    if (X_[var]==0).sum()>0:
                        self.sqrttransformer.variables_.remove(var)
                X_ = self.sqrttransformer.transform(X_)
        if hasattr(self, 'yeojohnsontransformer'):
            X_ = self.yeojohnsontransformer.transform(X_)
        if hasattr(self, 'squaretransformer'):
            X_ = self.squaretransformer.transform(X_)
        if hasattr(self, 'quantiletransformer'):
            X_[self.quantilevariables] = pd.DataFrame(
                self.quantiletransformer.transform(X_[self.quantilevariables]), columns = self.quantilevariables)
        return X_


class ScalingTransformer(BaseEstimator, TransformerMixin, CheckGaussian):
    
    
    def __init__(self, scaler):
        '''
            Method Name: __init__
            Description: This method initializes instance of ScalingTransformer class
            Output: None
            Parameters:
            - scaler: String that represents method of performing feature scaling. 
            (Accepted values are 'Standard', 'MinMax', 'Robust' and 'Combine')
        '''
        super(ScalingTransformer, self).__init__()
        self.scaler = scaler


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method fits dataset onto respective scalers selected.
            Output: self
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        if self.scaler == 'Standard':
            self.copyscaler = StandardScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'MinMax':
            self.copyscaler = MinMaxScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'Robust':
            self.copyscaler = RobustScaler()
            self.copyscaler.fit(X_)
        elif self.scaler == 'Combine':
            self.check_gaussian(X_)
            self.copyscaler = ColumnTransformer(
                [('std_scaler',StandardScaler(),self.gaussian_columns),('minmax_scaler',MinMaxScaler(),self.non_gaussian_columns)],remainder='passthrough',n_jobs=1)
            self.copyscaler.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using respective scalers.
            Output: Transformed features from dataset in dataframe format.
            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        if self.scaler != 'Combine':
            X_ = pd.DataFrame(
                self.copyscaler.transform(X_), columns = X.columns)
        else:
            X_ = pd.DataFrame(
                self.copyscaler.transform(X_), columns = self.gaussian_columns + self.non_gaussian_columns)
            X_ = X_[X.columns.tolist()]
        return X_