import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


def ModelHist_plot(hist,plot1,plot2,leg1,leg2):
    plt.figure()
    plt.plot(hist.history[plot1])
    plt.plot(hist.history[plot2])
    plt.legend([leg1, leg2])
    plt.show()


def Time_eval(y_test,predicted,xlab='Time',ylab='Data',leg=['Actual', 'Predicted']):
    plt.figure()
    plt.plot(y_test,color='red')
    plt.plot(predicted,color='blue')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(leg)
    plt.show()

def Time_eval_inverse(mms,y_test,predicted,xlab='Time',ylab='Data',leg=['Actual', 'Predicted']):
    actual_price = mms.inverse_transform(y_test)
    predicted_price = mms.inverse_transform(predicted)
    plt.figure()
    plt.plot(y_test,color='red')
    plt.plot(predicted,color='blue')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(leg)
    plt.show() 


def subplot(df1,df2,col,title1,title2,ylab,xlab1,xlab2):
    fig = plt.figure(figsize=[15,8])
    ax1=fig.add_subplot(2,1,1)
    ax2=fig.add_subplot(2,1,2)
    df1[col].plot.line(ax=ax1)
    df2[col].plot.line(ax=ax2)
    ax1.set_title(title1)
    ax1.set_ylabel(ylab)
    ax1.set_xlabel(xlab1)
    ax2.set_title(title2)
    ax2.set_ylabel(ylab)
    ax2.set_xlabel(xlab2)
    plt.tight_layout()
    plt.show()