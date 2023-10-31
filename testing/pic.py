import math
from scipy.stats import norm
import numpy as np

alpha=0.05
z=norm.isf(q=alpha)
def calculate(the_list):
    Mean=[]
    Relative_half_width=[]
    Var=[]
    acc=[]
    var_old=0
    mean_old=0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i]=0.0
        n=i+1
        mean_new=mean_old+(the_list[i]-mean_old)/n
        Mean.append(mean_new)
        var_new=(n-1)*var_old/n+(n-1)*(the_list[i]-mean_old)**2/(n*n)
        Var.append(1.96*(np.sqrt(var_new/n)))
        Relative_half_width.append(z*(np.sqrt(var_new/n)/(mean_new+1e-30)))
        var_old=var_new
        mean_old=mean_new
    return Mean,Relative_half_width,Var