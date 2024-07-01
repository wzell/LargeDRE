import numpy as np
from scipy.special import xlogy

DEG = 1.

def exact_beta(mean_s,sigma_s, mean_t, sigma_t,x):
    return (sigma_s/sigma_t)*np.exp(((((x-mean_s)/sigma_s)**2) - (((x-mean_t)/sigma_t)**2))/2)

def generate_gauss_dataset(seed,n,m,mu_S,sigma_S,mu_T,sigma_T):
    np.random.seed(seed)
    source_X = np.random.normal(mu_S, sigma_S, n)
    target_X = np.random.normal(mu_T, sigma_T, m)
    return source_X, target_X

def kernel_mat(source_X, target_X):
    m = target_X.shape[0]
    n = source_X.shape[0]
    Kb = np.zeros((m+n,m+n))
    for i in range(m):
        for j in range(m):
            Kb[i,j] = Kb[j,i] = ker(target_X[i],target_X[j])
    for i in np.arange(m,m+n):
        for j in np.arange(m,m+n):
            Kb[i,j] = Kb[j,i] = ker(source_X[i-m],source_X[j-m])
    for i in np.arange(m):
        for j in np.arange(m,m+n):
            Kb[i,j] = Kb[j,i] = ker(target_X[i],source_X[j-m])
    for i in np.arange(m,m+n):
        for j in np.arange(m):
            Kb[i,j] = Kb[j,i] = ker(source_X[i-m],target_X[j])
    return Kb
            
def ker(x,t):
    return 1 + np.exp(-((x-t)**2)/2)

def estimated_beta_analytic(source_X,target_X,alpha,x,w):
    m = target_X.shape[0]
    n = source_X.shape[0]
    estimate = np.zeros(x.shape[0])
    for k in range(x.shape[0]):
        sum1 = 0
        for i in range(n):
            sum1 += w[i]*ker(x[k],source_X[i])
        sum2 = 0
        for j in range(m):
            sum2 += ker(x[k],target_X[j])/(alpha*m)
        estimate[k] = sum1 + sum2
    return estimate

def loss_Q3(source_X, target_X, Kb, alpha, method):
    m = target_X.shape[0]
    n = source_X.shape[0]
    sum = 0
    for k,xs in enumerate(source_X):
        inner_sum = 0
        inner_sum += np.dot(alpha[m:], Kb[k+m, m:m+n])
        inner_sum += np.dot(alpha[:m], Kb[k+m, :len(target_X)])
        if method == 'KuLSIF':
            sum+=1/2*inner_sum**2
        elif method == 'ExpWeight':
            if inner_sum<0:
                inner_sum=0
            sum+= 1/2. *xlogy(inner_sum, 2.* inner_sum )-1/2.*inner_sum
    return 1/n*sum 

def loss_P3(source_X, target_X, Kb, alpha, method):# menon: source, -1, e^v (exp), 1/2 v^2 (kulsif)
    m = target_X.shape[0]
    n = source_X.shape[0]
    sum = 0
    for k,xt in enumerate(target_X):
        inner_sum = 0
        inner_sum += np.dot(alpha[m:m+len(source_X)], Kb[k, m:m+len(source_X)])
        inner_sum += np.dot(alpha[:len(target_X)], Kb[k, :len(target_X)])
        if method == 'KuLSIF':
            sum+= -inner_sum
        elif method == 'ExpWeight':
            sum+= -inner_sum
    return 1/m*sum

def loss_regularized3(source_X, target_X, Kb, alpha, reg_param, method):
    val = loss_P3(source_X, target_X, Kb, alpha, method)+loss_Q3(source_X, target_X, Kb, alpha, method=method)
    val += reg_param / 2 * alpha.dot(Kb).dot(np.transpose(alpha))
    return val

def evaluate2(source_X,target_X,alpha,x,w,method):
    m = target_X.shape[0]
    n = source_X.shape[0]
    estimate = np.zeros(x.shape[0])
    # Define the kernel matrix for source_X and x
    ker_matrix_source = np.array([[ker(xsi, xk) for xk in x] for xsi in source_X])
    ker_matrix_target = np.array([[ker(xti, xk) for xk in x] for xti in target_X])

    # Calculate the weighted sums
    sum1 = np.dot(w[m:m+len(source_X)], ker_matrix_source)
    sum2 = np.dot(w[:len(target_X)], ker_matrix_target)

    # Combine the sums according to the specified method
    if method == 'KuLSIF':
        estimate = sum1+sum2
        estimate[estimate<0]=0
    elif method == 'exp':
        estimate = np.exp(2 * (sum1 + sum2))
    elif method == 'KLest':
        estimate = np.exp(sum1 + sum2)
    elif method == 'ExpWeight':
        ss = sum1+sum2
        for i,val in enumerate(ss):
            if val>=1/2.:
                estimate[i] = 0.5 * np.log(2 * val)
            else:
                estimate[i] = 0.5 * np.log(2 * 1/2.)
                if estimate[i]<0: print(val)
    elif method== 'poly':
        deg = DEG
        estimate = ((1.+deg)*(sum1+sum2))**(1./(1.+deg))
    return estimate








