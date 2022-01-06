# Standardize a vector or rows of a matrix
def standardize(A):
  return (A - A.mean(axis=0))/A.std(axis=0)

# Split data into train and test sets
def train_test_split(X,y, test_ratio):
  # Shuffling both vector in unison
  p = np.random.permutation(X.shape[0])
  X_shuffled = X[p]
  y_shuffled = y[p]

  n_test = int(0.3*X.shape[0])

  X_test = X_shuffled[:n_test,:]
  y_test = y_shuffled[:n_test]
  X_train = X_shuffled[n_test:,:]
  y_train = y_shuffled[n_test:]
  return X_train, y_train, X_test, y_test

# Calculating cross-entropy
def cross_entropy(t,y):
  return - np.mean( t * np.log(y) + (1 - t) * np.log(1-y) )

# Function to perform gradient descent
def gradient_descent(x0,func,gradient,learning_rate=0.001,max_iter=2000):
  x_new = x0
  values = [func(x0)]
  i = 0
  while( i < max_iter ):
    x_new = x_new - learning_rate*gradient(x_new)
    values.append(func(x_new))
    i += 1
  return x_new, values
