library(tidyverse) # for plotting only

# no scientific notation for common numbers
options(digits = 3, scipen=5)

sigmoid <- function(x){
  1/(1+exp(-x))
}
d_sigmoid <- function(x){
  sigmoid(x)*(1-sigmoid(x))
}


# Data Matrix
X = matrix(c(0,0, 0,1,1,0,1,1), nrow=2)
X
# true labels
Y = matrix(c(0, 1, 1, 1), nrow = 1)

data <-data.frame(x1=X[1,], x2= X[2,], label=Y[1,])

# element wise loss function
# y and yhat should be row vectors of equal length
# note that this function fails for yhat values of exactly 0 or 1,
# but those will never be produced by a sigmoid activation funciton
loss <- function(yhat, y){
  l = -(y * log(yhat) + (1-y)* log(1- yhat))
  # save l in gloval environment for later use
  assign("l", l, envir=globalenv())
  return(l)
}

# empirical risk function
# y and yhat should be row vectors of equal length
risk <- function(yhat, y){
  L = sum(loss(yhat, y))/length(yhat)
  # save L in global environment for later use
  assign("L", L, envir=globalenv())
  return(L)
}

# initialize neural net parameters
# layer 1: 2 hidden nodes --> W1 is 2x2, b1 is 2x1
W1 = matrix(c(1,0,0,1), nrow=2)
b1 = matrix(c(0,0))

# layer 2: 1  node --> W2 is 2x1, b2 is 1x1
W2 = matrix(c(1,-1), nrow = 1)
b2 = matrix(c(0))


# define feed forward pass
forward <- function(x){
  Z1 = sweep(W1 %*% x , 1, -b1) # read: W1X + b1
  A1 = sigmoid(Z1)
  Z2 = sweep(W2 %*% A1, 1, -b2)
  A2 = sigmoid(Z2)
  
  
  # update values in global environment for later use in backward pass
  assign('A0',  x, envir = globalenv()) # avoid possible name conflict w/ 'x'
  assign('Z1', Z1, envir = globalenv())
  assign('A1', A1, envir = globalenv())
  assign('Z2', Z2, envir = globalenv())
  assign('A2', A2, envir = globalenv())
  
  return(A2)
}


# define backward pass
backward <- function(y, learning_rate=1){
  n = length(y)
  dZ2 = A2 - y
  dW2 = (dZ2 %*% t(A1))/n
  db2 = rowSums(dZ2) / n
  dZ1 = t(W2) %*% dZ2 * (A1 * (1-A1))
  dW1 = (dZ1 %*% t(A0))
  db1 = rowSums(dZ1) / n
  
  # update parameters
  assign('W1', W1 - learning_rate * dW1,  envir = globalenv())
  assign('b1', b1 - learning_rate * db1,  envir = globalenv())
  assign('W2', W2 - learning_rate * dW2,  envir = globalenv())
  assign('b2', b2 - learning_rate * db2,  envir = globalenv())
}

# function for plotting nn outputs (you don't need to understand this)
plot_nn <- function(data = NULL){
  grid <- as_tibble(expand.grid(data.frame(x1=seq(0,1,by=.01), x2=seq(0,1,by=.01))))
  grid$yhat = as.numeric(forward(t(grid %>% select(x1, x2))))
  
  g<- ggplot(grid, aes(x=x1, y=x2, col=yhat)) + 
    geom_point() + theme_minimal() +
    coord_equal() + 
    scale_color_viridis_c()
  
  if (!is.null(data)){
    data$label <- factor(data$label)
    g <- g + geom_point(data=data, aes(x=x1, y=x2, shape=label), col='red', size=4)
  }
  
  g
}

plot_nn(data=data)

# train the neural net --> 100 iterations of gradient descent
for(i in 1:100){
  print(risk(forward(X), Y))
  backward(Y)
}
forward(X)

plot_nn(data=data)


# let's try a somewhat more complicated dataset.
# Can our neural network learn to correctly classify it?


newX = matrix(c(0,0, 0,0.5, 0,1, .5,0 , .5,.5, .5,1, 1,0, 1,.5, 1,1), nrow=2)
newY = matrix(c(0,0,1,1,1,0,0,0,1), nrow=1)
newdata <-data.frame(x1=newX[1,], x2= newX[2,], label=newY[1,])

plot_nn(data=newdata)

for(i in 1:1000){
  print(risk(forward(newX), newY))
  backward(newY)
}
forward(newX)

plot_nn(data = newdata)



# how about if we add more nodes in the hidden layer? (10 instead of 2)

W1 = matrix(rnorm(20), nrow=10)
b1 = matrix(rnorm(10), nrow =10)
W2 = matrix(rnorm(10), nrow=1)
b2 = matrix(rnorm(1), nrow =1)

plot_nn(data = newdata)

for(i in 1:1000){
  print(risk(forward(newX), newY))
  backward(newY)
}
forward(newX)

plot_nn(data=newdata)
