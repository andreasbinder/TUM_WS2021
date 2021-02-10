#EM Algorithm 

#function for f(x,mu,sigma)

#a)
myf <- function(x,mu,sigma){
  firstFac <- (1/(sigma*sqrt(2*pi)))
  secondFac <- exp((-1)*((x-mu)^2)/(2*sigma^2))
  return(firstFac*secondFac)
}

#b) 
value <- c(.76,.86,1.12,3.05,3.51,3.75)
mu_a <- 1.12
sigma_a <- 1
p_a <- .5 

mu_b <- 3.05
sigma_b <- 1 
p_b <- .5 


#c) 
em <- function(value, mu_a, sigma_a, p_a, mu_b, sigma_b, p_b, niter){
  for(i in 1:niter){
    #Calculate likelihoods 
    p_x_a <- sapply(value,myf,mu=mu_a,sigma=sigma_a)
    p_x_b <- sapply(value,myf,mu=mu_b,sigma=sigma_b)
    
    p_x <- p_x_a*p_a + p_x_b*p_b
    p_a_x <- p_x_a*p_a/p_x
    p_b_x <- p_x_b*p_b/p_x
    
    #Update parameters 
    mu_a <- sum(p_a_x*value)/sum(p_a_x)
    sigma_a <- sqrt(sum(p_a_x*((value-mu_a)^2))/(sum(p_a_x)))
    p_a <- sum(p_a_x)/sum(p_a_x+p_b_x)
    
    mu_b <- sum(p_b_x*value)/sum(p_b_x)
    sigma_b <- sqrt(sum(p_b_x*((value-mu_b)^2))/(sum(p_b_x)))
    p_b <- sum(p_b_x)/sum(p_a_x+p_b_x)
    
  }
  return(list(p_a_x, p_b_x))
}

em(value, mu_a, sigma_a, p_a, mu_b, sigma_b, p_b, 2)

#d) experiment with different values 
em(value, .76,1,.5,3.75,1,.5,10)
em(value, .86,1,.5,1.12,1,.5,10)

em(value, .76,1,.5,3.75,1,.5,50)
em(value, .86,1,.5,1.12,1,.5,50)

em(value, .76,1,.5,3.75,1,.5,100)
em(value, .86,1,.5,1.12,1,.5,100)

#Choosing reasonable start values is crucial. When choosing these properly (i.e. .76,3.75), the algorithm converges to a clear solution quite quickly.
#However, if our start values are "badly" chosen (i.e. .86,1.12) it might happen that there is no 
#convergence after 10 iterations. Increasing the number of iterations, the algorithm might
#converge to a clear solution eventually. Here, the first three data points (almost surely) belong to cluster A
#and the last three data points (almost surely) belong to cluster B  
