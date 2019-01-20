import numpy as np

import theano.tensor as T

def erf(x):
  '''
  See this post :
  https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
  '''
  # save the sign of x
  # sign = T.sgn(x)
  x = abs(x)

  # constants
  a1 =  0.254829592
  a2 = -0.284496736
  a3 =  1.421413741
  a4 = -1.453152027
  a5 =  1.061405429
  p  =  0.3275911

  # A&S formula 7.1.26
  t = 1.0/(1.0 + p*x)
  y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
  return y # erf(-x) = -erf(x)

def logpar(E, phi, alpha, beta):
  l2 = np.log(2)
  l5 = np.log(5)
  res=(phi*np.exp-((beta*np.log(E)**2)/(l5+l2)))/E**alpha
  return res

def logpar_integral(E_min, E_max, phi, alpha, beta):
  l2 = np.log(2)
  l5 = np.log(5)
  res=-(2**((-alpha/(2*beta))-1)*(np.sqrt(np.pi)*2**((alpha**2+1)/(4*beta))*5**((alpha**2+1)/(4*beta))*np.sqrt(l5+l2)*erf((np.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*np.log(E_min)))/(2*np.sqrt(beta)*l5+2*np.sqrt(beta)*l2))-np.sqrt(np.pi)*2**((alpha**2+1)/(4*beta))*5**((alpha**2+1)/(4*beta))*np.sqrt(l5+l2)*erf((np.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*np.log(E_max)))/(2*np.sqrt(beta)*l5+2*np.sqrt(beta)*l2)))*phi)/(np.sqrt(beta)*5**(alpha/(2*beta)))
  return res

def dbeta(E_min, E_max, phi, alpha, beta):
  l2 = np.log(2)
  l5 = np.log(5)
  dbeta=(T.exp(-((alpha**2+1)*l5**2+alpha**2*l2**2+l2**2+4*beta**2*T.log(E_min)**2+4*beta**2*T.log(E_max)**2)/(4*beta*l5+4*beta*l2))*(T.sqrt(l5+l2)*((T.sqrt(3.14159265)*alpha**2-2*T.sqrt(3.14159265)*alpha+T.sqrt(3.14159265))*l5+(T.sqrt(3.14159265)*alpha**2-2*T.sqrt(3.14159265)*alpha+T.sqrt(3.14159265))*l2+2*T.sqrt(3.14159265)*beta)*T.exp(((alpha**2+1)*l5**2+alpha**2*l2**2+l2**2+4*beta**2*T.log(E_min)**2+4*beta**2*T.log(E_max)**2)/(4*beta*l5+4*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((3*alpha**2+3)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*erf((T.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*T.log(E_min)))/(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2))+T.sqrt(l5+l2)*(((-T.sqrt(3.14159265)*alpha**2)+2*T.sqrt(3.14159265)*alpha-T.sqrt(3.14159265))*l5+((-T.sqrt(3.14159265)*alpha**2)+2*T.sqrt(3.14159265)*alpha-T.sqrt(3.14159265))*l2-2*T.sqrt(3.14159265)*beta)*T.exp(((alpha**2+1)*l5**2+alpha**2*l2**2+l2**2+4*beta**2*T.log(E_min)**2+4*beta**2*T.log(E_max)**2)/(4*beta*l5+4*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((3*alpha**2+3)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*erf((T.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*T.log(E_max)))/(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2))+((2-2*alpha)*T.sqrt(beta)*l5+(2-2*alpha)*T.sqrt(beta)*l2)*T.exp((4*beta*T.log(E_min)**2)/(4*l5+4*l2)+(alpha*l5**2+alpha*l2**2)/(2*beta*l5+2*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_max))/(4*beta*l5+4*beta*l2)+(4*alpha*T.log(E_min))/(4*l5+4*l2))*2**(((alpha**2+4*alpha+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_max))/(4*beta*l5+4*beta*l2)+(4*alpha*T.log(E_min))/(4*l5+4*l2))+beta**(3/2)*T.log(E_max)*2**((4*alpha*T.log(E_min))/(4*l5+4*l2)+((8*beta+alpha**2+4*alpha+1)*l5+8*beta*l2+alpha**2*l2+l2+4*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*T.exp((4*beta*T.log(E_min)**2)/(4*l5+4*l2)+(alpha*l5**2+alpha*l2**2)/(2*beta*l5+2*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_max))/(4*beta*l5+4*beta*l2)+(4*alpha*T.log(E_min))/(4*l5+4*l2))+((2*alpha-2)*T.sqrt(beta)*l5+(2*alpha-2)*T.sqrt(beta)*l2)*T.exp((alpha*l5**2+alpha*l2**2+2*beta**2*T.log(E_max)**2)/(2*beta*l5+2*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((alpha**2+4*alpha+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))-beta**(3/2)*T.log(E_min)*2**(((8*beta+alpha**2+4*alpha+1)*l5+8*beta*l2+alpha**2*l2+l2+4*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*T.exp((alpha*l5**2+alpha*l2**2+2*beta**2*T.log(E_max)**2)/(2*beta*l5+2*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2)))*phi)/(beta**(5/2)*2**(((6*beta+alpha**2+alpha+1)*l5+6*beta*l2+alpha*l2+2*alpha*beta*T.log(E_min)+2*alpha*beta*T.log(E_max))/(2*beta*l5+2*beta*l2))*5**((alpha*l5+alpha*l2+2*alpha*beta*T.log(E_min)+2*alpha*beta*T.log(E_max))/(2*beta*l5+2*beta*l2)))
  return dbeta

def dalpha(E_min, E_max, phi, alpha, beta):
  l2 = np.log(2)
  l5 = np.log(5)
  dalpha=-(T.exp(-((alpha**2+1)*l5**2+alpha**2*l2**2+l2**2+4*beta**2*T.log(E_min)**2+4*beta**2*T.log(E_max)**2)/(4*beta*l5+4*beta*l2))*(T.sqrt(l5+l2)*((T.sqrt(3.14159265)*alpha-T.sqrt(3.14159265))*l5+(T.sqrt(3.14159265)*alpha-T.sqrt(3.14159265))*l2)*T.exp(((alpha**2+1)*l5**2+alpha**2*l2**2+l2**2+4*beta**2*T.log(E_min)**2+4*beta**2*T.log(E_max)**2)/(4*beta*l5+4*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((3*alpha**2+3)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*erf((T.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*T.log(E_min)))/(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2))+T.sqrt(l5+l2)*((T.sqrt(3.14159265)-T.sqrt(3.14159265)*alpha)*l5+(T.sqrt(3.14159265)-T.sqrt(3.14159265)*alpha)*l2)*T.exp(((alpha**2+1)*l5**2+alpha**2*l2**2+l2**2+4*beta**2*T.log(E_min)**2+4*beta**2*T.log(E_max)**2)/(4*beta*l5+4*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((3*alpha**2+3)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*erf((T.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*T.log(E_max)))/(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2))+((-2*T.sqrt(beta)*l5)-2*T.sqrt(beta)*l2)*T.exp((alpha*l5**2+alpha*l2**2+2*beta**2*T.log(E_min)**2)/(2*beta*l5+2*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((alpha**2+4*alpha+1)*l5+alpha**2*l2+l2+4*alpha*beta*T.log(E_min)+4*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))+(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2)*T.exp((alpha*l5**2+alpha*l2**2+2*beta**2*T.log(E_max)**2)/(2*beta*l5+2*beta*l2))*5**(((alpha**2+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2))*2**(((alpha**2+4*alpha+1)*l5+alpha**2*l2+l2+4*beta*T.log(E_min)+4*alpha*beta*T.log(E_max))/(4*beta*l5+4*beta*l2)))*phi)/(beta**(3/2)*2**(((4*beta+alpha**2+alpha+1)*l5+4*beta*l2+alpha*l2+2*alpha*beta*T.log(E_min)+2*alpha*beta*T.log(E_max))/(2*beta*l5+2*beta*l2))*5**((alpha*l5+alpha*l2+2*alpha*beta*T.log(E_min)+2*alpha*beta*T.log(E_max))/(2*beta*l5+2*beta*l2)))
  return dalpha

def dphi(E_min, E_max, phi, alpha, beta):
  l2 = np.log(2)
  l5 = np.log(5)
  dphi=-(2**((-alpha/(2*beta))-1)*(T.sqrt(3.14159265)*2**((alpha**2+1)/(4*beta))*5**((alpha**2+1)/(4*beta))*T.sqrt(l5+l2)*erf((T.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*T.log(E_min)))/(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2))-T.sqrt(3.14159265)*2**((alpha**2+1)/(4*beta))*5**((alpha**2+1)/(4*beta))*T.sqrt(l5+l2)*erf((T.sqrt(l5+l2)*((alpha-1)*l5+(alpha-1)*l2+2*beta*T.log(E_max)))/(2*T.sqrt(beta)*l5+2*T.sqrt(beta)*l2))))/(T.sqrt(beta)*5**(alpha/(2*beta)))
  return dphi
