# Creating a Transfer Function of a Continuous Time LTI System Using Python


## Prerequisites and Setup


```python
import numpy as np
import control as ct
```

## Method 1: Defining by Polynomial Coefficients



```python
#Import required Python packages
import numpy as np
import control as ct
#==================================================
# Given G(s)=(2s+1)/(s^2+4s+6)
# Define numerator and denominator coefficients as arrays
num = np.array([2,1])          
den = np.array([1,4,6])
#====================================================
# Create transfer function
G = ct.tf(num, den)
print('G=',G)
```

    G= 
       2 s + 1
    -------------
    s^2 + 4 s + 6
    
    

## Method 2: Algebraic Definition using the s Operator


```python
#Import required Python packages
import numpy as np
import control as ct
#=========================================================================
# Define the Laplace variable 's'
s=ct.TransferFunction.s #Define the 's' variable using the 'TransferFunction' constructor
#==================================================================
# Define the transfer function G(s) algebraically
# G = (2s+1) / (s^2+4s+6)
G = (2*s+1) / (s**2+4*s+6)
print('G(s) =', G)
```

    G(s) = 
       2 s + 1
    -------------
    s^2 + 4 s + 6
    
    

## Method 3:  When the transfer function is expressed in pole, zero,gain form



```python
import control as ct
#=========================================
# Example: H(s)=10(s+1)/((s+2)(s+5)(s+7))
# zero at s= -1, poles at s= -2, -5, -7 and the gain is 10
#Create transfer function
H = ct.zpk([-1], [-2,-5,-7], 10)
print('H(s)=',H)
```

    H(s)= 
           10 s + 10
    ------------------------
    s^3 + 14 s^2 + 59 s + 70
    
    

# Method 4: Creating from State-Space Representation

### Create State-Space model from Transfer Function


```python
import numpy as np
import control as ct
#========================================================
# System Matrices for a 2nd order system demonstration:
A = np.array([[0.0, 1.0], 
              [-1.0, -3.0]])
B = np.array([[0.0], 
              [1.0]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])
#=====================================================
# Create the State-Space model of the system
ss_model = ct.ss(A, B, C, D)
print('A=',A)
print('B=',B)
print('C=',C)
print('D=',D)
```

    A= [[ 0.  1.]
     [-1. -3.]]
    B= [[0.]
     [1.]]
    C= [[1. 0.]]
    D= [[0.]]
    

### Convert the State-Space model to a Transfer Function


```python
# Convert the State-Space model to a Transfer Function
H_ss_to_tf = ct.ss2tf(ss_model)
print('TF from State-Space Model H(s) =', H_ss_to_tf)
```

    TF from State-Space Model H(s) = 
          1
    -------------
    s^2 + 3 s + 1
    
    

### Convert the Transfer Function to a State-Space model 


```python
# Convert the Transfer Function to a State-Space model  
H_tf_to_ss = ct.tf2ss(H_ss_to_tf)
print('TF from State-Space Model H(s) =', H_tf_to_ss)
```

    TF from State-Space Model H(s) = A = [[-3. -1.]
         [ 1.  0.]]
    
    B = [[1.]
         [0.]]
    
    C = [[0. 1.]]
    
    D = [[0.]]
    
    

#  Combining Transfer Functions (Series and Feedback)


# Define Subsystems


```python
# Define G1(s) = 1 / (s + 1)
num1 = np.array([1])
den1 = np.array([1, 1])
G1 = ct.tf(num1, den1)
#================================
# Define G2(s) = 5 / (2*s + 1)
num2 = np.array([5])
den2 = np.array([2, 1])
G2 = ct.tf(num2, den2)
#===============================
print('G1(s) =', G1,' and G2(s)=',G2)

```

    G1(s) = 
      1
    -----
    s + 1
      and G2(s)= 
       5
    -------
    2 s + 1
    
    

# Series (Cascade) Combination


```python
# Series Combination
H_series = ct.series(G1, G2)
print('H_series(s) = G1(s)G2(s) =', H_series)
```

    H_series(s) = G1(s)G2(s) = 
           5
    ---------------
    2 s^2 + 3 s + 1
    
    


```python
#Parallel CombinationG1+G2
H_Parallel=ct.parallel(G1,G2)
print('H_Parallel(G1+G2)=',H_Parallel)
```

    H_Parallel(G1+G2)= 
        7 s + 6
    ---------------
    2 s^2 + 3 s + 1
    
    


```python
#Parallel CombinationG1-G2
H_Parallel=ct.parallel(G1,-1*G2)
print('H_Parallel(G1-G2)=',H_Parallel)
```

    H_Parallel(G1-G2)= 
       -3 s - 4
    ---------------
    2 s^2 + 3 s + 1
    
    

#  Feedback Combination (Closed Loop)


```python
# Negative Feedback Combination: G is the forward path, H is the feedback path
# T(s) = G / (1 + G*H)
# Define G1(s) = 1 / (s + 1)
num1 = np.array([1])
den1 = np.array([1, 1])
G = ct.tf(num1, den1)
# Define G2(s) = 5 / (2*s + 1)
num2 = np.array([5])
den2 = np.array([2, 1])
#=======================================
```


```python
H = ct.tf(num2, den2)
print('G(s) =', G)
print('H(s) =', H)
H_feedback = ct.feedback(G, H)
print('\nH_feedback(s) = G / (1 + G*H) =', H_feedback)
```

    G(s) = 
      1
    -----
    s + 1
    
    H(s) = 
       5
    -------
    2 s + 1
    
    
    H_feedback(s) = G / (1 + G*H) = 
        2 s + 1
    ---------------
    2 s^2 + 3 s + 6
    
    


```python
# Negative Feedback Combination: G is the forward path, H is the feedback path
# T(s) = G / (1 - G*H)
H_feedback_positive = ct.feedback(G, H,1)
print('\nH_feedback(s) = G / (1 - G*H) =', H_feedback_positive)
```

    
    H_feedback(s) = G / (1 - G*H) = 
        2 s + 1
    ---------------
    2 s^2 + 3 s - 4
    
    

#  Modeling Time Delay using Padé Approximation

Python Code (Padé Approximation for Time Delay)
Consider the Air Heater parameters: Kh=3.5,theta1=22 and time delay  theta_d=2, approximation order=5.


```python
# Process Parameters
Kh = 3.5
theta_t = 22
theta_d = 2 # Time Delay (tau)
#=======================================
# 1. Define the base first-order transfer function (without delay)
# H1(s) = Kh/(theta_t(s)+1)=3.5 / (22*s + 1)
num = np.array([Kh])
den = np.array([theta_t, 1])
H1 = ct.tf(num, den)
```


```python
# 2. Define the order of the Padé Approximation 
N_pade = 5 
[num_pade, den_pade] = ct.pade(theta_d, N_pade)
# 3. Create the Transfer Function representing the delay
Hpade = ct.tf(num_pade, den_pade)
print('\nHpade(s) (Time Delay Approximation) =', Hpade)
# 4. Combine the systems in series to get the full system H(s)
H_delay = ct.series(H1, Hpade)
print('\nFull System H(s) = H1(s) * Hpad(s) =', H_delay)
```

    
    Hpade(s) (Time Delay Approximation) = 
    -s^5 + 15 s^4 - 105 s^3 + 420 s^2 - 945 s + 945
    -----------------------------------------------
    s^5 + 15 s^4 + 105 s^3 + 420 s^2 + 945 s + 945
    
    
    Full System H(s) = H1(s) * Hpad(s) = 
            -3.5 s^5 + 52.5 s^4 - 367.5 s^3 + 1470 s^2 - 3308 s + 3308
    --------------------------------------------------------------------------
    22 s^6 + 331 s^5 + 2325 s^4 + 9345 s^3 + 2.121e+04 s^2 + 2.174e+04 s + 945
    
    


```python

```
