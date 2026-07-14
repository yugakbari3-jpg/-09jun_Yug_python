"""def addition(a,b):
    print("Sum:",a+b)
def subtraction(a,b):
    print("Sum:",a-b)
def multiplication(a,b):
    print("Sum:",a*b)
def division(a,b):
    print("Sum:",a/b)

n1 = int(input("Enter your Number1:"))
n2 = int(input("Enter your Number2:"))

n = input("Enter the opreation you want:")
if n == "addition":
    addition(n1,n2)
elif n == "subtraction":
    subtraction(n1,n2)
elif n == "multiplication":
    multiplication(n1,n2)
elif n == "Division":
    division
else:
    print('Error ocurred')"""

def addition(a,b):
    print("Sum:",a+b)
def subtraction(a,b):
    print("Sum:",a-b)
def multiplication(a,b):
    print("Sum:",a*b)
def division(a,b):
    print("Sum:",a/b)

n1 = int(input("Enter your Number1:"))
n2 = int(input("Enter your Number2:"))

n = input("Enter the opreation you want:")
if n == "+":
    addition(n1,n2)
elif n == "-":
    subtraction(n1,n2)
elif n == "*":
    multiplication(n1,n2)
elif n == "/":
    division
else:
    print('Error ocurred')