A = int(input("Enter number A: "))
B = int(input("Enter the number B:"))

user = input("Enter thr opreator(+,-,/,*): ")

if user == "+":
    print("The answer is:", A+B)
elif user == "-":
    print("The answer is:", A-B)
elif user == "*":
    print("The anser is:", A*B)
else:
    print("The answer is:", A/B)

