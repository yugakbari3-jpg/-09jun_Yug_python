score = int(input("Enter you favorite cricket team score:"))

if score > 200:
    print("'High Score!'")
elif score >= 150 or score <= 199:
    print("'Good score'")
elif score >= 100 or score >= 149:
    print("'Average'")
else:
    print("'Needs Improveement'")
