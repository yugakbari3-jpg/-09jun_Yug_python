students = {}

n = int(input("Enter number of students : "))

for i in range(n):
    stud_id = input("Enter student ID: ")

    student = {
        "Name": input("Enter Name: "),

        "Grade": input("Enter Grade: "),

        "Subject": input("Enter Subject: ")
    }
    students[stud_id] = student

print(students)