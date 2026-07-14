def getdata(id,name,city):
    print("ID",id)
    print("Name:",name)
    print("City:",city)

n = int(input("Enter no of Students:"))
for i in range(n):
    n1 = int(input("Enter your ID:"))
    n2 = input("Enter your Name:")
    n3 = input("Enter your City:")

getdata(n1,n2,n3)