friends = {}

username = input("Enter username: ")
followers = float(input("Enter followers (in K): "))
    
friends[username] = followers

for username in friends:
    print(f"{username}: {friends[username]}K followers")