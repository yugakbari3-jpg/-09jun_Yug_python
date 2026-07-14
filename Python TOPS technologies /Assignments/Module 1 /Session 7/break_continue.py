x = ['Hi', 'Spam', 'Hello', 'Spam', 'How are you?']
for x in x:
    if x== 'Spam':
        continue
    if x== "How are you?":
        break
    print(x)
