myset = {'Apple', 'Samasung', 'oppo', 'oneplus', 'Nothing'}

#print(myset)

if 'Apple' in myset:
    print("Yes...")
else:
    print('No...')
#=================================#
"""for i in myset:
    print(i)"""
#--------------------------------------#

myset.add('Gran Velocita')
myset.update(['Calculator','Amazon','Ajio Luxe'])
myset.remoeve('f')
myset.pop()
myset.clear()
print(myset)