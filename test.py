from itertools import chain

lst1 = ['a', 'b', 'c']
lst2 = ['x', 'y','z']

lst3 = []
for x in  lst1:
  for y in lst2:
    lst3.append((x, y))

print(lst3)
#print(list(chain.from_iterable([lst1, lst2])))
