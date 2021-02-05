import numpy as np

print("Numpy version : " + np.__version__)

print("- lists ----------------------------")
mylist = [[1, 2], [3, 4], [5, 6], [7, 8]]
print(np.arange(0, 10, 2))
print(np.array(mylist))
print(np.zeros((3, 7)))
print(np.linspace(0, 7, 10))
print(np.eye(5))

print("- random ---------------------------")
print(np.random.randint(3, 7, 5))
np.random.seed(42)
print(np.random.rand(3))
np.random.seed(42)
print(np.random.rand(3))

print("- shape ----------------------------")
arr = np.arange(25)
ranarr = np.random.randint(0, 50, 10)
print(np.array(mylist).shape)
print(np.array(mylist)[0,1])
print(arr.reshape(5,5))
print(ranarr.max())
print(ranarr.argmax())

print("- index ----------------------------")
print(arr[3])
print(arr[:3])
print(arr[3:9])
print(arr)
arr[5:7] = 00
print(arr)
arr_cpy = arr.copy()

print("- bool -----------------------------")
bool_arr = arr > 4
print(bool_arr)

print("- operator  ------------------------")
print(arr-2)
print(arr.sum())
print(arr.mean())
print(arr.max())
print(arr.min())
print(arr.var())
print(arr.std())

arr2 = np.arange(0,25).reshape(5, 5)
print(arr2)
print(arr2.sum(axis=1))

print("- list of list  ------------------------")

lst1 = ['a', 'b', 'c']
lst2 = ['x', 'y','z']

lst3 = []
for x in  lst1:
  for y in lst2:
    lst3.append((x, y))

print(lst3)

print("end of program")
