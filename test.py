import numpy as np
# import argparse
# parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
# parser.add_argument("--random-scale", action="store_true",
#                         help="Whether to randomly scale the inputs during the training.")
# args = parser.parse_args()
# print(args.random_scale)
# i = ['a', 'b']
# l = [1, 2]
# b = dict([i,l])
# print(b['a'])

# class A():
#     def __init__(self,len=6):
#         self.len = len
#
#     def ab(self):
#         for i in range(len):
#             yield i
#
# a = A()
# c = a.ab()
# print(a)
a = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
c = np.array([1, 2, 3])
a_mask = a >= 4
b = a[a_mask]
print(b.shape)
print(c.shape)
print(a_mask)
