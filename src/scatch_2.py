# from datetime import datetime
# print(
#     (datetime.strptime('1/1/09', '%m/%d/%y')).year
# )


import math
mean = 10 
sigma = -20 

sigma_sq = math.pow(sigma, 2)
esig_sq = math.exp(sigma_sq)

mode = math.exp(mean - sigma_sq)
lmean = math.exp(mean + math.pow(sigma, 2)/2)

skewness = (esig_sq + 2) * math.sqrt((esig_sq - 1))
skunk = (esig_sq + 2) * (esig_sq - 1) ** (1/2)

print(f'mode: {mode}')
print(f'lmean: {lmean}')
print(f'lower: {mode / 2}')
lower = mode / 2
print(f'skewness: {skewness}')
print(f'upper: {lmean + skewness}')
upper = lmean + skewness
print(f'upper > lower: {upper > lower}')


# # for i in reversed(range(7)):
# #     print(i)


# from z3 import *
# x = Int('x')
# y = Int('y')

# print(simplify(If(1 == 1, x, y) + ArithRef(1)))


# x = [[], 2, [3]]
# x.append([1,2,3])
# print(x)
