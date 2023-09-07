import numpy as np

# a = np.random.randn(5)
# print('a', a)

# expa = np.exp(a)
# print('expa', expa)

# answer = expa / expa.sum()

# print('answer', answer)
# print(sum(answer))


A = np.random.randn(100,5)
print('A', A)

expA = np.exp(A)
print('expA', expA)

answerA = expA / expA.sum(axis=1, keepdims = True)

print('answer', answerA)
print(sum(answer))