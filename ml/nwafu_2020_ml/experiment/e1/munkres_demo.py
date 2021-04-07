from munkres import Munkres, print_matrix

matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
m = Munkres()
indexes = m.compute(matrix)
print(indexes)
print_matrix(matrix, msg='Lowest cost through this matrix')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print('(%d,%d)->%d' % (row, column, value))
print('total cost:%d' % total)