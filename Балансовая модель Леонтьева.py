import numpy as np
import pandas as pd
from tabulate import tabulate

# Матрица коэффициентов прямых затрат
A = np.array([[0.2,0.3,0.1],
            [0,0.3,0.3],
             [0.4,0.1,0.2]])

# Вектор объемов конечной продукции
Y = [1000,800,700]

#Уравнения межотраслевого баланса
print('Уравнения межотраслевого баланса:')
print('x1=', A[0][0],'x1 + ', A[0][1],'x2 + ', A[0][2], 'x3 + ', Y[0])
print('x2=', A[1][0],'x1 + ', A[1][1],'x2 + ', A[1][2], 'x3 + ', Y[1])
print('x3=', A[2][0],'x1 + ', A[2][1],'x2 + ', A[2][2], 'x3 + ', Y[2])

def matrix(array):
     for i in range(len(array)):
         for j in range(len(array[i])):
             print(array[i][j], end=' ')
         print()
#E-A

E = np.eye(3)
E_A = E-A
print('\nE-A:')
matrix(E_A)

#Проверка условия Хаукинса-Саймона
prom_matrix = np.delete(E_A,0,0)
mini_matrix1 = np.delete(prom_matrix,0,1)
det1 = round(np.linalg.det(mini_matrix1),3)

prom_matrix = np.delete(E_A,1,0)
mini_matrix2 = np.delete(prom_matrix,1,1)
det2 = round(np.linalg.det(mini_matrix2),3)

prom_matrix = np.delete(E_A,2,0)
mini_matrix3 = np.delete(prom_matrix,2,1)
det3 = round(np.linalg.det(mini_matrix3),3)

print('\nУгловые определители:')
print('Определитель 1:',det1,'\nОпределитель 2:',det2,'\nОпределитель 3:', det3)

# Обратная матрица D=(E_A)^(-1)
D = np.linalg.inv(E_A)
D_okrugl = np.around(D,3)

print('\nD:',D_okrugl)

# Матрица Х

X = D.dot(Y)
X_okrugl = np.around(X,2)

print('\nX:',X_okrugl)

# Матрица потоков средств производства В
B = []

n=0
while n < 3:
    bij = A[n] * X[n]
    B.append(bij)
    n = n + 1

B_okrugl = np.around(B,3)

print('B:')
print(matrix(B_okrugl))

# P
P = []
promezh = []
summ_column = 0

for j in range(len(B)):
    summ_column = 0
    for i in range(len(B)):
        summ_column += B[i][j]
    promezh.append(summ_column)


for i in range(len(promezh)):
    p_prom = X[i] - promezh[i]
    P.append(p_prom)

print('P:',P)

#Таблица
name = ['потребители / производители','1', '2', '3', 'конечный продукт', 'валовой продукт']

for x in range(len(B_okrugl)):
    df = pd.DataFrame(B_okrugl)

df.loc[len(df.index)] = P
df.loc[len(df.index)] = X_okrugl

new_array = np.append(X_okrugl, ['', ''])
new_Y = np.append(Y, ['', ''])
df.insert(3, 'конечный продукт',new_Y, False)
df.insert(4, ' ', new_array, False)
df.index = ['1', '2', '3', 'общий доход', "валовой продукт"]
print(tabulate(df, headers=name, tablefmt='fancy_grid', stralign='center'))

# Увеличение конечного продукта 1-й и 2-й отрасли на n%
print('Увеличение конечного продукта 1-й и 2-й отрасли на 3%')
n = 1.03
Y_new = [Y[0]*n,Y[1]*n,Y[2]]

# Матрица Х
X_new = D.dot(Y_new)
X_okrugl_new = np.around(X_new,2)

# Матрица потоков средств производства В
B_new = []
n=0
while n < 3:
    bij = A[n] * X_new[n]
    B_new.append(bij)
    n = n + 1

B_okrugl_new = np.around(B_new,3)

# P
P_new = []
promezh = []
summ_column = 0

for j in range(len(B_new)):
    summ_column = 0
    for i in range(len(B_new)):
        summ_column += B_new[i][j]
    promezh.append(summ_column)


for i in range(len(promezh)):
    p_prom = X_new[i] - promezh[i]
    P_new.append(p_prom)


#Таблица
name = ['потребители / производители','1', '2', '3', 'конечный продукт', 'валовой продукт']

for x in range(len(B_okrugl_new)):
    df = pd.DataFrame(B_okrugl_new)

df.loc[len(df.index)] = P_new
df.loc[len(df.index)] = X_okrugl_new

new_array = np.append(X_okrugl_new, ['', ''])
new_Y = np.append(Y_new, ['', ''])
df.insert(3, 'конечный продукт',new_Y, False)
df.insert(4, ' ', new_array, False)
df.index = ['1', '2', '3', 'общий доход', "валовой продукт"]
print(tabulate(df, headers=name, tablefmt='fancy_grid', stralign='center'))
