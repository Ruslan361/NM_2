def Thomas_Shelby_algorythm(matrix_A, vector_b):
    size = len(vector_b)
    y = []
    alpha = [0,]
    beta = [0,]

    A = [0,]
    B = [0,]
    C = [0,]

    for i in range(1, size-1):
        A.append(matrix_A[i][0])
        C.append(-matrix_A[i][1])
        B.append(matrix_A[i][2])

    alpha.append(0)
    beta.append(vector_b[0])  #мю_1
    for i in range(1, size-1):
        alpha.append(B[i]/(C[i] - A[i]*alpha[i]))
        beta.append((-vector_b[i] + A[i]*beta[i])/(C[i] - A[i]*alpha[i]))

    y.append(vector_b[-1])
    for i in range(size-1, 0, -1):   #идем от послепредпоследнего элемента до предпослепервого
        y.insert(0, alpha[i]*y[0]+beta[i])
    return y