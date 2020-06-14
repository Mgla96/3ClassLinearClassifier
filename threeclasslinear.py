# Starter code for CS 165B HW2 Spring 2019
import numpy as np


def run_train_test(training_input, testing_input):
    """
    Implementing simple three-class linear classifier 
    """
    # dictionary
    sol = {
        "tpr": 0,
        "fpr": 0,
        "error_rate": 0,
        "accuracy": 0,
        "precision": 0
    }
    # training
    # num of features in each instance aka how many dimensions ex 1D, 2D, 3D
    D = training_input[0][0]
    N1 = training_input[0][1]  # num of examples in class A
    N2 = training_input[0][2]  # num of examples in class B
    N3 = training_input[0][3]  # num of examples in class C
    training_input.pop(0)
    aTraining = np.array(training_input[0:N1])
    bTraining = np.array(training_input[N1:N1+N2])
    cTraining = np.array(training_input[N1+N2:N1+N2+N3])

    aCentroid = np.zeros(D)
    bCentroid = np.zeros(D)
    cCentroid = np.zeros(D)
    for i in range(D):
        aTotal = 0
        bTotal = 0
        cTotal = 0
        for j in range(len(aTraining)):
            aTotal += aTraining[j][i]
        for k in range(len(bTraining)):
            bTotal += bTraining[k][i]
        for l in range(len(cTraining)):
            cTotal += cTraining[l][i]
        aCentroid[i] = aTotal/float(N1)
        bCentroid[i] = bTotal/float(N2)
        cCentroid[i] = cTotal/float(N3)

    # discriminant function for A/B,B/C,A/C
    wab = np.zeros(D)
    wbc = np.zeros(D)
    wac = np.zeros(D)
    wab = aCentroid-bCentroid
    wbc = bCentroid-cCentroid
    wac = aCentroid-cCentroid
    tab = np.dot((np.transpose(aCentroid+bCentroid)), wab)/2
    tbc = np.dot((np.transpose(bCentroid+cCentroid)), wbc)/2
    tac = np.dot((np.transpose(aCentroid+cCentroid)), wac)/2

    # DTest = testing_input[0][0] #num of features in each instance aka how many dimensions ex 1D, 2D, 3D
    N1Test = testing_input[0][1]  # num of examples in class A
    N2Test = testing_input[0][2]  # num of examples in class B
    N3Test = testing_input[0][3]  # num of examples in class C
    testing_input.pop(0)
    aTest = np.array(testing_input[:N1Test])
    bTest = np.array(testing_input[N1Test:N1Test+N2Test])
    cTest = np.array(testing_input[N1Test+N2Test:N1Test+N2Test+N3Test])

    # Use discriminant function to decide "A" or "B" and then depending on answer
    # decide "A or C" or "B or "C"
    # keep track of true positives, true negatives, false positives, and false negatives
    TPA = 0
    FPA = 0
    FNA = 0
    TNA = 0

    TPB = 0
    FPB = 0
    FNB = 0
    TNB = 0

    TPC = 0
    FPC = 0
    FNC = 0
    TNC = 0

    for i in range(len(aTest)):
        if(np.dot(aTest[i], wab) >= tab):  # a
            if (np.dot(aTest[i], wac) >= tac):  # A and A
                TPA += 1  # correct
                TNB += 1  # correct
                TNC += 1  # correct
                print("A in A")
            else:  # c
                FPC += 1  # correct
                FNA += 1  # correct
                TNB += 1  # correct
                print("C in A")
        else:  # b
            if(np.dot(aTest[i], wbc) >= tbc):  # b
                FPB += 1  # correct said it's B but it's not B
                FNA += 1  # correct said it's not A but it is A
                TNC += 1  # correct said it's not C and it's not C
                print("B in A")
            else:  # c
                FPC += 1  # correct said it's C but it's not C
                FNA += 1  # correct said it's not A but it is A
                TNB += 1  # correct said it's not B and it isn't B
                print("C in A")

    for i in range(len(bTest)):
        if(np.dot(bTest[i], wab) >= tab):  # a
            if (np.dot(bTest[i], wac) >= tac):  # a
                FPA += 1  # correct said it's a but it's not A
                FNB += 1  # correct said not B but it was B
                TNC += 1  # correct said not c and it's not c
                print("A in B")
            else:  # c
                TNA += 1  # Correct said it's not A and it's not A
                FNB += 1  # correct said it's not B but it is B
                FPC += 1  # correct said it's c but it's not c
                print("C in B")
        else:  # b
            if(np.dot(bTest[i], wbc) >= tbc):  # B and B
                TPB += 1  # correct
                TNA += 1  # correct
                TNC += 1  # correct
                print("B in B")  # **
            else:  # c?
                FPC += 1  # correct said c but not c
                TNA += 1  # correct said not a and not a
                FNB += 1  # correct said not b but should be b
                print("C2 in B")  # **

    for i in range(len(cTest)):
        if(np.dot(cTest[i], wab) >= tab):  # a
            if (np.dot(aTest[i], wac) >= tac):  # a
                FPA += 1  # correct said it was a but it's not a
                TNB += 1  # correct said not b and it's not b
                FNC += 1  # correct said not c but it was c
                print("A in C")
            else:  # c
                TPC += 1  # correct
                TNA += 1  # correct
                TNB += 1  # correct
                print("C in C")
        else:  # b
            if(np.dot(cTest[i], wbc) >= tbc):  # b
                TNA += 1  # correct said not a and isn't a
                FPB += 1  # correct said b but it isn't b
                FNC += 1  # said not C but it was C
                print("B in C")  # **
            else:  # c
                TPC += 1  # correct
                TNA += 1  # correct
                TNB += 1  # correct
                print("C2 in C")  # **
    sol["tpr"] = (TPA+TPB+TPC) / float(TPA+TPB+TPC+FNA +
                                       FNB+FNC)  # true positive / total positive
    # false positive / total negative and positive
    sol["fpr"] = (FPA+FPB+FPC) / float(TNA+TNB+TNC+FPA+FPB+FPC)
    # false positive and negative / total negative and positive
    sol["error_rate"] = (FPA+FPB+FPC+FNA+FNB+FNC) / \
        float(TPA+TPB+TPC+FPA+FPB+FPC+TNA+TNB+TNC+FNA+FNB+FNC)
    # true positive and true negative / total positive and negative
    sol["accuracy"] = (TPA+TPB+TPC+TNA+TNB+TNC) / \
        float(TPA+TPB+TPC+FPA+FPB+FPC+TNA+TNB+TNC+FNA+FNB+FNC)
    # true positives / estimated positive aka true positives + false positives
    sol["precision"] = (TPA+TPB+TPC) / float(TPA+TPB+TPC+FPA+FPB+FPC)
    print(sol)
    return sol
