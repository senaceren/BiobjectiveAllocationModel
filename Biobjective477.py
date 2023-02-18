import csv

import numpy as np
import pandas as pd

#indices
W = [0,1,2,3,4]
L = [0,1,2,3,4]
K = [k for k in range(1, 10)]
#C = [1,2,3,4,5]
ar = [1,2,3,4]
val = str("value")

P = 99999999999999999999
H = 200
F = 9999999999

Arc = [(i,j) for i in W for j in L]
wareskuzone = [(i,j,k) for i in W for j in L for k in K]
waresku = [(i,k) for i in W for k in K]
zonesku = [(j,k) for j in L for k in K]
waretoware = [(i1,i2) for i1 in W for i2 in W]
inter = [(i1,i2,k) for i1 in ar for i2 in W for k in K]

#Parameters

Travel = np.array([[4.32272793586998, 8.082408004452557, 3.783474882452186, 12.075338714744266, 0.4989462540292099],
[4.32272793586998, 8.082408004452557, 3.783474882452186, 12.075338714744266, 0.4989462540292099],
[4.32272793586998, 8.082408004452557, 3.783474882452186, 12.075338714744266, 0.4989462540292099],
[4.32272793586998, 8.082408004452557, 3.783474882452186, 12.075338714744266, 0.4989462540292099],
[4.32272793586998, 8.082408004452557, 3.783474882452186, 12.075338714744266, 0.4989462540292099],
[4.32272793586998, 8.082408004452557, 3.783474882452186, 12.075338714744266, 0.4989462540292099]])
T = {(i,j): Travel[i][j] for i,j in Arc}

Demand = np.random.randint(70,90, size = (5,10))
D = {(j,k): Demand[j][k] for j,k in zonesku}

#Supply = np.random.randint(1999,5999, size = (10))
Supply = np.array([450, 450, 450, 450,450, 450, 450, 450, 450, 450])
S ={k: Supply[k] for k in K}

Volume = np.random.randint(1,5, size = (10))
V = {k: Volume[k] for k in K}

Initial = np.random.randint(15,30, size = (6,10))
I0 = {(i,k): Initial[i][k] for i,k in waresku}

Capacity = np.random.randint(9999699,9999999, size=(6))
C = {i: Capacity[i] for i in W}

rows = 5
cols = 5
TravCost = np.empty((rows, cols))
Warehouse = np.random.randint(5,15, size=(5,5))
FixCost = 50
KmCost = 2
TravCost = KmCost*Warehouse + FixCost
#print(TravCost)

TC = {(i1,i2): TravCost[i1,i2] for i1 in W for i2 in W}

def epsilon_constraint_allocation(epsilon):

    from gurobipy import Model, GRB, quicksum

    mdl = Model("Biobjective")

    y = mdl.addVars(wareskuzone, lb=0.0, vtype=GRB.INTEGER, name="y")
    B = mdl.addVars(zonesku, lb=0.0, vtype=GRB.INTEGER, name="B")
    E = mdl.addVars(K, lb=0.0, vtype=GRB.INTEGER, name="E")
    Z = mdl.addVars(inter, lb=0.0, vtype=GRB.INTEGER, name="Z")
    N = mdl.addVars(waretoware, lb=0.0, vtype=GRB.INTEGER, name="N")
    A = mdl.addVars(waresku, lb=0.0, vtype=GRB.INTEGER, name="A")

    mdl.modelSense = GRB.MINIMIZE
    obj = quicksum((T[i,j]*quicksum(y[i,j,k] for k in K)) for i in W for j in L) + P*quicksum(B[j,k] for j,k in zonesku) + F*quicksum(E[k] for k in K)
    mdl.setObjective(obj)

    # 1
    mdl.addConstrs(quicksum(y[i, j, k] for i in W) + B[j, k] >= D[j, k] for j, k in zonesku);
    # 2
    mdl.addConstrs(
        quicksum(y[0, j, k] for j in L) <= A[0, k] + quicksum(Z[i, 0, k] for i in ar) - quicksum(A[i, k] for i in ar) +
        I0[0, k] for k in K);
    # 3
    mdl.addConstrs(quicksum(y[i, j, k] for j in L) <= A[i, k] + quicksum(Z[i1, i, k] for i1 in ar) - quicksum(
        Z[i, i2, k] for i2 in W) + I0[i, k] for k in K for i in ar);
    # 4
    mdl.addConstrs(quicksum(
        V[k] * (I0[i, k] + A[i, k] + quicksum(Z[i1, i, k] for i1 in ar) - quicksum(Z[i, i2, k] for i2 in W)) for k in
        K) <= C[i] for i in ar);
    # 5
    mdl.addConstrs(quicksum(
        V[k] * (I0[0, k] + A[0, k] + quicksum(Z[i, 0, k] for i in ar) - quicksum(A[i, k] for i in ar)) for k in K) <= C[
                       0] for i in W);
    # 6
    mdl.addConstrs(
        I0[i, k] + A[i, k] + quicksum(Z[i1, i, k] for i1 in ar) >= quicksum(Z[i, i2, k] for i2 in W) for k in K for i in
        ar);
    # 7
    mdl.addConstrs(I0[0, k] + A[0, k] + quicksum(Z[i, 0, k] for i in ar) >= quicksum(A[i, k] for i in ar) for k in K);
    # 8
    mdl.addConstrs(quicksum(V[k] * Z[i1, i2, k] for k in K) == H * N[i1, i2] for i2 in W for i1 in ar);
    # 9
    mdl.addConstrs(Z[i, i, k] == 0 for i in ar for k in K)
    # 10
    mdl.addConstrs(quicksum(A[i, k] for i in W) == S[k] - E[k] for k in K)
    # 11
    mdl.addConstrs(
        quicksum(V[k] * E[k] for k in K) >= quicksum(V[k] * I0[i, k] for k in K) + quicksum(V[k] * S[k] for k in K) - C[
            i] for i in W)

    mdl.optimize()

    #10
    LHS = {(val): quicksum(TC[i1, i2]*N[i1,i2] for i1 in ar for i2 in W)}
    mdl.addConstr(LHS[(val)] <= epsilon)

    mdl.optimize()

    #var_names = []
    #var_values = []

    #for var in mdl.getVars():
        #if var.X != 0:
            #var_names.append(str(var.varName))
            #var_values.append(var.X)

    #with open("denemeout.csv", "w") as myfile:
        #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #wr.writerows(zip(var_names, var_values))


    if mdl.Status == GRB.Status.INFEASIBLE:
        return("Inf")
    return [obj.getValue(), LHS[val].getValue()]

points = []
epsilon = 999999999999999999
while epsilon_constraint_allocation(epsilon) != "Inf":
    points.append(epsilon_constraint_allocation(epsilon))
    epsilon = epsilon_constraint_allocation(epsilon)[1] - 1
print(points)

#Sanırım error'u N değerleri çıkmadığından yaşıyorum bazı senaryolarda.