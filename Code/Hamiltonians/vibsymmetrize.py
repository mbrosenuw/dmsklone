import numpy as np


def generate_primitive_basis(n_max):
    """Generate the list of primitive basis states (n1, n2)."""
    states = [(n1, n2) for n1 in range(-n_max, n_max + 1) for n2 in range(-n_max, n_max + 1)]
    return states


def symmetrized_basis(n1, n2, gamma):
    """Return the symmetrized basis state as a vector in the primitive basis."""
    exist = True
    if n1 != np.abs(n2):
        if gamma == 0:
            coeff = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
        elif gamma == 1:
            coeff = [1 / 2, -1 / 2, 1 / 2, -1 / 2]
        elif gamma == 2:
            coeff = [1 / 2, 1 / 2, -1 / 2, -1 / 2]
        elif gamma == 3:
            coeff = [1 / 2, -1 / 2, -1 / 2, 1 / 2]
        

        states = [
            (n1, n2),
            (-n1, -n2),
            (n2, n1),
            (-n2, -n1)
        ]
    else:
        if n1 == 0:
            if gamma == 0:
                coeff = [1]
            else:
                coeff = [0]
                exist = False
            states = [
                (n1, n1)
            ]
        elif n1 == -n2:
            if gamma == 0:
                coeff = [1 / np.sqrt(2), 1 / np.sqrt(2)]
            elif gamma == 3:
                coeff = [1 / np.sqrt(2), -1 / np.sqrt(2)]
            else:
                coeff = [0, 0]
                exist = False

            states = [
                (n1, -n1),
                (-n1, n1)
            ]
        else:
            if gamma == 0:
                coeff = [1/np.sqrt(2), 1/np.sqrt(2)]
            elif gamma == 1:
                coeff = [1/np.sqrt(2), -1/np.sqrt(2)]
            else:
                coeff = [0, 0]
                exist = False

            states = [
                (n1, n1),
                (-n1, -n1)
            ]
    return coeff, states, exist


def gettrans(n_max):
    """Build the transformation matrix from primitive to symmetrized basis."""
    primitive_basis = generate_primitive_basis(n_max)
    # for b in primitive_basis: print(b)
    symmetrized_basis_list = []
    T = []
    for gamma in [0, 1, 2, 3]:
        subT = [[],[],[],[]]
        subbasis = [[],[],[],[]]
        for n1 in range(0, n_max + 1):
            for n2 in range(-n1, n1 + 1):
                coeff, states, exist = symmetrized_basis(n1, n2, gamma)
                if exist == True:
                    row = np.zeros(len(primitive_basis))
                    for c, (p1, p2) in zip(coeff, states):
                        if (p1, p2) in primitive_basis:
                            idx = primitive_basis.index((p1, p2))
                            row[idx] += c
                    # modn1 = (np.mod(n1,3) == 0)
                    # modn2 = (np.mod(n2, 3) == 0)
                    # if modn1 and modn2: l = 0
                    # elif modn1 ^ modn2: l = 1
                    # else: l = 2
                    modn1 = np.mod(n1,3)
                    modn2 = np.mod(n2, 3)
                    if modn1 == 0 and modn2 == 0:
                        l = 0
                    elif modn1 == 1 and modn2 == 0:
                        l = 1
                    elif modn1 == 0 and modn2 == 1:
                        l = 1
                    elif modn1 == 2 and modn2 == 0:
                        l = 1
                    elif modn1 == 0 and modn2 == 2:
                        l = 1
                    elif modn1 == 1 and modn2 == 1:
                        l = 2
                    elif modn1 == 2 and modn2 == 2:
                        l = 2
                    elif modn1 == 1 and modn2 == 2:
                        l = 3
                    elif modn1 == 2 and modn2 == 1:
                        l = 3

                    subT[l].append(row)
                    subbasis[l].append((n1, n2, l, gamma))
        T.extend([row for lchunk in subT for row in lchunk])
        symmetrized_basis_list.extend([b for lchunk in subbasis for b in lchunk])
    return np.array(T), symmetrized_basis_list




