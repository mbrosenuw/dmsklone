import numpy as np
from .vibsymmetrize import gettrans
from scipy.sparse import csr_matrix
from Code.general import timing
import scipy.sparse as sp


class torham():
    def __init__(self, d1, d2, F, Fprime, V3, V6=0, V3p=0, V3m=0, chop=None):
        if chop is None:
            chop = (d1 + 1) * (d2 + 1)
        # print('getting torsion hamiltonian')
        with timing("Constructing Torsional Hamiltonian") as t:
            chop = chop
            In1 = np.eye(d1 + 1)
            In2 = np.eye(d2 + 1)
            Htor = vibH(F, Fprime, V3, V6, V3p, V3m, d1, d2)
            S, symmbasis = gettrans(int(d1 / 2))
            Htor = np.round(S @ Htor @ S.T, 8)
            P, symmbasis = sortfullbasis(symmbasis)
            Htor = P @ Htor @ P.T
            blocks = gettorblocks(symmbasis)
            T = np.zeros(Htor.shape)
            penergies = np.zeros(T.shape[0])
            diagbasis = [None] * len(symmbasis)

        with timing("Diagonalizing Torsional Hamiltonian") as t:
            for (l, gamma), block in blocks.items():
                idxs = block['start']
                idxe = block['end'] + 1
                penergies[idxs:idxe], T[idxs:idxe, idxs:idxe] = np.linalg.eigh(Htor[idxs:idxe, idxs:idxe])
                diagbasis[idxs:idxe] = [(i, l, gamma) for i in range(idxe - idxs)]

        with timing("Truncating Torsional Hamiltonian") as t:
            chopidxs = np.argpartition(penergies, chop)[:chop]
            chopidxs = np.sort(chopidxs)
            T = sp.csr_matrix(T)
            T_csc = T.tocsc()
            TofK = T_csc[:, chopidxs].tocsr()
            chopbasis = [diagbasis[i] for i in chopidxs]
            S = csr_matrix(S)
            self.symmbasis = symmbasis
            self.blocks = blocks
            self.fblocks = gettorblocks4(chopbasis)
            self.basis = chopbasis
            self.TofK = TofK
            self.Htor = TofK.conj().T @ Htor @ TofK
            self.P1 = TofK.conj().T @ P @ S @ sp.kron(p(d1), In2, format='csr') @ S.T @ P.T @ TofK
            self.P2 = TofK.conj().T @ P @S @ sp.kron(In1, p(d2), format='csr') @ S.T @ P.T @ TofK
            self.I = TofK.conj().T @ P @S @ sp.kron(In1, In1, format='csr') @ S.T @ P.T @ TofK


class torham2():
    def __init__(self, d1, d2, F, Fprime, V3, V6=0, V3p=0, V3m=0, chop=None):
        if chop is None:
            chop = (d1 + 1) * (d2 + 1)
        # print('getting torsion hamiltonian')
        with timing("Constructing Torsional Hamiltonian") as t:
            chop = chop
            In1 = np.eye(d1 + 1)
            In2 = np.eye(d2 + 1)
            Htor = vibH(F, Fprime, V3, V6, V3p, V3m, d1, d2)
            prim = pbasis(d1)
            S, symmbasis = sortprimbasis(prim)
            Htor = np.round(S @ Htor @ S.T, 20)
            pmbasis = relabel(symmbasis)
            S, pmmbasis = sortprimbasis(pmbasis)
            Htor = np.round(S @ Htor @ S.T, 20)
            e_vals, e_vecs = sp.linalg.eigsh(sp.csr_matrix(Htor), k=20, which='SA')
            print("Lowest torsional energy levels (cm^-1):")
            for i, e in enumerate(e_vals):
                print(f"{i + 1:2d}: {e:.10f}")
            pHtor = Htor
            blocks = gettorblocks2(symmbasis)
            T = np.zeros(Htor.shape)
            penergies = np.zeros(T.shape[0])
            diagbasis = [None] * len(symmbasis)

        with timing("Diagonalizing Torsional Hamiltonian") as t:
            for (l), block in blocks.items():
                idxs = block['start']
                idxe = block['end'] + 1
                penergies[idxs:idxe], T[idxs:idxe, idxs:idxe] = np.linalg.eigh(Htor[idxs:idxe, idxs:idxe])
                diagbasis[idxs:idxe] = [(i, l) for i in range(idxe - idxs)]

        with timing("Truncating Torsional Hamiltonian") as t:
            chopidxs = np.argpartition(penergies, chop)[:chop]
            chopidxs = np.sort(chopidxs)
            T = sp.csr_matrix(T)
            T_csc = T.tocsc()
            TofK = T_csc[:, chopidxs].tocsr()
            chopbasis = [diagbasis[i] for i in chopidxs]
            S = csr_matrix(S)
            self.symmbasis = symmbasis
            self.blocks = blocks
            self.basis = chopbasis
            self.TofK = TofK
            self.pHtor = csr_matrix(pHtor)
            self.Htor = TofK.conj().T @ Htor @ TofK
            self.P1 = TofK.conj().T @ S @ sp.kron(p(d1), In2, format='csr') @ S.T @ TofK
            self.P2 = TofK.conj().T @ S @ sp.kron(In1, p(d2), format='csr') @ S.T @ TofK
            self.I = TofK.conj().T @ S @ sp.kron(In1, In1, format='csr') @ S.T @ TofK
            self.fblocks = gettorblocks3(self.basis)


def pbasis(n_max):
    states = []
    n_max = int(n_max / 2)
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            modn1 = np.mod(n1, 3)
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
                l = 2
            elif modn1 == 2 and modn2 == 1:
                l = 2

            # modn1 = (np.mod(n1, 3) == 0)
            # modn2 = (np.mod(n2, 3) == 0)
            # if modn1 and modn2:
            #     l = 0
            # elif modn1 ^ modn2:
            #     l = 1
            # else:
            #     l = 2
            states.append((n1, n2, l))
    return states

def relabel(basis):
    new_basis = []
    for b in basis:
        new_basis.append((b[0] + b[1],b[0] - b[1], b[2]))
    return new_basis


def gettorblocks(basis):
    blocks = {}
    for idx, (n1, n2, l, gamma) in enumerate(basis):
        if (l, gamma) not in blocks:
            blocks[(l, gamma)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(l, gamma)]["end"] = idx
            blocks[(l, gamma)]["count"] += 1
    # for (l, gamma), block in blocks.items():
    #     print(
    #         f"Combination {(l,gamma)} starts at {block['start']}, "
    #         f"ends at {block['end']}, and occurs {block['count']} times."
    #     )
    return blocks


def gettorblocks2(basis):
    blocks = {}
    for idx, (n1, n2, l) in enumerate(basis):
        if (l) not in blocks:
            blocks[(l)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(l)]["end"] = idx
            blocks[(l)]["count"] += 1
    return blocks


def gettorblocks3(basis):
    blocks = {}
    for idx, (m, l) in enumerate(basis):
        if (l) not in blocks:
            blocks[(l)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(l)]["end"] = idx
            blocks[(l)]["count"] += 1
    return blocks

def gettorblocks4(basis):
    blocks = {}
    for idx, (m, l,g) in enumerate(basis):
        if (l,g) not in blocks:
            blocks[(l,g)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(l,g)]["end"] = idx
            blocks[(l,g)]["count"] += 1
    return blocks


def p(d):
    mat = np.zeros((d + 1, d + 1))
    for i in range(d + 1):
        mat[i][i] = (i - d / 2)
    return mat


def vn(d, m):
    mat = np.zeros((d + 1, d + 1))
    for n in range(d + 1):
        if n + m < d + 1:
            mat[n + m, n] = 1
        if n - m >= 0:
            mat[n - m, n] = 1
    # print(mat)
    return mat


def vnpm(d, m):
    mat = np.zeros((d + 1, d + 1))
    if m > 0:
        for n in range(d + 1):
            if n + m < d + 1:
                mat[n + m, n] = 1
    elif m < 0:
        for n in range(d + 1):
            if n + m >= 0:
                mat[n + m, n] = 1
    return mat


def Tvib(d1, d2, F, Fprime):
    In1 = np.eye(d1 + 1)
    In2 = np.eye(d2 + 1)
    p1sq = np.kron(np.linalg.matrix_power(p(d1), 2), In2)
    p2sq = np.kron(In1, np.linalg.matrix_power(p(d2), 2))
    p1p2 = np.kron(p(d1), p(d2))
    Tmat = F * (p1sq + p2sq) + 2 * Fprime * p1p2
    # e_vals, e_vecs = sp.linalg.eigsh(csr_matrix(Tmat), k=20, which='SA')
    # print("Lowest torsional energy levels (cm^-1):")
    # for i, e in enumerate(e_vals):
    #     print(f"{i + 1:2d}: {e:.6f}")
    return Tmat


def Vvib(d1, d2, V3, V6, V3p, V3m):
    In1 = np.eye(d1 + 1)
    In2 = np.eye(d2 + 1)
    t1 = 2 * np.kron(In1, In2)
    t2 = 1 / 2 * np.kron(vn(d1, 3), In2)
    t3 = 1 / 2 * np.kron(In1, vn(d2, 3))
    p7 = V3 / 2 * (t1- t2 - t3)
    # p7 = V3 / 2 * (t1 )

    t4 = 1 / 2 * np.kron(vn(d1, 6), In2)
    t5 = 1 / 2 * np.kron(In1, vn(d2, 6))
    p11 = V6/2 * (t1 - t4 - t5)

    t6 = 1 / 2 * np.kron(vnpm(d1, 3), vnpm(d2, 3))
    t7 = 1 / 2 * np.kron(vnpm(d1, -3), vnpm(d2, -3))
    p15 = V3p * (t1 - t6 - t7)

    t6 = 1 / 2 * np.kron(vnpm(d1, 3), vnpm(d2, -3))
    t7 = 1 / 2 * np.kron(vnpm(d1, -3), vnpm(d2, 3))
    p16 = V3m * (t1 - t6 - t7)
    # e_vals, e_vecs = sp.linalg.eigsh(csr_matrix(p7), k=20, which='SA')
    # print("Lowest torsional energy levels (cm^-1):")
    # for i, e in enumerate(e_vals):
    #     print(f"{i + 1:2d}: {e:.6f}")
    # return p7 + p11
    return p7 + p11 + p15 + p16


def isherm(matrix):
    return np.allclose(matrix, matrix.conj().T)


def vibH(F, Fprime, V3, V6, V3p, V3m, d1, d2):
    T = Tvib(d1, d2, F, Fprime)
    print('T is hermitian:', isherm(T))
    V = Vvib(d1, d2, V3, V6, V3p, V3m)
    print('V is hermitian:', isherm(V))
    # print(np.linalg.eigh(T))
    # print(np.linalg.eigh(V))
    H = T + V
    # print(np.linalg.eigh(H))
    # return H
    return H


def sortfullbasis(basis):
    # Updated to sort on both x[6] (primary) and x[1] (secondary)
    sorted_basis = sorted(basis, key=lambda x: (x[2], x[3], x[0], x[1]))
    sorted_indices = [basis.index(element) for element in sorted_basis]
    n = len(basis)
    P = np.zeros((n, n))
    for i, k in enumerate(sorted_indices):
        P[i, k] = 1
    return P, sorted_basis


def sortprimbasis(basis):
    def sign_priority(val):
        return 0 if val < 0 else 1

    sorted_basis = sorted(
        basis,
        key=lambda x: (abs(x[2]), x[0], x[1])
    )

    sorted_indices = [basis.index(element) for element in sorted_basis]
    n = len(basis)
    P = np.zeros((n, n))
    for i, k in enumerate(sorted_indices):
        P[i, k] = 1
    return P, sorted_basis
