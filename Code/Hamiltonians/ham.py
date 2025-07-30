import numpy as np
import Code.pyrotations as pyr
from .torham import torham
import scipy.sparse as sp
from Code.general import timing


class ham():
    def __init__(self, d1, d2, jmin, jmax, F, Fprime, Qx, Qz, consts, uconsts, mu, V3, V6, V3p, V3m, chop):
        self.vars = [f"d1 = {d1}",
                     f"d2 = {d2}",
                     f"jmin = {jmin}",
                     f"jmax = {jmax}",
                     f"F = {F}",
                     f"Fprime = {Fprime}",
                     f"Qx = {Qx}",
                     f"Qz = {Qz}",
                     f"consts = {consts}",
                     f"uconsts = {uconsts}",
                     f"mu = {mu}",
                     f"V3 = {V3}",
                     f"chop = {chop}"]

        with timing("Torsional Hamiltonian Calculation") as t:
            torsions = torham(d1, d2, F, Fprime, V3, V6, V3p, V3m, chop)
            torsions2 = torham(d1, d2, F, Fprime, V3*0.99, V6, V3p, V3m, chop)
            self.torsions = torsions

        with timing("Rotational Hamiltonian Calculation") as t:
            rotations = pyr.rotation(consts, uconsts, jmin, jmax, mu)
            self.rotations = rotations

        with timing("Lower State Hamiltonian Calculation") as t:
            self.lsubham = evalham(torsions, rotations.lsubham, Qx, Qz)

        with timing("Upper State Hamiltonian Calculation") as t:
            self.usubham = evalham(torsions2, rotations.usubham, Qx, Qz)

        tdipole = torsions2.TofK.conj().T @ sp.eye((d1+1)*(d2+1), (d1+1)*(d2+1), format='csr') @ torsions.TofK
        self.pdipole = sp.kron(tdipole, rotations.dipole)
        self.opdipole = self.usubham.P @ self.pdipole @ self.lsubham.P.T
        coo = self.opdipole.tocoo()
        mask = np.abs(coo.data) >= 1e-10
        self.opdipole = sp.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=self.opdipole.shape)
        self.dipole = self.usubham.wfns.T @ self.opdipole @ self.lsubham.wfns
        coo = self.dipole.tocoo()
        mask = np.abs(coo.data) >= 1e-10
        self.dipole = sp.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=self.dipole.shape)

class subham:
    def __init__(self, diagbasis, energies, wfns, P, blocks, basis, penergies):
        self.diagbasis = diagbasis
        self.energies = energies
        self.wfns = wfns
        self.P = P
        self.blocks = blocks
        self.basis = basis
        self.penergies = penergies

def evalham(torsions, rsubham, Qx, Qz):
    print('Constructing Hamiltonian')
    tbasis = torsions.basis
    rbasis = rsubham.diagbasis
    P1plusP2 = torsions.P1 + torsions.P2
    P1minusP2 = torsions.P1 - torsions.P2
    t1 = sp.kron(torsions.I, rsubham.Htot)
    t2 = -2 * Qz * sp.kron(P1plusP2, rsubham.jb)
    t3 = -2 * Qx * sp.kron(P1minusP2, rsubham.jc)
    t4 = sp.kron(torsions.Htor, rsubham.I)
    basis = fullbasis(tbasis, rbasis)
    Htot = t1 + t2 + t3 + t4
    print(Htot.shape)
    penergies = (t1 + t4).diagonal()

    # reorder basis !
    P, basis,penergies = sortfullbasis(basis, penergies)
    Htot = P @ Htot @ P.T

    has_nonzero_diag = np.all(np.abs((t2 + t3).diagonal()) < 1e-8)
    print("All diagonal Coriolis elements zero:", has_nonzero_diag)
    blocks = getfullblocks(basis)
    wfns = sp.csr_matrix(Htot.shape, dtype=complex)
    energies = np.zeros(Htot.shape[0])
    diagbasis = [None] * len(basis)
    print('starting diagonalization')
    for (l, j, gamma), block in blocks.items():
        idxs = block['start']
        idxe = block['end'] + 1
        print(idxe-idxs)
        energies[idxs:idxe], eigenvectors = np.linalg.eigh(Htot[idxs:idxe, idxs:idxe].toarray())
        ev = sp.csr_matrix(eigenvectors)
        wfns[idxs:idxe, idxs:idxe] = ev
        diagbasis[idxs:idxe] = [(i, l, j, gamma) for i in range(idxe - idxs)]

    return subham(diagbasis, energies, wfns, P, blocks, basis, penergies)


def getfullblocks(basis):
    blocks = {}
    for idx, (m, l, gamma1, p, j, gamma2, gamma0) in enumerate(basis):
        if (l, j, gamma0) not in blocks:
            blocks[(l, j, gamma0)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(l, j, gamma0)]["end"] = idx
            blocks[(l, j, gamma0)]["count"] += 1
    return blocks


def sortfullbasis(basis, energies):
    indices = np.arange(len(basis))
    sortable = list(zip(indices, basis, energies))
    sorted_basis = sorted(sortable, key=lambda x: (x[1][1], x[1][4], x[1][6], x[1][2], x[2]))
    sorted_indices, sorted_basis, sorted_energies = zip(*sorted_basis)
    n = len(basis)
    row_indices = np.arange(n)
    P = sp.csr_matrix((np.ones(n), (row_indices, sorted_indices)), shape=(n, n))

    return P, list(sorted_basis), list(sorted_energies)


def fullbasis(torbasis, rbasis):
    metric_tensor = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    b = [(m, l, gamma1, p, j, gamma2, metric_tensor[gamma1][gamma2]) for (m, l, gamma1) in torbasis for (p, j, gamma2)
         in rbasis]
    return b
