import numpy as np

# convert vector to peptide
class feature2peptide:
    def __init__(self):
        self.aa = "ACDEFGHIKLMNPQRSTVWY"
    def vector_to_aa(self, v):
        idx = np.where(v != 0)
        return self.aa[idx[0][0]]

    def rnn_feature_to_peptide(self, X, peplen):
        peptide = self.vector_to_aa(X[0,:20])
        charge = np.where(X[0,82:88])[0][0]+1
        for i in range(peplen-1):
            peptide += self.vector_to_aa(X[i,20:40])
        return (peptide, charge)
        