import scipy.io
from scipy.sparse import csr_matrix

data = scipy.io.mmread('./data/Cora.mtx')
data = csr_matrix(data)
i = 1
