// -*- mode: c++; coding: utf-8-unix -*-
/// 
/// @file matrix.hh
///
///

#ifndef THERMAL_MATRIX_HH_
#define THERMAL_MATRIX_HH_

struct SparseMatrix_t {
	int cur_idx;
	int cur_row;
	std::vector<int>& m_rowptr;
	std::vector<int>& m_idxs;
	std::vector<FLOAT_t>& m_elems;
	int border_min;
	int border_max;

    inline void next_row(void) {
		m_rowptr[cur_row] = cur_idx;
		cur_row += 1;
	}

	SparseMatrix_t(std::vector<int> & rowptr, std::vector<int> & idxs, std::vector<FLOAT_t> & elems)
		: cur_idx(0), cur_row(0), m_rowptr(rowptr), m_idxs(idxs), m_elems(elems),
		  border_min(2147483647), border_max(-1) {
	}
};

extern void init_differential_matrix(SparseMatrix_t & mtx);
extern void matrix_reorder_CuthillMckee(SparseMatrix_t & dst_mtx, SparseMatrix_t & src_mtx, std::vector<int> & perm_fwd, std::vector<int> & perm_rev);
extern void split_matrix(SparseMatrix_t & dst_mtx0, SparseMatrix_t & dst_mtx1, SparseMatrix_t & src_mtx);

#endif /* THERMAL_MATRIX_HH_ */

// Local Variables:
// indent-tabs-mode: t
// tab-width: 4
// End:
