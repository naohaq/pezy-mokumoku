// -*- mode: c++; coding: utf-8-unix -*- 
///
/// @file matrix.cpp
///
///

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <algorithm>

#include "sim_conf.hh"

#include "matrix.hh"

namespace {

inline void
set_mtx_elem(SparseMatrix_t & mtx, int x, int y, int z, FLOAT_t w)
{
    int idx = mtx.cur_idx;
    mtx.m_idxs[idx] = crd2idx(x,y,z);
    mtx.m_elems[idx] = w;
    mtx.cur_idx += 1;
}

struct custom_comp_t {
	std::vector<int> & m_degs;
	bool operator()(const int & a, const int & b) const {
		return (m_degs[a] < m_degs[b]);
	}

	custom_comp_t(std::vector<int> & degs) : m_degs(degs) { }
};

}

void
split_matrix(SparseMatrix_t & dst_mtx0, SparseMatrix_t & dst_mtx1, SparseMatrix_t & src_mtx)
{
	int nrow = src_mtx.m_rowptr.size( ) - 1;
	std::vector<int> rem_idxs0;
	std::vector<int> rem_idxs1;

	dst_mtx0.cur_idx = 0;
	dst_mtx0.cur_row = 0;
	for (int i=0; i<nrow/2; i+=1) {
		dst_mtx0.m_rowptr[i] = src_mtx.m_rowptr[i];
		for (int j=src_mtx.m_rowptr[i]; j<src_mtx.m_rowptr[i+1]; j+=1) {
			int idx = src_mtx.m_idxs[j];
			if (idx >= nrow/2) {
				if (idx < dst_mtx0.border_min) {
					dst_mtx0.border_min = idx;
				}
				if (idx > dst_mtx0.border_max) {
					dst_mtx0.border_max = idx;
				}
			}

			dst_mtx0.m_idxs[j]  = idx;
			dst_mtx0.m_elems[j] = src_mtx.m_elems[j];
		}
	}
	dst_mtx0.m_rowptr[nrow/2] = src_mtx.m_rowptr[nrow/2];

	int row_offs = nrow/2;
	int idx_offs = src_mtx.m_rowptr[row_offs];
	dst_mtx1.cur_idx = 0;
	dst_mtx1.cur_row = 0;
	for (int i=row_offs; i<nrow; i+=1) {
		dst_mtx1.m_rowptr[i - row_offs] = src_mtx.m_rowptr[i] - idx_offs;
		for (int j=src_mtx.m_rowptr[i]; j<src_mtx.m_rowptr[i+1]; j+=1) {
			int idx = src_mtx.m_idxs[j];
			if (idx < row_offs) {
				if (idx < dst_mtx1.border_min) {
					dst_mtx1.border_min = idx;
				}
				if (idx > dst_mtx1.border_max) {
					dst_mtx1.border_max = idx;
				}
			}

			dst_mtx1.m_idxs[j - idx_offs] = idx;
			dst_mtx1.m_elems[j - idx_offs] = src_mtx.m_elems[j];
		}
	}
	dst_mtx0.m_rowptr[nrow - row_offs] = src_mtx.m_rowptr[nrow] - idx_offs;

}

void
matrix_reorder_CuthillMckee(SparseMatrix_t & dst_mtx, SparseMatrix_t & src_mtx, std::vector<int> & perm_fwd, std::vector<int> & perm_rev)
{
    int nrow = src_mtx.m_rowptr.size( ) - 1;
    std::vector<bool> visited(nrow, false);
    std::vector<int> degs(nrow);
    std::vector<int> vtxs[2];
	custom_comp_t cmp_deg(degs);
	int cur_idx = 0;
    int f = 0;
    int b = 1 - f;

    for (int i=0; i<nrow; i+=1) {
		degs[i] = src_mtx.m_rowptr[i+1] - src_mtx.m_rowptr[i];
    }

    vtxs[f].push_back(0);
    visited[0] = true;

    while (vtxs[f].size( ) > 0) {
		for (const auto& v : vtxs[f]) {
			perm_fwd[cur_idx] = v;
			perm_rev[v] = cur_idx;
			cur_idx += 1;

			for (int i=src_mtx.m_rowptr[v]; i < src_mtx.m_rowptr[v+1]; i+=1) {
				int w = src_mtx.m_idxs[i];
				if (! visited[w]) {
					vtxs[b].push_back(w);
					visited[w] = true;
				}
			}
		}

		std::sort(vtxs[b].begin( ), vtxs[b].end( ), cmp_deg);
		vtxs[f].clear( );
		f = b;
		b = 1 - f;
    }

	for (int i=0; i<nrow; i+=1) {
		int v = perm_fwd[i];
		dst_mtx.next_row( );
		for (int j=src_mtx.m_rowptr[v]; j < src_mtx.m_rowptr[v+1]; j+=1) {
			int k = dst_mtx.cur_idx;
			dst_mtx.m_idxs[k]  = perm_rev[src_mtx.m_idxs[j]];
			dst_mtx.m_elems[k] = src_mtx.m_elems[j];
			dst_mtx.cur_idx += 1;
		}
	}

	dst_mtx.next_row( );
	std::cout << "--" << std::endl;
    std::cout << "Non-zero elements: " << dst_mtx.cur_idx << std::endl;
    std::cout << "Rows: " << dst_mtx.cur_row-1 << std::endl;
}

void
init_differential_matrix(SparseMatrix_t & mtx)
{
    mtx.next_row( );
    set_mtx_elem(mtx, 0, 0, 0, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, 0, 0, 1, s_cz);
    set_mtx_elem(mtx, 0, 1, 0, s_cy);
    set_mtx_elem(mtx, 1, 0, 0, s_cx);

    for (int z=1; z<NZ-1; z+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, 0, 0, z-1, s_cz);
        set_mtx_elem(mtx, 0, 0, z  , - s_cx - s_cy - 2.0*s_cz);
        set_mtx_elem(mtx, 0, 0, z+1, s_cz);
        set_mtx_elem(mtx, 0, 1, z  , s_cy);
        set_mtx_elem(mtx, 1, 0, z  , s_cx);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, 0, 0, NZ-2, s_cz);
    set_mtx_elem(mtx, 0, 0, NZ-1, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, 0, 1, NZ-1, s_cy);
    set_mtx_elem(mtx, 1, 0, NZ-1, s_cx);

    for (int y=1; y<NY-1; y+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, 0, y-1, 0, s_cy);
        set_mtx_elem(mtx, 0, y  , 0, - s_cx - 2.0*s_cy - s_cz);
        set_mtx_elem(mtx, 0, y+1, 0, s_cy);
        set_mtx_elem(mtx, 0, y  , 1, s_cz);
        set_mtx_elem(mtx, 1, y  , 0, s_cx);

        for (int z=1; z<NZ-1; z+=1) {
			mtx.next_row( );
            set_mtx_elem(mtx, 0, y-1, z  , s_cy);
            set_mtx_elem(mtx, 0, y  , z-1, s_cz);
            set_mtx_elem(mtx, 0, y  , z  , - s_cx - 2.0*s_cy - 2.0*s_cz);
            set_mtx_elem(mtx, 0, y  , z+1, s_cz);
            set_mtx_elem(mtx, 0, y+1, z  , s_cy);
            set_mtx_elem(mtx, 1, y  , z  , s_cx);
        }

		mtx.next_row( );
        set_mtx_elem(mtx, 0, y-1, NZ-1, s_cy);
        set_mtx_elem(mtx, 0, y  , NZ-2, s_cz);
        set_mtx_elem(mtx, 0, y  , NZ-1, - s_cx - 2.0*s_cy - s_cz);
        set_mtx_elem(mtx, 0, y+1, NZ-1, s_cy);
        set_mtx_elem(mtx, 1, y  , NZ-1, s_cx);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, 0, NY-2, 0, s_cy);
    set_mtx_elem(mtx, 0, NY-1, 0, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, 0, NY-1, 1, s_cz);
    set_mtx_elem(mtx, 1, NY-1, 0, s_cx);

    for (int z=1; z<NZ-1; z+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, 0, NY-2, z  , s_cy);
        set_mtx_elem(mtx, 0, NY-1, z-1, s_cz);
        set_mtx_elem(mtx, 0, NY-1, z  , - s_cx - s_cy - 2.0*s_cz);
        set_mtx_elem(mtx, 0, NY-1, z+1, s_cz);
        set_mtx_elem(mtx, 1, NY-1, z  , s_cx);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, 0, NY-2, NZ-1, s_cy);
    set_mtx_elem(mtx, 0, NY-1, NZ-2, s_cz);
    set_mtx_elem(mtx, 0, NY-1, NZ-1, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, 1, NY-1, NZ-1, s_cx);

    for (int x=1; x<NX-1; x+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, x-1, 0, 0, s_cx);
        set_mtx_elem(mtx, x  , 0, 0, - 2.0*s_cx - s_cy - s_cz);
        set_mtx_elem(mtx, x  , 0, 1, s_cz);
        set_mtx_elem(mtx, x  , 1, 0, s_cy);
        set_mtx_elem(mtx, x+1, 0, 0, s_cx);

        for (int z=1; z<NZ-1; z+=1) {
			mtx.next_row( );
            set_mtx_elem(mtx, x-1, 0, z  , s_cx);
            set_mtx_elem(mtx, x  , 0, z-1, s_cz);
            set_mtx_elem(mtx, x  , 0, z  , - 2.0*s_cx - s_cy - 2.0*s_cz);
            set_mtx_elem(mtx, x  , 0, z+1, s_cz);
            set_mtx_elem(mtx, x  , 1, z  , s_cy);
            set_mtx_elem(mtx, x+1, 0, z  , s_cx);
        }

		mtx.next_row( );
        set_mtx_elem(mtx, x-1, 0, NZ-1, s_cx);
        set_mtx_elem(mtx, x  , 0, NZ-2, s_cz);
        set_mtx_elem(mtx, x  , 0, NZ-1, - 2.0*s_cx - s_cy - s_cz);
        set_mtx_elem(mtx, x  , 1, NZ-1, s_cy);
        set_mtx_elem(mtx, x+1, 0, NZ-1, s_cx);

        for (int y=1; y<NY-1; y+=1) {
			mtx.next_row( );
            set_mtx_elem(mtx, x-1, y  , 0, s_cx);
            set_mtx_elem(mtx, x  , y-1, 0, s_cy);
            set_mtx_elem(mtx, x  , y  , 0, - 2.0*s_cx - 2.0*s_cy - s_cz);
            set_mtx_elem(mtx, x  , y  , 1, s_cz);
            set_mtx_elem(mtx, x  , y+1, 0, s_cy);
            set_mtx_elem(mtx, x+1, y  , 0, s_cx);

            for (int z=1; z<NZ-1; z+=1) {
				mtx.next_row( );
                set_mtx_elem(mtx, x-1, y  , z  , s_cx);
                set_mtx_elem(mtx, x  , y-1, z  , s_cy);
                set_mtx_elem(mtx, x  , y  , z-1, s_cz);
                set_mtx_elem(mtx, x  , y  , z  , - 2.0*s_cx - 2.0*s_cy - 2.0*s_cz);
                set_mtx_elem(mtx, x  , y  , z+1, s_cz);
                set_mtx_elem(mtx, x  , y+1, z  , s_cy);
                set_mtx_elem(mtx, x+1, y  , z  , s_cx);
            }

			mtx.next_row( );
            set_mtx_elem(mtx, x-1, y  , NZ-1, s_cx);
            set_mtx_elem(mtx, x  , y-1, NZ-1, s_cy);
            set_mtx_elem(mtx, x  , y  , NZ-2, s_cz);
            set_mtx_elem(mtx, x  , y  , NZ-1, - 2.0*s_cx - 2.0*s_cy - s_cz);
            set_mtx_elem(mtx, x  , y+1, NZ-1, s_cy);
            set_mtx_elem(mtx, x+1, y  , NZ-1, s_cx);
        }

		mtx.next_row( );
        set_mtx_elem(mtx, x-1, NY-1, 0, s_cx);
        set_mtx_elem(mtx, x  , NY-2, 0, s_cy);
        set_mtx_elem(mtx, x  , NY-1, 0, - 2.0*s_cx - s_cy - s_cz);
        set_mtx_elem(mtx, x  , NY-1, 1, s_cz);
        set_mtx_elem(mtx, x+1, NY-1, 0, s_cx);

        for (int z=1; z<NZ-1; z+=1) {
			mtx.next_row( );
            set_mtx_elem(mtx, x-1, NY-1, z  , s_cx);
            set_mtx_elem(mtx, x  , NY-2, z  , s_cy);
            set_mtx_elem(mtx, x  , NY-1, z-1, s_cz);
            set_mtx_elem(mtx, x  , NY-1, z  , - 2.0*s_cx - s_cy - 2.0*s_cz);
            set_mtx_elem(mtx, x  , NY-1, z+1, s_cz);
            set_mtx_elem(mtx, x+1, NY-1, z  , s_cx);
        }

		mtx.next_row( );
        set_mtx_elem(mtx, x-1, NY-1, NZ-1, s_cx);
        set_mtx_elem(mtx, x  , NY-2, NZ-1, s_cy);
        set_mtx_elem(mtx, x  , NY-1, NZ-2, s_cz);
        set_mtx_elem(mtx, x  , NY-1, NZ-1, - 2.0*s_cx - s_cy - s_cz);
        set_mtx_elem(mtx, x+1, NY-1, NZ-1, s_cx);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, NX-2, 0, 0, s_cx);
    set_mtx_elem(mtx, NX-1, 0, 0, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, NX-1, 0, 1, s_cz);
    set_mtx_elem(mtx, NX-1, 1, 0, s_cy);

    for (int z=1; z<NZ-1; z+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, NX-2, 0, z  , s_cx);
        set_mtx_elem(mtx, NX-1, 0, z-1, s_cz);
        set_mtx_elem(mtx, NX-1, 0, z  , - s_cx - s_cy - 2.0*s_cz);
        set_mtx_elem(mtx, NX-1, 0, z+1, s_cz);
        set_mtx_elem(mtx, NX-1, 1, z  , s_cy);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, NX-2, 0, NZ-1, s_cx);
    set_mtx_elem(mtx, NX-1, 0, NZ-2, s_cz);
    set_mtx_elem(mtx, NX-1, 0, NZ-1, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, NX-1, 1, NZ-1, s_cy);

    for (int y=1; y<NY-1; y+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, NX-2, y  , 0, s_cx);
        set_mtx_elem(mtx, NX-1, y-1, 0, s_cy);
        set_mtx_elem(mtx, NX-1, y  , 0, - s_cx - 2.0*s_cy - s_cz);
        set_mtx_elem(mtx, NX-1, y+1, 0, s_cy);
        set_mtx_elem(mtx, NX-1, y  , 1, s_cz);

        for (int z=1; z<NZ-1; z+=1) {
			mtx.next_row( );
            set_mtx_elem(mtx, NX-2, y  , z  , s_cx);
            set_mtx_elem(mtx, NX-1, y-1, z  , s_cy);
            set_mtx_elem(mtx, NX-1, y  , z-1, s_cz);
            set_mtx_elem(mtx, NX-1, y  , z  , - s_cx - 2.0*s_cy - 2.0*s_cz);
            set_mtx_elem(mtx, NX-1, y  , z+1, s_cz);
            set_mtx_elem(mtx, NX-1, y+1, z  , s_cy);
        }

		mtx.next_row( );
        set_mtx_elem(mtx, NX-2, y  , NZ-1, s_cx);
        set_mtx_elem(mtx, NX-1, y-1, NZ-1, s_cy);
        set_mtx_elem(mtx, NX-1, y  , NZ-2, s_cz);
        set_mtx_elem(mtx, NX-1, y  , NZ-1, - s_cx - 2.0*s_cy - s_cz);
        set_mtx_elem(mtx, NX-1, y+1, NZ-1, s_cy);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, NX-2, NY-1, 0, s_cx);
    set_mtx_elem(mtx, NX-1, NY-2, 0, s_cy);
    set_mtx_elem(mtx, NX-1, NY-1, 0, - s_cx - s_cy - s_cz);
    set_mtx_elem(mtx, NX-1, NY-1, 1, s_cz);

    for (int z=1; z<NZ-1; z+=1) {
		mtx.next_row( );
        set_mtx_elem(mtx, NX-2, NY-1, z  , s_cx);
        set_mtx_elem(mtx, NX-1, NY-2, z  , s_cy);
        set_mtx_elem(mtx, NX-1, NY-1, z-1, s_cz);
        set_mtx_elem(mtx, NX-1, NY-1, z  , - s_cx - s_cy - 2.0*s_cz);
        set_mtx_elem(mtx, NX-1, NY-1, z+1, s_cz);
    }

    mtx.next_row( );
    set_mtx_elem(mtx, NX-2, NY-1, NZ-1, s_cx);
    set_mtx_elem(mtx, NX-1, NY-2, NZ-1, s_cy);
    set_mtx_elem(mtx, NX-1, NY-1, NZ-2, s_cz);
    set_mtx_elem(mtx, NX-1, NY-1, NZ-1, - s_cx - s_cy - s_cz);

    mtx.next_row( );
    std::cout << "Non-zero elements: " << mtx.cur_idx << std::endl;
    std::cout << "Rows: " << mtx.cur_row-1 << std::endl;
}



// Local Variables:
// indent-tabs-mode: t
// tab-width: 4
// End:
