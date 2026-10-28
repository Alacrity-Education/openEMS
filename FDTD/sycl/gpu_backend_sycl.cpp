/*
*	Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*	Copyright (C) 2026 Alexandru Iercosan-Lucaci (alex.lucaci@alacrity.ro)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Environment knobs of the SYCL backend (read once when a grid is set up; --verbose=2 prints the decisions).
 * Every knob changes how the backend computes, never what: unless a knob says otherwise its settings are
 * bit-identical to each other. Speeds are from an Arc B580: "small PCB" is a PCB model of 3.06 Mcells (337 x 227
 * x 40, Mur on six faces, four copper sheet planes).
 *
 * OPENEMS_SYCL_WINDOW = 0 (default) | auto | rule | T,W
 *     The timesteps as blocks of T steps marched along x in windows of W x planes that stay in the L2 cache, each
 *     kind of block recorded once as a SYCL command graph and replayed; the voltage and current frontiers step back
 *     one plane per step, so the halo is T+1 planes. rule (and auto): T = 8, W = 0.65 L2 / plane bytes - (T+1);
 *     where that leaves fewer than T planes a window, T = 4, then 2 (on the 18 MiB L2 of the B580 the halo of T = 8
 *     takes most of it). T,W: forced (W may be NX). 0: one timestep at a time. Grids with TF/SF, Debye, absorbing
 *     sheets, cylinder coordinates, multi-grids or Mur planes outside the voltage update keep the single step.
 * OPENEMS_SYCL_MUR_SG = 32 (default) | 16
 *     Development: the sub-group size of the voltage kernel that updates the Mur z faces inside the main update
 *     (Mur mode 2, the line and shift line of a z line exchange their values by sub-group shifts). small PCB:
 *     3095 MC/s with 32, 2937 with 16.
 * OPENEMS_SYCL_FUSED_STEP = unset (default) | 0 | 1
 *     The voltage and current updates fused into one pass; unset: on grids without UPML, Mur, absorbing sheets,
 *     TF/SF and dispersive materials; 1: on every Cartesian grid the extensions allow (with a fix-up of the
 *     currents next to the voltages the extensions change); 0: never.
 * OPENEMS_SYCL_FUSED_XC, OPENEMS_SYCL_FUSED_TY, OPENEMS_SYCL_FUSED_TZ
 *     Development: the tile of the fused step (XC x lines per work-group, default 4; TY rows along y, 4, 8 or 16,
 *     default 8; TZ work-items along z, 16 or 32, default 32).
 * OPENEMS_SYCL_FUSED_REGIONS = 1 (default) | 0 | 2
 *     The UPML z slabs in the fused kernel with aligned tiles; 0 left to the region kernels; 2 one z tiling.
 * OPENEMS_SYCL_INDEX = 32 | 64
 *     The index width of the kernels; default 32 where the fields fit.
 * OPENEMS_SYCL_UPML_FUSION = 1 (default) | 0
 *     The UPML regions fused with the main updates; 0 the separate pre/post kernels.
 * OPENEMS_SYCL_COEFF = sets (default) | full | factor, OPENEMS_SYCL_COEFF_ULP = 2 (default)
 *     The storage of the update coefficients: deduplicated sets, full arrays, or sets divided by the mesh geometry
 *     (Cartesian meshes; within COEFF_ULP units in the last place of the stored coefficients, not bit-identical).
 * OPENEMS_SYCL_KERNEL_TIMES = 0 (default) | 1
 *     Development: a table of kernel times at the end of the run, waiting after every launch: kernel shares,
 *     never speeds.
 */

#include <sycl/sycl.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "sycl_internal.h"
#include "gpu_build_id.h"
#include "FDTD/operator.h"
#include "FDTD/operator_cylinder.h"
#include "FDTD/extensions/engine_ext_upml.h"
#include "FDTD/extensions/engine_ext_excitation.h"
#include "FDTD/extensions/engine_ext_lumpedRLC.h"
#include "FDTD/extensions/engine_ext_steadystate.h"
#include "FDTD/extensions/engine_ext_mur_abc.h"
#include "FDTD/extensions/engine_ext_absorbing_bc.h"
#include "FDTD/extensions/engine_ext_tfsf.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"
#include "FDTD/extensions/engine_ext_debyematerial.h"
#include "FDTD/extensions/engine_ext_dispersive.h"
#include "FDTD/operator_cylinder.h"

// Main FDTD updates, operation by operation the same as Engine::UpdateVoltages/UpdateCurrents,
// over the nodes in [B, E). One work-item per mesh node: z along dimension 2 (contiguous), y along
// dimension 1, x along dimension 0 (one work-group per x line). The coefficients are read with main_coeff().
// Nodes below Z.z_lo or from Z.z_hi on belong to UPML regions along z (see SYCL_ZSlabs).
// The index type I is unsigned int if the fields fit (see GPU_Backend_SYCL::Impl::index32), else size_t.
// FACTOR: the factorized coefficients (coeff_mode 3/4), else modes 0-2 (see main_coeff()).
template<typename I, bool FACTOR>
inline void update_voltages(const sycl::nd_item<3>& it, float* __restrict__ volt, const float* __restrict__ curr,
                            const void* index, const float* ca, const float* cb, unsigned int mode,
                            const SYCL_GridDim& N, const SYCL_GridDim& B, const SYCL_GridDim& E, const SYCL_ZSlabs& Z)
{
	// with UPML regions along z: slab lo, main nodes and slab hi in whole sub-groups each (no divergence)
	const unsigned int tz = it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	unsigned int z = B.nz + tz;
	if (Z.t_int)
	{
		if (tz<Z.t_lo)
		{
			if (B.nz+tz>=Z.z_lo)
				return;
		}
		else if (tz<Z.t_lo+Z.t_int)
		{
			z = Z.z_lo + (tz-Z.t_lo);
			if (z>=Z.z_hi)
				return;
		}
		else
			z = Z.z_hi + (tz-Z.t_lo-Z.t_int);
	}
	const unsigned int y = B.ny + it.get_group(1)*it.get_local_range(1) + it.get_local_id(1);
	const unsigned int x = B.nx + it.get_group(0);
	if (z>=E.nz || y>=E.ny || x>=E.nx)
		return;
	if (z<Z.z_lo)
	{
		upml_fused_volt_node<FACTOR>(volt, volt, curr, Z.volt_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		return;
	}
	if (z>=Z.z_hi)
	{
		upml_fused_volt_node<FACTOR>(volt, volt, curr, Z.volt_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		return;
	}
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nz : 0;   // shift to the previous line, none on the first
	const I ym = (y>0) ? N.nz : 0;
	const I zm = (z>0) ? 1 : 0;
	const SYCL_MainCoeffT<I, FACTOR> C = main_coeff<I, FACTOR>(index, ca, cb, mode, sn, i, 0, N, x, y, z);
	const float* vv = C.a;
	const float* vi = C.b;
	const I s = C.s;
	float v;

	//for x
	v  = volt[i] * vv[0];
	v += vi[0] * (curr[2*sn+i] - curr[2*sn+i-ym] - curr[sn+i] + curr[sn+i-zm]);
	volt[i] = v;

	//for y
	v  = volt[sn+i] * vv[s];
	v += vi[s] * (curr[i] - curr[i-zm] - curr[2*sn+i] + curr[2*sn+i-xm]);
	volt[sn+i] = v;

	//for z
	v  = volt[2*sn+i] * vv[2*s];
	v += vi[2*s] * (curr[sn+i] - curr[sn+i-xm] - curr[i] + curr[i-ym]);
	volt[2*sn+i] = v;
}

// Mur mode 1: the voltage update with the first order Mur ABC of the active planes F, which
// mode 0 runs as mur_pre, mur_post and mur_apply (sycl_ext_mur_abc.cpp, Engine_Ext_Mur_ABC), as two launches over
// disjoint nodes: update_voltages_inner off the planes, mur_shell on the line and shift line of each. The Mur value of
// component c of a cell L on the line of a plane is
//     v = V_old(S) - k*V_old(L);  v += k*V_new(S)
// with S the cell next to L on the shift line and V_new(S) its main update. The work-item of S writes it: it holds
// V_old(S) and computes V_new(S); the work-item of L leaves component c alone. Every voltage has one writer, and
// the only work-item that reads its old value is that writer, so no work-item waits for another.
// Component c is tangential to the planes normal to the two other axes a = c+1 and b = c+2 (mod 3). An edge
// cell E on the lines of an a-plane and of a b-plane takes the value of the plane applied last, the apply hooks
// running in the extension order: the plane of the highest rank (SYCL_MurFace::rank; with Mur on all faces the
// one of the lowest face: x-low wins over y-low and z-low, see Engine::SortExtensionByPriority()). Its shift
// cell is on the line of the other plane, whose Mur value replaces the main update there, while E needs that
// main update (the Mur post hooks all read before the first apply): the work-item of the diagonal cell D (on the
// shift lines of both planes) writes E and both line cells next to it, and computes the main update of the
// shift cell of E again from its old value, with the statements of the main update (volt_update_at()).
// So for each component c a work-item on no line of an active a- or b-plane writes its own main update, on the
// shift line of an a-plane the Mur value of the a-line cell next to it, the same for b, and on both shift lines
// the edge E. The normal component of a line cell is the main update of its own work-item.
template<unsigned int c, typename I, bool FACTOR>
inline void mur_volt_component(float* volt, const float* curr, const void* index, const float* ca, const float* cb,
                               unsigned int mode, const SYCL_GridDim& N, const SYCL_MurFace* F, unsigned int active,
                               unsigned int x, unsigned int y, unsigned int z, const SYCL_MainCoeffT<I, FACTOR>& C,
                               I sn, I i, I xm, I ym, I zm)
{
	const unsigned int a = (c+1)%3;
	const unsigned int b = (c+2)%3;
	const unsigned int p[3] = {x, y, z};
	const unsigned int n[3] = {N.nx, N.ny, N.nz};
	const bool a0 = (active>>(2*a)) & 1u;
	const bool a1 = (active>>(2*a+1)) & 1u;
	const bool b0 = (active>>(2*b)) & 1u;
	const bool b1 = (active>>(2*b+1)) & 1u;
	if ((a0 && (p[a]==F[2*a].line)) || (a1 && (p[a]==F[2*a+1].line)) || (b0 && (p[b]==F[2*b].line)) || (b1 && (p[b]==F[2*b+1].line)))
		return;   // a line cell: written by the work-item of its shift cell
	const bool sa0 = a0 && (p[a]==F[2*a].shift);
	const bool sa = sa0 || (a1 && (p[a]==F[2*a+1].shift));
	const bool sb0 = b0 && (p[b]==F[2*b].shift);
	const bool sb = sb0 || (b1 && (p[b]==F[2*b+1].shift));
	const float old = volt[c*sn+i];
	const float nv = volt_update_node<I, FACTOR>(c, old, curr, C, sn, i, xm, ym, zm);
	if (!sa && !sb)
	{
		volt[c*sn+i] = nv;
		return;
	}
	// the a-plane: c is its component nyPP, coefficient (i,j) = (p[b], p[c]); the b-plane: c is its nyP, (i,j) = (p[c], p[a])
	const SYCL_MurFace& A = F[sa0 ? 2*a : 2*a+1];
	const SYCL_MurFace& B = F[sb0 ? 2*b : 2*b+1];
	const unsigned int la = A.line;
	const unsigned int lb = B.line;
	// the line cells La (a-line), Lb (b-line) and the edge E, all read before any is written
	unsigned int q[3] = {x, y, z};
	q[a] = la;
	const I ia = c*sn + nijk<I>(N, 0, q[0], q[1], q[2]);
	q[b] = lb;
	const I ie = c*sn + nijk<I>(N, 0, q[0], q[1], q[2]);
	q[a] = p[a];
	const I ib = c*sn + nijk<I>(N, 0, q[0], q[1], q[2]);
	const float old_a = sa ? volt[ia] : 0.0f;
	const float old_b = sb ? volt[ib] : 0.0f;
	float v_e = 0.0f;
	if (sa && sb)
	{
		const float old_e = volt[ie];
		if (A.rank>B.rank)
		{
			// the a-plane is applied last: its shift cell of E is Lb (q = Lb)
			const float k = A.c_nyPP[(size_t)lb*n[c] + p[c]];
			const float new_b = volt_update_at<I, FACTOR>(c, old_b, curr, index, ca, cb, mode, N, q[0], q[1], q[2]);
			v_e = old_b - k * old_e;
			v_e = v_e + k * new_b;
		}
		else
		{
			// the b-plane is applied last: its shift cell of E is La
			q[a] = la;
			q[b] = p[b];
			const float k = B.c_nyP[(size_t)p[c]*n[a] + la];
			const float new_a = volt_update_at<I, FACTOR>(c, old_a, curr, index, ca, cb, mode, N, q[0], q[1], q[2]);
			v_e = old_a - k * old_e;
			v_e = v_e + k * new_a;
		}
	}
	volt[c*sn+i] = nv;
	if (sa)
	{
		const float k = A.c_nyPP[(size_t)p[b]*n[c] + p[c]];
		float v = old - k * old_a;
		v = v + k * nv;
		volt[ia] = v;
	}
	if (sb)
	{
		const float k = B.c_nyP[(size_t)p[c]*n[a] + p[a]];
		float v = old - k * old_b;
		v = v + k * nv;
		volt[ib] = v;
	}
	if (sa && sb)
		volt[ie] = v_e;
}

// all components of node (x, y, z) on the line or the shift line of a Mur plane
template<typename I, bool FACTOR>
inline void mur_volt_node(float* volt, const float* curr, const void* index, const float* ca, const float* cb,
                          unsigned int mode, const SYCL_GridDim& N, const SYCL_MurFace* F, unsigned int active,
                          unsigned int x, unsigned int y, unsigned int z)
{
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nz : 0;
	const I ym = (y>0) ? N.nz : 0;
	const I zm = (z>0) ? 1 : 0;
	const SYCL_MainCoeffT<I, FACTOR> C = main_coeff<I, FACTOR>(index, ca, cb, mode, sn, i, 0, N, x, y, z);
	mur_volt_component<0, I, FACTOR>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z, C, sn, i, xm, ym, zm);
	mur_volt_component<1, I, FACTOR>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z, C, sn, i, xm, ym, zm);
	mur_volt_component<2, I, FACTOR>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z, C, sn, i, xm, ym, zm);
}

// Mur mode 2: the Mur values of the tangential components x and y of the nodes L on the line of an active z
// plane, from the node S on its shift line in the same z line, the neighbouring work-item of the same sub-group
// (z is the contiguous work-item dimension; DecideMurFused() refuses the mode when a line and its shift line fall
// into different sub-groups or are not adjacent, MurZFaceRefusal()). Called by every work-item of
// update_voltages_inner<..., true> (a sub-group function: \a node false where the work-item has no node of the
// interior of the z faces or inside them), with the old values o_x, o_y and the main updates v_x, v_y of its own
// node (index xy = x*ny + y of the plane's coefficients, as the Mur extension indexes them: i along nyP = x, j
// along nyPP = y). The values of S come over by a shift by one work-item; L replaces its main update with
//     v = V_old(S) - k*V_old(L);  v = v + k*V_new(S)
// the statements of mur_pre and mur_post, which mur_apply writes. The z component of L stays its main update (the
// z planes write only x and y), and S keeps its main update, as every main node. Node sets: this kernel writes the
// nodes with x and y inside [S.L, S.H) on every z line of the main range (the z faces' lines and shift lines
// included); mur_shell the others (the x-face slabs, then the y-face slabs inside x: the 2-line bands along the four
// edges of each z face, their shift cells too), which never read a voltage this kernel writes.
inline void mur_zface_lanes(const sycl::sub_group& sg, const SYCL_MurZFaces& M, bool node, unsigned int z, size_t xy,
                            float o_x, float o_y, float& v_x, float& v_y)
{
	for (unsigned int f=0; f<2; ++f)
	{
		if (!((M.active>>f) & 1u))
			continue;   // the same for all work-items
		// the shift line above the line (z-low): the values of the next work-item, else of the previous one
		const bool up = M.shift[f]>M.line[f];
		const float so_x = up ? sycl::shift_group_left(sg, o_x, 1) : sycl::shift_group_right(sg, o_x, 1);
		const float so_y = up ? sycl::shift_group_left(sg, o_y, 1) : sycl::shift_group_right(sg, o_y, 1);
		const float sn_x = up ? sycl::shift_group_left(sg, v_x, 1) : sycl::shift_group_right(sg, v_x, 1);
		const float sn_y = up ? sycl::shift_group_left(sg, v_y, 1) : sycl::shift_group_right(sg, v_y, 1);
		if (node && (z==M.line[f]))
		{
			const float k_x = M.c_x[f][xy];
			float v = so_x - k_x * o_x;
			v = v + k_x * sn_x;
			v_x = v;
			const float k_y = M.c_y[f][xy];
			v = so_y - k_y * o_y;
			v = v + k_y * sn_y;
			v_y = v;
		}
	}
}

// The main update of the nodes inside the box [S.L, S.H), the statements of update_voltages: in one kernel with
// the Mur nodes (all the same, see mur_shell) the Mur path would cost every work-group its registers, and the
// sub-groups on the outer z lines of every line would run both paths. ZF (Mur mode 2): the box covers the z faces
// too, whose interior gets its Mur values from mur_zface_lanes(), a sub-group function: no work-item returns early.
template<typename I, bool FACTOR, bool ZF>
inline void update_voltages_inner(const sycl::nd_item<3>& it, float* __restrict__ volt, const float* __restrict__ curr,
                                  const void* index, const float* ca, const float* cb, unsigned int mode,
                                  const SYCL_GridDim& N, const SYCL_GridDim& B, const SYCL_GridDim& E, const SYCL_ZSlabs& Z,
                                  const SYCL_MurShell& S, const SYCL_MurZFaces& M)
{
	const unsigned int tz = it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	unsigned int z = B.nz + tz;
	bool live = true;
	if (Z.t_int)
	{
		if (tz<Z.t_lo)
		{
			if (B.nz+tz>=Z.z_lo)
				live = false;
		}
		else if (tz<Z.t_lo+Z.t_int)
		{
			z = Z.z_lo + (tz-Z.t_lo);
			if (z>=Z.z_hi)
				live = false;
		}
		else
			z = Z.z_hi + (tz-Z.t_lo-Z.t_int);
	}
	const unsigned int y = B.ny + it.get_group(1)*it.get_local_range(1) + it.get_local_id(1);
	const unsigned int x = B.nx + it.get_group(0);
	live = live && (z<E.nz) && (y<E.ny) && (x<E.nx);
	if (live && (z<Z.z_lo))
	{
		upml_fused_volt_node<FACTOR>(volt, volt, curr, Z.volt_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		live = false;
	}
	else if (live && (z>=Z.z_hi))
	{
		upml_fused_volt_node<FACTOR>(volt, volt, curr, Z.volt_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		live = false;
	}
	// a node of mur_shell otherwise
	const bool node = live && (x>=S.L.nx) && (x<S.H.nx) && (y>=S.L.ny) && (y<S.H.ny) && (z>=S.L.nz) && (z<S.H.nz);
	if constexpr (!ZF)
	{
		(void)M;
		if (!node)
			return;
	}
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = node ? nijk<I>(N, 0, x, y, z) : 0;
	float o_x = 0, o_y = 0, v_x = 0, v_y = 0;
	if (node)
	{
		const I xm = (x>0) ? (I)N.ny*N.nz : 0;   // shift to the previous line, none on the first
		const I ym = (y>0) ? N.nz : 0;
		const I zm = (z>0) ? 1 : 0;
		const SYCL_MainCoeffT<I, FACTOR> C = main_coeff<I, FACTOR>(index, ca, cb, mode, sn, i, 0, N, x, y, z);
		o_x = volt[i];
		o_y = volt[sn+i];
		v_x = volt_update_node<I, FACTOR>(0, o_x, curr, C, sn, i, xm, ym, zm);
		v_y = volt_update_node<I, FACTOR>(1, o_y, curr, C, sn, i, xm, ym, zm);
		volt[2*sn+i] = volt_update_node<I, FACTOR>(2, volt[2*sn+i], curr, C, sn, i, xm, ym, zm);
	}
	if constexpr (ZF)
		mur_zface_lanes(it.get_sub_group(), M, node, z, (size_t)x*N.ny + y, o_x, o_y, v_x, v_y);
	if (node)
	{
		volt[i]    = v_x;
		volt[sn+i] = v_y;
	}
}

// the j-th of the n0 + (E-H) lines of an axis outside [L, H) (n0 = L-B)
inline unsigned int mur_shell_line(unsigned int j, unsigned int B, unsigned int L, unsigned int H)
{
	return (j<L-B) ? B+j : H+(j-(L-B));
}

// The main nodes outside the box [S.L, S.H): the lines and shift lines of the Mur planes (inactive ones too),
// one work-item per node, as the slabs of the x faces (all y, z), of the y faces (x inside) and of the z faces
// (x and y inside; with Mur mode 2 S spans all z lines, so there are none: update_voltages_inner<..., true>
// has them). Launched right after update_voltages_inner, on nodes disjoint from its own.
template<typename I, bool FACTOR>
inline void mur_shell(const sycl::nd_item<3>& it, float* volt, const float* curr, const void* index, const float* ca, const float* cb,
                      unsigned int mode, const SYCL_GridDim& N, const SYCL_MurShell& S, const SYCL_MurFace* F, unsigned int active)
{
	size_t k = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	const unsigned int fx = S.E.nx-S.B.nx, fy = S.E.ny-S.B.ny, fz = S.E.nz-S.B.nz;   // full extents
	const unsigned int ix = S.H.nx-S.L.nx, iy = S.H.ny-S.L.ny;                         // inner extents
	const unsigned int cx = fx-ix, cy = fy-iy, cz = fz-(S.H.nz-S.L.nz);                // outer lines
	const size_t sx = (size_t)cx*fy*fz, sy = (size_t)ix*cy*fz, sz = (size_t)ix*iy*cz;
	unsigned int x, y, z;
	if (k<sx)
	{
		x = mur_shell_line((unsigned int)(k/((size_t)fy*fz)), S.B.nx, S.L.nx, S.H.nx);
		k %= (size_t)fy*fz;
		y = S.B.ny + (unsigned int)(k/fz);
		z = S.B.nz + (unsigned int)(k%fz);
	}
	else if ((k -= sx)<sy)
	{
		x = S.L.nx + (unsigned int)(k/((size_t)cy*fz));
		k %= (size_t)cy*fz;
		y = mur_shell_line((unsigned int)(k/fz), S.B.ny, S.L.ny, S.H.ny);
		z = S.B.nz + (unsigned int)(k%fz);
	}
	else if ((k -= sy)<sz)
	{
		x = S.L.nx + (unsigned int)(k/((size_t)iy*cz));
		k %= (size_t)iy*cz;
		y = S.L.ny + (unsigned int)(k/cz);
		z = mur_shell_line((unsigned int)(k%cz), S.B.nz, S.L.nz, S.H.nz);
	}
	else
		return;
	mur_volt_node<I, FACTOR>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z);
}

// the currents on the last mesh line of each direction are not updated
template<typename I, bool FACTOR>
inline void update_currents(const sycl::nd_item<3>& it, float* __restrict__ curr, const float* __restrict__ volt,
                            const void* index, const float* ca, const float* cb, unsigned int mode,
                            const SYCL_GridDim& N, const SYCL_GridDim& B, const SYCL_GridDim& E, const SYCL_ZSlabs& Z)
{
	// with UPML regions along z: slab lo, main nodes and slab hi in whole sub-groups each (no divergence)
	const unsigned int tz = it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	unsigned int z = B.nz + tz;
	if (Z.t_int)
	{
		if (tz<Z.t_lo)
		{
			if (B.nz+tz>=Z.z_lo)
				return;
		}
		else if (tz<Z.t_lo+Z.t_int)
		{
			z = Z.z_lo + (tz-Z.t_lo);
			if (z>=Z.z_hi)
				return;
		}
		else
			z = Z.z_hi + (tz-Z.t_lo-Z.t_int);
	}
	const unsigned int y = B.ny + it.get_group(1)*it.get_local_range(1) + it.get_local_id(1);
	const unsigned int x = B.nx + it.get_group(0);
	if (z>=E.nz || y>=E.ny || x>=E.nx)
		return;
	if (z<Z.z_lo)
	{
		upml_fused_curr_node<FACTOR>(curr, curr, volt, Z.curr_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		return;
	}
	if (z>=Z.z_hi)
	{
		upml_fused_curr_node<FACTOR>(curr, curr, volt, Z.curr_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		return;
	}
	if (z+1>=N.nz || y+1>=N.ny || x+1>=N.nx)
		return;
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xp = (I)N.ny*N.nz;
	const I yp = N.nz;
	const SYCL_MainCoeffT<I, FACTOR> C = main_coeff<I, FACTOR>(index, ca, cb, mode, sn, i, 6, N, x, y, z);
	const float* ii = C.a;
	const float* iv = C.b;
	const I s = C.s;
	float c;

	//for x
	c  = curr[i] * ii[0];
	c += iv[0] * (volt[2*sn+i] - volt[2*sn+i+yp] - volt[sn+i] + volt[sn+i+1]);
	curr[i] = c;

	//for y
	c  = curr[sn+i] * ii[s];
	c += iv[s] * (volt[i] - volt[i+1] - volt[2*sn+i] + volt[2*sn+i+xp]);
	curr[sn+i] = c;

	//for z
	c  = curr[2*sn+i] * ii[2*s];
	c += iv[2*s] * (volt[sn+i] - volt[sn+i+xp] - volt[i] + volt[i+yp]);
	curr[2*sn+i] = c;
}

// ---- fused step

// the voltages of node (x,y,z), the same operations as update_voltages
template<typename I, bool FACTOR>
inline void fused_volt(const float* __restrict__ volt, const float* __restrict__ curr,
                       const void* index, const float* va, const float* vb, unsigned int mode,
                       const SYCL_GridDim& N, unsigned int x, unsigned int y, unsigned int z, float v[3])
{
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nz : 0;
	const I ym = (y>0) ? N.nz : 0;
	const I zm = (z>0) ? 1 : 0;
	const SYCL_MainCoeffT<I, FACTOR> C = main_coeff<I, FACTOR>(index, va, vb, mode, sn, i, 0, N, x, y, z);
	float t;
	t  = volt[i] * C.a[0];
	t += C.b[0] * (curr[2*sn+i] - curr[2*sn+i-ym] - curr[sn+i] + curr[sn+i-zm]);
	v[0] = t;
	t  = volt[sn+i] * C.a[C.s];
	t += C.b[C.s] * (curr[i] - curr[i-zm] - curr[2*sn+i] + curr[2*sn+i-xm]);
	v[1] = t;
	t  = volt[2*sn+i] * C.a[2*C.s];
	t += C.b[2*C.s] * (curr[sn+i] - curr[sn+i-xm] - curr[i] + curr[i-ym]);
	v[2] = t;
}

// The tile is as wide as a sub-group, so that a sub-group is one row of it: 32 work-items by
// default, 16 (the native SIMD width of Xe2) with OPENEMS_SYCL_FUSED_TZ=16. Both widths are
// compiled, the backend picks one at the first launch (see UpdateVoltages).
// OPENEMS_SYCL_FUSED_TZ and _TY override the shape, to measure it on a new GPU.
#define FUSED_TZ 32   // work-items along z, the last one computes the border voltages only
#define FUSED_TY 8    // work-items along y, the last row computes the border voltages only
#define FUSED_XC 4    // x lines per work-group (default): short work-groups are faster, despite the voltages of one more x line

// The arguments of update_fused but the UPML z slabs (passed separately, see launch_fused),
// so that a launch is written once for every tile shape
struct SYCL_FusedArgs
{
	const float* volt_in;
	const float* curr_in;
	float* volt_out;
	float* curr_out;
	const void* index;
	const float* va;
	const float* vb;
	const float* ia;
	const float* ib;
	unsigned int mode;
	SYCL_GridDim N, B, E;
	unsigned int xc;
	//! main nodes z_lo <= z < z_hi, the UPML z slabs below and above (see SYCL_ZSlabs)
	unsigned int z_lo, z_hi;
	//! the z tiles: t_slab slab tiles, then the main tiles from zm_start up to zm_end (see update_fused)
	unsigned int t_slab, zm_start, zm_end;
	//! the slab tiles of slab lo and hi: lines of sw work-items, sr lines (the last one the border),
	//! sy tiles along y, sn tiles (see launch_fused)
	unsigned int sw_lo, sw_hi, sr_lo, sr_hi, sy_lo, sy_hi, sn_lo, sn_hi;
};

// Fused step: the voltages and currents of the main nodes [B, E) in one pass, from the current
// fields (volt_in, curr_in) into the next buffers (volt_out, curr_out); every value is computed
// with the same operations as update_voltages/update_currents.
// One kernel for the whole timestep: the current update reads the new voltages from local
// memory, instead of a second pass reading them back from global memory.
// A work-group marches along xc x lines of a (TZ-1)x(TY-1) tile in (z,y). At each x it computes
// the new voltages of the tile and its border (+z, +y) into local memory and of the next x line
// (registers), then the new currents of the tile. The border voltages of other tiles are
// computed again (they only read the current fields), those outside [B, E) (the UPML regions
// along x and y) come from volt_out, where the region kernels stored them before.
// With the UPML z slabs in the kernel (SLABS, see SYCL_ZSlabs), [B, E) covers them along z:
// a slab node is computed with the UPML update (upml_volt_value, upml_curr_value), from the
// current voltage flux into the second flux buffer (written by the node's own work-group only,
// as other work-groups compute it again as a border voltage) and the current flux in place.
// The first t_slab tiles along z hold the slab tiles: a slab is a few nodes deep (e.g. 8), so a
// slab tile packs its y lines one after the other, lines of sw = depth+1 work-items (the last one
// the border along z), sr lines (the last one the border along y): e.g. 28 lines of 9 in 256
// work-items, nearly all busy, and only one line in 27 computed again as a border. The main tiles
// cover [z_lo, zm_end), the lines of slab hi start at zm_end = z_hi-1, its last main node: so the
// main tiles (and their border nodes) take the main code path only, the slab tiles the UPML path
// but for their first (slab hi) or border (slab lo) node, which takes the cheaper main path.
// With zm_start = B.nz, zm_end = E.nz and no slab tiles, one tiling covers all, and rows crossing
// z_lo or z_hi run both code paths.
// sV is the local memory tile sV[3][TY*TZ]: the place of a work-item is its local linear id
// ty*TZ + tz, a row (line along z) is rowlen (TZ, or sw in slab tiles) places long.
// FACTOR: coeff_mode 3/4 (see main_coeff()).
template<typename I, unsigned int TZ, unsigned int TY, bool SLABS, bool FACTOR>
inline void update_fused(const sycl::nd_item<3>& it, const sycl::local_accessor<float, 1>& sV,
                         const SYCL_FusedArgs& a, const SYCL_ZSlabs& Z)
{
	const float* __restrict__ volt_in = a.volt_in;
	const float* __restrict__ curr_in = a.curr_in;
	float* volt_out = a.volt_out;
	float* __restrict__ curr_out = a.curr_out;
	const SYCL_GridDim& N = a.N;
	const SYCL_GridDim& B = a.B;
	const SYCL_GridDim& E = a.E;
	const unsigned int tz = it.get_local_id(2), ty = it.get_local_id(1);
	const unsigned int place = ty*TZ + tz;
	// the tile: its first node (y0, z0), the end of its class along z (slab lo, main nodes, slab hi),
	// its lines (rows of rowlen work-items, the last row the border) and the place of the work-item in it
	const unsigned int gz = it.get_group(2);
	unsigned int y0, z0, z_end, rows, row, col, rowlen;
	bool lo;   // slab nodes of the tile are in slab lo, else in slab hi; the same for the work-group
	if (SLABS && (gz<a.t_slab))
	{
		const unsigned int k = gz*(unsigned int)it.get_group_range(1) + (unsigned int)it.get_group(1);
		if (k>=a.sn_lo+a.sn_hi)
			return;   // the whole work-group: no barrier waits for it
		lo = k<a.sn_lo;
		const unsigned int kk = lo ? k : k-a.sn_lo;
		const unsigned int sy = lo ? a.sy_lo : a.sy_hi;
		rowlen = lo ? a.sw_lo : a.sw_hi;
		rows = lo ? a.sr_lo : a.sr_hi;
		row = place/rowlen;
		col = place%rowlen;
		y0 = B.ny + (kk%sy)*(rows-1);
		z0 = (lo ? B.nz : a.zm_end) + (kk/sy)*(rowlen-1);
		z_end = lo ? a.z_lo : E.nz;
	}
	else
	{
		rowlen = TZ;
		rows = TY;
		row = ty;
		col = tz;
		y0 = B.ny + (unsigned int)it.get_group(1)*(TY-1);
		z0 = a.zm_start + (gz-a.t_slab)*(TZ-1);
		z_end = a.zm_end;
		lo = z0<a.z_lo;   // one tiling only: the main nodes are at least TZ long (see DecideFusedStep())
	}
	const unsigned int z = z0 + col;
	const unsigned int y = y0 + row;
	const unsigned int xs = B.nx + (unsigned int)it.get_group(0)*a.xc;
	const unsigned int xe = sycl::min(xs+a.xc, E.nx);
	const bool live = (z0<z_end) && (z<=z_end) && (row<rows);   // a node of the tile or its border
	const bool main_yz = live && (z<E.nz) && (y<E.ny);   // a node line computed here
	const bool owner = (z<z_end) && (y<E.ny) && (col<rowlen-1) && (row<rows-1);   // written by this work-group
	const bool read_yz = live && (y<=E.ny) && (z<N.nz) && (y<N.ny);   // a border line outside [B, E): from volt_out
	const bool slab = SLABS && ((z<a.z_lo) || (z>=a.z_hi));   // a node of a UPML z slab
	const I sn = (I)N.nx*N.ny*N.nz;
	auto sv = [&](unsigned int c, unsigned int p) -> float& {return sV[c*TY*TZ + p];};

	// the UPML z slab of the slab nodes of the work-item (by value: see upml_pick())
	UPMLFusedArgs Av, Ac;
	UPMLRegion R;
	if constexpr (SLABS)
	{
		Av = upml_pick(lo, Z.volt_lo, Z.volt_hi);
		Ac = upml_pick(lo, Z.curr_lo, Z.curr_hi);
		R = upml_pick(lo, Z.R_lo, Z.R_hi);
	}
	else
	{
		(void)lo;
		Av = Ac = UPMLFusedArgs();
		R = UPMLRegion();
	}

	// the new voltages at x: computed on the nodes of [B, E), else from volt_out (or unused)
	auto volt_at = [&](unsigned int x, float v[3])
	{
		if (main_yz && (x<E.nx))
		{
			if (!slab)
				fused_volt<I, FACTOR>(volt_in, curr_in, a.index, a.va, a.vb, a.mode, N, x, y, z, v);
			else if constexpr (SLABS)
				upml_volt_value<I, FACTOR>(volt_in, curr_in, Av, N, R, x, y, z, owner && (x<xe), v);
			if (owner && (x<xe))
			{
				const I i = nijk<I>(N, 0, x, y, z);
				volt_out[i] = v[0]; volt_out[sn+i] = v[1]; volt_out[2*sn+i] = v[2];
			}
		}
		else if (read_yz && (x<N.nx))
		{
			const I i = nijk<I>(N, 0, x, y, z);
			v[0] = volt_out[i]; v[1] = volt_out[sn+i]; v[2] = volt_out[2*sn+i];
		}
		else
			v[0] = v[1] = v[2] = 0;
	};

	float v[3];
	volt_at(xs, v);
	sv(0, place) = v[0]; sv(1, place) = v[1]; sv(2, place) = v[2];
	sycl::group_barrier(it.get_group());
	for (unsigned int x=xs; x<xe; ++x)
	{
		float v1[3];   // the new voltages at x+1
		volt_at(x+1, v1);
		// the currents at x, not updated on the last mesh lines (see update_currents)
		if (owner)
		{
			const bool update = (x+1<N.nx) && (y+1<N.ny) && (z+1<N.nz);
			float curl[3] = {0, 0, 0};
			if (update)
			{
				const unsigned int zp = place+1, yp = place+rowlen;
				curl[0] = (sv(2, place) - sv(2, yp) - sv(1, place) + sv(1, zp));
				curl[1] = (sv(0, place) - sv(0, zp) - sv(2, place) + v1[2]);
				curl[2] = (sv(1, place) - v1[1] - sv(0, place) + sv(0, yp));
			}
			if (slab)
			{
				if constexpr (SLABS)
					upml_curr_value<I, FACTOR>(curr_in, curr_out, Ac, N, R, x, y, z, update, curl);
			}
			else if (update)
			{
				const I i = nijk<I>(N, 0, x, y, z);
				const SYCL_MainCoeffT<I, FACTOR> C = main_coeff<I, FACTOR>(a.index, a.ia, a.ib, a.mode, sn, i, 6, N, x, y, z);
				float c;
				//for x
				c  = curr_in[i] * C.a[0];
				c += C.b[0] * curl[0];
				curr_out[i] = c;
				//for y
				c  = curr_in[sn+i] * C.a[C.s];
				c += C.b[C.s] * curl[1];
				curr_out[sn+i] = c;
				//for z
				c  = curr_in[2*sn+i] * C.a[2*C.s];
				c += C.b[2*C.s] * curl[2];
				curr_out[2*sn+i] = c;
			}
		}
		sycl::group_barrier(it.get_group());
		sv(0, place) = v1[0]; sv(1, place) = v1[1]; sv(2, place) = v1[2];
		sycl::group_barrier(it.get_group());
	}
}

// One launch of update_fused: work-groups of (1, TY, TZ) work-items (TZ along z, dimension 2),
// one sub-group of TZ per tile row. Z: the UPML z slabs in the kernel, or NULL. aligned: slab
// tiles and main tiles from z_lo (see update_fused), else one tiling from B.nz.
template<typename I, unsigned int TZ, unsigned int TY, bool FACTOR>
static void launch_fused(GPU_Backend_SYCL::Impl* d, const SYCL_FusedArgs& args, const SYCL_ZSlabs* Z, bool aligned)
{
	SYCL_FusedArgs a = args;
	auto tiles = [](unsigned int n, unsigned int w) {return (n+w-2)/(w-1);};
	const size_t grid_y = (a.E.ny-a.B.ny+TY-2)/(TY-1);
	a.sn_lo = a.sn_hi = 0;
	a.sw_lo = a.sw_hi = a.sr_lo = a.sr_hi = a.sy_lo = a.sy_hi = 2;
	if (Z && aligned)
	{
		// slab hi from its last main node on (see update_fused)
		a.zm_start = a.z_lo;
		a.zm_end = (a.E.nz>a.z_hi) ? a.z_hi-1 : a.E.nz;
		// a slab tile: lines of up to TZ work-items, as many as fit the work-group
		auto slab = [&](unsigned int depth, unsigned int& w, unsigned int& r, unsigned int& ny, unsigned int& n)
		{
			if (depth==0)
				return;
			w = std::min(depth+1, TZ);
			r = (TY*TZ)/w;
			ny = tiles(a.E.ny-a.B.ny, r);
			n = ny*tiles(depth, w);
		};
		slab(a.z_lo-a.B.nz, a.sw_lo, a.sr_lo, a.sy_lo, a.sn_lo);
		slab(a.E.nz-a.zm_end, a.sw_hi, a.sr_hi, a.sy_hi, a.sn_hi);
		a.t_slab = (a.sn_lo+a.sn_hi+grid_y-1)/grid_y;
	}
	else
	{
		a.t_slab = 0;
		a.zm_start = a.B.nz;
		a.zm_end = a.E.nz;
	}
	const size_t grid_z = a.t_slab + tiles(a.zm_end-a.zm_start, TZ);
	const size_t grid_x = (a.E.nx-a.B.nx+a.xc-1)/a.xc;
	if (grid_x==0 || grid_y==0 || grid_z==0)
		return;
	const sycl::range<3> local(1, TY, TZ);
	const sycl::range<3> global(grid_x, grid_y*TY, grid_z*TZ);
	const bool timed = SYCL_KernelTimes::Enabled();
	std::chrono::steady_clock::time_point start;
	if (timed)
	{
		d->Flush();
		start = std::chrono::steady_clock::now();
	}
	try
	{
		d->Queue().submit([&](sycl::handler& h)
		{
			sycl::local_accessor<float, 1> sV(sycl::range<1>(3*TY*TZ), h);
			if (Z)
			{
				const SYCL_ZSlabs slabs = *Z;
				h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it)
					[[sycl::reqd_work_group_size(1, TY, TZ)]] [[sycl::reqd_sub_group_size(TZ)]]
				{
					update_fused<I, TZ, TY, true, FACTOR>(it, sV, a, slabs);
				});
			}
			else
				h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it)
					[[sycl::reqd_work_group_size(1, TY, TZ)]] [[sycl::reqd_sub_group_size(TZ)]]
				{
					const SYCL_ZSlabs none = {};   // never read
					update_fused<I, TZ, TY, false, FACTOR>(it, sV, a, none);
				});
		});
	}
	catch (const sycl::exception& e)
	{
		throw std::runtime_error(std::string("GPU_Backend_SYCL: update_fused failed: ") + e.what());
	}
	if (timed)
	{
		d->Flush();
		SYCL_KernelTimes::Add("update_fused", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
	}
}

template<typename I, unsigned int TZ, bool FACTOR>
static void launch_fused_ty(GPU_Backend_SYCL::Impl* d, const SYCL_FusedArgs& a, const SYCL_ZSlabs* Z, bool aligned, unsigned int ty)
{
	switch (ty)
	{
		case 4:  launch_fused<I, TZ, 4, FACTOR>(d, a, Z, aligned); break;
		case 16: launch_fused<I, TZ, 16, FACTOR>(d, a, Z, aligned); break;
		default: launch_fused<I, TZ, 8, FACTOR>(d, a, Z, aligned); break;
	}
}

// the index type of the fused step, see GPU_Backend_SYCL::Impl::index32, and the coefficient form (factor: coeff_mode 3/4)
template<unsigned int TZ>
static void launch_fused_tz(GPU_Backend_SYCL::Impl* d, const SYCL_FusedArgs& a, const SYCL_ZSlabs* Z, bool aligned, unsigned int ty, bool index32,
                            bool factor)
{
	if (index32 && factor)
		launch_fused_ty<unsigned int, TZ, true>(d, a, Z, aligned, ty);
	else if (index32)
		launch_fused_ty<unsigned int, TZ, false>(d, a, Z, aligned, ty);
	else if (factor)
		launch_fused_ty<size_t, TZ, true>(d, a, Z, aligned, ty);
	else
		launch_fused_ty<size_t, TZ, false>(d, a, Z, aligned, ty);
}

// the currents of the given main nodes again (update_currents), after the voltage extensions
// changed voltages they read (see GPU_Backend_SYCL::Impl::fixup)
template<bool FACTOR>
inline void update_currents_nodes(const sycl::nd_item<3>& it, const float* __restrict__ curr_in, float* __restrict__ curr_out,
                                  const float* __restrict__ volt, const size_t* nodes, unsigned int count,
                                  const void* index, const float* ia, const float* ib, unsigned int mode, const SYCL_GridDim& N)
{
	const unsigned int k = it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (k>=count)
		return;
	const size_t sn = (size_t)N.nx*N.ny*N.nz;
	const size_t i  = nodes[k];
	const size_t xp = (size_t)N.ny*N.nz;
	const size_t yp = N.nz;
	const unsigned int x = i/xp, y = (i/yp)%N.ny, z = i%N.nz;
	const SYCL_MainCoeffT<size_t, FACTOR> C = main_coeff<size_t, FACTOR>(index, ia, ib, mode, sn, i, 6, N, x, y, z);
	float c;
	c  = curr_in[i] * C.a[0];
	c += C.b[0] * (volt[2*sn+i] - volt[2*sn+i+yp] - volt[sn+i] + volt[sn+i+1]);
	curr_out[i] = c;
	c  = curr_in[sn+i] * C.a[C.s];
	c += C.b[C.s] * (volt[i] - volt[i+1] - volt[2*sn+i] + volt[2*sn+i+xp]);
	curr_out[sn+i] = c;
	c  = curr_in[2*sn+i] * C.a[2*C.s];
	c += C.b[2*C.s] * (volt[sn+i] - volt[sn+i+xp] - volt[i] + volt[i+yp]);
	curr_out[2*sn+i] = c;
}

// the raw values of a dump region (see SYCL_GatherRegion) for a snapshot: only these go over
// PCIe, not the whole fields; the host interpolates them
inline void gather_region(const sycl::nd_item<3>& it, const float* __restrict__ f, const SYCL_GatherRegion& R, const SYCL_GridDim& N,
                          float* __restrict__ out)
{
	const size_t e = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (e<R.total)
		out[e] = f[region_field_index(R, N, e)];
}

// frequency domain dumps: sums[k*total + e] += raw value(e) * w[k] for the weights of one timestep,
// a float times a complex float, then a complex sum; the host interpolates the sums
#define DFT_WEIGHTS 32   // weights per launch, captured by value in the kernel lambda (256 bytes)
struct SYCL_DFTWeights { float2 w[DFT_WEIGHTS]; };

inline void field_dft(const sycl::nd_item<3>& it, const float* __restrict__ f, const SYCL_GatherRegion& R, const SYCL_GridDim& N,
                      const SYCL_DFTWeights& W, unsigned int nw, float2* __restrict__ sums)
{
	const size_t e = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (e>=R.total)
		return;
	const float v = f[region_field_index(R, N, e)];
	for (unsigned int k=0; k<nw; ++k)
	{
		float2 s = sums[k*R.total + e];
		s.x += v*W.w[k].x;
		s.y += v*W.w[k].y;
		sums[k*R.total + e] = s;
	}
}

// Field energy: the squared voltages and currents of the nodes below L, summed along x
// in float, one work-item per (y,z) line. The host sums the lines in double.
inline void energy_lines(const sycl::nd_item<3>& it, const float* __restrict__ volt, const float* __restrict__ curr, float2* partial,
                         const SYCL_GridDim& N, const SYCL_GridDim& L)
{
	const unsigned int z = it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	const unsigned int y = it.get_group(1)*it.get_local_range(1) + it.get_local_id(1);
	if (z>=L.nz || y>=L.ny)
		return;
	const size_t sn = (size_t)N.nx*N.ny*N.nz;
	float E = 0, H = 0;
	for (unsigned int x=0; x<L.nx; ++x)
	{
		const size_t i = nijk(N, 0, x, y, z);
		E += volt[i]*volt[i];
		E += volt[sn+i]*volt[sn+i];
		E += volt[2*sn+i]*volt[2*sn+i];
		H += curr[i]*curr[i];
		H += curr[sn+i]*curr[sn+i];
		H += curr[2*sn+i]*curr[2*sn+i];
	}
	float2 p;
	p.x = E;
	p.y = H;
	partial[y*L.nz + z] = p;
}

// The launches: the kernel arguments are captured by value (all trivially copyable)
template<typename I, bool FACTOR>
static void launch_update_voltages(GPU_Backend_SYCL::Impl* d, unsigned int threads_z, const SYCL_GridDim& B, const SYCL_GridDim& E)
{
	const bool c = d->coeff_mode>0;
	float* volt = d->volt;
	const float* curr = d->curr;
	const void* index = d->index;
	const float* ca = c ? d->coeff : d->vv;
	const float* cb = c ? d->coeff : d->vi;
	const unsigned int mode = d->coeff_mode;
	const SYCL_GridDim N = d->dim;
	const SYCL_GridDim Bk = B, Ek = E;
	const SYCL_ZSlabs Z = d->zslabs;
	SYCL_Launch(d, "update_voltages", threads_z, E.ny-B.ny, E.nx-B.nx, [=](sycl::nd_item<3> it)
	{
		update_voltages<I, FACTOR>(it, volt, curr, index, ca, cb, mode, N, Bk, Ek, Z);
	});
}

// Mur modes 1 and 2: update_voltages_inner, then mur_shell (see mur_volt_component)
template<typename I, bool FACTOR, bool ZF, unsigned int SG>
static void launch_update_voltages_inner(GPU_Backend_SYCL::Impl* d, unsigned int threads_z, const SYCL_GridDim& B, const SYCL_GridDim& E,
                                         const SYCL_MurShell& S, const SYCL_MurZFaces& M)
{
	const bool c = d->coeff_mode>0;
	float* volt = d->volt;
	const float* curr = d->curr;
	const void* index = d->index;
	const float* ca = c ? d->coeff : d->vv;
	const float* cb = c ? d->coeff : d->vi;
	const unsigned int mode = d->coeff_mode;
	const SYCL_GridDim N = d->dim;
	const SYCL_GridDim Bk = B, Ek = E;
	const SYCL_ZSlabs Z = d->zslabs;
	const SYCL_MurShell Sk = S;
	const SYCL_MurZFaces Mk = M;
	if constexpr (ZF)
		SYCL_Launch(d, "update_voltages_inner", threads_z, E.ny-B.ny, E.nx-B.nx, [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]]
		{
			update_voltages_inner<I, FACTOR, true>(it, volt, curr, index, ca, cb, mode, N, Bk, Ek, Z, Sk, Mk);
		});
	else
		SYCL_Launch(d, "update_voltages_inner", threads_z, E.ny-B.ny, E.nx-B.nx, [=](sycl::nd_item<3> it)
		{
			update_voltages_inner<I, FACTOR, false>(it, volt, curr, index, ca, cb, mode, N, Bk, Ek, Z, Sk, Mk);
		});
}

template<typename I, bool FACTOR>
static void launch_mur_shell(GPU_Backend_SYCL::Impl* d, const SYCL_MurShell& S, size_t count, unsigned int active)
{
	const bool c = d->coeff_mode>0;
	float* volt = d->volt;
	const float* curr = d->curr;
	const void* index = d->index;
	const float* ca = c ? d->coeff : d->vv;
	const float* cb = c ? d->coeff : d->vi;
	const unsigned int mode = d->coeff_mode;
	const SYCL_GridDim N = d->dim;
	const SYCL_MurShell Sk = S;
	const SYCL_MurFace* F = d->mur_dev;
	SYCL_Launch(d, "mur_shell", count, 1, 1, [=](sycl::nd_item<3> it)
	{
		mur_shell<I, FACTOR>(it, volt, curr, index, ca, cb, mode, N, Sk, F, active);
	});
}

// The voltage update of Mur modes 1 and 2 on the main nodes [B, E) with the Mur planes of \a active, and the shell
// nodes inside [x0,x1) x [y0,y1) (the whole grid for the single step, a window for the windowed blocks)
template<typename I, bool FACTOR>
static void launch_mur_volt(GPU_Backend_SYCL::Impl* d, unsigned int threads_z, const SYCL_GridDim& B, const SYCL_GridDim& E,
                            unsigned int active, unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
{
	// the nodes off the Mur planes, then those on them (disjoint, see mur_volt_component); mode 2: the interior
	// of the z faces in the first kernel (see mur_zface_lanes), the box spans all z lines, so mur_shell has no
	// z-face slabs (its work-items past the count would otherwise run into them)
	SYCL_MurShell S = d->mur_shell;
	SYCL_MurZFaces M = d->mur_zf;
	if (d->mur_zf_on)
	{
		S.L.nz = d->main_start.nz;
		S.H.nz = d->main_stop.nz;
		M.active = (active>>4) & 3u;
		if (d->mur_zf_sg==16)
			launch_update_voltages_inner<I, FACTOR, true, 16>(d, threads_z, B, E, S, M);
		else
			launch_update_voltages_inner<I, FACTOR, true, 32>(d, threads_z, B, E, S, M);
	}
	else
		launch_update_voltages_inner<I, FACTOR, false, 32>(d, threads_z, B, E, S, M);
	// the shell nodes inside the window: the box [S.B, S.E) and the inner box cut to it (the same shell, intersected)
	SYCL_MurShell W = S;
	W.B.nx = std::max(S.B.nx, x0); W.E.nx = std::min(S.E.nx, x1);
	W.B.ny = std::max(S.B.ny, y0); W.E.ny = std::min(S.E.ny, y1);
	if ((W.E.nx<=W.B.nx) || (W.E.ny<=W.B.ny))
		return;
	W.L.nx = std::min(std::max(S.L.nx, W.B.nx), W.E.nx); W.H.nx = std::min(std::max(S.H.nx, W.L.nx), W.E.nx);
	W.L.ny = std::min(std::max(S.L.ny, W.B.ny), W.E.ny); W.H.ny = std::min(std::max(S.H.ny, W.L.ny), W.E.ny);
	const size_t count = (size_t)(W.E.nx-W.B.nx)*(W.E.ny-W.B.ny)*(W.E.nz-W.B.nz) - (size_t)(W.H.nx-W.L.nx)*(W.H.ny-W.L.ny)*(W.H.nz-W.L.nz);
	launch_mur_shell<I, FACTOR>(d, W, count, active);
}

template<typename I, bool FACTOR>
static void launch_update_currents(GPU_Backend_SYCL::Impl* d, unsigned int threads_z, const SYCL_GridDim& B, const SYCL_GridDim& E)
{
	const bool c = d->coeff_mode>0;
	float* curr = d->curr;
	const float* volt = d->volt;
	const void* index = d->index;
	const float* ca = c ? d->coeff : d->ii;
	const float* cb = c ? d->coeff : d->iv;
	const unsigned int mode = d->coeff_mode;
	const SYCL_GridDim N = d->dim;
	const SYCL_GridDim Bk = B, Ek = E;
	const SYCL_ZSlabs Z = d->zslabs;
	SYCL_Launch(d, "update_currents", threads_z, E.ny-B.ny, E.nx-B.nx, [=](sycl::nd_item<3> it)
	{
		update_currents<I, FACTOR>(it, curr, volt, index, ca, cb, mode, N, Bk, Ek, Z);
	});
}

/***************************** helpers *****************************/

namespace
{
struct KernelTimeTable
{
	std::mutex mutex;
	std::map<std::string, std::pair<double, size_t>> times;   // total ms, launches
	~KernelTimeTable()
	{
		if (times.empty())
			return;
		double total = 0;
		for (const auto& t : times)
			total += t.second.first;
		std::vector<std::pair<double, std::string>> sorted;
		for (const auto& t : times)
			sorted.push_back(std::make_pair(t.second.first, t.first));
		std::sort(sorted.rbegin(), sorted.rend());
		fprintf(stderr, "SYCL kernel times (%.1f ms total):\n", total);
		for (const auto& t : sorted)
		{
			const size_t n = times[t.second].second;
			fprintf(stderr, "  %6.1f %%  %10.1f ms  %8zu x %8.1f us  %s\n", 100*t.first/total, t.first, n, 1000*t.first/n, t.second.c_str());
		}
	}
};
KernelTimeTable& KernelTimes() {static KernelTimeTable table; return table;}
}

bool SYCL_KernelTimes::Enabled()
{
	static const bool enabled = getenv("OPENEMS_SYCL_KERNEL_TIMES") && (atoi(getenv("OPENEMS_SYCL_KERNEL_TIMES"))!=0);
	return enabled;
}

void SYCL_KernelTimes::Add(const char* name, double ms)
{
	KernelTimeTable& t = KernelTimes();
	std::lock_guard<std::mutex> lock(t.mutex);
	std::pair<double, size_t>& entry = t.times[name];
	entry.first += ms;
	++entry.second;
}

SYCL_Context::SYCL_Context(const sycl::device& dev)
	: context(std::make_shared<sycl::context>(dev)),
	  device(dev),
	  compute_q(*context, dev, [this](sycl::exception_list errors) {AsyncHandler(errors);}, sycl::property::queue::in_order()),
	  copy_q(*context, dev, [this](sycl::exception_list errors) {AsyncHandler(errors);}, sycl::property::queue::in_order())
{
	name = dev.get_info<sycl::info::device::name>();
	// no sub-group operations in the kernels: the size only pads the UPML slabs (see MainRange())
	sub_group_size = 0;
	for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>())
		if ((s==32) || ((sub_group_size!=32) && (s>sub_group_size)))
			sub_group_size = s;
	if (sub_group_size==0)
		sub_group_size = 32;
	max_parameter_size = dev.get_info<sycl::info::device::max_parameter_size>();
	max_mem_alloc_size = dev.get_info<sycl::info::device::max_mem_alloc_size>();
}

SYCL_Context::~SYCL_Context()
{
	try
	{
		compute_q.wait();
		copy_q.wait();
	}
	catch (...)
	{
	}
}

void SYCL_Context::AsyncHandler(sycl::exception_list errors)
{
	for (const std::exception_ptr& e : errors)
	{
		std::string what;
		try
		{
			std::rethrow_exception(e);
		}
		catch (const std::exception& ex)
		{
			what = ex.what();
		}
		catch (...)
		{
			what = "unknown error";
		}
		std::cerr << "GPU_Backend_SYCL: asynchronous device error: " << what << std::endl;
		std::lock_guard<std::mutex> lock(m_AsyncMutex);
		if (m_AsyncError.empty())
			m_AsyncError = what;
	}
}

void SYCL_Context::ThrowAsyncError()
{
	std::lock_guard<std::mutex> lock(m_AsyncMutex);
	if (!m_AsyncError.empty())
		throw std::runtime_error("GPU_Backend_SYCL: device error: " + m_AsyncError);
}

GPU_Backend_SYCL::Impl::Impl()
{
	dim.nx = dim.ny = dim.nz = 0;
	numCells = 0;
	volt = curr = vv = vi = ii = iv = NULL;
	coeff_mode = 0;
	index = NULL;
	coeff = NULL;
	energy = NULL;
	energy_count = 0;
	main_start = main_stop = dim;
	upml_fused = -1;
	zslab_lo = zslab_hi = false;
	memset(&zslabs, 0, sizeof(zslabs));
	fused_step = -1;
	fused_blockers = 0;
	fused_optin = 0;
	volt_next = curr_next = NULL;
	fixup = NULL;
	index32 = false;
	fixup_count = 0;
	snap_nv = snap_n = 0;
	snap_dev = NULL;
	snap_host = NULL;
	snap_taken = false;
	mur_mode = -1;
	memset(&mur, 0, sizeof(mur));
	mur_dev = NULL;
	memset(&mur_shell, 0, sizeof(mur_shell));
	mur_shell_count = 0;
	mur_released = false;
	mur_zf_on = false;
	memset(&mur_zf, 0, sizeof(mur_zf));
	mur_zf_sg = 32;
	window = SYCL_ParseWindowKnob(NULL, 0);
	window_used = false;
	step_base = NULL;
	window_ready = -1;
	win_T = win_W = 0;
	l2_bytes = 0;
	coeff_table_bytes = 0;
	launches = 0;
	window_graphs_on = false;
	window_logged = false;
	window_steps = window_blocks = window_chunks = window_block_launches = 0;
}

GPU_Backend_SYCL::Impl::~Impl()
{
	if (!ctx)
		return;
	try
	{
		ctx->compute_q.wait();
		ctx->copy_q.wait();
	}
	catch (...)
	{
	}
	FreeSnapshots();
	for (size_t n=0; n<m_Allocations.size(); ++n)
	{
		try
		{
			sycl::free(m_Allocations.at(n), *ctx->context);
		}
		catch (...)
		{
		}
	}
}

bool GPU_Backend_SYCL::Impl::DecideFusedStep()
{
	if (fused_step>=0)
		return fused_step>0;
	fused_step = 0;
	// OPENEMS_SYCL_FUSED_STEP: unset: the default policy below, 0: never, 1: whenever the extensions allow it
	const char* env = getenv("OPENEMS_SYCL_FUSED_STEP");
	if (env && (atoi(env)==0))
	{
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (OPENEMS_SYCL_FUSED_STEP=0)" << std::endl;
		return false;
	}
	if (fused_blockers>0)
	{
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (" << fused_blockers
			          << " blocking extension(s) or grid(s))" << std::endl;
		return false;
	}
	// The fused step does not write the currents on the last mesh lines (never updated), so a
	// current an extension changes there after the current update would keep its change in one of
	// the two buffers only
	for (size_t k=0; k<curr_modified.size(); ++k)
	{
		const size_t node = curr_modified[k] % numCells;
		if ((node/((size_t)dim.ny*dim.nz)+1>=dim.nx) || ((node/dim.nz)%dim.ny+1>=dim.ny) || (node%dim.nz+1>=dim.nz))
		{
			if (g_settings.GetVerboseLevel()>1)
				std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (an extension changes currents on the last mesh lines)" << std::endl;
			return false;
		}
	}
	// Default policy: no fused step on a grid with UPML regions. On the B580 the fused step (the
	// UPML z slabs in its kernel, the regions along x and y in one region kernel per half-step) is
	// faster on a uniform grid, but not on a graded one with many coefficient sets, where its border
	// voltages computed again cost about as much as the second pass of the separate updates saves
	// (benchmark/bench.py, MC/s, 2 repeats, separate / fused):
	//   pml S 5969 / 6317, pml M 8069 / 8944, inhomogeneous S 3097 / 2892, inhomogeneous M 4322 / 4346
	// Without UPML the fused step is faster (ideal S 11004 vs 10885, M 13553 vs 11758).
	if (!upml.empty() && !(env && (atoi(env)==1)))
	{
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (UPML regions; OPENEMS_SYCL_FUSED_STEP=1 fuses them)" << std::endl;
		return false;
	}
	// Nor with the extensions whose voltage changes need a large fix-up (the planes of the Mur ABC,
	// absorbing sheets and TF/SF, the cells of the dispersive materials): the fix-up recomputes the
	// currents next to them in a second pass, scattered for planes normal to z, and costs more than
	// the fused step saves (MC/s, separate / fused, fixed number of timesteps): bench dispersive
	// S 4465 / 3819, M 5451 / 4305; bench ideal with Mur on all faces S 7939 / 7105, M 9899 / 8848;
	// a 921x708x44 model with Mur and four conducting sheets 6546 / 5732; the PCB feed line model
	// of 921x708x44 nodes (Mur, conducting sheets, lumped ports) 1957 / 1541 at timestep 8000.
	if ((fused_optin>0) && !(env && (atoi(env)==1)))
	{
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (" << fused_optin
			          << " Mur, absorbing, TF/SF or dispersive extension(s); OPENEMS_SYCL_FUSED_STEP=1 fuses them)" << std::endl;
		return false;
	}

	// The kernel arguments (captured by value) must fit the device's max_parameter_size: the
	// arguments of SYCL_FusedArgs, the UPML z slabs and a margin for the local memory accessor
	// and the ranges, about 850 bytes. The B580 reports 2048 bytes.
	// The kernel reads the slabs at fixed places (upml_pick() selects by value): a by-value struct
	// indexed at run time would be copied to private memory by every work-item, which made an
	// earlier variant with all six regions indexed by number 30 times slower than the separate updates.
	const size_t margin = 256;
	if (sizeof(SYCL_FusedArgs) + margin > ctx->max_parameter_size)
	{
		std::cout << "GPU_Backend_SYCL: the kernel arguments of the fused step do not fit the device ("
		          << sizeof(SYCL_FusedArgs) + margin << " > " << ctx->max_parameter_size << " bytes), using separate updates" << std::endl;
		return false;
	}
	// OPENEMS_SYCL_FUSED_REGIONS=0: the UPML z slabs stay with the region kernels (into the next
	// buffers, see SYCL_Ext_UPML) as the regions along x and y always do, the kernel updates the
	// main nodes only (slower on the B580: the z slabs as region kernels have short rows).
	// 2: the z slabs in the kernel with one z tiling (rows crossing the slab ends run both code
	// paths, to measure the alignment, see update_fused). Unset or 1: in the kernel, aligned tiles.
	const int regions = getenv("OPENEMS_SYCL_FUSED_REGIONS") ? atoi(getenv("OPENEMS_SYCL_FUSED_REGIONS")) : 1;
	if (regions==0)
		zslab_lo = zslab_hi = false;
	// The kernel picks the slab of a tile by its first node: with one tiling the main nodes must be
	// at least a tile row (32) long along z, so that no tile has nodes of both slabs; the slab tiles
	// need one main node (the border of slab lo).
	if ((zslab_lo || zslab_hi) && ((sizeof(SYCL_FusedArgs) + sizeof(SYCL_ZSlabs) + margin > ctx->max_parameter_size)
	                               || (main_stop.nz < main_start.nz + ((regions==2) ? 32 : 1))))
		zslab_lo = zslab_hi = false;

	// The currents of the nodes that read a voltage changed between the half-steps are computed
	// again, on main nodes. The regions along x and y are updated by the region kernels after the
	// voltage extensions; a UPML z slab in the kernel cannot be (its current flux is updated in
	// place): then its region kernels update it too.
	// (the nodes of the changed voltages first: a dispersive material registers three per node)
	std::vector<size_t> changed(volt_modified.size());
	for (size_t k=0; k<volt_modified.size(); ++k)
		changed[k] = volt_modified[k] % numCells;
	std::sort(changed.begin(), changed.end());
	changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
	std::vector<size_t> list;
	list.reserve(2*changed.size());
	for (size_t k=0; k<changed.size(); ++k)
	{
		const size_t node = changed[k];
		const unsigned int p[3] = {(unsigned int)(node/((size_t)dim.ny*dim.nz)), (unsigned int)((node/dim.nz)%dim.ny), (unsigned int)(node%dim.nz)};
		const unsigned int start[3] = {main_start.nx, main_start.ny, main_start.nz};
		const unsigned int stop[3] = {main_stop.nx, main_stop.ny, main_stop.nz};
		const unsigned int n_lines[3] = {dim.nx, dim.ny, dim.nz};
		for (int m=-1; m<3; ++m)   // the node itself and its lower neighbours
		{
			unsigned int c[3] = {p[0], p[1], p[2]};
			if (m>=0)
			{
				if (c[m]==0)
					continue;
				--c[m];
			}
			bool ok = true;
			for (int n=0; n<3; ++n)
				ok &= (c[n]+1<n_lines[n]);
			for (int n=0; n<2; ++n)
				ok &= (c[n]>=start[n]) && (c[n]<stop[n]);
			if (!ok)
				continue;
			if (c[2]<start[2])
				zslab_lo = false;   // a node of slab lo: its region kernels update it
			else if (c[2]>=stop[2])
				zslab_hi = false;
			else
				list.push_back(((size_t)c[0]*dim.ny + c[1])*dim.nz + c[2]);
		}
	}
	std::sort(list.begin(), list.end());
	list.erase(std::unique(list.begin(), list.end()), list.end());
	// The buffers are allocated last, after the field dumps: without them the run is only slower.
	volt_next = TryAlloc<float>(3*numCells);
	curr_next = TryAlloc<float>(3*numCells);
	if ((volt_next==NULL) || (curr_next==NULL))
	{
		Free(volt_next);
		Free(curr_next);
		volt_next = curr_next = NULL;
		std::cout << "GPU_Backend_SYCL: not enough device memory for the fused step, using separate updates" << std::endl;
		return false;
	}
	if (list.size()>UINT_MAX)
	{
		Free(volt_next);
		Free(curr_next);
		volt_next = curr_next = NULL;
		std::cout << "GPU_Backend_SYCL: too many nodes changed by the extensions for the fused step, using separate updates" << std::endl;
		return false;
	}
	fixup_count = list.size();
	fixup = Alloc<size_t>(fixup_count, list.data());

	// both buffers start with the current fields
	SYCL_Check("fused step buffers", [&] {Queue().memcpy(volt_next, volt, 3*numCells*sizeof(float));});
	SYCL_Check("fused step buffers", [&] {Queue().memcpy(curr_next, curr, 3*numCells*sizeof(float));});
	Flush();
	fused_step = 1;
	if (g_settings.GetVerboseLevel()>1)
	{
		std::cout << "GPU_Backend_SYCL: fused voltage and current updates" << (index32 ? ", 32 bit indices" : "");
		if (zslab_lo || zslab_hi)
			std::cout << " (" << zslab_lo+zslab_hi << " UPML regions along z in the kernel)";
		if (fixup_count)
		{
			std::ostringstream pct;
			pct << std::fixed << std::setprecision(2) << 100.0*fixup_count/numCells;
			std::cout << ", fix-up of " << fixup_count << " node(s) (" << pct.str() << "% of the nodes)";
		}
		std::cout << std::endl;
	}
	return true;
}

std::string GPU_Backend_SYCL::Impl::MurRefusal() const
{
	if (mur_released)
		return "host fallback";
	if (fused_step>0)
		return "fused step";
	const unsigned int n_lines[3] = {dim.nx, dim.ny, dim.nz};
	const unsigned int start[3] = {main_start.nx, main_start.ny, main_start.nz};
	const unsigned int stop[3] = {main_stop.nx, main_stop.ny, main_stop.nz};
	// the line cells and the shift cells, updated in the main kernel, must be main nodes, outside the UPML regions
	unsigned int faces = 0;
	for (size_t k=0; k<mur_planes.size(); ++k)
	{
		const SYCL_MurPlane& P = mur_planes[k];
		for (int n=0; n<3; ++n)
		{
			const bool in = (n==P.ny) ? ((P.line>=start[n]) && (P.line<stop[n]) && (P.shift>=start[n]) && (P.shift<stop[n]))
			                          : ((start[n]==0) && (stop[n]==n_lines[n]));
			if (!in)
				return "a Mur plane crosses a UPML region";
		}
		const unsigned int f = 2*P.ny + (P.line>0 ? 1 : 0);
		if (faces & (1u<<f))
			return "two Mur planes on one face";
		faces |= 1u<<f;
	}
	for (int n=0; n<3; ++n)
		if (((faces>>(2*n)) & 3u)==3u && (n_lines[n]<4))
			return "fewer than 4 lines between two Mur planes";
	size_t last_mur = 0;
	for (size_t k=0; k<mur_planes.size(); ++k)
		last_mur = std::max(last_mur, mur_planes[k].seq);
	// a voltage (flat index g) on the line of a Mur plane, of a component the plane writes
	auto on_line = [&](size_t g, bool any_component) {
		const unsigned int c = (unsigned int)(g/numCells);
		const size_t node = g % numCells;
		const unsigned int q[3] = {(unsigned int)(node/((size_t)dim.ny*dim.nz)), (unsigned int)((node/dim.nz)%dim.ny), (unsigned int)(node%dim.nz)};
		for (size_t k=0; k<mur_planes.size(); ++k)
			if ((q[mur_planes[k].ny]==mur_planes[k].line) && (any_component || ((int)c!=mur_planes[k].ny)))
				return true;
		return false;
	};
	// The other extensions: the Mur values are written in the main update, before the post and apply hooks
	// of all extensions, so none of them may read or write the voltages of the line cells, unless its hook
	// runs only after the Mur apply hooks (the excitation). The UPML region updates are outside the main nodes.
	for (size_t e=0; e<ext_log.size(); ++e)
	{
		Engine_Extension* eng_ext = ext_log[e].eng_ext;
		if (dynamic_cast<Engine_Ext_Mur_ABC*>(eng_ext))
			continue;
		if (dynamic_cast<Engine_Ext_UPML*>(eng_ext))
		{
			if (upml_fused<=0)
				return "UPML not fused";
			continue;
		}
		const bool disp = dynamic_cast<Engine_Ext_Dispersive*>(eng_ext)!=NULL;   // Lorentz/Drude materials, sheets, Debye materials
		const bool exc = dynamic_cast<Engine_Ext_Excitation*>(eng_ext)!=NULL;
		const bool rlc = dynamic_cast<Engine_Ext_LumpedRLC*>(eng_ext)!=NULL;
		const bool ssd = dynamic_cast<Engine_Ext_SteadyState*>(eng_ext)!=NULL;
		if (!disp && !exc && !rlc && !ssd)
			return eng_ext->GetExtensionName();
		if (ssd)
			return "steady-state detection reads the fields before the Mur apply";
		if (exc && (e>last_mur))
			continue;   // only an apply hook, after those of the Mur planes
		if (exc || rlc)
		{
			for (size_t k=ext_log[e].vm_begin; k<ext_log[e].vm_end; ++k)
				if (on_line(volt_modified[k], false))
					return eng_ext->GetExtensionName() + " on a Mur plane";
			continue;
		}
		for (size_t k=ext_log[e].nodes_begin; k<ext_log[e].nodes_end; ++k)
			if (on_line(ext_nodes[k], true))
				return eng_ext->GetExtensionName() + " on a Mur plane";
	}
	return std::string();
}

// true if the work-items of the z positions tz_a and tz_b (the work-item index along z, dimension 2 of work-groups
// of \a block, see MainRange()) are in the same sub-group of \a w work-items, for every y row of the work-group
// (DPC++ forms the sub-groups of consecutive local linear ids, z the fastest dimension)
static bool SYCL_SameSubGroupZ(unsigned int tz_a, unsigned int tz_b, const SYCL_Dim3& block, unsigned int w)
{
	if (tz_a/block.x != tz_b/block.x)
		return false;   // different work-groups
	const unsigned int a = tz_a%block.x, b = tz_b%block.x;
	for (unsigned int k=0; k<block.y*block.z; ++k)
		if ((k*block.x + a)/w != (k*block.x + b)/w)
			return false;
	return true;
}

std::string GPU_Backend_SYCL::Impl::MurZFaceRefusal()
{
	memset(&mur_zf, 0, sizeof(mur_zf));
	if (dim.nz<6)
		return "fewer than 6 z lines";
	bool any = false;
	for (size_t k=0; k<mur_planes.size(); ++k)
	{
		const SYCL_MurPlane& P = mur_planes[k];
		if (P.ny!=2)
			continue;
		const unsigned int f = (P.line>0) ? 1 : 0;
		if ((P.shift+1!=P.line) && (P.line+1!=P.shift))
			return "the shift line of a z plane is not next to its line";
		mur_zf.c_x[f] = P.c_nyP;
		mur_zf.c_y[f] = P.c_nyPP;
		mur_zf.line[f] = P.line;
		mur_zf.shift[f] = P.shift;
		any = true;
	}
	if (!any)
		return "no Mur plane on a z face";
	// the launch of update_voltages_inner (UpdateVoltages()): the work-item index along z of a main node, and the work-group
	SYCL_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	const SYCL_Dim3 block = Block(threads_z, E.ny-B.ny);
	// OPENEMS_SYCL_MUR_SG=16: sub-groups of 16 (development; default 32, see SYCL_Context::sub_group_size)
	mur_zf_sg = (getenv("OPENEMS_SYCL_MUR_SG") && (atoi(getenv("OPENEMS_SYCL_MUR_SG"))==16)) ? 16 : 32;
	bool offered = false;
	for (size_t s : ctx->device.get_info<sycl::info::device::sub_group_sizes>())
		offered = offered || (s==mur_zf_sg);
	if (!offered)
		return "no sub-groups of " + std::to_string(mur_zf_sg) + " on this device";
	auto tz = [&](unsigned int z) {return zslabs.t_int ? zslabs.t_lo + (z - zslabs.z_lo) : z - B.nz;};
	for (unsigned int f=0; f<2; ++f)
		if ((mur_zf.c_x[f]!=NULL) && !SYCL_SameSubGroupZ(tz(mur_zf.line[f]), tz(mur_zf.shift[f]), block, mur_zf_sg))
			return std::string("the line and the shift line of the z-") + (f ? "high" : "low") + " plane are in different sub-groups";
	return std::string();
}

void GPU_Backend_SYCL::Impl::DecideMurFused()
{
	if (mur_mode>=0)
		return;
	mur_mode = 0;
	const bool verbose = g_settings.GetVerboseLevel()>1;
	if (!mur_planes.empty())
	{
		const std::string reason = MurRefusal();
		if (reason.empty())
			mur_mode = 1;
		else if (verbose)
		{
			std::cout << "GPU_Backend_SYCL: Mur fused: 2 unavailable: " << reason << std::endl;
			std::cout << "GPU_Backend_SYCL: Mur fused: 1 unavailable: " << reason << std::endl;
		}
	}
	if (mur_mode==1)
	{
		// the plane applied last has the highest rank (see mur_volt_component)
		const unsigned int active = mur.active;
		memset(&mur, 0, sizeof(mur));
		mur.active = active;
		for (size_t k=0; k<mur_planes.size(); ++k)
		{
			const SYCL_MurPlane& P = mur_planes[k];
			SYCL_MurFace& F = mur.f[2*P.ny + (P.line>0 ? 1 : 0)];
			F.c_nyP = P.c_nyP;
			F.c_nyPP = P.c_nyPP;
			F.line = P.line;
			F.shift = P.shift;
			F.rank = (unsigned int)P.seq;
		}
		mur_dev = Alloc<SYCL_MurFace>(6, mur.f);
		// the box of update_voltages_inner: the main nodes but the lines and shift lines of the planes
		const unsigned int b[3] = {main_start.nx, main_start.ny, main_start.nz};
		const unsigned int e[3] = {main_stop.nx, main_stop.ny, main_stop.nz};
		unsigned int l[3], h[3];
		for (int n=0; n<3; ++n)
		{
			l[n] = b[n];
			h[n] = e[n];
		}
		for (size_t k=0; k<mur_planes.size(); ++k)
		{
			const SYCL_MurPlane& P = mur_planes[k];
			if (P.line<P.shift)
				l[P.ny] = std::max(l[P.ny], P.shift+1);
			else
				h[P.ny] = std::min(h[P.ny], P.shift);
		}
		size_t inner = 1;
		for (int n=0; n<3; ++n)
		{
			h[n] = std::max(h[n], l[n]);   // fewer lines than the two layers of both planes: no inner nodes
			inner *= h[n]-l[n];
		}
		mur_shell.B = main_start;
		mur_shell.E = main_stop;
		mur_shell.L.nx = l[0]; mur_shell.L.ny = l[1]; mur_shell.L.nz = l[2];
		mur_shell.H.nx = h[0]; mur_shell.H.ny = h[1]; mur_shell.H.nz = h[2];
		mur_shell_count = (size_t)(e[0]-b[0])*(e[1]-b[1])*(e[2]-b[2]) - inner;
		const std::string reason = MurZFaceRefusal();
		if (reason.empty())
		{
			// the z faces' interior moves to update_voltages_inner: mur_shell keeps its x- and y-face slabs
			mur_zf_on = true;
			mur_shell_count = (size_t)(e[0]-b[0] - (h[0]-l[0]))*(e[1]-b[1])*(e[2]-b[2])
			                + (size_t)(h[0]-l[0])*(e[1]-b[1] - (h[1]-l[1]))*(e[2]-b[2]);
		}
		else if (verbose)
			std::cout << "GPU_Backend_SYCL: Mur fused: 2 unavailable: " << reason << std::endl;
	}
	if (verbose)
	{
		if (mur_zf_on)
			std::cout << "GPU_Backend_SYCL: Mur fused: 2 (" << mur_planes.size() << " planes, z faces in the kernel, sub-groups of "
			          << mur_zf_sg << ")" << std::endl;
		else
			std::cout << "GPU_Backend_SYCL: Mur fused: " << mur_mode << " (" << mur_planes.size() << " planes)" << std::endl;
		if (mur_mode==1)
		{
			static const char* const faces[6] = {"x-low", "x-high", "y-low", "y-high", "z-low", "z-high"};
			std::vector<std::pair<size_t, int> > order;
			for (size_t k=0; k<mur_planes.size(); ++k)
				order.push_back(std::make_pair(mur_planes[k].seq, 2*mur_planes[k].ny + (mur_planes[k].line>0 ? 1 : 0)));
			std::sort(order.begin(), order.end());
			std::cout << "GPU_Backend_SYCL: Mur planes in apply order (edges take the last):";
			for (size_t k=0; k<order.size(); ++k)
				std::cout << " " << faces[order[k].second];
			std::cout << std::endl;
		}
	}
}

void GPU_Backend_SYCL::Impl::ResetZSlabs()
{
	zslab_lo = zslab_hi = false;
	zslabs.z_lo = main_start.nz;
	zslabs.z_hi = main_stop.nz;
}

void GPU_Backend_SYCL::Impl::MainRange(SYCL_GridDim& B, SYCL_GridDim& E, unsigned int& threads_z)
{
	B = main_start;
	E = main_stop;
	if (zslab_lo)
		B.nz = zslabs.R_lo.sz;
	if (zslab_hi)
		E.nz = zslabs.R_hi.sz + zslabs.R_hi.lz;
	threads_z = E.nz - B.nz;
	zslabs.t_lo = zslabs.t_int = 0;
	// Slab lo, main nodes and slab hi in whole sub-groups each: a sub-group shared by UPML and
	// main nodes runs both code paths
	if (zslab_lo || zslab_hi)
	{
		const unsigned int w = ctx->sub_group_size;
		auto groups = [w](unsigned int n) {return (n+w-1)/w*w;};
		zslabs.t_lo = groups(zslabs.z_lo - B.nz);
		zslabs.t_int = groups(zslabs.z_hi - zslabs.z_lo);
		threads_z = zslabs.t_lo + zslabs.t_int + groups(E.nz - zslabs.z_hi);
	}
}

void GPU_Backend_SYCL::Impl::FreeSnapshots()
{
	for (size_t r=0; r<snap_lines.size(); ++r)
		Free(snap_lines[r]);
	snap_lines.clear();
	snap_regions[0].clear();
	snap_regions[1].clear();
	if (ctx)
	{
		try
		{
			if (snap_dev)
				sycl::free(snap_dev, *ctx->context);
			if (snap_host)
				sycl::free(snap_host, *ctx->context);
		}
		catch (...)
		{
		}
	}
	snap_dev = snap_host = NULL;
	snap_nv = snap_n = 0;
	snap_taken = false;
}

void GPU_Backend_SYCL::Impl::Flush()
{
	SYCL_Check("queue synchronization", [&] {Queue().wait_and_throw();});
	ctx->ThrowAsyncError();
}

SYCL_Dim3 GPU_Backend_SYCL::Impl::Block(size_t ni, size_t nj)
{
	if (nj<=1)
		return SYCL_Dim3{256, 1, 1};
	if (ni<=16)
		return SYCL_Dim3{16, 16, 1};
	return SYCL_Dim3{32, 8, 1};
}

SYCL_Dim3 GPU_Backend_SYCL::Impl::Grid(SYCL_Dim3 block, size_t ni, size_t nj, size_t nk)
{
	return SYCL_Dim3{(ni+block.x-1)/block.x, (nj+block.y-1)/block.y, nk};
}

void* GPU_Backend_SYCL::Impl::UploadRegion(const Engine_GatherRegion& region, SYCL_GatherRegion& dev)
{
	std::vector<unsigned int> lines;
	for (int n=0; n<3; ++n)
		for (int k=0; k<3; ++k)
			lines.insert(lines.end(), region.lines[n][k].begin(), region.lines[n][k].end());
	unsigned int* buf = TryAlloc<unsigned int>(lines.size());
	if (!buf)
		return NULL;
	if (!lines.empty())
		SYCL_Check("memcpy", [&] {Queue().memcpy(buf, lines.data(), lines.size()*sizeof(unsigned int)).wait();});
	size_t pos = 0, start = 0;
	for (int n=0; n<3; ++n)
	{
		for (int k=0; k<3; ++k)
		{
			dev.lines[n][k] = buf + pos;
			dev.size[n][k] = region.lines[n][k].size();
			pos += dev.size[n][k];
		}
		dev.start[n] = start;
		start += region.Size(n);
	}
	dev.total = start;
	return buf;
}

/***************************** GPU_Backend_SYCL *****************************/

namespace
{
//! The Level Zero GPUs, found once; empty without a runtime or driver
const std::vector<sycl::device>& Devices()
{
	// never freed: the SYCL runtime may be torn down before the static destructors run
	static const std::vector<sycl::device>* devices = []()
	{
		std::vector<sycl::device>* list = new std::vector<sycl::device>();
		try
		{
			for (const sycl::device& dev : sycl::device::get_devices(sycl::info::device_type::gpu))
				if (dev.get_backend()==sycl::backend::ext_oneapi_level_zero)
					list->push_back(dev);
		}
		catch (...)
		{
			list->clear();
		}
		return list;
	}();
	return *devices;
}
}

int GPU_Backend_SYCL::DeviceCount()
{
	try
	{
		return Devices().size();
	}
	catch (...)
	{
		return 0;
	}
}

bool GPU_Backend_SYCL::DeviceInfo(int n, GPU_DeviceInfo* info)
{
	try
	{
		if ((n<0) || (n>=DeviceCount()))
			return false;
		const sycl::device& dev = Devices().at(n);
		snprintf(info->name, sizeof(info->name), "%s", dev.get_info<sycl::info::device::name>().c_str());
		info->memory = dev.get_info<sycl::info::device::global_mem_size>();
		// deprecated in SYCL 2020 without a replacement that says the same: no aspect tells
		// that the device memory is the host memory
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		info->integrated = dev.get_info<sycl::info::device::host_unified_memory>();
#pragma clang diagnostic pop
		return true;
	}
	catch (...)
	{
		return false;
	}
}

namespace
{
GPU_Backend* NewSYCLBackend(int n)
{
	return GPU_Backend_SYCL::New(n);
}
}

const GPU_Plugin* GPU_Backend_SYCL::Plugin()
{
	static const GPU_Plugin plugin = {OPENEMS_GPU_BUILD_ID, OPENEMS_GPU_PLUGIN_NAME, DeviceCount, DeviceInfo, NewSYCLBackend};
	return &plugin;
}

GPU_Backend_SYCL* GPU_Backend_SYCL::New(int device)
{
	// called through the plugin entry: nothing may throw from here
	if ((device<0) || (device>=DeviceCount()))
		return NULL;
	Impl* impl = NULL;
	std::string name = "device " + std::to_string(device);
	try
	{
		const sycl::device& dev = Devices().at(device);
		name = dev.get_info<sycl::info::device::name>();
		impl = new Impl();
		impl->ctx = std::make_shared<SYCL_Context>(dev);
	}
	catch (const std::exception& e)
	{
		std::cerr << "GPU_Backend_SYCL: cannot use " << name << ": " << e.what() << std::endl;
		delete impl;
		return NULL;
	}
	catch (...)
	{
		std::cerr << "GPU_Backend_SYCL: cannot use " << name << std::endl;
		delete impl;
		return NULL;
	}

	// A GPU this build has no code for (neither the AOT image nor one the SPIR-V compiles to)
	// fails here, when the module is built for it: run the smallest kernel there is, so that
	// the CPU is left to fall back to.
	try
	{
		sycl::queue& q = impl->ctx->compute_q;
		int* probe = sycl::malloc_device<int>(1, q);
		if (probe==NULL)
			throw std::runtime_error("malloc_device failed");
		try
		{
			q.single_task([=]() {*probe = 1;}).wait_and_throw();
			impl->ctx->ThrowAsyncError();
		}
		catch (...)
		{
			sycl::free(probe, q);
			throw;
		}
		sycl::free(probe, q);
	}
	catch (const std::exception& e)
	{
		std::cerr << "GPU_Backend_SYCL: no kernels for " << name << " in this build: " << e.what() << std::endl;
		delete impl;
		return NULL;
	}
	catch (...)
	{
		std::cerr << "GPU_Backend_SYCL: no kernels for " << name << " in this build" << std::endl;
		delete impl;
		return NULL;
	}
	return new GPU_Backend_SYCL(impl);
}

GPU_Backend* GPU_Backend_SYCL::NewSubGridBackend()
{
	// no fused step with sub-grids (cylindrical multi-grid)
	d->fused_blockers++;
	Impl* impl = new Impl();
	impl->fused_blockers++;
	impl->ctx = d->ctx;
	return new GPU_Backend_SYCL(impl);
}

GPU_Backend_SYCL::GPU_Backend_SYCL(Impl* impl)
{
	d = impl;
}

GPU_Backend_SYCL::~GPU_Backend_SYCL()
{
	delete d;
}

std::string GPU_Backend_SYCL::GetName() const
{
	return "SYCL (" + d->ctx->name + ")";
}

// The coefficient statistics line of the verbose output (one per grid): the nodes, the storage
// mode, the number of sets and the bytes of coefficient traffic per node (the set index, 2 or 4
// bytes, plus the small cached table; the 12 floats of the full arrays, 48 bytes).
static std::string CoefficientStatistics(size_t nodes, int mode, size_t sets, size_t table_bytes)
{
	std::ostringstream o;
	o << "GPU_Backend_SYCL: coefficients: " << nodes << " nodes, ";
	if (mode==0)
		o << "full arrays, " << sets << " sets, " << 12*sizeof(float) << " bytes per node of coefficient traffic";
	else
		o << "sets (" << (mode==1 ? 16 : 32) << " bit index), " << sets << " sets, " << (mode==1 ? 2 : 4)
		  << " bytes per node of coefficient traffic + table " << table_bytes << " bytes";
	return o.str();
}

namespace
{
//! Result of FactorizeCoefficients()
struct SYCL_FactorResult
{
	std::vector<float> buffer;    //!< the device buffer, see SYCL_FACTOR_HDR
	GPU_CoeffSets index;          //!< the set index per node (mode 1/2: 16/32 bit), table empty
	size_t normalized, exact;     //!< number of normalized and of exact sets
	size_t exceptions;            //!< nodes with an exact set
	size_t exact_nodes;           //!< nodes with a normalized set that reproduces all coefficients bit for bit
};

//! distance in units in the last place of two finite floats of the same sign (else large)
inline uint32_t UlpDistance(float a, float b)
{
	if (a==b)
		return 0;
	uint32_t ua, ub;
	std::memcpy(&ua, &a, 4);
	std::memcpy(&ub, &b, 4);
	if ((ua ^ ub) & 0x80000000u)
		return UINT32_MAX;
	return (ua>ub) ? ua-ub : ub-ua;
}
}

// Factorized coefficients (OPENEMS_SYCL_COEFF=factor, Cartesian meshes): on a graded mesh the
// exact sets of a node's 12 coefficients are nearly as many as the nodes, since vi and iv carry
// the mesh spacing: vi(n) = M_v(material) * len_n / (dual_p * dual_q) (Operator::Calc_ECPos,
// Calc_ECOperatorPos), iv(n) the same with primal and dual swapped, while vv and ii depend on the
// material only. So the sets hold vv, ii and vi, iv divided by that geometry (in double), and the
// kernels multiply it back from 1-D tables (main_coeff(), SYCL_FactorMul()). The normalized values
// of one material differ only by the rounding of the stored floats: they are grouped by their value
// rounded to 20 mantissa bits (relative 1e-6) and replaced by the middle of the group, plus values
// that reproduce sampled members exactly. Every node is checked by reconstructing its coefficients
// as the kernel does; a node with a coefficient more than max_ulp (OPENEMS_SYCL_COEFF_ULP, default 2)
// units in the last place off keeps its exact 12 floats as an exact set (appended to the table, not
// multiplied). Returns false if the sets do not pay (as GPU_FindSets).
static bool FactorizeCoefficients(const Operator* op, const unsigned int numLines[3], uint32_t max_ulp, SYCL_FactorResult& R)
{
	const size_t L = (size_t)numLines[0]+numLines[1]+numLines[2];
	const size_t nodes = (size_t)numLines[0]*numLines[1]*numLines[2];
	// the geometry tables in double, and as the kernel reads them (high and low floats)
	std::vector<double> geo_d(4*L, 0.0);
	std::vector<float> geo(8*L, 0.0f);
	const size_t off[3] = {0, numLines[0], (size_t)numLines[0]+numLines[1]};
	for (int a=0; a<3; ++a)
		for (unsigned int p=0; p<numLines[a]; ++p)
		{
			unsigned int pos[3] = {0, 0, 0};
			pos[a] = p;
			const double primal = op->GetEdgeLength(a, pos, false);
			const double dual = op->GetEdgeLength(a, pos, true);
			const double v[4] = {primal, dual ? 1.0/dual : 0.0, dual, primal ? 1.0/primal : 0.0};
			for (int t=0; t<4; ++t)
			{
				const float hi = (float)v[t];
				geo_d[t*L + off[a] + p] = v[t];
				geo[2*t*L + off[a] + p] = hi;
				geo[(2*t+1)*L + off[a] + p] = (float)(v[t] - (double)hi);
			}
		}
	// the geometry of the edge: f = the high and low floats of length, inverse p, inverse q; g in double
	struct Geometry
	{
		float f[6];
		double g;
		bool operator==(const Geometry& o) const {return std::equal(f, f+6, o.f);}
	};
	// of component c (0: vi, 1: iv) of direction n at xyz
	auto geometry = [&](int c, int n, const unsigned int xyz[3], Geometry& G)
	{
		const int ax[3] = {n, (n==0) ? 1 : 0, (n==2) ? 1 : 2};
		G.g = 1.0;
		for (int k=0; k<3; ++k)
		{
			const int t = 2*c + (k ? 1 : 0);
			const size_t o = off[ax[k]] + xyz[ax[k]];
			G.f[2*k] = geo[2*t*L + o];
			G.f[2*k+1] = geo[(2*t+1)*L + o];
			G.g *= geo_d[t*L + o];
		}
	};
	// the operations of main_coeff()
	auto reconstruct = [](float r, const Geometry& G) -> float
	{
		return SYCL_FactorMul(r, G.f[0], G.f[1], G.f[2], G.f[3], G.f[4], G.f[5]);
	};
	auto key_of = [](double v) -> uint64_t
	{
		uint64_t u;
		std::memcpy(&u, &v, 8);
		return (u + (1ull<<31)) & ~((1ull<<32)-1);   // 20 of the 52 mantissa bits, rounded
	};
	// the coefficients of a node, the geometry and normalized values of vi, iv (index 3*c+n; NaN:
	// cannot be normalized)
	struct Node
	{
		float raw[12];
		Geometry G[6];
		double norm[6];
	};
	auto load = [&](size_t i, Node& N)
	{
		GPU_NodeCoefficients(op, numLines, i, N.raw);
		const unsigned int xyz[3] = {(unsigned int)(i/((size_t)numLines[1]*numLines[2])),
		                             (unsigned int)((i/numLines[2]) % numLines[1]), (unsigned int)(i % numLines[2])};
		for (int k=0; k<6; ++k)
		{
			geometry(k/3, k%3, xyz, N.G[k]);
			const float b = N.raw[(k<3 ? 3 : 9) + k%3];
			const double g = N.G[k].g;
			N.norm[k] = (b==0) ? 0.0 : ((g>0 && std::isfinite(g)) ? (double)b/g : std::numeric_limits<double>::quiet_NaN());
		}
	};

	// pass 1: the groups of normalized values: their range, and their distinct members (the
	// stored coefficient and its geometry) with their counts, up to MAX_MEMBERS per group and
	// MAX_TRACKED in all (host memory)
	const unsigned int MAX_MEMBERS = 256, MAX_EXTRA = 32;
	const size_t MAX_TRACKED = 4000000;
	struct MemberKey
	{
		uint64_t key;
		uint32_t b;
		uint32_t f[6];
		bool operator==(const MemberKey& o) const {return (key==o.key) && (b==o.b) && std::equal(f, f+6, o.f);}
	};
	struct MemberHash
	{
		size_t operator()(const MemberKey& m) const
		{
			uint64_t h = (1469598103934665603ULL ^ m.key) * 1099511628211ULL;   // FNV-1a
			h = (h ^ m.b) * 1099511628211ULL;
			for (int k=0; k<6; ++k)
				h = (h ^ m.f[k]) * 1099511628211ULL;
			return h;
		}
	};
	struct MemberInfo { size_t count; double v; };
	struct Group { double lo, hi; unsigned int n; };
	std::unordered_map<uint64_t, Group> groups;
	std::unordered_map<MemberKey, MemberInfo, MemberHash> members;
	{
		Node N;
		for (size_t i=0; i<nodes; ++i)
		{
			load(i, N);
			for (int k=0; k<6; ++k)
			{
				const double v = N.norm[k];
				if (std::isnan(v) || (v==0))
					continue;
				const float b = N.raw[(k<3 ? 3 : 9) + k%3];
				MemberKey mk;
				mk.key = key_of(v);
				std::memcpy(&mk.b, &b, 4);
				std::memcpy(mk.f, N.G[k].f, sizeof(mk.f));
				auto gi = groups.find(mk.key);
				if (gi==groups.end())
					gi = groups.emplace(mk.key, Group{v, v, 0}).first;
				Group& g = gi->second;
				g.lo = std::min(g.lo, v);
				g.hi = std::max(g.hi, v);
				auto mi = members.find(mk);
				if (mi!=members.end())
					mi->second.count++;
				else if ((g.n<MAX_MEMBERS) && (members.size()<MAX_TRACKED))
				{
					members.emplace(mk, MemberInfo{1, v});
					g.n++;
				}
			}
		}
	}
	// the members of each group, the most frequent first
	std::unordered_map<uint64_t, std::vector<std::pair<size_t, const MemberKey*>>> by_group;
	for (const auto& m : members)
		by_group[m.first.key].push_back(std::make_pair(m.second.count, &m.first));
	// the normalized values of a group: first the middle of its range, then, for the most frequent
	// members it does not reproduce exactly, values that do (within 2 ulp of the member's own
	// value; at most MAX_EXTRA), so that a mesh with few distinct spacings is reproduced bit for bit
	std::unordered_map<uint64_t, std::vector<float>> rep;
	rep.reserve(groups.size());
	for (const auto& kv : groups)
	{
		const Group& g = kv.second;
		std::vector<float>& reps = rep[kv.first];
		reps.push_back((float)(0.5*(g.lo + g.hi)));
		auto bg = by_group.find(kv.first);
		if (bg==by_group.end())
			continue;
		std::vector<std::pair<size_t, const MemberKey*>>& list = bg->second;
		std::sort(list.begin(), list.end(), [](const std::pair<size_t, const MemberKey*>& x, const std::pair<size_t, const MemberKey*>& y)
		          {return x.first>y.first;});
		for (const auto& e : list)
		{
			if (reps.size()>MAX_EXTRA)
				break;
			const MemberKey& m = *e.second;
			Geometry G;
			std::memcpy(G.f, m.f, sizeof(G.f));
			float b;
			std::memcpy(&b, &m.b, 4);
			bool hit = false;
			for (float r : reps)
				hit = hit || (reconstruct(r, G)==b);
			if (hit)
				continue;
			float c = (float)members[m].v;
			c = std::nextafter(std::nextafter(c, -INFINITY), -INFINITY);
			for (int t=0; t<5; ++t, c = std::nextafter(c, INFINITY))
				if (reconstruct(c, G)==b)
				{
					reps.push_back(c);
					break;
				}
		}
	}
	by_group.clear();
	members.clear();
	groups.clear();

	// pass 2: the normalized sets; nodes that do not reconstruct get a placeholder set (all NaN)
	std::vector<size_t> exc;
	size_t exact_nodes = 0;   // nodes whose coefficients are all reproduced bit for bit
	Node N;
	const bool ok = GPU_FindSets(nodes, 12, [&](size_t i, float* values)
	{
		load(i, N);
		bool good = true, exact = true;
		for (int k=0; k<6 && good; ++k)
		{
			const int c = (k<3 ? 3 : 9) + k%3;
			const float b = N.raw[c];
			if (std::isnan(N.norm[k]))
			{
				good = false;
				break;
			}
			// the normalized value of the group that reconstructs b exactly, else the nearest
			float r = 0.0f;
			uint32_t dist = 0;
			if (N.norm[k]!=0)
			{
				dist = UINT32_MAX;
				for (float cand : rep[key_of(N.norm[k])])
				{
					const uint32_t d = UlpDistance(reconstruct(cand, N.G[k]), b);
					if (d<dist)
					{
						dist = d;
						r = cand;
					}
					if (d==0)
						break;
				}
			}
			good = dist<=max_ulp;
			exact = exact && (dist==0);
			values[c] = r;
		}
		if (!good)
		{
			exc.push_back(i);
			std::fill(values, values+12, std::numeric_limits<float>::quiet_NaN());
			return;
		}
		exact_nodes += exact;
		for (int n=0; n<3; ++n)
		{
			values[n] = N.raw[n];
			values[6+n] = N.raw[6+n];
		}
	}, R.index);
	if (!ok)
		return false;
	R.normalized = R.index.count;
	R.exceptions = exc.size();
	R.exact_nodes = exact_nodes;
	std::vector<float> table;
	table.swap(R.index.table);

	// the exact sets of the exceptions, deduplicated (else one per node)
	GPU_CoeffSets ex;
	if (!exc.empty() && !GPU_FindSets(exc.size(), 12, [&](size_t k, float* values) {GPU_NodeCoefficients(op, numLines, exc[k], values);}, ex))
	{
		ex.count = exc.size();
		ex.index.resize(exc.size());
		ex.table.resize(12*exc.size());
		for (size_t k=0; k<exc.size(); ++k)
		{
			ex.index[k] = k;
			GPU_NodeCoefficients(op, numLines, exc[k], &ex.table[12*k]);
		}
	}
	R.exact = exc.empty() ? 0 : ex.count;
	for (size_t k=0; k<exc.size(); ++k)
		R.index.index[exc[k]] = (uint32_t)(R.normalized + ex.index[k]);
	const size_t total = R.normalized + R.exact;
	if (total > std::numeric_limits<uint32_t>::max())
		return false;
	R.index.count = total;
	R.index.mode = (total <= (size_t)std::numeric_limits<uint16_t>::max()+1) ? 1 : 2;

	// the buffer: header, geometry, normalized sets, exact sets
	R.buffer.assign(SYCL_FACTOR_HDR, 0.0f);
	const uint32_t exact_start = (uint32_t)R.normalized;
	std::memcpy(&R.buffer[0], &exact_start, 4);
	R.buffer.insert(R.buffer.end(), geo.begin(), geo.end());
	R.buffer.insert(R.buffer.end(), table.begin(), table.end());
	if (R.exact)
		R.buffer.insert(R.buffer.end(), ex.table.begin(), ex.table.end());
	return true;
}

bool GPU_Backend_SYCL::Init(const Operator* op)
{
	unsigned int numLines[3];
	for (int n=0; n<3; ++n)
		numLines[n] = op->GetNumberOfLines(n, true);
	d->dim.nx = numLines[0];
	d->dim.ny = numLines[1];
	d->dim.nz = numLines[2];
	d->numCells = (size_t)numLines[0]*numLines[1]*numLines[2];
	if (g_settings.GetVerboseLevel()>1)
	{
		const sycl::device& dev = d->ctx->device;
		std::cout << "GPU_Backend_SYCL: " << d->ctx->name << ": " << dev.get_info<sycl::info::device::max_compute_units>()
		          << " compute units, L2 cache " << dev.get_info<sycl::info::device::global_mem_cache_size>()/(1024*1024)
		          << " MiB, sub-group size " << d->ctx->sub_group_size << std::endl;
	}
	d->window = SYCL_ParseWindowKnob(getenv("OPENEMS_SYCL_WINDOW"), d->dim.nx);
	d->l2_bytes = d->ctx->device.get_info<sycl::info::device::global_mem_cache_size>();
	d->main_start.nx = d->main_start.ny = d->main_start.nz = 0;
	d->main_stop = d->dim;
	d->upml_fused = -1;
	d->ResetZSlabs();
	// no fused step on a cylindrical mesh
	if (dynamic_cast<const Operator_Cylinder*>(op))
		d->fused_blockers++;

	// SYCL has no limit on the number of work-groups along x (the grid z of HIP), but a field
	// buffer must fit one allocation of the device
	const size_t field_bytes = 3*d->numCells*sizeof(float);
	if (field_bytes > d->ctx->max_mem_alloc_size)
	{
		std::cerr << "GPU_Backend_SYCL::Init: Error: a field buffer of " << field_bytes << " bytes, the device allows at most "
		          << d->ctx->max_mem_alloc_size << " bytes per allocation; use a CPU engine" << std::endl;
		return false;
	}

	// The kernels index with 32 bit while the fields fit (Intel GPUs emulate 64 bit integer
	// arithmetic); OPENEMS_SYCL_INDEX=32 or 64 overrides.
	const bool fits32 = 3*d->numCells <= UINT_MAX;
	d->index32 = fits32;
	if (const char* env = getenv("OPENEMS_SYCL_INDEX"))
		d->index32 = (atoi(env)==32);
	if (d->index32 && !fits32)
	{
		std::cerr << "GPU_Backend_SYCL: the mesh is too large for 32 bit indices, the kernels use 64 bit" << std::endl;
		d->index32 = false;
	}

	const size_t count = 3*d->numCells;
	d->volt = d->Alloc<float>(count);
	d->curr = d->Alloc<float>(count);
	if (d->window.on)
		d->step_base = d->Alloc<int>(1);

	const char* index_bits = d->index32 ? ", 32 bit indices" : ", 64 bit indices";

	// upload the final operator coefficients, including all changes by operator extensions.
	// OPENEMS_SYCL_COEFF selects the storage (the statistics line of verbose level 2 names it):
	//   sets (default): every distinct set of the 12 coefficients of a node once and a 16/32 bit
	//     set index per node (most nodes share one of a few sets: same material and mesh spacing),
	//     the full arrays if the sets do not pay (a graded mesh with many distinct spacings);
	//   factor: sets of the coefficients divided by the mesh geometry, which the kernels multiply
	//     back from 1-D tables (see FactorizeCoefficients()): few sets on graded meshes too.
	//     Cartesian meshes only (else sets); the reconstruction may differ from the stored
	//     coefficients by up to OPENEMS_SYCL_COEFF_ULP (default 2) units in the last place;
	//   full: the full arrays (48 bytes per node).
	const char* coeff_env = getenv("OPENEMS_SYCL_COEFF");
	const std::string coeff_req = coeff_env ? coeff_env : "sets";
	if (coeff_req=="factor")
	{
		if (dynamic_cast<const Operator_Cylinder*>(op))
		{
			if (g_settings.GetVerboseLevel()>1)
				std::cout << "GPU_Backend_SYCL: factorized coefficients need a Cartesian mesh, using sets" << std::endl;
		}
		else
		{
			const uint32_t max_ulp = getenv("OPENEMS_SYCL_COEFF_ULP") ? (uint32_t)std::max(0, atoi(getenv("OPENEMS_SYCL_COEFF_ULP"))) : 2;
			SYCL_FactorResult F;
			const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
			if (FactorizeCoefficients(op, numLines, max_ulp, F))
			{
				d->coeff_mode = 2 + F.index.mode;
				d->index = d->AllocIndex(F.index);
				d->coeff = d->Alloc<float>(F.buffer.size(), F.buffer.data());
				d->coeff_table_bytes = F.buffer.size()*sizeof(float);
				if (g_settings.GetVerboseLevel()>1)
				{
					const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
					std::cout << "GPU_Backend_SYCL: coefficients: " << d->numCells << " nodes, factorized ("
					          << (F.index.mode==1 ? 16 : 32) << " bit index), " << F.normalized << " normalized sets + "
					          << F.exact << " exact sets (" << F.exceptions << " exception nodes, max " << max_ulp << " ulp; "
					          << F.exact_nodes << " nodes reproduced bit for bit), "
					          << (F.index.mode==1 ? 2 : 4) << " bytes per node of coefficient traffic + table "
					          << F.buffer.size()*sizeof(float) << " bytes (geometry " << 32*((size_t)numLines[0]+numLines[1]+numLines[2])
					          << " bytes)" << index_bits << ", " << sec << " s" << std::endl;
				}
				return true;
			}
			if (g_settings.GetVerboseLevel()>1)
				std::cout << "GPU_Backend_SYCL: factorized coefficients do not pay, using sets" << std::endl;
		}
	}
	else if ((coeff_req!="sets") && (coeff_req!="full"))
		std::cerr << "GPU_Backend_SYCL: unknown OPENEMS_SYCL_COEFF=" << coeff_req << " (sets, factor or full), using sets" << std::endl;
	GPU_CoeffSets sets;
	if ((coeff_req!="full") && GPU_FindSets(d->numCells, 12, [&](size_t i, float* values) {GPU_NodeCoefficients(op, numLines, i, values);}, sets))
	{
		d->coeff_mode = sets.mode;
		d->index = d->AllocIndex(sets);
		d->coeff = d->Alloc<float>(sets.table.size(), sets.table.data());
		d->coeff_table_bytes = sets.table.size()*sizeof(float);
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: " << sets.count << " distinct coefficient sets, compressed update coefficients ("
			          << (sets.mode==1 ? 16 : 32) << " bit index)" << index_bits << std::endl;
		if (g_settings.GetVerboseLevel()>1)
			std::cout << CoefficientStatistics(d->numCells, sets.mode, sets.count, sets.table.size()*sizeof(float)) << std::endl;
		return true;
	}

	// the full arrays, one host copy at a time
	d->coeff_mode = 0;
	std::vector<float> values;
	auto upload = [&](int c) {GPU_CoefficientArray(op, numLines, c, values); return d->Alloc<float>(count, values.data());};
	d->vv = upload(0);
	d->vi = upload(1);
	d->ii = upload(2);
	d->iv = upload(3);
	if (g_settings.GetVerboseLevel()>1)
	{
		std::cout << "GPU_Backend_SYCL: full coefficient arrays" << index_bits << std::endl;
		std::cout << CoefficientStatistics(d->numCells, 0, 0, 0) << std::endl;
	}
	return true;
}

// The separate main voltage update of the nodes [B, E) (all of them for the single step, those of a window for the
// windowed blocks), with the Mur planes of \a active in it (Mur modes 1 and 2), the index type and coefficient form of
// the grid
static void MainVoltages(GPU_Backend_SYCL::Impl* d, unsigned int threads_z, const SYCL_GridDim& B, const SYCL_GridDim& E,
                         unsigned int active, unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
{
	const bool factor = d->coeff_mode>=3;
	if ((d->mur_mode==1) && active)
	{
		if (d->index32 && factor)
			launch_mur_volt<unsigned int, true>(d, threads_z, B, E, active, x0, x1, y0, y1);
		else if (d->index32)
			launch_mur_volt<unsigned int, false>(d, threads_z, B, E, active, x0, x1, y0, y1);
		else if (factor)
			launch_mur_volt<size_t, true>(d, threads_z, B, E, active, x0, x1, y0, y1);
		else
			launch_mur_volt<size_t, false>(d, threads_z, B, E, active, x0, x1, y0, y1);
		return;
	}
	if (d->index32 && factor)
		launch_update_voltages<unsigned int, true>(d, threads_z, B, E);
	else if (d->index32)
		launch_update_voltages<unsigned int, false>(d, threads_z, B, E);
	else if (factor)
		launch_update_voltages<size_t, true>(d, threads_z, B, E);
	else
		launch_update_voltages<size_t, false>(d, threads_z, B, E);
}

static void MainCurrents(GPU_Backend_SYCL::Impl* d, unsigned int threads_z, const SYCL_GridDim& B, const SYCL_GridDim& E)
{
	const bool factor = d->coeff_mode>=3;
	if (d->index32 && factor)
		launch_update_currents<unsigned int, true>(d, threads_z, B, E);
	else if (d->index32)
		launch_update_currents<unsigned int, false>(d, threads_z, B, E);
	else if (factor)
		launch_update_currents<size_t, true>(d, threads_z, B, E);
	else
		launch_update_currents<size_t, false>(d, threads_z, B, E);
}

void GPU_Backend_SYCL::UpdateVoltages()
{
	const bool c = d->coeff_mode>0;
	const bool fused = d->DecideFusedStep();
	d->DecideMurFused();   // once, after the pre hooks of the first timestep (the UPML decides its regions there)
	if (fused)
	{
		// the main nodes and the UPML z slabs in the kernel (see update_fused)
		SYCL_GridDim B = d->main_start;
		SYCL_GridDim E = d->main_stop;
		if (d->zslab_lo)
			B.nz = d->zslabs.R_lo.sz;
		if (d->zslab_hi)
			E.nz = d->zslabs.R_hi.sz + d->zslabs.R_hi.lz;
		static const unsigned int xc = getenv("OPENEMS_SYCL_FUSED_XC") ? std::max(1, atoi(getenv("OPENEMS_SYCL_FUSED_XC"))) : FUSED_XC;
		// the tile is a sub-group wide (see FUSED_TZ); OPENEMS_SYCL_FUSED_TY sets the other side
		static const unsigned int ty = getenv("OPENEMS_SYCL_FUSED_TY") ? std::max(2, atoi(getenv("OPENEMS_SYCL_FUSED_TY"))) : FUSED_TY;
		static const unsigned int tz = (getenv("OPENEMS_SYCL_FUSED_TZ") && (atoi(getenv("OPENEMS_SYCL_FUSED_TZ"))==16)) ? 16 : FUSED_TZ;
		// OPENEMS_SYCL_FUSED_REGIONS=2: one z tiling, see DecideFusedStep()
		static const bool aligned = !(getenv("OPENEMS_SYCL_FUSED_REGIONS") && (atoi(getenv("OPENEMS_SYCL_FUSED_REGIONS"))==2));
		const SYCL_FusedArgs a = { (const float*)d->volt, (const float*)d->curr, d->volt_next, d->curr_next, (const void*)d->index,
		                           (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi),
		                           (const float*)(c ? d->coeff : d->ii), (const float*)(c ? d->coeff : d->iv),
		                           d->coeff_mode, d->dim, B, E, xc, d->zslabs.z_lo, d->zslabs.z_hi, 0, 0, 0,
		                           0, 0, 0, 0, 0, 0, 0, 0 };
		// a slab missing in the kernel gets the arguments of the other one: the kernel picks them
		// (upml_pick) and the device faults on pointers of a slab that is not there (e.g. a UPML on
		// the upper z end only)
		SYCL_ZSlabs slabs = d->zslabs;
		if (!d->zslab_lo)
		{
			slabs.R_lo = slabs.R_hi;
			slabs.volt_lo = slabs.volt_hi;
			slabs.curr_lo = slabs.curr_hi;
		}
		else if (!d->zslab_hi)
		{
			slabs.R_hi = slabs.R_lo;
			slabs.volt_hi = slabs.volt_lo;
			slabs.curr_hi = slabs.curr_lo;
		}
		const SYCL_ZSlabs* Z = (d->zslab_lo || d->zslab_hi) ? &slabs : NULL;
		if (tz==16)
			launch_fused_tz<16>(d, a, Z, aligned, ty, d->index32, d->coeff_mode>=3);
		else
			launch_fused_tz<32>(d, a, Z, aligned, ty, d->index32, d->coeff_mode>=3);
		// only the voltages: UpdateCurrents swaps the currents, once the fix-up has read the old ones
		std::swap(d->volt, d->volt_next);
		return;
	}
	SYCL_GridDim B, E;
	unsigned int threads_z;
	d->MainRange(B, E, threads_z);
	MainVoltages(d, threads_z, B, E, d->mur.active, 0, d->dim.nx, 0, d->dim.ny);
}

void GPU_Backend_SYCL::UpdateCurrents()
{
	if (d->fused_step>0)
	{
		// done by update_fused, but for the currents next to voltages changed by extensions since
		if (d->fixup_count)
		{
			const bool c = d->coeff_mode>0;
			const float* curr_in = d->curr;
			float* curr_out = d->curr_next;
			const float* volt = d->volt;
			const size_t* nodes = d->fixup;
			const unsigned int count = d->fixup_count;
			const void* index = d->index;
			const float* ia = c ? d->coeff : d->ii;
			const float* ib = c ? d->coeff : d->iv;
			const unsigned int mode = d->coeff_mode;
			const SYCL_GridDim N = d->dim;
			if (mode>=3)
				SYCL_Launch(d, "update_currents_nodes", count, 1, 1, [=](sycl::nd_item<3> it)
				{
					update_currents_nodes<true>(it, curr_in, curr_out, volt, nodes, count, index, ia, ib, mode, N);
				});
			else
				SYCL_Launch(d, "update_currents_nodes", count, 1, 1, [=](sycl::nd_item<3> it)
				{
					update_currents_nodes<false>(it, curr_in, curr_out, volt, nodes, count, index, ia, ib, mode, N);
				});
		}
		std::swap(d->curr, d->curr_next);
		return;
	}
	SYCL_GridDim B, E;
	unsigned int threads_z;
	d->MainRange(B, E, threads_z);
	MainCurrents(d, threads_z, B, E);
}

/************************** windowed blocks (OPENEMS_SYCL_WINDOW) **************************/

// OPENEMS_SYCL_WINDOW when it is not set
static const char* const SYCL_WINDOW_DEFAULT = "0";

GPU_Backend_SYCL::Impl::WindowKnob SYCL_ParseWindowKnob(const char* text, unsigned int nx)
{
	GPU_Backend_SYCL::Impl::WindowKnob k;
	k.on = k.valid = k.automatic = k.tune = false;
	k.T = k.W = 0;
	k.text = text ? text : SYCL_WINDOW_DEFAULT;
	if (k.text.empty() || (k.text=="0"))
		return k;
	k.on = true;
	if ((k.text=="auto") || (k.text=="rule"))
	{
		k.valid = k.automatic = true;
		k.tune = (k.text=="auto");
		k.T = 8;
		return k;
	}
	// T,W
	const std::string::size_type comma = k.text.find(',');
	if (comma==std::string::npos)
		return k;
	const std::string t = k.text.substr(0, comma), w = k.text.substr(comma+1);
	auto number = [](const std::string& v, unsigned int& out)
	{
		if (v.empty() || (v.size()>9) || (v.find_first_not_of("0123456789")!=std::string::npos))
			return false;
		out = (unsigned int)atoi(v.c_str());
		return true;
	};
	if (!number(t, k.T))
		return k;
	if (w=="NX")
		k.W = nx;
	else if (!number(w, k.W))
		return k;
	k.valid = (k.T>0) && (k.T<=256) && (k.W>0);
	return k;
}

unsigned int SYCL_WindowRule(size_t l2_bytes, size_t plane_bytes, unsigned int T, unsigned int nx, size_t table_bytes,
                             double margin, unsigned int halo)
{
	if ((plane_bytes==0) || (nx==0))
		return std::max(nx, 1u);
	double budget = margin*(double)l2_bytes;
	if ((double)table_bytes<budget/3)
		budget -= (double)table_bytes;
	const double w = std::floor(budget/(double)plane_bytes) - (halo ? (double)halo : (T + 1.0));
	const unsigned int W = (w<1) ? 1u : (unsigned int)std::min(w, (double)nx);
	return std::max(1u, std::min(W, nx));
}

unsigned int SYCL_WindowRuleTW(size_t l2_bytes, size_t plane_bytes, unsigned int& T, unsigned int nx, size_t table_bytes)
{
	// fewer than T planes a window: the next smaller T has a shorter halo and leaves more of the budget to the window
	while (true)
	{
		const unsigned int W = SYCL_WindowRule(l2_bytes, plane_bytes, T, nx, table_bytes);
		if ((W>=T) || (W>=nx) || (T<=2))
			return W;
		T = (T>8) ? 8 : (T>4) ? 4 : 2;
	}
}

unsigned int GPU_Backend_SYCL::Impl::MurActive(unsigned int ts) const
{
	unsigned int active = 0;
	for (size_t k=0; k<mur_planes.size(); ++k)
		if (ts>=mur_planes[k].start_ts)
			active |= 1u<<(2*mur_planes[k].ny + (mur_planes[k].line>0 ? 1 : 0));
	return active;
}

unsigned int GPU_Backend_SYCL::Impl::MurNextChange(unsigned int ts) const
{
	unsigned int next = UINT_MAX;
	for (size_t k=0; k<mur_planes.size(); ++k)
		if (mur_planes[k].start_ts>ts)
			next = std::min(next, mur_planes[k].start_ts);
	return next;
}

void GPU_Backend_SYCL::Impl::WindowVoltages(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1, unsigned int mur_active)
{
	// the launches of UpdateVoltages() on the main range cut to the window; with the Mur planes in the update,
	// update_voltages_inner skips the nodes outside the box [S.L, S.H) as before and mur_shell covers the shell
	// nodes inside the window
	SYCL_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	B.nx = std::max(B.nx, x0); E.nx = std::min(E.nx, x1);
	B.ny = std::max(B.ny, y0); E.ny = std::min(E.ny, y1);
	if ((E.nx<=B.nx) || (E.ny<=B.ny))
		return;
	MainVoltages(this, threads_z, B, E, (mur_mode==1) ? mur_active : 0, x0, x1, y0, y1);
}

void GPU_Backend_SYCL::Impl::WindowCurrents(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
{
	SYCL_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	B.nx = std::max(B.nx, x0); E.nx = std::min(E.nx, x1);
	B.ny = std::max(B.ny, y0); E.ny = std::min(E.ny, y1);
	if ((E.nx<=B.nx) || (E.ny<=B.ny))
		return;
	MainCurrents(this, threads_z, B, E);
}

bool GPU_Backend_SYCL::Impl::WindowPrepare()
{
	if (window_ready>=0)
		return window_ready>0;
	window_ready = 0;
	const bool verbose = g_settings.GetVerboseLevel()>1;
	// the decisions the single step makes in the hooks and the first voltage update, before anything is recorded
	for (size_t k=0; k<window_exts.size(); ++k)
		if (SYCL_WindowHooks* h = dynamic_cast<SYCL_WindowHooks*>(window_exts[k]))
			h->WindowPrepare();
	DecideMurFused();
	if (!mur_planes.empty() && (mur_mode!=1))
	{
		// the three launches per plane of mode 0 work on whole planes: not in the windows
		if (verbose)
			std::cout << "GPU_Backend_SYCL: window: refused: the Mur planes are not in the voltage update (Mur fused: 0)" << std::endl;
		window_used = false;
		return false;
	}
	// plane bytes: the fields and the coefficient index (or the full coefficient arrays) of one x plane
	const size_t plane_bytes = (size_t)dim.ny*dim.nz*(24 + CoeffBytesPerNode());
	if (window.automatic)
	{
		win_T = window.T;
		win_W = SYCL_WindowRuleTW(l2_bytes, plane_bytes, win_T, dim.nx, coeff_table_bytes);
		win_source = "rule";
	}
	else
	{
		win_T = window.T;
		win_W = std::min(window.W, dim.nx);
		win_source = "forced";
	}
	// a V frontier must not fall between the line and the shift line of a Mur plane on an x face (the work-item of
	// the shift line writes the line, mur_volt_component): snapped down below both (see WindowFrontiers())
	win_forbidden.clear();
	for (size_t k=0; k<mur_planes.size(); ++k)
		if (mur_planes[k].ny==0)
			for (unsigned int e=std::min(mur_planes[k].line, mur_planes[k].shift)+1; e<=std::max(mur_planes[k].line, mur_planes[k].shift); ++e)
				win_forbidden.push_back(e);
	// the blocks are recorded as SYCL command graphs (sycl_ext_oneapi_graph) where the device has them; not with
	// the kernel times, which wait after every launch
	window_graphs_on = !SYCL_KernelTimes::Enabled()
	                   && (ctx->device.has(sycl::aspect::ext_oneapi_graph) || ctx->device.has(sycl::aspect::ext_oneapi_limited_graph));
	if (verbose && !window_graphs_on && !SYCL_KernelTimes::Enabled())
		std::cout << "GPU_Backend_SYCL: window: the device has no SYCL command graphs, plain launches" << std::endl;
	window_ready = 1;
	return true;
}

void GPU_Backend_SYCL::Impl::WindowFrontiers(unsigned int F, unsigned int L, std::vector<unsigned int>& eV, std::vector<unsigned int>& eI) const
{
	// Frontiers along x of a window: after its launches of step b the voltages of the planes below eV[b] and the
	// currents of the planes below eI[b] are at step b of the block. The voltage update of plane x reads the currents
	// of x-1 and x, the current update of x the voltages of x and x+1 (none on the last plane): so eI[b] = eV[b] - 1
	// (all planes once eV[b] = nx), and eV[b+1] = eI[b] (the voltages of b+1 need the currents of b of x-1 and x; the
	// currents of b that read a voltage are done before it changes). The window's front F (eV[0]) steps back one plane
	// per timestep; a frontier that would split a Mur pair (win_forbidden) is snapped down, which keeps both rules.
	const long long nx = dim.nx;
	auto snap = [&](long long e)
	{
		e = std::max(0LL, std::min(e, nx));
		while ((e>0) && (std::find(win_forbidden.begin(), win_forbidden.end(), (unsigned int)e)!=win_forbidden.end()))
			--e;
		return (unsigned int)e;
	};
	eV.assign(L, 0);
	eI.assign(L, 0);
	for (unsigned int b=0; b<L; ++b)
	{
		const long long cand = (b==0) ? (long long)F : std::min((long long)F - b, (long long)eI[b-1]);
		eV[b] = snap(cand);
		eI[b] = (eV[b]>=nx) ? (unsigned int)nx : (eV[b]>0 ? eV[b]-1 : 0);
	}
}

void GPU_Backend_SYCL::Impl::WindowBlock(unsigned int ts, unsigned int L, unsigned int W, GPU_Extension* host)
{
	// The windows march along x: window k covers the planes up to (k+1)*W at step 0 of the block, the last one all.
	// Per window and step, the voltage half-step on the planes its V frontier newly covers, then the current half-step on
	// those of its I frontier; each half-step in the hook order of Engine_GPU::VoltageHalfStep()/CurrentHalfStep() (pre
	// in reverse, the update, post, apply), every hook restricted to the half-step's planes. A cell sees the operations
	// of the single step in the same order; only the order across cells changes (see WindowFrontiers()).
	const std::vector<GPU_Extension*>& X = window_exts;
	const int count = (int)X.size();
	const unsigned int nx = dim.nx, ny = dim.ny;
	const unsigned int mur_active = (mur_mode==1) ? MurActive(ts) : 0;
	SYCL_WindowHooks* host_hooks = host ? dynamic_cast<SYCL_WindowHooks*>(host) : NULL;
	std::vector<unsigned int> pV(L, 0), pI(L, 0), eV, eI;
	for (unsigned int k=0; ; ++k)
	{
		const unsigned long long Fk = (unsigned long long)(k+1)*W;
		const bool last = Fk>=nx;
		WindowFrontiers(last ? nx + L : (unsigned int)Fk, L, eV, eI);
		for (unsigned int b=0; b<L; ++b)
		{
			if (eV[b]>pV[b])
			{
				const unsigned int x0 = pV[b], x1 = eV[b];
				for (int n=count-1; n>=0; --n)
					X[n]->LaunchInWindow(GPU_WindowPreVoltage, b, x0, x1, 0, ny);
				WindowVoltages(x0, x1, 0, ny, mur_active);
				for (int n=0; n<count; ++n)
					X[n]->LaunchInWindow(GPU_WindowPostVoltage, b, x0, x1, 0, ny);
				for (int n=0; n<count; ++n)
				{
					X[n]->LaunchInWindow(GPU_WindowApplyVoltage, b, x0, x1, 0, ny);
					if (host_hooks && (X[n]==host))
						host_hooks->HostStep(ts + b);
				}
			}
			if (eI[b]>pI[b])
			{
				const unsigned int x0 = pI[b], x1 = eI[b];
				for (int n=count-1; n>=0; --n)
					X[n]->LaunchInWindow(GPU_WindowPreCurrent, b, x0, x1, 0, ny);
				WindowCurrents(x0, x1, 0, ny);
				for (int n=0; n<count; ++n)
					X[n]->LaunchInWindow(GPU_WindowPostCurrent, b, x0, x1, 0, ny);
				for (int n=0; n<count; ++n)
					X[n]->LaunchInWindow(GPU_WindowApplyCurrent, b, x0, x1, 0, ny);
			}
		}
		pV = eV;
		pI = eI;
		if (last)
			break;
	}
}

void GPU_Backend_SYCL::Impl::WindowDropGraphs(unsigned int keep_W)
{
	for (std::map<std::vector<unsigned int>, WindowGraph>::iterator it=window_graphs.begin(); it!=window_graphs.end(); )
		if (it->first[1]!=keep_W)
			window_graphs.erase(it++);
		else
			++it;
}

void GPU_Backend_SYCL::Impl::WindowRun(unsigned int start_ts, unsigned int n)
{
	if (n==0)
		return;
	namespace sx = sycl::ext::oneapi::experimental;
	const std::vector<GPU_Extension*>& X = window_exts;
	std::vector<SYCL_WindowHooks*> hooks(X.size(), NULL);
	for (size_t k=0; k<X.size(); ++k)
		hooks[k] = dynamic_cast<SYCL_WindowHooks*>(X[k]);
	const bool verbose = g_settings.GetVerboseLevel()>1;
	// record a block of L timesteps from ts into a graph (its launches are recorded, not run); false if that fails
	auto record = [&](unsigned int ts, unsigned int L, WindowGraph& g)
	{
		const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		const unsigned long long l0 = launches;
		std::string what;
		try
		{
			sx::command_graph<sx::graph_state::modifiable> graph(Queue().get_context(), Queue().get_device());
			graph.begin_recording(Queue());
			try
			{
				WindowBlock(ts, L, win_W, NULL);
			}
			catch (const std::exception& e)
			{
				what = e.what();
			}
			graph.end_recording(Queue());
			if (what.empty())
				g.exec = std::make_shared<SYCL_ExecGraph>(graph.finalize());
		}
		catch (const std::exception& e)
		{
			if (what.empty())
				what = e.what();
		}
		if (!what.empty())
		{
			std::cout << "GPU_Backend_SYCL: window: graph recording failed (" << what << "), the same launches without graphs" << std::endl;
			window_graphs_on = false;
			return false;
		}
		g.launches = launches - l0;
		g.capture_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		if (verbose && window_logged)
		{
			std::ostringstream line;
			line << "GPU_Backend_SYCL: window: graph " << window_graphs.size()+1 << " for blocks of " << L << " steps (" << g.launches
			     << " launches, recorded in " << std::fixed << std::setprecision(1) << g.capture_ms << " ms)";
			std::cout << line.str() << std::endl;
		}
		return true;
	};
	unsigned int ts = start_ts;
	const unsigned int end = start_ts + n;
	while (ts<end)
	{
		// the next timestep an extension needs the host in (run alone, over the whole grid)
		unsigned int host_ts = UINT_MAX;
		GPU_Extension* host = NULL;
		for (size_t k=0; k<X.size(); ++k)
			if (hooks[k])
			{
				const unsigned int h = hooks[k]->NextHostStep(ts);
				if (h<host_ts)
				{
					host_ts = h;
					host = X[k];
				}
			}
		// the timestep of the block, read by the window launches of the extensions (outside a graph, whose
		// arguments are frozen)
		int* base = step_base;
		const int step = (int)ts;
		SYCL_Check("set_step", [&] {Queue().single_task([=]() {*base = step;}); ++launches;});
		unsigned int L = 1;
		if (host_ts==ts)
			WindowBlock(ts, 1, dim.nx, host);
		else
		{
			L = std::min(win_T, end - ts);
			if (host_ts!=UINT_MAX)
				L = std::min(L, host_ts - ts);
			if (mur_mode==1)
			{
				const unsigned int change = MurNextChange(ts);
				if (change!=UINT_MAX)
					L = std::min(L, change - ts);   // the active Mur planes are an argument of the launches
			}
			bool done = false;
			if (window_graphs_on)
			{
				std::vector<unsigned int> key(3);
				key[0] = L;
				key[1] = win_W;
				key[2] = (mur_mode==1) ? MurActive(ts) : 0;
				std::map<std::vector<unsigned int>, WindowGraph>::iterator it = window_graphs.find(key);
				if ((it==window_graphs.end()) && (window_graphs.size()<32))
				{
					WindowGraph g;
					if (record(ts, L, g))
						it = window_graphs.insert(std::make_pair(key, g)).first;
				}
				if (it!=window_graphs.end())
				{
					SYCL_Check("graph replay", [&] {Queue().ext_oneapi_graph(*it->second.exec);});
					done = true;
				}
			}
			if (!done)
			{
				const unsigned long long l0 = launches;
				WindowBlock(ts, L, win_W, NULL);
				if (L==win_T)
					window_block_launches = launches - l0;
			}
		}
		for (size_t k=0; k<X.size(); ++k)
			X[k]->EndOfBlock(ts + L - 1);
		ts += L;
		++window_blocks;
	}
	window_steps += n;
}

bool GPU_Backend_SYCL::SetWindowExtensions(const std::vector<GPU_Extension*>& exts, const std::string& refusal)
{
	const bool verbose = g_settings.GetVerboseLevel()>1;
	d->window_used = false;
	d->window_exts.clear();
	if (!d->window.on)
	{
		if (verbose)
			std::cout << "GPU_Backend_SYCL: window: off" << std::endl;
		return false;
	}
	if (!refusal.empty())
	{
		if (verbose)
			std::cout << "GPU_Backend_SYCL: window: refused: " << refusal << std::endl;
		return false;
	}
	if (!d->window.valid || (d->step_base==NULL))
	{
		if (verbose)
			std::cout << "GPU_Backend_SYCL: window: invalid value " << d->window.text << " (0, auto, rule or T,W), single-step" << std::endl;
		return false;
	}
	for (size_t n=0; n<exts.size(); ++n)
		if (!exts[n]->WindowCapable())
		{
			if (verbose)
				std::cout << "GPU_Backend_SYCL: window: refused: an extension is not window-capable" << std::endl;
			return false;
		}
	// the windowed blocks run the separate kernels; whether the single step would fuse (see DecideFusedStep())
	const char* fused_env = getenv("OPENEMS_SYCL_FUSED_STEP");
	const bool fused_on = fused_env && (atoi(fused_env)==1);
	const bool fused = (d->fused_step>0) || ((d->fused_step<0) && (d->fused_blockers==0) && !(fused_env && (atoi(fused_env)==0))
	                                         && (fused_on || (d->upml.empty() && (d->fused_optin==0))));
	d->window_exts = exts;
	d->window_used = true;
	if (d->fused_step<0)
		d->fused_step = 0;
	if (verbose && fused)
		std::cout << "GPU_Backend_SYCL: window: separate updates, the fused step is not used on the windowed path" << std::endl;
	return true;
}

bool GPU_Backend_SYCL::IterateWindowed(unsigned int start_ts, unsigned int n)
{
	if (!d->window_used)
		return false;
	if (d->fused_step>0)
		return false;   // decided before the windowed blocks: the fields are in its double buffers
	if (!d->WindowPrepare())
		return false;
	++d->window_chunks;
	d->WindowRun(start_ts, n);
	if (!d->window_logged && (g_settings.GetVerboseLevel()>1))
	{
		// after the first chunk: the launches of a block of T steps (with its set_step), the graphs so far
		d->window_logged = true;
		// the launches per step of the longest block (a graph's, or the last block of T steps without one)
		unsigned long long per_block = d->window_block_launches;
		unsigned int per_len = d->win_T;
		double capture_ms = 0;
		for (std::map<std::vector<unsigned int>, Impl::WindowGraph>::const_iterator it=d->window_graphs.begin(); it!=d->window_graphs.end(); ++it)
		{
			if ((per_block==0) || (it->first[0]>per_len) || ((it->first[0]==per_len) && (it->second.launches>0)))
			{
				per_block = it->second.launches;
				per_len = it->first[0];
			}
			capture_ms += it->second.capture_ms;
		}
		const unsigned int nx = d->dim.nx;
		std::ostringstream line;
		line << "GPU_Backend_SYCL: window: T=" << d->win_T << " W=" << d->win_W << " planes=" << nx
		     << " halo=" << d->win_T+1 << " windows=" << (nx + d->win_W - 1)/d->win_W << " launches/step=";
		if (per_block)
			line << std::fixed << std::setprecision(1) << (per_block + 1.0)/per_len;
		else
			line << "?";
		line << " graphs=" << d->window_graphs.size();
		if (!d->window_graphs.empty())
			line << " (recorded in " << std::fixed << std::setprecision(1) << capture_ms << " ms)";
		else if (SYCL_KernelTimes::Enabled())
			line << " (kernel times: plain launches)";
		line << " (" << d->win_source << ", " << d->dim.ny*(size_t)d->dim.nz*(24 + d->CoeffBytesPerNode())
		     << " bytes per plane, coefficient table " << d->coeff_table_bytes << " bytes, L2 " << d->l2_bytes/(1024*1024) << " MiB, " << d->window_exts.size() << " window-capable extensions)";
		std::cout << line.str() << std::endl;
	}
	return true;
}

static void CheckSize(size_t host_count, size_t device_count)
{
	if (host_count!=device_count)
		throw std::runtime_error("GPU_Backend_SYCL: host and device field size mismatch");
}

void GPU_Backend_SYCL::DownloadVoltages(ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CheckSize(volt.size(), 3*d->numCells);
	d->PinHostMemory(volt.data(), volt.size()*sizeof(FDTD_FLOAT));
	SYCL_Check("download", [&] {d->Queue().memcpy(volt.data(), d->volt, volt.size()*sizeof(float));});
	d->Flush();
}

void GPU_Backend_SYCL::DownloadCurrents(ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CheckSize(curr.size(), 3*d->numCells);
	d->PinHostMemory(curr.data(), curr.size()*sizeof(FDTD_FLOAT));
	SYCL_Check("download", [&] {d->Queue().memcpy(curr.data(), d->curr, curr.size()*sizeof(float));});
	d->Flush();
}

void GPU_Backend_SYCL::UploadVoltages(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CheckSize(volt.size(), 3*d->numCells);
	d->PinHostMemory((void*)volt.data(), volt.size()*sizeof(FDTD_FLOAT));
	SYCL_Check("upload", [&] {d->Queue().memcpy(d->volt, volt.data(), volt.size()*sizeof(float));});
	d->Flush();
}

void GPU_Backend_SYCL::UploadCurrents(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CheckSize(curr.size(), 3*d->numCells);
	d->PinHostMemory((void*)curr.data(), curr.size()*sizeof(FDTD_FLOAT));
	SYCL_Check("upload", [&] {d->Queue().memcpy(d->curr, curr.data(), curr.size()*sizeof(float));});
	d->Flush();
}

bool GPU_Backend_SYCL::DownloadRange(bool currents, size_t offset, size_t count, FDTD_FLOAT* dst)
{
	if (offset+count > 3*d->numCells)
		return false;
	const float* src = (currents ? d->curr : d->volt) + offset;
	if (count)
		SYCL_Check("range download", [&] {d->Queue().memcpy(dst, src, count*sizeof(float));});
	d->Flush();
	return true;
}

void GPU_Backend_SYCL::Synchronize()
{
	d->Flush();
}

bool GPU_Backend_SYCL::SetSnapshotGather(const std::vector<Engine_GatherRegion>& volt_regions, const std::vector<Engine_GatherRegion>& curr_regions)
{
	SYCL_Check("queue synchronization", [&] {d->ctx->copy_q.wait_and_throw();});
	d->FreeSnapshots();
	size_t nv = 0, n = 0;
	for (size_t r=0; r<volt_regions.size(); ++r)
		nv += volt_regions[r].Size();
	n = nv;
	for (size_t r=0; r<curr_regions.size(); ++r)
		n += curr_regions[r].Size();
	if ((n==0) || (n>UINT_MAX))
		return false;
	// not fatal: without snapshots the dumps read the fields from the host mirror
	bool ok = true;
	try
	{
		d->snap_dev = sycl::malloc_device<float>(n, d->ctx->device, *d->ctx->context);
		d->snap_host = sycl::malloc_host<float>(n, *d->ctx->context);
	}
	catch (const sycl::exception&)
	{
		ok = false;
	}
	ok = ok && d->snap_dev && d->snap_host;
	const std::vector<Engine_GatherRegion>* regions[2] = {&volt_regions, &curr_regions};
	for (int f=0; f<2 && ok; ++f)
		for (size_t r=0; r<regions[f]->size() && ok; ++r)
		{
			SYCL_GatherRegion dev;
			void* buf = d->UploadRegion(regions[f]->at(r), dev);
			ok = (buf!=NULL);
			if (ok)
			{
				d->snap_lines.push_back(buf);
				d->snap_regions[f].push_back(dev);
			}
		}
	if (!ok)
	{
		d->FreeSnapshots();
		std::cerr << "GPU_Backend_SYCL: not enough memory for field snapshots, the field dumps wait for the device" << std::endl;
		return false;
	}
	d->snap_nv = nv;
	d->snap_n = n;
	return true;
}

bool GPU_Backend_SYCL::SnapshotFields(const FDTD_FLOAT* &volt, const FDTD_FLOAT* &curr)
{
	if (!d->snap_n)
		return false;
	// the raw values on the compute queue, once the last snapshot was downloaded from snap_dev
	if (d->snap_taken)
		SYCL_Check("barrier", [&] {d->Queue().ext_oneapi_submit_barrier({d->snap_done});});
	size_t offset = 0;
	const SYCL_GridDim N = d->dim;
	for (int f=0; f<2; ++f)
		for (size_t r=0; r<d->snap_regions[f].size(); ++r)
		{
			const SYCL_GatherRegion R = d->snap_regions[f][r];
			const float* field = f ? d->curr : d->volt;
			float* out = d->snap_dev + offset;
			SYCL_Launch(d, "gather_region", R.total, 1, 1, [=](sycl::nd_item<3> it)
			{
				gather_region(it, field, R, N, out);
			});
			offset += R.total;
		}
	SYCL_Check("barrier", [&] {d->snap_evaluated = d->Queue().ext_oneapi_submit_barrier();});
	// the download on the copy queue, while the compute queue continues with the next timesteps
	sycl::queue& copy = d->ctx->copy_q;
	SYCL_Check("snapshot download", [&]
	{
		copy.ext_oneapi_submit_barrier({d->snap_evaluated});
		d->snap_done = copy.memcpy(d->snap_host, d->snap_dev, d->snap_n*sizeof(float));
	});
	d->snap_taken = true;
	volt = d->snap_host;
	curr = d->snap_host + d->snap_nv;
	return true;
}

int GPU_Backend_SYCL::AddFieldDFT(bool h_field, const Engine_GatherRegion& region, unsigned int count)
{
	const size_t n = region.Size();
	if ((n==0) || (count==0))
		return -1;
	Impl::FieldDFT dft;
	dft.count = count;
	dft.currents = h_field;
	void* lines = d->UploadRegion(region, dft.region);
	dft.sums = lines ? d->TryAlloc<float2>(n*count) : NULL;
	if (!dft.sums)
	{
		// not fatal: without device sums the dump sums the fields on the host
		d->Free(lines);
		std::cerr << "GPU_Backend_SYCL: not enough memory for a frequency domain dump on the device" << std::endl;
		return -1;
	}
	SYCL_Check("memset", [&] {d->Queue().memset(dft.sums, 0, n*count*sizeof(float2));});
	d->field_dfts.push_back(dft);
	return d->field_dfts.size()-1;
}

void GPU_Backend_SYCL::AccumulateFieldDFT(int id, const std::vector<std::complex<float>>& weights)
{
	if ((id<0) || ((size_t)id>=d->field_dfts.size()))
		return;
	const Impl::FieldDFT& dft = d->field_dfts[id];
	const unsigned int count = std::min((size_t)dft.count, weights.size());
	const SYCL_GatherRegion R = dft.region;
	const SYCL_GridDim N = d->dim;
	const float* field = dft.currents ? d->curr : d->volt;
	for (unsigned int k0=0; k0<count; k0+=DFT_WEIGHTS)
	{
		SYCL_DFTWeights W;
		const unsigned int nw = std::min(count-k0, (unsigned int)DFT_WEIGHTS);
		for (unsigned int k=0; k<nw; ++k)
		{
			W.w[k].x = weights[k0+k].real();
			W.w[k].y = weights[k0+k].imag();
		}
		float2* sums = dft.sums + (size_t)k0*R.total;
		SYCL_Launch(d, "field_dft", R.total, 1, 1, [=](sycl::nd_item<3> it)
		{
			field_dft(it, field, R, N, W, nw, sums);
		});
	}
}

bool GPU_Backend_SYCL::ReadFieldDFT(int id, unsigned int k, std::vector<std::complex<float>>& sums)
{
	if ((id<0) || ((size_t)id>=d->field_dfts.size()) || (k>=d->field_dfts[id].count))
		return false;
	const Impl::FieldDFT& dft = d->field_dfts[id];
	sums.resize(dft.region.total);
	// std::complex<float> has the layout of float2
	SYCL_Check("download", [&] {d->Queue().memcpy(sums.data(), dft.sums + (size_t)k*dft.region.total, sums.size()*sizeof(float2));});
	d->Flush();
	return true;
}

void GPU_Backend_SYCL::WaitSnapshot()
{
	// called on the dump thread: report errors instead of throwing (a default event, no snapshot taken, is complete)
	try
	{
		d->snap_done.wait();
	}
	catch (const std::exception& e)
	{
		std::cerr << "GPU_Backend_SYCL: field snapshot failed: " << e.what() << std::endl;
	}
}

bool GPU_Backend_SYCL::CalcFastEnergy(const unsigned int numNodes[3], double& E_energy, double& H_energy)
{
	E_energy = H_energy = 0;
	SYCL_GridDim L = {numNodes[0], numNodes[1], numNodes[2]};
	if ((L.nx>d->dim.nx) || (L.ny>d->dim.ny) || (L.nz>d->dim.nz))
		return false;
	const size_t count = (size_t)L.ny*L.nz;
	if (L.nx==0 || count==0)
		return true;
	if (d->energy_count<count)
	{
		d->energy = d->Alloc<float2>(count);
		d->energy_count = count;
		d->energy_host.resize(count);   // small, not pinned
	}
	const float* volt = d->volt;
	const float* curr = d->curr;
	float2* partial = d->energy;
	const SYCL_GridDim N = d->dim;
	SYCL_Launch(d, "energy_lines", L.nz, L.ny, 1, [=](sycl::nd_item<3> it)
	{
		energy_lines(it, volt, curr, partial, N, L);
	});
	SYCL_Check("energy download", [&] {d->Queue().memcpy(d->energy_host.data(), d->energy, count*sizeof(float2));});
	d->Flush();
	for (size_t k=0; k<count; ++k)
	{
		E_energy += d->energy_host[k].x;
		H_energy += d->energy_host[k].y;
	}
	return true;
}

// all SYCL extensions, see sycl_internal.h
static const SYCL_ExtensionFactory SYCL_EXTENSIONS[] = {
	SYCL_CreateExt_Excitation,
	SYCL_CreateExt_UPML,
	SYCL_CreateExt_Mur_ABC,
	SYCL_CreateExt_LorentzMaterial,
	SYCL_CreateExt_DebyeMaterial,
	SYCL_CreateExt_LumpedRLC,
	SYCL_CreateExt_TFSF,
	SYCL_CreateExt_Absorbing_BC,
	SYCL_CreateExt_SteadyState,
	SYCL_CreateExt_Cylinder,
};

GPU_Extension* GPU_Backend_SYCL::CreateExtension(Engine_Extension* eng_ext, Engine* eng)
{
	// The fused step (see update_fused) knows the UPML, the excitation, lumped RLC elements, the
	// steady-state detection, and, fused only on request (see fused_optin), the Mur ABC, absorbing
	// sheets, TF/SF, Lorentz/Drude materials and conducting sheets and Debye materials: these run on
	// the fields in place as without it, the currents next to the voltages they change are
	// recomputed (volt_modified, see DecideFusedStep()). The others (the cylinder and multi-grid
	// extensions) block it.
	if (dynamic_cast<Engine_Ext_Mur_ABC*>(eng_ext) || dynamic_cast<Engine_Ext_Absorbing_BC*>(eng_ext) || dynamic_cast<Engine_Ext_TFSF*>(eng_ext)
	    || dynamic_cast<Engine_Ext_LorentzMaterial*>(eng_ext) || dynamic_cast<Engine_Ext_DebyeMaterial*>(eng_ext))
		d->fused_optin++;
	else if (!dynamic_cast<Engine_Ext_UPML*>(eng_ext) && !dynamic_cast<Engine_Ext_Excitation*>(eng_ext) && !dynamic_cast<Engine_Ext_LumpedRLC*>(eng_ext)
	         && !dynamic_cast<Engine_Ext_SteadyState*>(eng_ext))
		d->fused_blockers++;
	// in the extension order of the engine (see Impl::ext_log)
	const Impl::ExtRecord record = {eng_ext, d->volt_modified.size(), 0, d->ext_nodes.size(), 0};
	for (size_t n=0; n<sizeof(SYCL_EXTENSIONS)/sizeof(SYCL_EXTENSIONS[0]); ++n)
	{
		if (SYCL_EXTENSIONS[n]==NULL)
			continue;
		GPU_Extension* gpu_ext = SYCL_EXTENSIONS[n](d, eng_ext, eng);
		if (gpu_ext)
		{
			d->ext_log.push_back(record);
			d->ext_log.back().vm_end = d->volt_modified.size();
			d->ext_log.back().nodes_end = d->ext_nodes.size();
			return gpu_ext;
		}
	}
	d->fused_blockers++;   // host fallback
	return NULL;
}
