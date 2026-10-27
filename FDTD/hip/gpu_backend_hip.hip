/*
*	Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
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
 * Environment knobs of the HIP backend (read once when a grid is set up; --verbose=2 prints the decisions).
 * Every knob changes how the backend computes, never what: unless a knob says otherwise its settings are
 * bit-identical to each other. Speeds are from an RTX 5090: "PCB" is a PCB model of 28.7 Mcells (921 x 708 x
 * 44, Mur on six faces, four copper sheet planes), "small PCB" a model of 3.06 Mcells (337 x 227 x 40, Mur,
 * four sheet planes).
 *
 * OPENEMS_HIP_WINDOW = auto (default) | 0 | rule | T,W[,B]
 *     The timesteps as blocks of T steps marched along x in windows of W x planes that stay in the L2 cache, each
 *     block captured once as a graph and replayed; the voltage and current frontiers step back one plane per step,
 *     so the halo is T+1 planes. auto: the rule (T = 8, W = 0.65 L2 / plane bytes - (T+1); where that leaves fewer
 *     than T planes a window, T = 4, then 2: on a small L2 the halo of T = 8 alone takes most of it), then the
 *     tuner times up to 13 (T, W) candidates (T 2 to 16, 0.45 to 0.85 of L2, and one window of all x planes) and
 *     refines the three best with W -/+25 % on the first timesteps of the run (real timesteps: every candidate is
 *     bit-identical) and keeps the fastest; every run tunes. rule: the rule without the tuner.
 *     0: one timestep at a time. T,W: forced (W may be NX; B is accepted and unused). Grids with TF/SF, Debye,
 *     absorbing sheets, cylinder coordinates, multi-grids or Mur planes outside the voltage update keep the single
 *     step; auto also keeps the fused step on grids where it folds the UPML regions into its kernel and one window
 *     covers all of x (bench pml S: 23839 MC/s fused against 19406 windowed). PCB: 22944 MC/s at T 8, W 34 (the
 *     rule's window while its halo was 2T+1) against 13877 single-step (+65 %); small PCB: 21806 against 13901
 *     (+57 %), tuned (T 16, W 193) 23906 against 21870 with that rule (+9.3 %).
 * OPENEMS_HIP_ZPAD = auto (default) | 0
 *     The z stride of the device layout rounded up to 64, so that every z line starts on the same alignment. auto: on
 *     grids where the fused step is blocked, unless that more than doubles the z lines or the padded fields and
 *     coefficients would take more than half of the free device memory; 0: off.
 * OPENEMS_HIP_SHAPE = auto (default) | 0
 *     The thread block of the separate main kernels: auto times candidates on the live grid at the first timestep;
 *     a candidate within 0.5 % of the fastest ties with it, and a tie goes to the first listed, 32x8 (on the PCB,
 *     32x8, 64x4 and 32x4 tie within 0.03 % there, while inside the windowed blocks 32x8 is 1.8 to 2.6 % faster).
 *     0 the former block. ZPAD and SHAPE together: small PCB 10226 to 11572 MC/s (+13 %), the PCB
 *     12822 to 13877 (+8.2 %).
 * OPENEMS_HIP_INDEX = 32 | 64
 *     The index width of the fused step; default 32 on NVIDIA where the grid fits (V100: 64 registers against 126,
 *     half the occupancy with 64), 64 on AMD.
 * OPENEMS_HIP_FUSED_STEP = unset (default) | 0
 *     The voltage and current updates fused into one pass, on grids without Mur, sheets and the other extensions
 *     that block it; 0 never. The windowed blocks always run the separate kernels. V100: pml S/M within 2 to 7 %
 *     of the two-pass path with the UPML fused into the main kernels.
 * OPENEMS_HIP_FUSED_TZ, OPENEMS_HIP_FUSED_TY, OPENEMS_HIP_FUSED_XC
 *     Development: the tile of the fused step (TZ 32 or 64 threads along z, AMD only; TY 4, 8 or 16 rows along y,
 *     default 8, other values run 8; XC x lines per block, default 4).
 * OPENEMS_HIP_UPML_FUSION = 1 (default) | 0
 *     The UPML regions fused with the main updates; 0 the separate pre/post kernels (V100: 27 to 32 % slower).
 * OPENEMS_HIP_KERNEL_TIMES = 0 (default) | 1
 *     Development: a block of kernel times at the end of the run, from events around every launch (no graphs);
 *     inflates the run time (small PCB by about 53 %): kernel shares, never speeds.
 */

#include "hip/hip_runtime.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "hip_internal.h"
#include "gpu_build_id.h"
#include "FDTD/operator.h"
#include "FDTD/operator_cylindermultigrid.h"
#include "FDTD/extensions/engine_ext_upml.h"
#include "FDTD/extensions/engine_ext_excitation.h"
#include "FDTD/extensions/engine_ext_lumpedRLC.h"
#include "FDTD/extensions/engine_ext_steadystate.h"
#include "FDTD/extensions/engine_ext_mur_abc.h"
#include "FDTD/extensions/engine_ext_dispersive.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_steadystate.h"

// Main FDTD updates, operation by operation the same as Engine::UpdateVoltages/UpdateCurrents,
// over the nodes in [B, E). One thread per mesh node: z along x-threads (contiguous), y along
// y-threads, x along the grid z-dimension. The coefficients are read with main_coeff().
// Nodes below Z.z_lo or from Z.z_hi on belong to UPML regions along z (see HIP_ZSlabs).
__global__ void update_voltages(float* __restrict__ volt, const float* __restrict__ curr,
                                const void* index, const float* ca, const float* cb, unsigned int mode,
                                HIP_GridDim N, HIP_GridDim B, HIP_GridDim E, HIP_ZSlabs Z)
{
	// with UPML regions along z: slab lo, main nodes and slab hi in whole warps each (no divergence)
	const unsigned int tz = blockIdx.x*blockDim.x + threadIdx.x;
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
	const unsigned int y = B.ny + blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = B.nx + blockIdx.z;
	if (z>=E.nz || y>=E.ny || x>=E.nx)
		return;
	if (z<Z.z_lo)
	{
		upml_fused_volt_node(volt, volt, curr, Z.volt_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		return;
	}
	if (z>=Z.z_hi)
	{
		upml_fused_volt_node(volt, volt, curr, Z.volt_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		return;
	}
	const size_t sn = (size_t)N.nx*N.ny*N.nzd;
	const size_t i  = nijk(N, 0, x, y, z);
	const size_t xm = (x>0) ? (size_t)N.ny*N.nzd : 0;   // shift to the previous line, none on the first
	const size_t ym = (y>0) ? N.nzd : 0;
	const size_t zm = (z>0) ? 1 : 0;
	const HIP_MainCoeff C = main_coeff(index, ca, cb, mode, sn, i, 0);
	const float* vv = C.a;
	const float* vi = C.b;
	const size_t s = C.s;
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
// mode 0 runs as mur_pre, mur_post and mur_apply (hip_ext_mur_abc.hip, Engine_Ext_Mur_ABC), as two launches over
// disjoint nodes: update_voltages_inner off the planes, mur_shell on the line and shift line of each. The Mur value of
// component c of a cell L on the line of a plane is
//     v = V_old(S) - k*V_old(L);  v += k*V_new(S)
// with S the cell next to L on the shift line and V_new(S) its main update. The thread of S writes it: it holds
// V_old(S) and computes V_new(S); the thread of L leaves component c alone. Every voltage has one writer, and
// the only thread that reads its old value is that writer, so no thread waits for another.
// Component c is tangential to the planes normal to the two other axes a = c+1 and b = c+2 (mod 3). An edge
// cell E on the lines of an a-plane and of a b-plane takes the value of the plane applied last, the apply hooks
// running in the extension order: the plane of the highest rank (HIP_MurFace::rank; with Mur on all faces the
// one of the lowest face: x-low wins over y-low and z-low, see Engine::SortExtensionByPriority()). Its shift
// cell is on the line of the other plane, whose Mur value replaces the main update there, while E needs that
// main update (the Mur post hooks all read before the first apply): the thread of the diagonal cell D (on the
// shift lines of both planes) writes E and both line cells next to it, and computes the main update of the
// shift cell of E again from its old value, with the statements of the main update (volt_update_at()).
// So for each component c a thread on no line of an active a- or b-plane writes its own main update, on the
// shift line of an a-plane the Mur value of the a-line cell next to it, the same for b, and on both shift lines
// the edge E. The normal component of a line cell is the main update of its own thread.
template<unsigned int c>
__device__ __forceinline__ void mur_volt_component(float* volt, const float* curr, const void* index, const float* ca, const float* cb,
                                                   unsigned int mode, const HIP_GridDim& N, const HIP_MurFace* F, unsigned int active,
                                                   unsigned int x, unsigned int y, unsigned int z, const HIP_MainCoeff& C,
                                                   size_t sn, size_t i, size_t xm, size_t ym, size_t zm)
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
		return;   // a line cell: written by the thread of its shift cell
	const bool sa0 = a0 && (p[a]==F[2*a].shift);
	const bool sa = sa0 || (a1 && (p[a]==F[2*a+1].shift));
	const bool sb0 = b0 && (p[b]==F[2*b].shift);
	const bool sb = sb0 || (b1 && (p[b]==F[2*b+1].shift));
	const float old = volt[c*sn+i];
	const float nv = volt_update_node(c, old, curr, C, sn, i, xm, ym, zm);
	if (!sa && !sb)
	{
		volt[c*sn+i] = nv;
		return;
	}
	// the a-plane: c is its component nyPP, coefficient (i,j) = (p[b], p[c]); the b-plane: c is its nyP, (i,j) = (p[c], p[a])
	const HIP_MurFace& A = F[sa0 ? 2*a : 2*a+1];
	const HIP_MurFace& B = F[sb0 ? 2*b : 2*b+1];
	const unsigned int la = A.line;
	const unsigned int lb = B.line;
	// the line cells La (a-line), Lb (b-line) and the edge E, all read before any is written
	unsigned int q[3] = {x, y, z};
	q[a] = la;
	const size_t ia = c*sn + nijk(N, 0, q[0], q[1], q[2]);
	q[b] = lb;
	const size_t ie = c*sn + nijk(N, 0, q[0], q[1], q[2]);
	q[a] = p[a];
	const size_t ib = c*sn + nijk(N, 0, q[0], q[1], q[2]);
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
			const float new_b = volt_update_at(c, old_b, curr, index, ca, cb, mode, N, q[0], q[1], q[2]);
			v_e = old_b - k * old_e;
			v_e = v_e + k * new_b;
		}
		else
		{
			// the b-plane is applied last: its shift cell of E is La
			q[a] = la;
			q[b] = p[b];
			const float k = B.c_nyP[(size_t)p[c]*n[a] + la];
			const float new_a = volt_update_at(c, old_a, curr, index, ca, cb, mode, N, q[0], q[1], q[2]);
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
__device__ __forceinline__ void mur_volt_node(float* volt, const float* curr, const void* index, const float* ca, const float* cb,
                                              unsigned int mode, const HIP_GridDim& N, const HIP_MurFace* F, unsigned int active,
                                              unsigned int x, unsigned int y, unsigned int z)
{
	const size_t sn = (size_t)N.nx*N.ny*N.nzd;
	const size_t i  = nijk(N, 0, x, y, z);
	const size_t xm = (x>0) ? (size_t)N.ny*N.nzd : 0;
	const size_t ym = (y>0) ? N.nzd : 0;
	const size_t zm = (z>0) ? 1 : 0;
	const HIP_MainCoeff C = main_coeff(index, ca, cb, mode, sn, i, 0);
	mur_volt_component<0>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z, C, sn, i, xm, ym, zm);
	mur_volt_component<1>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z, C, sn, i, xm, ym, zm);
	mur_volt_component<2>(volt, curr, index, ca, cb, mode, N, F, active, x, y, z, C, sn, i, xm, ym, zm);
}

// Mur mode 2: the Mur value of the tangential components x and y of a node L on the line of an active
// z plane, from the node S on its shift line in the same z line, held by another lane of the same warp (the lane
// test is in z, the lanes along z are contiguous; DecideMurFused() refuses the mode when a line and its shift line
// fall into different warps, MurZFaceRefusal()). Called by every thread of update_voltages_inner<true> on a node of
// the interior of the z faces and inside them, with the old values o_x, o_y and the main updates v_x, v_y of its own
// node (index xy = x*ny + y of the plane's coefficients, as the Mur extension indexes them: i along nyP = x, j along
// nyPP = y). The lanes of L and S exchange V_old and V_new through shuffles; L replaces its main update with
//     v = V_old(S) - k*V_old(L);  v = v + k*V_new(S)
// the statements of mur_pre and mur_post, which mur_apply writes. The z component of L stays its main update (the
// z planes write only x and y), and S keeps its main update, as every main node. Node sets: this kernel writes the
// nodes with x and y inside [S.L, S.H) on every z line of the main range (the z faces' lines and shift lines
// included); mur_shell the others (the x-face slabs, then the y-face slabs inside x: the 2-line bands along the four
// edges of each z face, their shift cells too), which never read a voltage this kernel writes. Interior warps
// (no lane on a z face line or shift line) skip the branch together.
__device__ __forceinline__ void mur_zface_lane(const HIP_MurZFaces& M, unsigned int z, size_t xy,
                                               float o_x, float o_y, float& v_x, float& v_y)
{
	for (unsigned int f=0; f<2; ++f)
	{
		if (!((M.active>>f) & 1u))
			continue;
		const bool on_line = (z==M.line[f]);
		if (!on_line && (z!=M.shift[f]))
			continue;
		// lanes along z: the partner's lane is this lane shifted by the z distance
		const unsigned int lane = ((threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x) % warpSize;
		const int dz = on_line ? (int)M.shift[f] - (int)M.line[f] : (int)M.line[f] - (int)M.shift[f];
		const int other = (int)lane + dz;
		const HIP_LaneMask mask = ((HIP_LaneMask)1 << lane) | ((HIP_LaneMask)1 << other);
		const float so_x = __shfl_sync(mask, o_x, other);
		const float so_y = __shfl_sync(mask, o_y, other);
		const float sn_x = __shfl_sync(mask, v_x, other);
		const float sn_y = __shfl_sync(mask, v_y, other);
		if (on_line)
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
// the Mur nodes (all the same, see mur_shell) the registers of the Mur path halve the occupancy of every block,
// and the warps on the outer z lines of every line run both paths. ZF (Mur mode 2): the box covers
// the z faces too, whose interior gets its Mur values from mur_zface_lane().
template<bool ZF>
__global__ void update_voltages_inner(float* __restrict__ volt, const float* __restrict__ curr,
                                      const void* index, const float* ca, const float* cb, unsigned int mode,
                                      HIP_GridDim N, HIP_GridDim B, HIP_GridDim E, HIP_ZSlabs Z, HIP_MurShell S,
                                      HIP_MurZFaces M)
{
	const unsigned int tz = blockIdx.x*blockDim.x + threadIdx.x;
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
	const unsigned int y = B.ny + blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = B.nx + blockIdx.z;
	if (z>=E.nz || y>=E.ny || x>=E.nx)
		return;
	if (z<Z.z_lo)
	{
		upml_fused_volt_node(volt, volt, curr, Z.volt_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		return;
	}
	if (z>=Z.z_hi)
	{
		upml_fused_volt_node(volt, volt, curr, Z.volt_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		return;
	}
	if ((x<S.L.nx) || (x>=S.H.nx) || (y<S.L.ny) || (y>=S.H.ny) || (z<S.L.nz) || (z>=S.H.nz))
		return;   // a node of mur_shell
	const size_t sn = (size_t)N.nx*N.ny*N.nzd;
	const size_t i  = nijk(N, 0, x, y, z);
	const size_t xm = (x>0) ? (size_t)N.ny*N.nzd : 0;   // shift to the previous line, none on the first
	const size_t ym = (y>0) ? N.nzd : 0;
	const size_t zm = (z>0) ? 1 : 0;
	const HIP_MainCoeff C = main_coeff(index, ca, cb, mode, sn, i, 0);
	if (!ZF)
	{
		volt[i]      = volt_update_node(0, volt[i], curr, C, sn, i, xm, ym, zm);
		volt[sn+i]   = volt_update_node(1, volt[sn+i], curr, C, sn, i, xm, ym, zm);
		volt[2*sn+i] = volt_update_node(2, volt[2*sn+i], curr, C, sn, i, xm, ym, zm);
		return;
	}
	const float o_x = volt[i];
	const float o_y = volt[sn+i];
	float v_x = volt_update_node(0, o_x, curr, C, sn, i, xm, ym, zm);
	float v_y = volt_update_node(1, o_y, curr, C, sn, i, xm, ym, zm);
	volt[2*sn+i] = volt_update_node(2, volt[2*sn+i], curr, C, sn, i, xm, ym, zm);
	mur_zface_lane(M, z, (size_t)x*N.ny + y, o_x, o_y, v_x, v_y);
	volt[i]    = v_x;
	volt[sn+i] = v_y;
}

// the j-th of the n0 + (E-H) lines of an axis outside [L, H) (n0 = L-B)
__device__ __forceinline__ unsigned int mur_shell_line(unsigned int j, unsigned int B, unsigned int L, unsigned int H)
{
	return (j<L-B) ? B+j : H+(j-(L-B));
}

// The main nodes outside the box [S.L, S.H): the lines and shift lines of the Mur planes (inactive ones too),
// one thread per node, as the slabs of the x faces (all y, z), of the y faces (x inside) and of the z faces
// (x and y inside; with Mur mode 2 S spans all z lines, so there are none: update_voltages_inner<true>
// has them).
// Launched right after update_voltages_inner, on nodes disjoint from its own.
__global__ void mur_shell(float* volt, const float* curr, const void* index, const float* ca, const float* cb, unsigned int mode,
                          HIP_GridDim N, HIP_MurShell S, const HIP_MurFace* F, unsigned int active)
{
	size_t k = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
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
	mur_volt_node(volt, curr, index, ca, cb, mode, N, F, active, x, y, z);
}

// the currents on the last mesh line of each direction are not updated
__global__ void update_currents(float* __restrict__ curr, const float* __restrict__ volt,
                                const void* index, const float* ca, const float* cb, unsigned int mode,
                                HIP_GridDim N, HIP_GridDim B, HIP_GridDim E, HIP_ZSlabs Z)
{
	// with UPML regions along z: slab lo, main nodes and slab hi in whole warps each (no divergence)
	const unsigned int tz = blockIdx.x*blockDim.x + threadIdx.x;
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
	const unsigned int y = B.ny + blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = B.nx + blockIdx.z;
	if (z>=E.nz || y>=E.ny || x>=E.nx)
		return;
	if (z<Z.z_lo)
	{
		upml_fused_curr_node(curr, curr, volt, Z.curr_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		return;
	}
	if (z>=Z.z_hi)
	{
		upml_fused_curr_node(curr, curr, volt, Z.curr_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		return;
	}
	if (z+1>=N.nz || y+1>=N.ny || x+1>=N.nx)
		return;
	const size_t sn = (size_t)N.nx*N.ny*N.nzd;
	const size_t i  = nijk(N, 0, x, y, z);
	const size_t xp = (size_t)N.ny*N.nzd;
	const size_t yp = N.nzd;
	const HIP_MainCoeff C = main_coeff(index, ca, cb, mode, sn, i, 6);
	const float* ii = C.a;
	const float* iv = C.b;
	const size_t s = C.s;
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
template<typename I>
__device__ __forceinline__ void fused_volt(const float* __restrict__ volt, const float* __restrict__ curr,
                                           const void* index, const float* va, const float* vb, unsigned int mode,
                                           const HIP_GridDim& N, unsigned int x, unsigned int y, unsigned int z, float v[3])
{
	const I sn = (I)N.nx*N.ny*N.nzd;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nzd : 0;
	const I ym = (y>0) ? N.nzd : 0;
	const I zm = (z>0) ? 1 : 0;
	const HIP_MainCoeffT<I> C = main_coeff<I>(index, va, vb, mode, sn, i, 0);
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

// The tile is as wide as a wavefront, so that a wavefront is one row of it: 32 threads on
// NVIDIA and on the RDNA GPUs of AMD, 64 on the CDNA ones. The kernel is compiled for both
// widths and the backend picks by the warp size of the device (see UpdateVoltages), so one
// binary serves either.
// OPENEMS_HIP_FUSED_TZ and _TY override the shape, to measure it on a new GPU.
#define FUSED_TZ 32   // threads along z, the last one computes the border voltages only
#define FUSED_TY 8    // threads along y, the last row computes the border voltages only
#define FUSED_XC 4    // x lines per block (default): short blocks are faster, despite the voltages of one more x line

// Fused step: the voltages and currents of the main nodes [B, E) in one pass, from the current
// fields (volt_in, curr_in) into the next buffers (volt_out, curr_out); every value is computed
// with the same operations as update_voltages/update_currents.
// One kernel for the whole timestep: the current update reads the new voltages from shared
// memory, instead of a second pass reading them back from global memory.
// A block marches along FUSED_XC x lines of a (FUSED_TZ-1)x(FUSED_TY-1) tile in (z,y). At each
// x it computes the new voltages of the tile and its border (+z, +y) into shared memory and of
// the next x line (registers), then the new currents of the tile. The border voltages of other
// tiles are computed again (they only read the current fields), those outside the main nodes
// (UPML regions) come from volt_out, where the UPML regions stored them before.
// With the UPML regions in the kernel (F.count>0, [B, E) is the whole grid), the region nodes
// are computed like the main nodes, from the current flux (the new voltage flux goes to a
// second buffer, written by the node's own block; the current flux in place).
template<typename I, unsigned int TZ, unsigned int TY>
__global__ void __launch_bounds__(TZ*TY) update_fused(
	const float* __restrict__ volt_in, const float* __restrict__ curr_in, float* volt_out, float* __restrict__ curr_out,
	const void* index, const float* va, const float* vb, const float* ia, const float* ib, unsigned int mode,
	HIP_GridDim N, HIP_GridDim B, HIP_GridDim E, HIP_FusedRegions F, unsigned int xc)
{
	__shared__ float sV[3][TY][TZ];
	const unsigned int tz = threadIdx.x, ty = threadIdx.y;
	const unsigned int z = B.nz + blockIdx.x*(TZ-1) + tz;
	const unsigned int y = B.ny + blockIdx.y*(TY-1) + ty;
	const unsigned int xs = B.nx + blockIdx.z*xc;
	const unsigned int xe = min(xs+xc, E.nx);
	const bool main_yz = (z<E.nz) && (y<E.ny);                          // a main node line
	const bool owner = main_yz && (tz<TZ-1) && (ty<TY-1);  // written by this block
	const bool in_grid = (z<N.nz) && (y<N.ny);
	const I sn = (I)N.nx*N.ny*N.nzd;

	// the new voltages at x: computed on main nodes, else from the UPML regions (or unused)
	auto volt_at = [&](unsigned int x, float v[3])
	{
		if (main_yz && (x<E.nx))
		{
			const int r = F.count ? fused_region_of(F, x, y, z) : -1;
			if (r<0)
				fused_volt<I>(volt_in, curr_in, index, va, vb, mode, N, x, y, z, v);
			else
				upml_volt_value<I>(volt_in, curr_in, F.volt[r], N, F.R[r], x, y, z, owner && (x<xe), v);
			if (owner && (x<xe))
			{
				const I i = nijk<I>(N, 0, x, y, z);
				volt_out[i] = v[0]; volt_out[sn+i] = v[1]; volt_out[2*sn+i] = v[2];
			}
		}
		else if (in_grid && (x<N.nx))
		{
			const I i = nijk<I>(N, 0, x, y, z);
			v[0] = volt_out[i]; v[1] = volt_out[sn+i]; v[2] = volt_out[2*sn+i];
		}
		else
			v[0] = v[1] = v[2] = 0;
	};

	float v[3];
	volt_at(xs, v);
	sV[0][ty][tz] = v[0]; sV[1][ty][tz] = v[1]; sV[2][ty][tz] = v[2];
	__syncthreads();
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
				curl[0] = (sV[2][ty][tz] - sV[2][ty+1][tz] - sV[1][ty][tz] + sV[1][ty][tz+1]);
				curl[1] = (sV[0][ty][tz] - sV[0][ty][tz+1] - sV[2][ty][tz] + v1[2]);
				curl[2] = (sV[1][ty][tz] - v1[1] - sV[0][ty][tz] + sV[0][ty+1][tz]);
			}
			const int r = F.count ? fused_region_of(F, x, y, z) : -1;
			if (r>=0)
				upml_curr_value<I>(curr_in, curr_out, F.curr[r], N, F.R[r], x, y, z, update, curl);
			else if (update)
			{
				const I i = nijk<I>(N, 0, x, y, z);
				const HIP_MainCoeffT<I> C = main_coeff<I>(index, ia, ib, mode, sn, i, 6);
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
		__syncthreads();
		sV[0][ty][tz] = v1[0]; sV[1][ty][tz] = v1[1]; sV[2][ty][tz] = v1[2];
		__syncthreads();
	}
}

// The arguments of update_fused, so that a launch is written once for every tile shape
struct HIP_FusedArgs
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
	HIP_GridDim N, B, E;
	HIP_FusedRegions F;
	unsigned int xc;
};

template<typename I, unsigned int TZ, unsigned int TY>
static void launch_fused(hipStream_t stream, const HIP_FusedArgs& a)
{
	const dim3 block(TZ, TY, 1);
	const dim3 grid((a.E.nz-a.B.nz+TZ-2)/(TZ-1), (a.E.ny-a.B.ny+TY-2)/(TY-1), (a.E.nx-a.B.nx+a.xc-1)/a.xc);
	update_fused<I, TZ, TY><<<grid, block, 0, stream>>>(a.volt_in, a.curr_in, a.volt_out, a.curr_out, a.index,
	                                                 a.va, a.vb, a.ia, a.ib, a.mode, a.N, a.B, a.E, a.F, a.xc);
}

template<typename I, unsigned int TZ>
static void launch_fused_ty(hipStream_t stream, const HIP_FusedArgs& a, unsigned int ty)
{
	switch (ty)
	{
		case 4:  launch_fused<I, TZ, 4>(stream, a); break;
		case 16: launch_fused<I, TZ, 16>(stream, a); break;
		default: launch_fused<I, TZ, 8>(stream, a); break;
	}
}

// the index type of the fused step, see GPU_Backend_HIP::Impl::index32
template<unsigned int TZ>
static void launch_fused_tz(hipStream_t stream, const HIP_FusedArgs& a, unsigned int ty, bool index32)
{
	if (index32)
		launch_fused_ty<unsigned int, TZ>(stream, a, ty);
	else
		launch_fused_ty<size_t, TZ>(stream, a, ty);
}

// the raw values of a dump region (see HIP_GatherRegion) for a snapshot: only these go over
// PCIe, not the whole fields; the host interpolates them
__global__ void gather_region(const float* __restrict__ f, HIP_GatherRegion R, HIP_GridDim N, float* __restrict__ out)
{
	const size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (e<R.total)
		out[e] = f[region_field_index(R, N, e)];
}

// frequency domain dumps: sums[k*total + e] += raw value(e) * w[k] for the weights of one timestep,
// a float times a complex float, then a complex sum; the host interpolates the sums
#define DFT_WEIGHTS 32   // weights per launch, passed by value in the kernel parameters
struct HIP_DFTWeights { float2 w[DFT_WEIGHTS]; };

__global__ void field_dft(const float* __restrict__ f, HIP_GatherRegion R, HIP_GridDim N,
                          HIP_DFTWeights W, unsigned int nw, float2* __restrict__ sums)
{
	const size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
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

// the currents of the given main nodes again (update_currents), after the voltage extensions
// changed voltages they read (see GPU_Backend_HIP::Impl::fixup)
__global__ void update_currents_nodes(const float* __restrict__ curr_in, float* __restrict__ curr_out, const float* __restrict__ volt,
                                      const size_t* nodes, unsigned int count,
                                      const void* index, const float* ia, const float* ib, unsigned int mode, HIP_GridDim N)
{
	const unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
	if (k>=count)
		return;
	const size_t sn = (size_t)N.nx*N.ny*N.nzd;
	const size_t i  = nodes[k];
	const size_t xp = (size_t)N.ny*N.nzd;
	const size_t yp = N.nzd;
	const HIP_MainCoeff C = main_coeff(index, ia, ib, mode, sn, i, 6);
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

// Field energy: the squared voltages and currents of the nodes below L, summed along x
// in float, one thread per (y,z) line. The host sums the lines in double.
__global__ void energy_lines(const float* __restrict__ volt, const float* __restrict__ curr, float2* partial,
                             HIP_GridDim N, HIP_GridDim L)
{
	const unsigned int z = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (z>=L.nz || y>=L.ny)
		return;
	const size_t sn = (size_t)N.nx*N.ny*N.nzd;
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
	partial[y*L.nz + z] = make_float2(E, H);
}

/***************************** helpers *****************************/

namespace
{
struct KernelTimeTable
{
	hipEvent_t start = NULL, stop = NULL;
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
		fprintf(stderr, "HIP kernel times (%.1f ms total):\n", total);
		for (const auto& t : sorted)
		{
			const size_t n = times[t.second].second;
			fprintf(stderr, "  %6.1f %%  %10.1f ms  %8zu x %8.1f us  %s\n", 100*t.first/total, t.first, n, 1000*t.first/n, t.second.c_str());
		}
	}
};
KernelTimeTable& KernelTimes() {static KernelTimeTable table; return table;}
}

bool HIP_KernelTimes::Enabled()
{
	static const bool enabled = getenv("OPENEMS_HIP_KERNEL_TIMES") && (atoi(getenv("OPENEMS_HIP_KERNEL_TIMES"))!=0);
	return enabled;
}

void HIP_KernelTimes::Begin(hipStream_t stream)
{
	KernelTimeTable& t = KernelTimes();
	if (!t.start)
	{
		(void)hipEventCreate(&t.start);
		(void)hipEventCreate(&t.stop);
	}
	(void)hipEventRecord(t.start, stream);
}

void HIP_KernelTimes::End(hipStream_t stream, const char* name)
{
	KernelTimeTable& t = KernelTimes();
	(void)hipEventRecord(t.stop, stream);
	(void)hipEventSynchronize(t.stop);
	float ms = 0;
	(void)hipEventElapsedTime(&ms, t.start, t.stop);
	std::pair<double, size_t>& entry = t.times[name];
	entry.first += ms;
	++entry.second;
}

void HIP_Check(hipError_t err, const char* what)
{
	if (err!=hipSuccess)
		throw std::runtime_error(std::string("GPU_Backend_HIP: ") + what + " failed: " + hipGetErrorString(err));
}

HIP_Context::HIP_Context()
{
	device = 0;
	stream = 0;
	copy_stream = 0;
	warp_size = 32;
}

HIP_Context::~HIP_Context()
{
	if (stream)
	{
		(void)hipStreamSynchronize(stream);
		(void)hipStreamDestroy(stream);
	}
	if (copy_stream)
	{
		(void)hipStreamSynchronize(copy_stream);
		(void)hipStreamDestroy(copy_stream);
	}
}

GPU_Backend_HIP::Impl::Impl()
{
	dim.nx = dim.ny = dim.nz = dim.nzd = 0;
	numCells = numCellsD = 0;
	shape_mode = 0;
	shape_tz = shape_ty = 0;
	shape_decided = false;
	shape_block = dim3(0, 0, 0);
	volt = curr = vv = vi = ii = iv = NULL;
	coeff_mode = 0;
	index = NULL;
	coeff = NULL;
	energy = NULL;
	energy_count = 0;
	fused_step = -1;
	fused_blockers = 0;
	fregions.count = 0;
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
	window = HIP_ParseWindowKnob(NULL, 0);
	window_used = false;
	step_base = NULL;
	window_ready = -1;
	win_T = win_W = win_B = 0;
	l2_bytes = 0;
	coeff_table_bytes = 0;
	launches = 0;
	window_graphs_on = false;
	window_logged = false;
	window_steps = window_blocks = window_chunks = window_block_launches = 0;
	tune_state = 0;
	tune_chunks = tune_rule_T = tune_rule_W = 0;
	tune_steps = 0;
	tune_ms = 0;
	tune_plane_bytes = 0;
	HIP_Check(hipEventCreateWithFlags(&snap_evaluated, hipEventDisableTiming), "hipEventCreate");
	HIP_Check(hipEventCreateWithFlags(&snap_done, hipEventDisableTiming), "hipEventCreate");
}

GPU_Backend_HIP::Impl::~Impl()
{
	if (ctx)
	{
		(void)hipStreamSynchronize(ctx->stream);
		(void)hipStreamSynchronize(ctx->copy_stream);
	}
	for (std::map<std::vector<unsigned int>, WindowGraph>::iterator it=window_graphs.begin(); it!=window_graphs.end(); ++it)
		(void)hipGraphExecDestroy(it->second.exec);
	if (window_steps && (tune_state>0) && (g_settings.GetVerboseLevel()>1))
		std::cout << "GPU_Backend_HIP: window: not tuned, the run ended during the tune (" << tune_steps << " timesteps of candidates)" << std::endl;
	if (window_steps && (g_settings.GetVerboseLevel()>1))
		std::cout << "GPU_Backend_HIP: window: " << window_steps << " timesteps in " << window_chunks << " chunks, " << window_blocks
		          << " blocks, " << window_graphs.size() << " graphs" << std::endl;
	FreeSnapshots();
	(void)hipEventDestroy(snap_evaluated);
	(void)hipEventDestroy(snap_done);
	for (size_t n=0; n<m_Allocations.size(); ++n)
		(void)hipFree(m_Allocations.at(n));
	for (std::set<void*>::iterator it=m_Pinned.begin(); it!=m_Pinned.end(); ++it)
		(void)hipHostUnregister(*it);
}

bool GPU_Backend_HIP::Impl::DecideFusedStep()
{
	if (fused_step>=0)
		return fused_step>0;
	fused_step = 0;
	const char* env = getenv("OPENEMS_HIP_FUSED_STEP");
	if ((fused_blockers>0) || (env && (atoi(env)==0)))
		return false;

	// The currents of the nodes that read a voltage changed between the half-steps are computed
	// again. With the UPML regions in the kernel, region nodes cannot (the current flux is
	// updated in place): then the region kernels update the regions.
	const bool folded = fregions.count>0;
	std::set<size_t> nodes;
	for (size_t k=0; k<volt_modified.size(); ++k)
	{
		const size_t node = volt_modified[k] % numCells;
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
				ok &= (folded || ((c[n]>=start[n]) && (c[n]<stop[n]))) && (c[n]+1<n_lines[n]);
			if (!ok)
				continue;
			if (folded && (fused_region_of(fregions, c[0], c[1], c[2])>=0))
				fregions.count = 0;   // a region node: no regions in the kernel
			nodes.insert(((size_t)c[0]*dim.ny + c[1])*dim.nz + c[2]);
		}
	}
	std::vector<size_t> list;
	for (std::set<size_t>::const_iterator it=nodes.begin(); it!=nodes.end(); ++it)
	{
		const unsigned int c[3] = {(unsigned int)(*it/((size_t)dim.ny*dim.nz)), (unsigned int)((*it/dim.nz)%dim.ny), (unsigned int)(*it%dim.nz)};
		if ((c[0]>=main_start.nx) && (c[0]<main_stop.nx) && (c[1]>=main_start.ny) && (c[1]<main_stop.ny) && (c[2]>=main_start.nz) && (c[2]<main_stop.nz))
			list.push_back(*it);   // main nodes only, the region kernels update the others after the extensions
		else if (fregions.count)
			list.push_back(*it);
	}
	// The buffers are allocated last, after the field dumps: without them the run is only slower.
	volt_next = TryAlloc<float>(3*numCellsD);
	curr_next = TryAlloc<float>(3*numCellsD);
	if ((volt_next==NULL) || (curr_next==NULL))
	{
		Free(volt_next);
		Free(curr_next);
		volt_next = curr_next = NULL;
		std::cout << "GPU_Backend_HIP: not enough device memory for the fused step, using separate updates" << std::endl;
		return false;
	}
	fixup_count = list.size();
	for (size_t k=0; k<list.size(); ++k)
		list[k] = Dev(list[k]);   // host flat indices above, see numCellsD
	fixup = Alloc<size_t>(fixup_count, list.data());

	// both buffers start with the current fields
	HIP_Check(hipMemcpyAsync(volt_next, volt, 3*numCellsD*sizeof(float), hipMemcpyDeviceToDevice, Stream()), "fused step buffers");
	HIP_Check(hipMemcpyAsync(curr_next, curr, 3*numCellsD*sizeof(float), hipMemcpyDeviceToDevice, Stream()), "fused step buffers");
	Flush();
	fused_step = 1;
	if (g_settings.GetVerboseLevel()>1)
	{
		std::cout << "GPU_Backend_HIP: fused voltage and current updates" << (index32 ? ", 32 bit indices" : "");
		if (fregions.count)
			std::cout << " (" << fregions.count << " UPML regions in the kernel)";
		std::cout << std::endl;
	}
	return true;
}

std::string GPU_Backend_HIP::Impl::MurRefusal() const
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
		const HIP_MurPlane& P = mur_planes[k];
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

// true if the threads of the z positions tz_a and tz_b (thread index along z: threadIdx.x of blocks of \a block
// threads, see MainRange()) are in the same warp of \a w threads, for every y and x thread of the block
static bool HIP_SameWarpZ(unsigned int tz_a, unsigned int tz_b, const dim3& block, unsigned int w)
{
	if (tz_a/block.x != tz_b/block.x)
		return false;   // different blocks
	const unsigned int a = tz_a%block.x, b = tz_b%block.x;
	for (unsigned int k=0; k<block.y*block.z; ++k)
		if ((k*block.x + a)/w != (k*block.x + b)/w)
			return false;
	return true;
}

std::string GPU_Backend_HIP::Impl::MurZFaceRefusal()
{
	memset(&mur_zf, 0, sizeof(mur_zf));
	if (dim.nz<6)
		return "fewer than 6 z lines";
	bool any = false;
	for (size_t k=0; k<mur_planes.size(); ++k)
	{
		const HIP_MurPlane& P = mur_planes[k];
		if (P.ny!=2)
			continue;
		const unsigned int f = (P.line>0) ? 1 : 0;
		mur_zf.c_x[f] = P.c_nyP;
		mur_zf.c_y[f] = P.c_nyPP;
		mur_zf.line[f] = P.line;
		mur_zf.shift[f] = P.shift;
		any = true;
	}
	if (!any)
		return "no Mur plane on a z face";
	// the launch of update_voltages_inner (UpdateVoltages()): the thread index along z of a main node, and the block
	HIP_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	// the block of the launch: DecideShape() ran before this decision (the fused step leaves it unset, then Block())
	const dim3 block = (shape_block.x>0) ? shape_block : Block(threads_z, E.ny-B.ny);
	auto tz = [&](unsigned int z) {return zslabs.t_int ? zslabs.t_lo + (z - zslabs.z_lo) : z - B.nz;};
	for (unsigned int f=0; f<2; ++f)
		if ((mur_zf.c_x[f]!=NULL) && !HIP_SameWarpZ(tz(mur_zf.line[f]), tz(mur_zf.shift[f]), block, ctx->warp_size))
			return std::string("the line and the shift line of the z-") + (f ? "high" : "low") + " plane are in different warps";
	return std::string();
}

void GPU_Backend_HIP::Impl::DecideMurFused()
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
			std::cout << "GPU_Backend_HIP: Mur fused: 2 unavailable: " << reason << std::endl;
			std::cout << "GPU_Backend_HIP: Mur fused: 1 unavailable: " << reason << std::endl;
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
			const HIP_MurPlane& P = mur_planes[k];
			HIP_MurFace& F = mur.f[2*P.ny + (P.line>0 ? 1 : 0)];
			F.c_nyP = P.c_nyP;
			F.c_nyPP = P.c_nyPP;
			F.line = P.line;
			F.shift = P.shift;
			F.rank = (unsigned int)P.seq;
		}
		mur_dev = Alloc<HIP_MurFace>(6, mur.f);
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
			const HIP_MurPlane& P = mur_planes[k];
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
			std::cout << "GPU_Backend_HIP: Mur fused: 2 unavailable: " << reason << std::endl;
	}
	if (verbose)
	{
		if (mur_zf_on)
			std::cout << "GPU_Backend_HIP: Mur fused: 2 (" << mur_planes.size() << " planes, z faces in the kernel)" << std::endl;
		else
			std::cout << "GPU_Backend_HIP: Mur fused: " << mur_mode << " (" << mur_planes.size() << " planes)" << std::endl;
		if (mur_mode==1)
		{
			static const char* const faces[6] = {"x-low", "x-high", "y-low", "y-high", "z-low", "z-high"};
			std::vector<std::pair<size_t, int> > order;
			for (size_t k=0; k<mur_planes.size(); ++k)
				order.push_back(std::make_pair(mur_planes[k].seq, 2*mur_planes[k].ny + (mur_planes[k].line>0 ? 1 : 0)));
			std::sort(order.begin(), order.end());
			std::cout << "GPU_Backend_HIP: Mur planes in apply order (edges take the last):";
			for (size_t k=0; k<order.size(); ++k)
				std::cout << " " << faces[order[k].second];
			std::cout << std::endl;
		}
	}
}

void GPU_Backend_HIP::Impl::ResetZSlabs()
{
	zslab_lo = zslab_hi = false;
	zslabs.z_lo = main_start.nz;
	zslabs.z_hi = main_stop.nz;
}

void GPU_Backend_HIP::Impl::MainRange(HIP_GridDim& B, HIP_GridDim& E, unsigned int& threads_z)
{
	B = main_start;
	E = main_stop;
	if (zslab_lo)
		B.nz = zslabs.R_lo.sz;
	if (zslab_hi)
		E.nz = zslabs.R_hi.sz + zslabs.R_hi.lz;
	threads_z = E.nz - B.nz;
	zslabs.t_lo = zslabs.t_int = 0;
	// Slab lo, main nodes and slab hi in whole warps each: a warp shared by UPML and main
	// nodes runs both code paths (on a RTX 5070 Ti 3 % slower for 300^3 cells)
	if (zslab_lo || zslab_hi)
	{
		const unsigned int w = ctx->warp_size;
		auto warps = [w](unsigned int n) {return (n+w-1)/w*w;};
		zslabs.t_lo = warps(zslabs.z_lo - B.nz);
		zslabs.t_int = warps(zslabs.z_hi - zslabs.z_lo);
		threads_z = zslabs.t_lo + zslabs.t_int + warps(E.nz - zslabs.z_hi);
	}
}

void GPU_Backend_HIP::Impl::DecideShape()
{
	if (shape_decided)
		return;
	shape_decided = true;
	const bool verbose = g_settings.GetVerboseLevel()>1;
	if (fused_step>0)
	{
		if (verbose)
			std::cout << "GPU_Backend_HIP: shape: unused (fused step)" << std::endl;
		return;
	}
	HIP_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	std::ostringstream reason;
	if (shape_mode==0)
		reason << "default";
	else
	{
		// auto: time the candidates on copies of the fields (the run's own fields and state stay untouched): per
		// shape, after a warm-up, rounds of update_voltages + update_currents pairs, the best round counts. The UPML
		// regions along z in the main kernels (HIP_ZSlabs) update their flux in place: the timed launches take every
		// node of the range as a main node instead (the same lines and reads, without the flux). The candidates in
		// the order of preference: a tie within 0.5 % of the fastest goes to the first listed (32x8: on a PCB of
		// 28.7 Mcells three shapes tied within 0.03 % here, and inside the windowed blocks 32x8 ran 1.8 to 2.6 %
		// faster than the other two)
		static const unsigned int candidates[][2] = {{32, 8}, {64, 4}, {32, 4}, {16, 16}, {128, 2}};
		const float tie = 1.005f;
		const unsigned int n_cand = sizeof(candidates)/sizeof(candidates[0]);
		const unsigned int pairs = 20, rounds = 2;
		const bool c = coeff_mode>0;
		const unsigned int nt = E.nz - B.nz, nj = E.ny - B.ny, nk = E.nx - B.nx;
		float* v = (nt && nj && nk) ? TryAlloc<float>(3*numCellsD) : NULL;
		float* i = v ? TryAlloc<float>(3*numCellsD) : NULL;
		hipEvent_t start = NULL, stop = NULL;
		const bool ok = (i!=NULL) && (hipEventCreate(&start)==hipSuccess) && (hipEventCreate(&stop)==hipSuccess);
		if (ok)
		{
			HIP_Check(hipMemcpyAsync(v, volt, 3*numCellsD*sizeof(float), hipMemcpyDeviceToDevice, Stream()), "shape tuning buffers");
			HIP_Check(hipMemcpyAsync(i, curr, 3*numCellsD*sizeof(float), hipMemcpyDeviceToDevice, Stream()), "shape tuning buffers");
			HIP_ZSlabs Z = zslabs;
			Z.t_lo = Z.t_int = 0;
			Z.z_lo = B.nz;
			Z.z_hi = E.nz;
			auto run = [&](unsigned int k, unsigned int n)
			{
				const dim3 block(candidates[k][0], candidates[k][1], 1);
				const dim3 grid = Grid(block, nt, nj, nk);
				for (unsigned int p=0; p<n; ++p)
				{
					update_voltages<<<grid, block, 0, Stream()>>>(v, (const float*)i, (const void*)index, (const float*)(c ? coeff : vv),
					                                               (const float*)(c ? coeff : vi), coeff_mode, dim, B, E, Z);
					update_currents<<<grid, block, 0, Stream()>>>(i, (const float*)v, (const void*)index, (const float*)(c ? coeff : ii),
					                                               (const float*)(c ? coeff : iv), coeff_mode, dim, B, E, Z);
				}
				CheckLaunch("shape tuning");
			};
			std::vector<float> best(n_cand, std::numeric_limits<float>::max());
			for (unsigned int k=0; k<n_cand; ++k)
				run(k, 2);
			for (unsigned int r=0; r<rounds; ++r)
				for (unsigned int k=0; k<n_cand; ++k)
				{
					HIP_Check(hipEventRecord(start, Stream()), "hipEventRecord");
					run(k, pairs);
					HIP_Check(hipEventRecord(stop, Stream()), "hipEventRecord");
					HIP_Check(hipEventSynchronize(stop), "hipEventSynchronize");
					float ms = 0;
					HIP_Check(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
					best[k] = std::min(best[k], 1000*ms/pairs);
				}
			unsigned int fastest = 0;
			for (unsigned int k=1; k<n_cand; ++k)
				if (best[k]<best[fastest])
					fastest = k;
			unsigned int b = 0;
			while (best[b]>tie*best[fastest])
				++b;
			shape_mode = 1;
			shape_tz = candidates[b][0];
			shape_ty = candidates[b][1];
			reason << "timed:";
			for (unsigned int k=0; k<n_cand; ++k)
				reason << " " << candidates[k][0] << "x" << candidates[k][1] << " " << std::fixed << std::setprecision(1) << best[k] << " us";
			reason << " per voltage and current update, best " << shape_tz << "x" << shape_ty;
			if (b!=fastest)
				reason << " (" << candidates[fastest][0] << "x" << candidates[fastest][1] << " within 0.5 %: the first listed wins)";
		}
		else
		{
			(void)hipGetLastError();
			shape_mode = 0;
			reason << "default, no nodes or not enough device memory to time the shapes";
		}
		if (start)
			(void)hipEventDestroy(start);
		if (stop)
			(void)hipEventDestroy(stop);
		Flush();
		Free(v);
		Free(i);
	}
	shape_block = MainBlock(threads_z, E.ny-B.ny);
	if (verbose)
		std::cout << "GPU_Backend_HIP: shape: " << shape_block.x << "x" << shape_block.y << " (" << reason.str() << ")" << std::endl;
}

void GPU_Backend_HIP::Impl::FreeSnapshots()
{
	for (size_t r=0; r<snap_lines.size(); ++r)
		Free(snap_lines[r]);
	snap_lines.clear();
	snap_regions[0].clear();
	snap_regions[1].clear();
	(void)hipFree(snap_dev);
	(void)hipHostFree(snap_host);
	snap_dev = snap_host = NULL;
	snap_nv = snap_n = 0;
	snap_taken = false;
}

void GPU_Backend_HIP::Impl::Flush()
{
	HIP_Check(hipStreamSynchronize(Stream()), "stream synchronization");
}

void GPU_Backend_HIP::Impl::CheckLaunch(const char* kernel)
{
	HIP_Check(hipGetLastError(), kernel);
}

void GPU_Backend_HIP::Impl::PinHostMemory(void* ptr, size_t bytes)
{
	if (m_Pinned.count(ptr))
		return;
	// not fatal: unpinned memory only makes the transfers slower
	if (hipHostRegister(ptr, bytes, hipHostRegisterDefault)==hipSuccess)
		m_Pinned.insert(ptr);
	else
		(void)hipGetLastError();
}

dim3 GPU_Backend_HIP::Impl::Block(size_t ni, size_t nj)
{
	if (nj<=1)
		return dim3(256, 1, 1);
	if (ni<=16)
		return dim3(16, 16, 1);
	return dim3(32, 8, 1);
}

dim3 GPU_Backend_HIP::Impl::Grid(dim3 block, size_t ni, size_t nj, size_t nk)
{
	return dim3((ni+block.x-1)/block.x, (nj+block.y-1)/block.y, nk);
}

/***************************** GPU_Backend_HIP *****************************/

//! Smallest kernel there is, to see whether this build has code for this device (see New())
__global__ void probe_device()
{
}

int GPU_Backend_HIP::DeviceCount()
{
	int count = 0;
	if (hipGetDeviceCount(&count)!=hipSuccess)
	{
		(void)hipGetLastError();
		return 0;
	}
	return count;
}

bool GPU_Backend_HIP::DeviceInfo(int n, GPU_DeviceInfo* info)
{
	hipDeviceProp_t prop;
	if (hipGetDeviceProperties(&prop, n)!=hipSuccess)
	{
		(void)hipGetLastError();
		return false;
	}
	snprintf(info->name, sizeof(info->name), "%s", prop.name);
	info->memory = prop.totalGlobalMem;
	info->integrated = prop.integrated!=0;
	return true;
}

namespace
{
GPU_Backend* NewHIPBackend(int n)
{
	return GPU_Backend_HIP::New(n);
}
}

const GPU_Plugin* GPU_Backend_HIP::Plugin()
{
	static const GPU_Plugin plugin = {OPENEMS_GPU_BUILD_ID, OPENEMS_GPU_PLUGIN_NAME, DeviceCount, DeviceInfo, NewHIPBackend};
	return &plugin;
}

#ifdef OPENEMS_HIP_ARCHITECTURES
namespace
{
//! Whether this build has code for the AMD architecture \a arch (gcnArchName, e.g. "gfx1100:sramecc-:xnack-")
bool BuiltForArch(std::string arch)
{
	arch = arch.substr(0, arch.find(':'));
	// a generic target covers a family of architectures (see the AMDGPU docs of LLVM)
	static const char* const generic[][2] = {
		{"gfx9-generic", "gfx90"}, {"gfx9-4-generic", "gfx94"}, {"gfx9-4-generic", "gfx95"},
		{"gfx10-1-generic", "gfx101"}, {"gfx10-3-generic", "gfx103"}, {"gfx11-generic", "gfx11"}, {"gfx12-generic", "gfx12"}};
	const std::string list = OPENEMS_HIP_ARCHITECTURES;
	size_t start = 0;
	while (start<list.size())
	{
		size_t end = list.find(',', start);
		if (end==std::string::npos)
			end = list.size();
		const std::string target = list.substr(start, std::min(list.find(':', start), end)-start);
		start = end+1;
		if (target==arch)
			return true;
		for (const auto& g : generic)
			if ((target==g[0]) && (arch.compare(0, strlen(g[1]), g[1])==0) && (arch!="gfx908") && (arch!="gfx90a"))
				return true;
	}
	return false;
}
}
#endif

GPU_Backend_HIP* GPU_Backend_HIP::New(int device)
{
	if ((device<0) || (device>=DeviceCount()) || (hipSetDevice(device)!=hipSuccess))
	{
		(void)hipGetLastError();
		return NULL;
	}
	Impl* impl = new Impl();
	impl->ctx = std::make_shared<HIP_Context>();
	impl->ctx->device = device;
	hipDeviceProp_t prop;
	HIP_Check(hipGetDeviceProperties(&prop, impl->ctx->device), "hipGetDeviceProperties");
	impl->ctx->name = prop.name;
	impl->ctx->warp_size = prop.warpSize > 0 ? prop.warpSize : 32;
#ifdef OPENEMS_HIP_ARCHITECTURES
	// Some AMD runtimes (ROCm 7.2 with assertions) abort in the query below on a device
	// the build has no code for, instead of returning an error: compare with the list first.
	if (!BuiltForArch(prop.gcnArchName))
	{
		std::cerr << "GPU_Backend_HIP: no kernels for " << impl->ctx->name << " (" << prop.gcnArchName
		          << ") in this build, which is for " << OPENEMS_HIP_ARCHITECTURES << std::endl;
		delete impl;
		return NULL;
	}
#endif
	HIP_Check(hipStreamCreateWithFlags(&impl->ctx->stream, hipStreamNonBlocking), "hipStreamCreate");
	HIP_Check(hipStreamCreateWithFlags(&impl->ctx->copy_stream, hipStreamNonBlocking), "hipStreamCreate");

	// A GPU whose architecture this build has no code for is found and described like any
	// other, and launching on it takes the process down with it rather than returning an
	// error. Ask about a kernel instead of running one: this reports the missing code and
	// leaves the CPU to fall back to.
	hipFuncAttributes attr;
	hipError_t err = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(probe_device));
	if (err!=hipSuccess)
	{
		// the architecture is the name to add to the build, but only the AMD platform
		// fills it in: the NVIDIA one leaves gcnArchName of the properties untouched
		std::string arch;
#ifdef __HIP_PLATFORM_AMD__
		arch = std::string(" (") + prop.gcnArchName + ")";
#endif
		std::cerr << "GPU_Backend_HIP: no kernels for " << impl->ctx->name << arch
		          << " in this build: " << hipGetErrorString(err) << std::endl;
		(void)hipGetLastError();
		delete impl;
		return NULL;
	}
	return new GPU_Backend_HIP(impl);
}

GPU_Backend* GPU_Backend_HIP::NewSubGridBackend()
{
	// no fused step with sub-grids (cylindrical multi-grid)
	d->fused_blockers++;
	Impl* impl = new Impl();
	impl->fused_blockers++;
	impl->ctx = d->ctx;
	return new GPU_Backend_HIP(impl);
}

GPU_Backend_HIP::GPU_Backend_HIP(Impl* impl)
{
	d = impl;
}

GPU_Backend_HIP::~GPU_Backend_HIP()
{
	delete d;
}

std::string GPU_Backend_HIP::GetName() const
{
	return "HIP (" + d->ctx->name + ")";
}

bool GPU_Backend_HIP::Init(const Operator* op)
{
	unsigned int numLines[3];
	for (int n=0; n<3; ++n)
		numLines[n] = op->GetNumberOfLines(n, true);
	d->dim.nx = numLines[0];
	d->dim.ny = numLines[1];
	d->dim.nz = numLines[2];
	d->dim.nzd = numLines[2];
	d->numCells = (size_t)numLines[0]*numLines[1]*numLines[2];
	// OPENEMS_HIP_ZPAD: the z lines of the device layout are stored with a stride rounded up to a multiple of 64, so that
	// every line starts on the same alignment (see nijk()). auto (default): padded if the fused step will be blocked (the
	// separate main kernels gain from it; the fused step does not), unless the padding would more than double the z
	// lines or the padded fields and full coefficient arrays (24 + 48 bytes a node, the most the main update allocates)
	// would take more than half of the free device memory; 0: off. The prediction of the fused step is the test of
	// CreateExtension() on the operator extensions (each becomes one engine extension), the sub-grids and
	// OPENEMS_HIP_FUSED_STEP=0; the other reasons are only known later (UPML that cannot fuse, an excited current on a
	// last line, memory): those grids keep the unpadded layout with the separate kernels.
	const unsigned int zpad = 64;
	const char* zpad_env = getenv("OPENEMS_HIP_ZPAD");
	const bool zpad_auto = !(zpad_env && (std::string(zpad_env)=="0"));
	if (zpad_env && zpad_auto && (std::string(zpad_env)!="auto"))
		std::cerr << "GPU_Backend_HIP: OPENEMS_HIP_ZPAD=" << zpad_env << " is not auto or 0, using auto" << std::endl;
	std::string zpad_reason;   // what blocks the fused step (auto pads), empty: nothing
	std::string zpad_skip;     // why auto does not pad although the fused step is blocked
	if (zpad_auto)
	{
		const char* fused_env = getenv("OPENEMS_HIP_FUSED_STEP");
		if (fused_env && (atoi(fused_env)==0))
			zpad_reason = "OPENEMS_HIP_FUSED_STEP=0";
		else if ((d->fused_blockers>0) || dynamic_cast<const Operator_CylinderMultiGrid*>(op))
			zpad_reason = "a sub-grid";   // a sub-grid (see NewSubGridBackend()), or the grid of a multi-grid operator, which gets one
		for (size_t n=0; (n<op->GetNumberOfExtentions()) && zpad_reason.empty(); ++n)
		{
			Operator_Extension* op_ext = op->GetExtension(n);
			if (!dynamic_cast<Operator_Ext_UPML*>(op_ext) && !dynamic_cast<Operator_Ext_Excitation*>(op_ext)
			    && !dynamic_cast<Operator_Ext_LumpedRLC*>(op_ext) && !dynamic_cast<Operator_Ext_SteadyState*>(op_ext))
				zpad_reason = op_ext->GetExtensionName();
		}
		if (!zpad_reason.empty())
		{
			const unsigned int nzd = (numLines[2] + zpad - 1)/zpad*zpad;
			const size_t padded_bytes = (size_t)numLines[0]*numLines[1]*nzd*(24 + 48);
			size_t free_bytes = 0, total_bytes = 0;
			std::ostringstream skip;
			if (nzd - numLines[2] > numLines[2])
				skip << "the padding would more than double the " << numLines[2] << " z lines";
			else if (hipMemGetInfo(&free_bytes, &total_bytes)!=hipSuccess)
			{
				(void)hipGetLastError();
				skip << "the free device memory is unknown";
			}
			else if (padded_bytes > free_bytes/2)
				skip << "the padded fields and coefficients (" << (padded_bytes>>20) << " MiB) would take more than half of the free device memory ("
				     << (free_bytes>>20) << " MiB)";
			zpad_skip = skip.str();
			if (zpad_skip.empty())
				d->dim.nzd = nzd;
		}
	}
	d->numCellsD = (size_t)numLines[0]*numLines[1]*d->dim.nzd;
	if (g_settings.GetVerboseLevel()>1)
	{
		if (d->dim.nzd!=d->dim.nz)
		{
			std::ostringstream pct;
			pct << std::fixed << std::setprecision(1) << 100.0*(d->dim.nzd - d->dim.nz)/d->dim.nz;
			std::cout << "GPU_Backend_HIP: z padding: " << d->dim.nz << " lines stored as " << d->dim.nzd << " (auto: the fused step is blocked by "
			          << zpad_reason << ", " << pct.str() << " % more field memory)" << std::endl;
		}
		else if (!zpad_auto)
			std::cout << "GPU_Backend_HIP: z padding: off" << std::endl;
		else if (zpad_reason.empty())
			std::cout << "GPU_Backend_HIP: z padding: off (auto: fused step)" << std::endl;
		else
			std::cout << "GPU_Backend_HIP: z padding: off (auto: the fused step is blocked by " << zpad_reason << ", but " << zpad_skip << ")" << std::endl;
	}
	// OPENEMS_HIP_SHAPE: auto (default: timed, see DecideShape()) or 0 Block()
	d->shape_mode = 2;
	if (const char* env = getenv("OPENEMS_HIP_SHAPE"))
	{
		const std::string shape = env;
		if (shape=="0")
			d->shape_mode = 0;
		else if (shape!="auto")
			std::cerr << "GPU_Backend_HIP: OPENEMS_HIP_SHAPE=" << shape << " is not 0 or auto, using auto" << std::endl;
	}
	d->main_start.nx = d->main_start.ny = d->main_start.nz = 0;
	d->main_stop = d->dim;
	d->upml_fused = -1;
	d->ResetZSlabs();
	d->window = HIP_ParseWindowKnob(getenv("OPENEMS_HIP_WINDOW"), d->dim.nx);

	// the main updates launch one block per x line along z, which the device limits (64k on AMD and NVIDIA);
	// from the properties: hipDeviceGetAttribute calls the CUDA driver API on NVIDIA, which the
	// plugin does not link (MSVC keeps that call, GCC folds it away)
	hipDeviceProp_t prop;
	HIP_Check(hipGetDeviceProperties(&prop, d->ctx->device), "hipGetDeviceProperties");
	if (g_settings.GetVerboseLevel()>1)
		std::cout << "GPU_Backend_HIP: " << prop.name << ": " << prop.multiProcessorCount << " multiprocessors, L2 cache "
		          << prop.l2CacheSize/(1024*1024) << " MiB, warp size " << d->ctx->warp_size << std::endl;
	d->l2_bytes = prop.l2CacheSize>0 ? (size_t)prop.l2CacheSize : 0;
	int max_grid_z = prop.maxGridSize[2];
	if ((max_grid_z>0) && (d->dim.nx>(unsigned int)max_grid_z))
	{
		std::cerr << "GPU_Backend_HIP::Init: Error: " << d->dim.nx << " mesh lines in x, the device allows at most "
		          << max_grid_z << "; swap the axes, or use a CPU engine" << std::endl;
		return false;
	}

	// The fused step indexes with 32 bit on NVIDIA while the fields fit (faster there), with
	// 64 bit on AMD; OPENEMS_HIP_INDEX=32 or 64 overrides.
	const bool fits32 = 3*d->numCellsD <= UINT_MAX;
#ifdef __HIP_PLATFORM_AMD__
	d->index32 = false;
#else
	d->index32 = fits32;
#endif
	if (const char* env = getenv("OPENEMS_HIP_INDEX"))
		d->index32 = (atoi(env)==32);
	if (d->index32 && !fits32)
	{
		std::cerr << "GPU_Backend_HIP: the mesh is too large for 32 bit indices, the fused step uses 64 bit" << std::endl;
		d->index32 = false;
	}

	const size_t count = 3*d->numCellsD;
	d->volt = d->Alloc<float>(count);
	d->curr = d->Alloc<float>(count);
	if (d->window.on)
		d->step_base = d->Alloc<int>(1);

	// upload the final operator coefficients, including all changes by operator extensions.
	// Most nodes share one of a few coefficient sets (same material and mesh spacing): store every distinct set of
	// the 12 coefficients of a node once and a 16 or 32 bit set index per node. Where the sets do not pay
	// (GPU_FindSets() refuses: too many distinct sets, e.g. a graded mesh with many spacings), the four full
	// arrays below (48 bytes per node); the kernels read either form (coeff_mode, logged at verbose level 2)
	GPU_CoeffSets sets;
	if (GPU_FindSets(d->numCells, 12, [&](size_t i, float* values) {GPU_NodeCoefficients(op, numLines, i, values);}, sets))
	{
		d->coeff_mode = sets.mode;
		d->index = d->AllocIndex(sets, d->dim.nzd!=d->dim.nz);   // per node of the device layout
		d->coeff = d->Alloc<float>(sets.table.size(), sets.table.data());
		d->coeff_table_bytes = sets.table.size()*sizeof(float);
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_HIP: " << sets.count << " distinct coefficient sets, compressed update coefficients ("
			          << (sets.mode==1 ? 16 : 32) << " bit index)" << std::endl;
		return true;
	}

	// the full arrays, one host copy at a time
	d->coeff_mode = 0;
	std::vector<float> values;
	auto upload = [&](int c)
	{
		GPU_CoefficientArray(op, numLines, c, values);
		if (d->dim.nzd!=d->dim.nz)
			values = d->Padded(values.data(), values.size());   // per node of the device layout
		return d->Alloc<float>(count, values.data());
	};
	d->vv = upload(0);
	d->vi = upload(1);
	d->ii = upload(2);
	d->iv = upload(3);
	return true;
}

void GPU_Backend_HIP::UpdateVoltages()
{
	const bool c = d->coeff_mode>0;
	const bool fused = d->DecideFusedStep();
	d->DecideShape();       // once, after the fused step and before DecideMurFused() (which may refuse a shape)
	d->DecideMurFused();   // once, after the pre hooks of the first timestep (the UPML decides its regions there)
	if (fused)
	{
		// the main nodes, or the whole grid with the UPML regions in the kernel (see update_fused)
		HIP_GridDim B = d->main_start;
		HIP_GridDim E = d->main_stop;
		if (d->fregions.count)
		{
			B.nx = B.ny = B.nz = 0;
			E = d->dim;
		}
		static const unsigned int xc = getenv("OPENEMS_HIP_FUSED_XC") ? std::max(1, atoi(getenv("OPENEMS_HIP_FUSED_XC"))) : FUSED_XC;
		const bool timed = HIP_KernelTimes::Enabled();
		if (timed)
			HIP_KernelTimes::Begin(d->Stream());
		// the tile is a wavefront wide (see FUSED_TZ); OPENEMS_HIP_FUSED_TY sets the other side
		static const unsigned int ty = getenv("OPENEMS_HIP_FUSED_TY") ? std::max(2, atoi(getenv("OPENEMS_HIP_FUSED_TY"))) : FUSED_TY;
		const HIP_FusedArgs a = { (const float*)d->volt, (const float*)d->curr, d->volt_next, d->curr_next, (const void*)d->index,
		                          (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi),
		                          (const float*)(c ? d->coeff : d->ii), (const float*)(c ? d->coeff : d->iv),
		                          d->coeff_mode, d->dim, B, E, d->fregions, xc };
#ifdef __HIP_PLATFORM_AMD__
		// only AMD has wavefronts of 64; compiling them for NVIDIA would be dead code
		static const unsigned int tz = getenv("OPENEMS_HIP_FUSED_TZ") ? atoi(getenv("OPENEMS_HIP_FUSED_TZ")) : 0;
		if (tz ? (tz>=64) : (d->ctx->warp_size>=64))
			launch_fused_tz<64>(d->Stream(), a, ty, d->index32);
		else
#endif
			launch_fused_tz<32>(d->Stream(), a, ty, d->index32);
		d->CheckLaunch("update_fused");
		if (timed)
			HIP_KernelTimes::End(d->Stream(), "update_fused");
		// only the voltages: UpdateCurrents swaps the currents, once the fix-up has read the old ones
		std::swap(d->volt, d->volt_next);
		return;
	}
	HIP_GridDim B, E;
	unsigned int threads_z;
	d->MainRange(B, E, threads_z);
	if ((d->mur_mode==1) && d->mur.active)
	{
		// the nodes off the Mur planes, then those on them (disjoint, see mur_volt_component); mode 2: the interior
		// of the z faces in the first kernel (see mur_zface_lane), the box spans all z lines, so mur_shell has no
		// z-face slabs (its threads past the count would otherwise run into them)
		HIP_MurShell S = d->mur_shell;
		if (d->mur_zf_on)
		{
			S.L.nz = d->main_start.nz;
			S.H.nz = d->main_stop.nz;
			HIP_MurZFaces M = d->mur_zf;
			M.active = (d->mur.active>>4) & 3u;
			HIP_LaunchMain(d, "update_voltages_inner", update_voltages_inner<true>, threads_z, E.ny-B.ny, E.nx-B.nx,
			            d->volt, (const float*)d->curr, (const void*)d->index, (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi), d->coeff_mode, d->dim, B, E, d->zslabs, S, M);
		}
		else
			HIP_LaunchMain(d, "update_voltages_inner", update_voltages_inner<false>, threads_z, E.ny-B.ny, E.nx-B.nx,
			            d->volt, (const float*)d->curr, (const void*)d->index, (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi), d->coeff_mode, d->dim, B, E, d->zslabs, S, d->mur_zf);
		HIP_Launch(d, "mur_shell", mur_shell, d->mur_shell_count, 1, 1,
		            d->volt, (const float*)d->curr, (const void*)d->index, (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi), d->coeff_mode, d->dim,
		            S, (const HIP_MurFace*)d->mur_dev, d->mur.active);
		return;
	}
	HIP_LaunchMain(d, "update_voltages", update_voltages, threads_z, E.ny-B.ny, E.nx-B.nx,
	            d->volt, (const float*)d->curr, (const void*)d->index, (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi), d->coeff_mode, d->dim, B, E, d->zslabs);
}

void GPU_Backend_HIP::UpdateCurrents()
{
	const bool c = d->coeff_mode>0;
	if (d->fused_step>0)
	{
		// done by update_fused, but for the currents next to voltages changed by extensions since
		if (d->fixup_count)
			HIP_Launch(d, "update_currents_nodes", update_currents_nodes, d->fixup_count, 1, 1, (const float*)d->curr, d->curr_next, (const float*)d->volt,
			            (const size_t*)d->fixup, d->fixup_count, (const void*)d->index,
			            (const float*)(c ? d->coeff : d->ii), (const float*)(c ? d->coeff : d->iv), d->coeff_mode, d->dim);
		std::swap(d->curr, d->curr_next);
		return;
	}
	HIP_GridDim B, E;
	unsigned int threads_z;
	d->MainRange(B, E, threads_z);
	HIP_LaunchMain(d, "update_currents", update_currents, threads_z, E.ny-B.ny, E.nx-B.nx,
	            d->curr, (const float*)d->volt, (const void*)d->index, (const float*)(c ? d->coeff : d->ii), (const float*)(c ? d->coeff : d->iv), d->coeff_mode, d->dim, B, E, d->zslabs);
}

/************************** windowed blocks (OPENEMS_HIP_WINDOW) **************************/

// The timestep of the block about to run, for the window-capable extensions (see GPU_Extension::LaunchInWindow()):
// written on the work stream before the block (outside a captured graph, whose arguments are frozen), so the
// launches of the block read the value of its launch
__global__ void set_step(int* step_base, int step)
{
	if (blockIdx.x*blockDim.x + threadIdx.x == 0)
		*step_base = step;
}

// OPENEMS_HIP_WINDOW when it is not set: the windowed blocks with the tuner; "0" keeps the single step
static const char* const HIP_WINDOW_DEFAULT = "auto";

GPU_Backend_HIP::Impl::WindowKnob HIP_ParseWindowKnob(const char* text, unsigned int nx)
{
	GPU_Backend_HIP::Impl::WindowKnob k;
	k.on = k.valid = k.automatic = k.tune = false;
	k.T = k.W = k.B = 0;
	k.text = text ? text : HIP_WINDOW_DEFAULT;
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
	// T,W[,B]
	std::vector<std::string> f;
	std::string::size_type a = 0;
	while (true)
	{
		const std::string::size_type b = k.text.find(',', a);
		f.push_back(k.text.substr(a, b==std::string::npos ? std::string::npos : b-a));
		if (b==std::string::npos)
			break;
		a = b+1;
	}
	auto number = [](const std::string& v, unsigned int& out)
	{
		if (v.empty() || (v.size()>9) || (v.find_first_not_of("0123456789")!=std::string::npos))
			return false;
		out = (unsigned int)atoi(v.c_str());
		return true;
	};
	if ((f.size()<2) || (f.size()>3) || !number(f[0], k.T))
		return k;
	if (f[1]=="NX")
		k.W = nx;
	else if (!number(f[1], k.W))
		return k;
	if ((f.size()==3) && !number(f[2], k.B))
		return k;
	k.valid = (k.T>0) && (k.T<=256) && (k.W>0);
	return k;
}

unsigned int HIP_WindowRule(size_t l2_bytes, size_t plane_bytes, unsigned int T, unsigned int nx, size_t table_bytes,
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

unsigned int HIP_WindowRuleTW(size_t l2_bytes, size_t plane_bytes, unsigned int& T, unsigned int nx, size_t table_bytes)
{
	// fewer than T planes a window: the next smaller T has a shorter halo and leaves more of the budget to the window
	while (true)
	{
		const unsigned int W = HIP_WindowRule(l2_bytes, plane_bytes, T, nx, table_bytes);
		if ((W>=T) || (W>=nx) || (T<=2))
			return W;
		T = (T>8) ? 8 : (T>4) ? 4 : 2;
	}
}

unsigned int GPU_Backend_HIP::Impl::MurActive(unsigned int ts) const
{
	unsigned int active = 0;
	for (size_t k=0; k<mur_planes.size(); ++k)
		if (ts>=mur_planes[k].start_ts)
			active |= 1u<<(2*mur_planes[k].ny + (mur_planes[k].line>0 ? 1 : 0));
	return active;
}

unsigned int GPU_Backend_HIP::Impl::MurNextChange(unsigned int ts) const
{
	unsigned int next = UINT_MAX;
	for (size_t k=0; k<mur_planes.size(); ++k)
		if (mur_planes[k].start_ts>ts)
			next = std::min(next, mur_planes[k].start_ts);
	return next;
}

void GPU_Backend_HIP::Impl::WindowVoltages(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1, unsigned int mur_active)
{
	const bool c = coeff_mode>0;
	HIP_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	B.nx = std::max(B.nx, x0); E.nx = std::min(E.nx, x1);
	B.ny = std::max(B.ny, y0); E.ny = std::min(E.ny, y1);
	if ((E.nx<=B.nx) || (E.ny<=B.ny))
		return;
	if ((mur_mode==1) && mur_active)
	{
		// the launches of UpdateVoltages() on the window: update_voltages_inner on the main range cut to the window,
		// which skips the nodes outside the box [S.L, S.H) as before, then mur_shell on the shell nodes inside the
		// window: the box [S.B, S.E) and the inner box cut to the window (the same shell, intersected)
		mur.active = mur_active;
		HIP_MurShell S = mur_shell;
		HIP_MurZFaces M = mur_zf;
		if (mur_zf_on)
		{
			S.L.nz = main_start.nz;
			S.H.nz = main_stop.nz;
			M.active = (mur_active>>4) & 3u;
			HIP_LaunchMain(this, "update_voltages_inner", update_voltages_inner<true>, threads_z, E.ny-B.ny, E.nx-B.nx,
			            volt, (const float*)curr, (const void*)index, (const float*)(c ? coeff : vv), (const float*)(c ? coeff : vi), coeff_mode, dim, B, E, zslabs, S, M);
		}
		else
			HIP_LaunchMain(this, "update_voltages_inner", update_voltages_inner<false>, threads_z, E.ny-B.ny, E.nx-B.nx,
			            volt, (const float*)curr, (const void*)index, (const float*)(c ? coeff : vv), (const float*)(c ? coeff : vi), coeff_mode, dim, B, E, zslabs, S, M);
		HIP_MurShell W = S;
		W.B.nx = std::max(S.B.nx, x0); W.E.nx = std::min(S.E.nx, x1);
		W.B.ny = std::max(S.B.ny, y0); W.E.ny = std::min(S.E.ny, y1);
		if ((W.E.nx<=W.B.nx) || (W.E.ny<=W.B.ny))
			return;
		W.L.nx = std::min(std::max(S.L.nx, W.B.nx), W.E.nx); W.H.nx = std::min(std::max(S.H.nx, W.L.nx), W.E.nx);
		W.L.ny = std::min(std::max(S.L.ny, W.B.ny), W.E.ny); W.H.ny = std::min(std::max(S.H.ny, W.L.ny), W.E.ny);
		const size_t count = (size_t)(W.E.nx-W.B.nx)*(W.E.ny-W.B.ny)*(W.E.nz-W.B.nz) - (size_t)(W.H.nx-W.L.nx)*(W.H.ny-W.L.ny)*(W.H.nz-W.L.nz);
		HIP_Launch(this, "mur_shell", ::mur_shell, count, 1, 1,
		            volt, (const float*)curr, (const void*)index, (const float*)(c ? coeff : vv), (const float*)(c ? coeff : vi), coeff_mode, dim,
		            W, (const HIP_MurFace*)mur_dev, mur_active);
		return;
	}
	HIP_LaunchMain(this, "update_voltages", update_voltages, threads_z, E.ny-B.ny, E.nx-B.nx,
	            volt, (const float*)curr, (const void*)index, (const float*)(c ? coeff : vv), (const float*)(c ? coeff : vi), coeff_mode, dim, B, E, zslabs);
}

void GPU_Backend_HIP::Impl::WindowCurrents(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
{
	const bool c = coeff_mode>0;
	HIP_GridDim B, E;
	unsigned int threads_z;
	MainRange(B, E, threads_z);
	B.nx = std::max(B.nx, x0); E.nx = std::min(E.nx, x1);
	B.ny = std::max(B.ny, y0); E.ny = std::min(E.ny, y1);
	if ((E.nx<=B.nx) || (E.ny<=B.ny))
		return;
	HIP_LaunchMain(this, "update_currents", update_currents, threads_z, E.ny-B.ny, E.nx-B.nx,
	            curr, (const float*)volt, (const void*)index, (const float*)(c ? coeff : ii), (const float*)(c ? coeff : iv), coeff_mode, dim, B, E, zslabs);
}

bool GPU_Backend_HIP::Impl::WindowPrepare()
{
	if (window_ready>=0)
		return window_ready>0;
	window_ready = 0;
	const bool verbose = g_settings.GetVerboseLevel()>1;
	// the decisions the single step makes in the hooks and the first voltage update, before anything is captured
	for (size_t k=0; k<window_exts.size(); ++k)
		if (HIP_WindowHooks* h = dynamic_cast<HIP_WindowHooks*>(window_exts[k]))
			h->WindowPrepare();
	DecideShape();
	DecideMurFused();
	if (!mur_planes.empty() && (mur_mode!=1))
	{
		// the three launches per plane of mode 0 work on whole planes: not in the windows
		if (verbose)
			std::cout << "GPU_Backend_HIP: window: refused: the Mur planes are not in the voltage update (Mur fused: 0)" << std::endl;
		window_used = false;
		return false;
	}
	// plane bytes: the fields and the coefficient index (or the full coefficient arrays) of one x plane
	const size_t plane_bytes = (size_t)dim.ny*dim.nzd*(24 + CoeffBytesPerNode());
	tune_plane_bytes = plane_bytes;
	if (window.automatic)
	{
		win_T = window.T;
		win_W = HIP_WindowRuleTW(l2_bytes, plane_bytes, win_T, dim.nx, coeff_table_bytes);
		tune_rule_T = win_T;
		tune_rule_W = win_W;
		win_source = "rule";
		if (window.tune)
			WindowTunePrepare();
	}
	else
	{
		win_T = window.T;
		win_W = std::min(window.W, dim.nx);
		win_source = "forced";
	}
	win_B = window.B;
	// a V frontier must not fall between the line and the shift line of a Mur plane on an x face (the thread of the
	// shift line writes the line, mur_volt_component): snapped down below both (see WindowFrontiers())
	win_forbidden.clear();
	for (size_t k=0; k<mur_planes.size(); ++k)
		if (mur_planes[k].ny==0)
			for (unsigned int e=std::min(mur_planes[k].line, mur_planes[k].shift)+1; e<=std::max(mur_planes[k].line, mur_planes[k].shift); ++e)
				win_forbidden.push_back(e);
	window_graphs_on = !HIP_KernelTimes::Enabled();
	if (verbose && (win_B>0))
	{
		std::ostringstream line;
		line << "GPU_Backend_HIP: window: B=" << win_B << " unused: the windows span all y lines (a slab of "
		     << (4*win_T+1) << " planes is " << std::fixed << std::setprecision(1) << (4.0*win_T+1)*plane_bytes/1e6
		     << " MB, L2 " << l2_bytes/(1024*1024) << " MiB)";
		std::cout << line.str() << std::endl;
	}
	window_ready = 1;
	return true;
}

void GPU_Backend_HIP::Impl::WindowFrontiers(unsigned int F, unsigned int L, std::vector<unsigned int>& eV, std::vector<unsigned int>& eI) const
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

void GPU_Backend_HIP::Impl::WindowBlock(unsigned int ts, unsigned int L, unsigned int W, GPU_Extension* host)
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
	HIP_WindowHooks* host_hooks = host ? dynamic_cast<HIP_WindowHooks*>(host) : NULL;
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

void GPU_Backend_HIP::Impl::WindowRun(unsigned int start_ts, unsigned int n)
{
	if (n==0)
		return;
	const std::vector<GPU_Extension*>& X = window_exts;
	std::vector<HIP_WindowHooks*> hooks(X.size(), NULL);
	for (size_t k=0; k<X.size(); ++k)
		hooks[k] = dynamic_cast<HIP_WindowHooks*>(X[k]);
	const bool verbose = g_settings.GetVerboseLevel()>1;
	// capture a block of L timesteps from ts into a graph (its launches are recorded, not run); false if that fails
	auto capture = [&](unsigned int ts, unsigned int L, WindowGraph& g)
	{
		const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		const unsigned long long l0 = launches;
		std::string what;
		hipError_t err = hipStreamBeginCapture(Stream(), hipStreamCaptureModeThreadLocal);
		if (err==hipSuccess)
		{
			try
			{
				WindowBlock(ts, L, win_W, NULL);
			}
			catch (const std::exception& e)
			{
				what = e.what();
			}
			hipGraph_t graph = NULL;
			err = hipStreamEndCapture(Stream(), &graph);
			if (what.empty() && (err!=hipSuccess))
				what = std::string("hipStreamEndCapture: ") + hipGetErrorString(err);
			if (what.empty())
			{
				err = hipGraphInstantiateWithFlags(&g.exec, graph, 0);
				if (err!=hipSuccess)
					what = std::string("hipGraphInstantiate: ") + hipGetErrorString(err);
			}
			if (graph)
				(void)hipGraphDestroy(graph);
		}
		else
			what = std::string("hipStreamBeginCapture: ") + hipGetErrorString(err);
		(void)hipGetLastError();
		if (!what.empty())
		{
			std::cout << "GPU_Backend_HIP: window: graph capture failed (" << what << "), the same launches without graphs" << std::endl;
			window_graphs_on = false;
			return false;
		}
		g.launches = launches - l0;
		g.capture_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		if (verbose && window_logged)
		{
			std::ostringstream line;
			line << "GPU_Backend_HIP: window: graph " << window_graphs.size()+1 << " for blocks of " << L << " steps (" << g.launches
			     << " launches, captured in " << std::fixed << std::setprecision(1) << g.capture_ms << " ms)";
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
		HIP_Launch(this, "set_step", set_step, 1, 1, 1, step_base, (int)ts);
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
					if (capture(ts, L, g))
						it = window_graphs.insert(std::make_pair(key, g)).first;
				}
				if (it!=window_graphs.end())
				{
					HIP_Check(hipGraphLaunch(it->second.exec, Stream()), "hipGraphLaunch");
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

float GPU_Backend_HIP::Impl::WindowTimedRun(unsigned int start_ts, unsigned int n)
{
	hipEvent_t start = NULL, stop = NULL;
	HIP_Check(hipEventCreate(&start), "hipEventCreate");
	HIP_Check(hipEventCreate(&stop), "hipEventCreate");
	HIP_Check(hipEventRecord(start, Stream()), "hipEventRecord");
	WindowRun(start_ts, n);
	HIP_Check(hipEventRecord(stop, Stream()), "hipEventRecord");
	HIP_Check(hipEventSynchronize(stop), "hipEventSynchronize");
	float ms = 0;
	HIP_Check(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
	(void)hipEventDestroy(start);
	(void)hipEventDestroy(stop);
	return ms;
}

// The tuner of OPENEMS_HIP_WINDOW=auto: a short list of (T, W) candidates, each timed on the live grid with one warm-up
// block (which captures its graphs) and two timed blocks of its T (WindowTimedRun()), the three best refined once with
// W -25 % and +25 %, the best measured one runs from then on. The candidates run real timesteps, in the chunks the
// engine hands over: every schedule is bit-identical to the single step (the frontiers of WindowFrontiers()), so the
// tune does not change the result and costs only the slower candidates' time; nothing is copied or restored.
static const unsigned int HIP_TUNE_MAX_CANDIDATES = 13;   // the first list (the refinement adds up to 6)
static const unsigned int HIP_TUNE_MAX_CHUNKS = 16;       // a tune not done after these chunks keeps its best so far

void GPU_Backend_HIP::Impl::WindowTunePrepare()
{
	const bool verbose = g_settings.GetVerboseLevel()>1;
	tune_cands.clear();
	tune_state = 0;
	// the candidates: the rule (its T and W), then T in {8, 16, 4, 2} with the halo of T+1 planes at 0.65, 0.45 and
	// 0.85 of L2; equal pairs once, at most HIP_TUNE_MAX_CANDIDATES
	auto add = [&](unsigned int T, unsigned int W)
	{
		for (size_t k=0; k<tune_cands.size(); ++k)
			if ((tune_cands[k].T==T) && (tune_cands[k].W==W))
				return;
		if (tune_cands.size()<HIP_TUNE_MAX_CANDIDATES)
		{
			WindowCandidate c = {T, W, -1};
			tune_cands.push_back(c);
		}
	};
	add(tune_rule_T, tune_rule_W);
	// one window over all of x, the single step's work in graphs: the fastest where the L2 gains less than the small
	// extension kernels of many windows cost (H100, A100: each window runs them again, a few us each)
	add(16, dim.nx);
	static const unsigned int Ts[] = {8, 16, 4, 2};
	static const double margins[] = {0.65, 0.45, 0.85};
	for (unsigned int t=0; t<4; ++t)
		for (unsigned int m=0; m<3; ++m)
			add(Ts[t], HIP_WindowRule(l2_bytes, tune_plane_bytes, Ts[t], dim.nx, coeff_table_bytes, margins[m], Ts[t] + 1));
	unsigned int steps = 0;
	for (size_t k=0; k<tune_cands.size(); ++k)
		steps += 3*tune_cands[k].T;
	tune_state = 1;
	win_source = "untuned";
	if (verbose)
		std::cout << "GPU_Backend_HIP: window: tuning on the first " << steps << " timesteps (" << tune_cands.size()
		          << " candidates of a warm-up and two timed blocks, then the three best with W -/+25 %; real timesteps, each"
		          << " candidate bit-identical)" << std::endl;
}

void GPU_Backend_HIP::Impl::WindowDropGraphs(unsigned int keep_W)
{
	for (std::map<std::vector<unsigned int>, WindowGraph>::iterator it=window_graphs.begin(); it!=window_graphs.end(); )
		if (it->first[1]!=keep_W)
		{
			(void)hipGraphExecDestroy(it->second.exec);
			window_graphs.erase(it++);
		}
		else
			++it;
}

unsigned int GPU_Backend_HIP::Impl::WindowTune(unsigned int start_ts, unsigned int n)
{
	unsigned int used = 0;
	++tune_chunks;
	while (tune_state>0)
	{
		// the first candidate not measured yet that fits into the rest of the chunk
		int next = -1;
		bool pending = false;
		for (size_t k=0; k<tune_cands.size(); ++k)
			if (tune_cands[k].mcs<0)
			{
				pending = true;
				if (3*tune_cands[k].T<=n-used)
				{
					next = (int)k;
					break;
				}
			}
		if (!pending)
		{
			if (tune_state==2)
			{
				WindowTuneFinish(true);
				break;
			}
			// the refinement: the three best measured, each with W -25 % and +25 % (at least one plane apart)
			std::vector<WindowCandidate> best = tune_cands;
			std::stable_sort(best.begin(), best.end(), [](const WindowCandidate& a, const WindowCandidate& b) {return a.mcs>b.mcs;});
			tune_state = 2;
			for (size_t k=0; (k<3) && (k<best.size()); ++k)
			{
				const unsigned int W = best[k].W;
				const unsigned int lo = std::max(1u, std::min(W - 1, (unsigned int)std::lround(0.75*W)));
				const unsigned int hi = std::min(dim.nx, std::max(W + 1, (unsigned int)std::lround(1.25*W)));
				const unsigned int ws[2] = {lo, hi};
				for (unsigned int j=0; j<2; ++j)
				{
					if ((ws[j]<1) || (ws[j]>dim.nx) || (ws[j]==W))
						continue;
					bool known = false;
					for (size_t m=0; m<tune_cands.size(); ++m)
						known = known || ((tune_cands[m].T==best[k].T) && (tune_cands[m].W==ws[j]));
					if (!known)
					{
						WindowCandidate c = {best[k].T, ws[j], -1};
						tune_cands.push_back(c);
					}
				}
			}
			continue;
		}
		if (next<0)
			break;   // the rest of the chunk is too short for the next candidate
		WindowCandidate& c = tune_cands[next];
		win_T = c.T;
		win_W = c.W;
		WindowRun(start_ts + used, c.T);   // warm-up: captures the graphs of this W
		const float ms = WindowTimedRun(start_ts + used + c.T, 2*c.T);
		c.mcs = (ms>0) ? (double)numCells*2.0*c.T/((double)ms*1e3) : 0;
		tune_ms += ms;
		tune_steps += 3*c.T;
		used += 3*c.T;
		WindowDropGraphs(UINT_MAX);   // at most 32 graphs: the next candidate starts empty
	}
	if ((tune_state>0) && (tune_chunks>=HIP_TUNE_MAX_CHUNKS))
		WindowTuneFinish(false);
	if (tune_state>0)
	{
		// the rest of this chunk: the best so far once the rule (the first candidate) is measured, the rule until then
		win_T = tune_rule_T;
		win_W = tune_rule_W;
		double best = -1;
		for (size_t k=0; (k<tune_cands.size()) && (tune_cands[0].mcs>=0); ++k)
			if (tune_cands[k].mcs>best)
			{
				best = tune_cands[k].mcs;
				win_T = tune_cands[k].T;
				win_W = tune_cands[k].W;
			}
		WindowDropGraphs(win_W);
	}
	return used;
}

void GPU_Backend_HIP::Impl::WindowTuneFinish(bool complete)
{
	const bool verbose = g_settings.GetVerboseLevel()>1;
	tune_state = 0;
	int best = -1;
	unsigned int measured = 0;
	for (size_t k=0; k<tune_cands.size(); ++k)
		if (tune_cands[k].mcs>=0)
		{
			++measured;
			if ((best<0) || (tune_cands[k].mcs>tune_cands[best].mcs))
				best = (int)k;
		}
	// a tune the chunks cut short moves away from the rule only when the rule (the first candidate) was timed too
	const bool rule_timed = !tune_cands.empty() && (tune_cands[0].mcs>=0);
	if ((best<0) || (!complete && !rule_timed))
	{
		win_T = tune_rule_T;
		win_W = tune_rule_W;
		win_source = "rule";
		if (verbose)
		{
			std::cout << "GPU_Backend_HIP: window: not tuned, the first " << tune_chunks << " chunks were too short for the "
			          << 3*win_T << " timesteps of the rule's candidate (" << measured << " of " << tune_cands.size()
			          << " candidates timed): the rule, T=" << win_T << " W=" << win_W << std::endl;
		}
		WindowDropGraphs(win_W);
		return;
	}
	win_T = tune_cands[best].T;
	win_W = tune_cands[best].W;
	win_source = "tuned";
	WindowDropGraphs(win_W);
	if (verbose)
	{
		std::ostringstream line;
		line << std::fixed << std::setprecision(0);
		line << "GPU_Backend_HIP: window: tuned T=" << win_T << " W=" << win_W << " B=0 shape=" << shape_block.x << "x" << shape_block.y
		     << " " << tune_cands[best].mcs << " MC/s (candidates:";
		for (size_t k=0; k<tune_cands.size(); ++k)
		{
			line << (k ? "; " : " ") << tune_cands[k].T << "," << tune_cands[k].W << " ";
			if (tune_cands[k].mcs>=0)
				line << tune_cands[k].mcs;
			else
				line << "-";
		}
		line << ")";
		std::cout << line.str() << std::endl;
		std::ostringstream cost;
		cost << "GPU_Backend_HIP: window: the tune ran " << tune_steps << " timesteps in " << tune_chunks << " chunk"
		     << (tune_chunks>1 ? "s" : "") << " (" << measured << " of " << tune_cands.size() << " candidates, timed blocks "
		     << std::fixed << std::setprecision(1) << tune_ms << " ms)" << (complete ? "" : ", stopped: the chunks were too short for the rest");
		std::cout << cost.str() << std::endl;
	}
}

bool GPU_Backend_HIP::SetWindowExtensions(const std::vector<GPU_Extension*>& exts, const std::string& refusal)
{
	const bool verbose = g_settings.GetVerboseLevel()>1;
	d->window_used = false;
	d->window_exts.clear();
	if (!d->window.on)
	{
		if (verbose)
			std::cout << "GPU_Backend_HIP: window: off" << std::endl;
		return false;
	}
	if (!refusal.empty())
	{
		if (verbose)
			std::cout << "GPU_Backend_HIP: window: refused: " << refusal << std::endl;
		return false;
	}
	if (!d->window.valid || (d->step_base==NULL))
	{
		if (verbose)
			std::cout << "GPU_Backend_HIP: window: invalid value " << d->window.text << " (0, auto, rule or T,W[,B]), single-step" << std::endl;
		return false;
	}
	for (size_t n=0; n<exts.size(); ++n)
		if (!exts[n]->WindowCapable())
		{
			if (verbose)
				std::cout << "GPU_Backend_HIP: window: refused: an extension is not window-capable" << std::endl;
			return false;
		}
	// the windowed blocks run the separate kernels; the fused step would be decided at the first voltage update
	const char* fused_env = getenv("OPENEMS_HIP_FUSED_STEP");
	const bool fused = (d->fused_step>0) || ((d->fused_step<0) && (d->fused_blockers==0) && !(fused_env && (atoi(fused_env)==0)));
	// auto keeps the fused step where it folds the UPML regions into its kernel and the rule's window covers all of x:
	// one window has no L2 gain, and the windowed blocks run the regions as separate kernels (RTX 5090, bench pml S:
	// 19406 MC/s windowed against 23839 fused; ideal S, without regions, gains 5 % from the windows' graphs)
	// (the UPML extensions decide their fusion at their first hook, after this: the regions of the usual layout fuse)
	const char* upml_env = getenv("OPENEMS_HIP_UPML_FUSION");
	if (fused && d->window.tune && !d->upml.empty() && !(upml_env && (atoi(upml_env)==0)))
	{
		const size_t plane_bytes = (size_t)d->dim.ny*d->dim.nzd*(24 + d->CoeffBytesPerNode());
		unsigned int T = d->window.T;
		const unsigned int W = HIP_WindowRuleTW(d->l2_bytes, plane_bytes, T, d->dim.nx, d->coeff_table_bytes);
		if (W>=d->dim.nx)
		{
			if (verbose)
				std::cout << "GPU_Backend_HIP: window: auto keeps the fused step: one window covers the grid (W=" << W << " of "
				          << d->dim.nx << " planes) and its " << d->upml.size() << " UPML regions go into the fused kernel" << std::endl;
			return false;
		}
	}
	d->window_exts = exts;
	d->window_used = true;
	if (d->fused_step<0)
		d->fused_step = 0;
	if (verbose && fused)
		std::cout << "GPU_Backend_HIP: window: separate updates, the fused step is not used on the windowed path" << std::endl;
	return true;
}

bool GPU_Backend_HIP::IterateWindowed(unsigned int start_ts, unsigned int n)
{
	if (!d->window_used)
		return false;
	if (d->fused_step>0)
		return false;   // decided before the windowed blocks: the fields are in its double buffers
	if (!d->WindowPrepare())
		return false;
	++d->window_chunks;
	unsigned int tuned = 0;
	if (d->tune_state>0)
		tuned = d->WindowTune(start_ts, n);   // real timesteps: the candidates' blocks
	d->WindowRun(start_ts + tuned, n - tuned);
	if (!d->window_logged && (d->tune_state==0) && (tuned<n) && (g_settings.GetVerboseLevel()>1))
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
		line << "GPU_Backend_HIP: window: T=" << d->win_T << " W=" << d->win_W << " B=" << d->win_B << " planes=" << nx
		     << " halo=" << d->win_T+1 << " windows=" << (nx + d->win_W - 1)/d->win_W << " launches/step=";
		if (per_block)
			line << std::fixed << std::setprecision(1) << (per_block + 1.0)/per_len;
		else
			line << "?";
		line << " graphs=" << d->window_graphs.size();
		if (!d->window_graphs.empty())
			line << " (captured in " << std::fixed << std::setprecision(1) << capture_ms << " ms)";
		else if (HIP_KernelTimes::Enabled())
			line << " (kernel times: plain launches)";
		line << " (" << d->win_source << ", " << d->dim.ny*(size_t)d->dim.nzd*(24 + d->CoeffBytesPerNode())
		     << " bytes per plane, coefficient table " << d->coeff_table_bytes << " bytes, L2 " << d->l2_bytes/(1024*1024) << " MiB, " << d->window_exts.size() << " window-capable extensions)";
		std::cout << line.str() << std::endl;
	}
	return true;
}

static void CheckSize(size_t host_count, size_t device_count)
{
	if (host_count!=device_count)
		throw std::runtime_error("GPU_Backend_HIP: host and device field size mismatch");
}

// OPENEMS_HIP_ZPAD: \a lines z lines of the host layout (stride nz) from or to the device layout (stride nzd), the padding untouched
static void CopyLines(GPU_Backend_HIP::Impl* d, void* dst, const void* src, size_t lines, hipMemcpyKind kind, const char* what)
{
	const size_t host_pitch = d->dim.nz*sizeof(float), dev_pitch = d->dim.nzd*sizeof(float);
	const bool to_host = (kind==hipMemcpyDeviceToHost);
	HIP_Check(hipMemcpy2DAsync(dst, to_host ? host_pitch : dev_pitch, src, to_host ? dev_pitch : host_pitch, host_pitch, lines, kind, d->Stream()), what);
}

void GPU_Backend_HIP::DownloadVoltages(ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CheckSize(volt.size(), 3*d->numCells);
	d->PinHostMemory(volt.data(), volt.size()*sizeof(FDTD_FLOAT));
	if (d->dim.nzd!=d->dim.nz)
		CopyLines(d, volt.data(), d->volt, volt.size()/d->dim.nz, hipMemcpyDeviceToHost, "download");
	else
		HIP_Check(hipMemcpyAsync(volt.data(), d->volt, volt.size()*sizeof(float), hipMemcpyDeviceToHost, d->Stream()), "download");
	d->Flush();
}

void GPU_Backend_HIP::DownloadCurrents(ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CheckSize(curr.size(), 3*d->numCells);
	d->PinHostMemory(curr.data(), curr.size()*sizeof(FDTD_FLOAT));
	if (d->dim.nzd!=d->dim.nz)
		CopyLines(d, curr.data(), d->curr, curr.size()/d->dim.nz, hipMemcpyDeviceToHost, "download");
	else
		HIP_Check(hipMemcpyAsync(curr.data(), d->curr, curr.size()*sizeof(float), hipMemcpyDeviceToHost, d->Stream()), "download");
	d->Flush();
}

void GPU_Backend_HIP::UploadVoltages(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CheckSize(volt.size(), 3*d->numCells);
	d->PinHostMemory(volt.data(), volt.size()*sizeof(FDTD_FLOAT));
	if (d->dim.nzd!=d->dim.nz)
		CopyLines(d, d->volt, volt.data(), volt.size()/d->dim.nz, hipMemcpyHostToDevice, "upload");
	else
		HIP_Check(hipMemcpyAsync(d->volt, volt.data(), volt.size()*sizeof(float), hipMemcpyHostToDevice, d->Stream()), "upload");
	d->Flush();
}

void GPU_Backend_HIP::UploadCurrents(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CheckSize(curr.size(), 3*d->numCells);
	d->PinHostMemory(curr.data(), curr.size()*sizeof(FDTD_FLOAT));
	if (d->dim.nzd!=d->dim.nz)
		CopyLines(d, d->curr, curr.data(), curr.size()/d->dim.nz, hipMemcpyHostToDevice, "upload");
	else
		HIP_Check(hipMemcpyAsync(d->curr, curr.data(), curr.size()*sizeof(float), hipMemcpyHostToDevice, d->Stream()), "upload");
	d->Flush();
}

bool GPU_Backend_HIP::DownloadRange(bool currents, size_t offset, size_t count, FDTD_FLOAT* dst)
{
	if (offset+count > 3*d->numCells)
		return false;
	if (d->dim.nzd!=d->dim.nz)
	{
		// OPENEMS_HIP_ZPAD: a part of one z line, or whole z lines (the caller, Engine_GPU, asks for one line);
		// else the host falls back
		const size_t nz = d->dim.nz;
		const float* src = (currents ? d->curr : d->volt) + d->Dev(offset);
		if (offset%nz + count <= nz)
			HIP_Check(hipMemcpyAsync(dst, src, count*sizeof(float), hipMemcpyDeviceToHost, d->Stream()), "range download");
		else if ((offset%nz==0) && (count%nz==0))
			CopyLines(d, dst, src, count/nz, hipMemcpyDeviceToHost, "range download");
		else
			return false;
		d->Flush();
		return true;
	}
	const float* src = (currents ? d->curr : d->volt) + offset;
	HIP_Check(hipMemcpyAsync(dst, src, count*sizeof(float), hipMemcpyDeviceToHost, d->Stream()), "range download");
	d->Flush();
	return true;
}

void GPU_Backend_HIP::Synchronize()
{
	d->Flush();
}

void* GPU_Backend_HIP::Impl::UploadRegion(const Engine_GatherRegion& region, HIP_GatherRegion& dev)
{
	std::vector<unsigned int> lines;
	for (int n=0; n<3; ++n)
		for (int k=0; k<3; ++k)
			lines.insert(lines.end(), region.lines[n][k].begin(), region.lines[n][k].end());
	unsigned int* buf = TryAlloc<unsigned int>(lines.size());
	if (!buf)
		return NULL;
	HIP_Check(hipMemcpy(buf, lines.data(), lines.size()*sizeof(unsigned int), hipMemcpyHostToDevice), "hipMemcpy");
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

bool GPU_Backend_HIP::SetSnapshotGather(const std::vector<Engine_GatherRegion>& volt_regions, const std::vector<Engine_GatherRegion>& curr_regions)
{
	HIP_Check(hipStreamSynchronize(d->ctx->copy_stream), "stream synchronization");
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
	bool ok = (hipMalloc(&d->snap_dev, n*sizeof(float))==hipSuccess)
	          && (hipHostMalloc(&d->snap_host, n*sizeof(float))==hipSuccess);
	const std::vector<Engine_GatherRegion>* regions[2] = {&volt_regions, &curr_regions};
	for (int f=0; f<2 && ok; ++f)
		for (size_t r=0; r<regions[f]->size() && ok; ++r)
		{
			HIP_GatherRegion dev;
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
		(void)hipGetLastError();
		d->FreeSnapshots();
		std::cerr << "GPU_Backend_HIP: not enough memory for field snapshots, the field dumps wait for the device" << std::endl;
		return false;
	}
	d->snap_nv = nv;
	d->snap_n = n;
	return true;
}

bool GPU_Backend_HIP::SnapshotFields(const FDTD_FLOAT* &volt, const FDTD_FLOAT* &curr)
{
	if (!d->snap_n)
		return false;
	// the raw values on the work stream, once the last snapshot was downloaded from snap_dev
	if (d->snap_taken)
		HIP_Check(hipStreamWaitEvent(d->Stream(), d->snap_done, 0), "hipStreamWaitEvent");
	size_t offset = 0;
	for (int f=0; f<2; ++f)
		for (size_t r=0; r<d->snap_regions[f].size(); ++r)
		{
			const HIP_GatherRegion& R = d->snap_regions[f][r];
			HIP_Launch(d, "gather_region", gather_region, R.total, 1, 1, (const float*)(f ? d->curr : d->volt), R, d->dim, d->snap_dev + offset);
			offset += R.total;
		}
	HIP_Check(hipEventRecord(d->snap_evaluated, d->Stream()), "hipEventRecord");
	// the download on the copy stream, while the work stream continues with the next timesteps
	hipStream_t copy = d->ctx->copy_stream;
	HIP_Check(hipStreamWaitEvent(copy, d->snap_evaluated, 0), "hipStreamWaitEvent");
	HIP_Check(hipMemcpyAsync(d->snap_host, d->snap_dev, d->snap_n*sizeof(float), hipMemcpyDeviceToHost, copy), "snapshot download");
	HIP_Check(hipEventRecord(d->snap_done, copy), "hipEventRecord");
	d->snap_taken = true;
	volt = d->snap_host;
	curr = d->snap_host + d->snap_nv;
	return true;
}

int GPU_Backend_HIP::AddFieldDFT(bool h_field, const Engine_GatherRegion& region, unsigned int count)
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
		std::cerr << "GPU_Backend_HIP: not enough memory for a frequency domain dump on the device" << std::endl;
		return -1;
	}
	HIP_Check(hipMemsetAsync(dft.sums, 0, n*count*sizeof(float2), d->Stream()), "hipMemset");
	d->field_dfts.push_back(dft);
	return d->field_dfts.size()-1;
}

void GPU_Backend_HIP::AccumulateFieldDFT(int id, const std::vector<std::complex<float>>& weights)
{
	if ((id<0) || ((size_t)id>=d->field_dfts.size()))
		return;
	const Impl::FieldDFT& dft = d->field_dfts[id];
	const unsigned int count = std::min((size_t)dft.count, weights.size());
	for (unsigned int k0=0; k0<count; k0+=DFT_WEIGHTS)
	{
		HIP_DFTWeights W;
		const unsigned int nw = std::min(count-k0, (unsigned int)DFT_WEIGHTS);
		for (unsigned int k=0; k<nw; ++k)
			W.w[k] = make_float2(weights[k0+k].real(), weights[k0+k].imag());
		HIP_Launch(d, "field_dft", field_dft, dft.region.total, 1, 1, (const float*)(dft.currents ? d->curr : d->volt),
		            dft.region, d->dim, W, nw, dft.sums + k0*dft.region.total);
	}
}

bool GPU_Backend_HIP::ReadFieldDFT(int id, unsigned int k, std::vector<std::complex<float>>& sums)
{
	if ((id<0) || ((size_t)id>=d->field_dfts.size()) || (k>=d->field_dfts[id].count))
		return false;
	const Impl::FieldDFT& dft = d->field_dfts[id];
	sums.resize(dft.region.total);
	HIP_Check(hipMemcpyAsync(sums.data(), dft.sums + k*dft.region.total, sums.size()*sizeof(float2), hipMemcpyDeviceToHost, d->Stream()), "download");
	d->Flush();
	return true;
}

void GPU_Backend_HIP::WaitSnapshot()
{
	// called on the dump thread: report errors instead of throwing
	const hipError_t err = hipEventSynchronize(d->snap_done);
	if (err!=hipSuccess)
		std::cerr << "GPU_Backend_HIP: field snapshot failed: " << hipGetErrorString(err) << std::endl;
}

bool GPU_Backend_HIP::CalcFastEnergy(const unsigned int numNodes[3], double& E_energy, double& H_energy)
{
	E_energy = H_energy = 0;
	HIP_GridDim L = {numNodes[0], numNodes[1], numNodes[2]};
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
	HIP_Launch(d, "energy_lines", energy_lines, L.nz, L.ny, 1, (const float*)d->volt, (const float*)d->curr, d->energy, d->dim, L);
	HIP_Check(hipMemcpyAsync(d->energy_host.data(), d->energy, count*sizeof(float2), hipMemcpyDeviceToHost, d->Stream()), "energy download");
	d->Flush();
	for (size_t k=0; k<count; ++k)
	{
		E_energy += d->energy_host[k].x;
		H_energy += d->energy_host[k].y;
	}
	return true;
}

// all HIP extensions, see hip_internal.h
static const HIP_ExtensionFactory HIP_EXTENSIONS[] = {
	HIP_CreateExt_Excitation,
	HIP_CreateExt_UPML,
	HIP_CreateExt_Mur_ABC,
	HIP_CreateExt_LorentzMaterial,
	HIP_CreateExt_DebyeMaterial,
	HIP_CreateExt_LumpedRLC,
	HIP_CreateExt_TFSF,
	HIP_CreateExt_Absorbing_BC,
	HIP_CreateExt_SteadyState,
	HIP_CreateExt_Cylinder,
};

GPU_Extension* GPU_Backend_HIP::CreateExtension(Engine_Extension* eng_ext, Engine* eng)
{
	// the fused step (see update_fused) knows the UPML, the excitation, lumped RLC elements and the steady-state detection
	if (!dynamic_cast<Engine_Ext_UPML*>(eng_ext) && !dynamic_cast<Engine_Ext_Excitation*>(eng_ext) && !dynamic_cast<Engine_Ext_LumpedRLC*>(eng_ext)
	    && !dynamic_cast<Engine_Ext_SteadyState*>(eng_ext))
		d->fused_blockers++;
	// in the extension order of the engine (see Impl::ext_log)
	const Impl::ExtRecord record = {eng_ext, d->volt_modified.size(), 0, d->ext_nodes.size(), 0};
	for (size_t n=0; n<sizeof(HIP_EXTENSIONS)/sizeof(HIP_EXTENSIONS[0]); ++n)
	{
		if (HIP_EXTENSIONS[n]==NULL)
			continue;
		GPU_Extension* gpu_ext = HIP_EXTENSIONS[n](d, eng_ext, eng);
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
