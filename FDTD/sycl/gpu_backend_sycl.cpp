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

#include <sycl/sycl.hpp>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "sycl_internal.h"
#include "gpu_build_id.h"
#include "FDTD/operator.h"
#include "FDTD/extensions/engine_ext_upml.h"
#include "FDTD/extensions/engine_ext_excitation.h"
#include "FDTD/extensions/engine_ext_lumpedRLC.h"
#include "FDTD/extensions/engine_ext_steadystate.h"

// Main FDTD updates, operation by operation the same as Engine::UpdateVoltages/UpdateCurrents,
// over the nodes in [B, E). One work-item per mesh node: z along dimension 2 (contiguous), y along
// dimension 1, x along dimension 0 (one work-group per x line). The coefficients are read with main_coeff().
// Nodes below Z.z_lo or from Z.z_hi on belong to UPML regions along z (see SYCL_ZSlabs).
// The index type I is unsigned int if the fields fit (see GPU_Backend_SYCL::Impl::index32), else size_t.
template<typename I>
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
		upml_fused_volt_node(volt, volt, curr, Z.volt_lo, N, Z.R_lo, x-Z.R_lo.sx, y-Z.R_lo.sy, z-Z.R_lo.sz);
		return;
	}
	if (z>=Z.z_hi)
	{
		upml_fused_volt_node(volt, volt, curr, Z.volt_hi, N, Z.R_hi, x-Z.R_hi.sx, y-Z.R_hi.sy, z-Z.R_hi.sz);
		return;
	}
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nz : 0;   // shift to the previous line, none on the first
	const I ym = (y>0) ? N.nz : 0;
	const I zm = (z>0) ? 1 : 0;
	const SYCL_MainCoeffT<I> C = main_coeff<I>(index, ca, cb, mode, sn, i, 0);
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

// the currents on the last mesh line of each direction are not updated
template<typename I>
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
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xp = (I)N.ny*N.nz;
	const I yp = N.nz;
	const SYCL_MainCoeffT<I> C = main_coeff<I>(index, ca, cb, mode, sn, i, 6);
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
template<typename I>
inline void fused_volt(const float* __restrict__ volt, const float* __restrict__ curr,
                       const void* index, const float* va, const float* vb, unsigned int mode,
                       const SYCL_GridDim& N, unsigned int x, unsigned int y, unsigned int z, float v[3])
{
	const I sn = (I)N.nx*N.ny*N.nz;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nz : 0;
	const I ym = (y>0) ? N.nz : 0;
	const I zm = (z>0) ? 1 : 0;
	const SYCL_MainCoeffT<I> C = main_coeff<I>(index, va, vb, mode, sn, i, 0);
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

// The arguments of update_fused but the UPML regions (passed separately, see launch_fused),
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
};

// Fused step: the voltages and currents of the main nodes [B, E) in one pass, from the current
// fields (volt_in, curr_in) into the next buffers (volt_out, curr_out); every value is computed
// with the same operations as update_voltages/update_currents.
// One kernel for the whole timestep: the current update reads the new voltages from local
// memory, instead of a second pass reading them back from global memory.
// A work-group marches along xc x lines of a (TZ-1)x(TY-1) tile in (z,y). At each x it computes
// the new voltages of the tile and its border (+z, +y) into local memory and of the next x line
// (registers), then the new currents of the tile. The border voltages of other tiles are
// computed again (they only read the current fields), those outside the main nodes (UPML
// regions) come from volt_out, where the UPML regions stored them before.
// With the UPML regions in the kernel (REG, F->count>0, [B, E) is the whole grid), the region
// nodes are computed like the main nodes, from the current flux (the new voltage flux goes to a
// second buffer, written by the node's own work-group; the current flux in place). F is the
// device copy of the regions (see Impl::FusedRegionsOnDevice()); without them (REG false) it is
// NULL and never read.
// sV is the local memory tile sV[3][TY][TZ], flattened: sV[c][ty][tz] is sV[(c*TY + ty)*TZ + tz].
template<typename I, unsigned int TZ, unsigned int TY, bool REG>
inline void update_fused(const sycl::nd_item<3>& it, const sycl::local_accessor<float, 1>& sV,
                         const SYCL_FusedArgs& a, const SYCL_FusedRegions* F)
{
	const float* __restrict__ volt_in = a.volt_in;
	const float* __restrict__ curr_in = a.curr_in;
	float* volt_out = a.volt_out;
	float* __restrict__ curr_out = a.curr_out;
	const SYCL_GridDim& N = a.N;
	const SYCL_GridDim& B = a.B;
	const SYCL_GridDim& E = a.E;
	const unsigned int tz = it.get_local_id(2), ty = it.get_local_id(1);
	const unsigned int z = B.nz + (unsigned int)it.get_group(2)*(TZ-1) + tz;
	const unsigned int y = B.ny + (unsigned int)it.get_group(1)*(TY-1) + ty;
	const unsigned int xs = B.nx + (unsigned int)it.get_group(0)*a.xc;
	const unsigned int xe = sycl::min(xs+a.xc, E.nx);
	const bool main_yz = (z<E.nz) && (y<E.ny);                          // a main node line
	const bool owner = main_yz && (tz<TZ-1) && (ty<TY-1);  // written by this work-group
	const bool in_grid = (z<N.nz) && (y<N.ny);
	const I sn = (I)N.nx*N.ny*N.nz;
	auto sv = [&](unsigned int c, unsigned int j, unsigned int k) -> float& {return sV[(c*TY + j)*TZ + k];};

	// the UPML region of node (x, y, z) in the kernel, -1: none; fused_region_of on a private
	// copy of the layout (F is global memory, which the compiler reads again after every store)
	SYCL_FusedRegions L;
	L.count = 0;
	if constexpr (REG)
	{
		L.count = F->count;
		L.lo = F->lo;
		L.hi = F->hi;
		for (int s=0; s<6; ++s)
			L.side[s] = F->side[s];
	}
	auto region_of = [&](unsigned int x) -> int
	{
		if constexpr (REG)
			return L.count ? fused_region_of(L, x, y, z) : -1;
		else
			return -1;
	};

	// the new voltages at x: computed on main nodes, else from the UPML regions (or unused)
	auto volt_at = [&](unsigned int x, float v[3])
	{
		if (main_yz && (x<E.nx))
		{
			const int r = region_of(x);
			if (r<0)
				fused_volt<I>(volt_in, curr_in, a.index, a.va, a.vb, a.mode, N, x, y, z, v);
			else if constexpr (REG)
			{
				const UPMLFusedArgs A = F->volt[r];
				const UPMLRegion R = F->R[r];
				upml_volt_value<I>(volt_in, curr_in, A, N, R, x, y, z, owner && (x<xe), v);
			}
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
	sv(0, ty, tz) = v[0]; sv(1, ty, tz) = v[1]; sv(2, ty, tz) = v[2];
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
				curl[0] = (sv(2, ty, tz) - sv(2, ty+1, tz) - sv(1, ty, tz) + sv(1, ty, tz+1));
				curl[1] = (sv(0, ty, tz) - sv(0, ty, tz+1) - sv(2, ty, tz) + v1[2]);
				curl[2] = (sv(1, ty, tz) - v1[1] - sv(0, ty, tz) + sv(0, ty+1, tz));
			}
			const int r = region_of(x);
			if (r>=0)
			{
				if constexpr (REG)
				{
					const UPMLFusedArgs A = F->curr[r];
					const UPMLRegion R = F->R[r];
					upml_curr_value<I>(curr_in, curr_out, A, N, R, x, y, z, update, curl);
				}
			}
			else if (update)
			{
				const I i = nijk<I>(N, 0, x, y, z);
				const SYCL_MainCoeffT<I> C = main_coeff<I>(a.index, a.ia, a.ib, a.mode, sn, i, 6);
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
		sv(0, ty, tz) = v1[0]; sv(1, ty, tz) = v1[1]; sv(2, ty, tz) = v1[2];
		sycl::group_barrier(it.get_group());
	}
}

// One launch of update_fused: work-groups of (1, TY, TZ) work-items (TZ along z, dimension 2),
// one sub-group of TZ per tile row. F: the device copy of the UPML regions in the kernel, or NULL.
template<typename I, unsigned int TZ, unsigned int TY>
static void launch_fused(GPU_Backend_SYCL::Impl* d, const SYCL_FusedArgs& a, const SYCL_FusedRegions* F)
{
	const size_t grid_z = (a.E.nz-a.B.nz+TZ-2)/(TZ-1);
	const size_t grid_y = (a.E.ny-a.B.ny+TY-2)/(TY-1);
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
			if (F)
				h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it)
					[[sycl::reqd_work_group_size(1, TY, TZ)]] [[sycl::reqd_sub_group_size(TZ)]]
				{
					update_fused<I, TZ, TY, true>(it, sV, a, F);
				});
			else
				h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it)
					[[sycl::reqd_work_group_size(1, TY, TZ)]] [[sycl::reqd_sub_group_size(TZ)]]
				{
					update_fused<I, TZ, TY, false>(it, sV, a, NULL);
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

template<typename I, unsigned int TZ>
static void launch_fused_ty(GPU_Backend_SYCL::Impl* d, const SYCL_FusedArgs& a, const SYCL_FusedRegions* F, unsigned int ty)
{
	switch (ty)
	{
		case 4:  launch_fused<I, TZ, 4>(d, a, F); break;
		case 16: launch_fused<I, TZ, 16>(d, a, F); break;
		default: launch_fused<I, TZ, 8>(d, a, F); break;
	}
}

// the index type of the fused step, see GPU_Backend_SYCL::Impl::index32
template<unsigned int TZ>
static void launch_fused_tz(GPU_Backend_SYCL::Impl* d, const SYCL_FusedArgs& a, const SYCL_FusedRegions* F, unsigned int ty, bool index32)
{
	if (index32)
		launch_fused_ty<unsigned int, TZ>(d, a, F, ty);
	else
		launch_fused_ty<size_t, TZ>(d, a, F, ty);
}

// the currents of the given main nodes again (update_currents), after the voltage extensions
// changed voltages they read (see GPU_Backend_SYCL::Impl::fixup)
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
	const SYCL_MainCoeff C = main_coeff(index, ia, ib, mode, sn, i, 6);
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
template<typename I>
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
		update_voltages<I>(it, volt, curr, index, ca, cb, mode, N, Bk, Ek, Z);
	});
}

template<typename I>
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
		update_currents<I>(it, curr, volt, index, ca, cb, mode, N, Bk, Ek, Z);
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
	memset(&fregions, 0, sizeof(fregions));
	fregions.count = 0;
	fregions_dev = NULL;
	memset(fregions_up, 0, sizeof(fregions_up));
	fregions_up_valid[0] = fregions_up_valid[1] = false;
	fregions_up_next = 0;
	volt_next = curr_next = NULL;
	fixup = NULL;
	index32 = false;
	fixup_count = 0;
	snap_nv = snap_n = 0;
	snap_dev = NULL;
	snap_host = NULL;
	snap_taken = false;
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
		return false;
	if (fused_blockers>0)
	{
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (" << fused_blockers
			          << " blocking extension(s) or grid(s))" << std::endl;
		return false;
	}
	// Default policy: no fused step on a grid with UPML regions. On the B580 the separate updates,
	// with the UPML regions along z in the main kernels, are faster than both fused variants
	// (benchmark/bench.py, MC/s, separate / fused with the regions in the kernel / fused with the
	// region kernels, OPENEMS_SYCL_FUSED_REGIONS=0):
	//   pml S 6006 / 4493 / 6154, pml M 8051 / 7133 / 8589, inhomogeneous S 3152 / 2953 / 1819
	// In the kernel the region nodes share sub-groups with the main nodes (both code paths run);
	// as region kernels the z slabs have short rows (about 70 us each on inhomogeneous S).
	// Without UPML the fused step is faster (ideal S 11114 vs 10793, M 13701 vs 11759).
	if (!upml.empty() && !(env && (atoi(env)==1)))
	{
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: separate voltage and current update kernels (UPML regions; OPENEMS_SYCL_FUSED_STEP=1 fuses them)" << std::endl;
		return false;
	}

	// The kernel arguments (captured by value) must fit the device's max_parameter_size: the
	// arguments of SYCL_FusedArgs, the region pointer and a margin for the local memory accessor
	// and the ranges, about 400 bytes. The B580 reports 2048 bytes.
	// The UPML regions (SYCL_FusedRegions, about 1.35 KB) are not among them, although with them
	// the arguments (about 1.5 KB) would fit the B580 too: the kernel picks a region by its
	// number, and a by-value struct indexed at run time is copied to private memory by every
	// work-item, which made the fused step 30 times slower than the separate updates. They are
	// in a device buffer instead (see FusedRegionsOnDevice()); without it, no regions in the kernel.
	const size_t margin = 256;
	const size_t arg_bytes = sizeof(SYCL_FusedArgs) + sizeof(const SYCL_FusedRegions*) + margin;
	if (arg_bytes > ctx->max_parameter_size)
	{
		std::cout << "GPU_Backend_SYCL: the kernel arguments of the fused step do not fit the device ("
		          << arg_bytes << " > " << ctx->max_parameter_size << " bytes), using separate updates" << std::endl;
		return false;
	}
	// OPENEMS_SYCL_FUSED_REGIONS=0: the UPML regions stay with the region kernels (into the next
	// buffers, see SYCL_Ext_UPML) and the kernel updates the main nodes only. On the B580 this is
	// faster on pml (no divergence between region and main nodes in a sub-group), slower on
	// inhomogeneous (the z slabs as region kernels), see the default policy above.
	if (const char* regions = getenv("OPENEMS_SYCL_FUSED_REGIONS"))
		if (atoi(regions)==0)
			fregions.count = 0;
	if (fregions.count && !fregions_dev)
	{
		fregions_dev = TryAlloc<SYCL_FusedRegions>(2);
		if (!fregions_dev)
			fregions.count = 0;   // the region kernels update the regions
	}

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
		if (fregions.count)
			std::cout << " (" << fregions.count << " UPML regions in the kernel)";
		std::cout << std::endl;
	}
	return true;
}

const SYCL_FusedRegions* GPU_Backend_SYCL::Impl::FusedRegionsOnDevice()
{
	// The UPML hooks set the voltage flux buffers of the step in fregions, alternating two: after
	// the first two steps both versions are on the device and the launch only picks one.
	for (unsigned int k=0; k<2; ++k)
		if (fregions_up_valid[k] && (memcmp(&fregions_up[k], &fregions, sizeof(fregions))==0))
			return fregions_dev + k;
	// in order with the kernels: a launch still reading the replaced copy runs before the copy
	const unsigned int k = fregions_up_next;
	fregions_up_next ^= 1;
	memcpy(&fregions_up[k], &fregions, sizeof(fregions));
	fregions_up_valid[k] = true;
	SYCL_Check("fused step regions", [&] {Queue().memcpy(fregions_dev + k, &fregions_up[k], sizeof(SYCL_FusedRegions)).wait();});
	return fregions_dev + k;
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

bool GPU_Backend_SYCL::Init(const Operator* op)
{
	unsigned int numLines[3];
	for (int n=0; n<3; ++n)
		numLines[n] = op->GetNumberOfLines(n, true);
	d->dim.nx = numLines[0];
	d->dim.ny = numLines[1];
	d->dim.nz = numLines[2];
	d->numCells = (size_t)numLines[0]*numLines[1]*numLines[2];
	d->main_start.nx = d->main_start.ny = d->main_start.nz = 0;
	d->main_stop = d->dim;
	d->upml_fused = -1;
	d->ResetZSlabs();

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

	const char* index_bits = d->index32 ? ", 32 bit indices" : ", 64 bit indices";

	// upload the final operator coefficients, including all changes by operator extensions.
	// Most nodes share one of a few coefficient sets (same material and mesh spacing): store
	// every distinct set of the 12 coefficients of a node once and a set index per node
	GPU_CoeffSets sets;
	if (GPU_FindSets(d->numCells, 12, [&](size_t i, float* values) {GPU_NodeCoefficients(op, numLines, i, values);}, sets))
	{
		d->coeff_mode = sets.mode;
		d->index = d->AllocIndex(sets);
		d->coeff = d->Alloc<float>(sets.table.size(), sets.table.data());
		if (g_settings.GetVerboseLevel()>1)
			std::cout << "GPU_Backend_SYCL: " << sets.count << " distinct coefficient sets, compressed update coefficients ("
			          << (sets.mode==1 ? 16 : 32) << " bit index)" << index_bits << std::endl;
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
		std::cout << "GPU_Backend_SYCL: full coefficient arrays" << index_bits << std::endl;
	return true;
}

void GPU_Backend_SYCL::UpdateVoltages()
{
	const bool c = d->coeff_mode>0;
	if (d->DecideFusedStep())
	{
		// the main nodes, or the whole grid with the UPML regions in the kernel (see update_fused)
		SYCL_GridDim B = d->main_start;
		SYCL_GridDim E = d->main_stop;
		if (d->fregions.count)
		{
			B.nx = B.ny = B.nz = 0;
			E = d->dim;
		}
		static const unsigned int xc = getenv("OPENEMS_SYCL_FUSED_XC") ? std::max(1, atoi(getenv("OPENEMS_SYCL_FUSED_XC"))) : FUSED_XC;
		// the tile is a sub-group wide (see FUSED_TZ); OPENEMS_SYCL_FUSED_TY sets the other side
		static const unsigned int ty = getenv("OPENEMS_SYCL_FUSED_TY") ? std::max(2, atoi(getenv("OPENEMS_SYCL_FUSED_TY"))) : FUSED_TY;
		static const unsigned int tz = (getenv("OPENEMS_SYCL_FUSED_TZ") && (atoi(getenv("OPENEMS_SYCL_FUSED_TZ"))==16)) ? 16 : FUSED_TZ;
		const SYCL_FusedArgs a = { (const float*)d->volt, (const float*)d->curr, d->volt_next, d->curr_next, (const void*)d->index,
		                           (const float*)(c ? d->coeff : d->vv), (const float*)(c ? d->coeff : d->vi),
		                           (const float*)(c ? d->coeff : d->ii), (const float*)(c ? d->coeff : d->iv),
		                           d->coeff_mode, d->dim, B, E, xc };
		const SYCL_FusedRegions* F = d->fregions.count ? d->FusedRegionsOnDevice() : NULL;
		if (tz==16)
			launch_fused_tz<16>(d, a, F, ty, d->index32);
		else
			launch_fused_tz<32>(d, a, F, ty, d->index32);
		// only the voltages: UpdateCurrents swaps the currents, once the fix-up has read the old ones
		std::swap(d->volt, d->volt_next);
		return;
	}
	SYCL_GridDim B, E;
	unsigned int threads_z;
	d->MainRange(B, E, threads_z);
	if (d->index32)
		launch_update_voltages<unsigned int>(d, threads_z, B, E);
	else
		launch_update_voltages<size_t>(d, threads_z, B, E);
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
			SYCL_Launch(d, "update_currents_nodes", count, 1, 1, [=](sycl::nd_item<3> it)
			{
				update_currents_nodes(it, curr_in, curr_out, volt, nodes, count, index, ia, ib, mode, N);
			});
		}
		std::swap(d->curr, d->curr_next);
		return;
	}
	SYCL_GridDim B, E;
	unsigned int threads_z;
	d->MainRange(B, E, threads_z);
	if (d->index32)
		launch_update_currents<unsigned int>(d, threads_z, B, E);
	else
		launch_update_currents<size_t>(d, threads_z, B, E);
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
	// the fused step (see update_fused) knows the UPML, the excitation, lumped RLC elements and the steady-state detection
	if (!dynamic_cast<Engine_Ext_UPML*>(eng_ext) && !dynamic_cast<Engine_Ext_Excitation*>(eng_ext) && !dynamic_cast<Engine_Ext_LumpedRLC*>(eng_ext)
	    && !dynamic_cast<Engine_Ext_SteadyState*>(eng_ext))
		d->fused_blockers++;
	for (size_t n=0; n<sizeof(SYCL_EXTENSIONS)/sizeof(SYCL_EXTENSIONS[0]); ++n)
	{
		if (SYCL_EXTENSIONS[n]==NULL)
			continue;
		GPU_Extension* gpu_ext = SYCL_EXTENSIONS[n](d, eng_ext, eng);
		if (gpu_ext)
			return gpu_ext;
	}
	d->fused_blockers++;   // host fallback
	return NULL;
}
