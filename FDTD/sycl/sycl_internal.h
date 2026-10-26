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

// SYCL only: shared by the SYCL backend and the SYCL extensions

#ifndef SYCL_INTERNAL_H
#define SYCL_INTERNAL_H

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_backend_sycl.h"
#include "FDTD/gpu_coeff_sets.h"

//! Grid size (number of mesh lines) as passed to the kernels
struct SYCL_GridDim
{
	unsigned int nx, ny, nz;
};

//! The kernels index the fields with the index type I: 32 bit while the fields fit, else size_t,
//! chosen in Init() (see GPU_Backend_SYCL::Impl::index32); the helpers take it as a template
//! parameter, size_t by default.
template<typename T> struct SYCL_NoDeduce { typedef T type; };

//! Fields and coefficients use the ArrayNIJK layout of the host:
//! index(n,x,y,z) = n*nx*ny*nz + x*ny*nz + y*nz + z
template<typename I = size_t>
inline I nijk(const SYCL_GridDim& N, unsigned int n, unsigned int x, unsigned int y, unsigned int z)
{
	return (((I)n*N.nx + x)*N.ny + y)*N.nz + z;
}

//! The raw values of a dump on the device, see Engine_GatherRegion: value e of component n is
//! the edge (n, lines[n][0][a], lines[n][1][b], lines[n][2][c]) with e = start[n] + (a*size[n][1] + b)*size[n][2] + c
struct SYCL_GatherRegion
{
	const unsigned int* lines[3][3];
	unsigned int size[3][3];
	size_t start[3];
	size_t total;
};

inline size_t region_field_index(const SYCL_GatherRegion& R, const SYCL_GridDim& N, size_t e)
{
	const int n = (e>=R.start[2]) ? 2 : ((e>=R.start[1]) ? 1 : 0);
	size_t l = e - R.start[n];
	const unsigned int c = l % R.size[n][2];
	l /= R.size[n][2];
	const unsigned int b = l % R.size[n][1];
	const unsigned int a = l / R.size[n][1];
	return nijk(N, n, R.lines[n][0][a], R.lines[n][1][b], R.lines[n][2][c]);
}

//! Main update coefficients of node i, see GPU_Backend_SYCL::Impl::coeff_mode:
//! a[n*s] and b[n*s] are vv/vi (voltages, set_offset 0) or ii/iv (currents, set_offset 6)
//! of direction n. Mode 0: ca/cb are the full arrays, else ca is the table of sets.
template<typename I> struct SYCL_MainCoeffT { const float* a; const float* b; I s; };
typedef SYCL_MainCoeffT<size_t> SYCL_MainCoeff;

template<typename I = size_t>
inline SYCL_MainCoeffT<I> main_coeff(const void* index, const float* ca, const float* cb, unsigned int mode,
                                     typename SYCL_NoDeduce<I>::type sn, typename SYCL_NoDeduce<I>::type i,
                                     unsigned int set_offset)
{
	SYCL_MainCoeffT<I> c;
	if (mode==0)
	{
		c.a = ca + i; c.b = cb + i; c.s = sn;
		return c;
	}
	const unsigned int set = (mode==1) ? (unsigned int)((const unsigned short*)index)[i] : ((const unsigned int*)index)[i];
	c.a = ca + (size_t)12*set + set_offset; c.b = c.a + 3; c.s = 1;   // 64 bit: more than 358M sets wrap in 32
	return c;
}

#include "sycl_upml.h"

//! Run \a f, converting a sycl::exception into a std::runtime_error naming \a what
template <typename F>
void SYCL_Check(const char* what, F f)
{
	try
	{
		f();
	}
	catch (const sycl::exception& e)
	{
		throw std::runtime_error(std::string("GPU_Backend_SYCL: ") + what + " failed: " + e.what());
	}
}

//! Device and queues, shared by the backends of all grids of a simulation (see NewSubGridBackend())
struct SYCL_Context
{
	std::shared_ptr<sycl::context> context;
	sycl::device device;
	sycl::queue compute_q;     //!< all kernels and the transfers of the work, in order
	sycl::queue copy_q;        //!< downloads of field snapshots, overlapping the work (see GPU_Backend_SYCL::SnapshotFields())
	std::string name;
	unsigned int sub_group_size;       //!< work-items that run in lockstep: 32 if the device has it, else its largest
	size_t max_parameter_size;         //!< bytes of kernel arguments
	unsigned long long max_mem_alloc_size;

	//! Both queues in order, on one context of \a dev; asynchronous errors are recorded (see ThrowAsyncError())
	SYCL_Context(const sycl::device& dev);
	~SYCL_Context();

	//! Throw a std::runtime_error if the device reported an asynchronous error
	void ThrowAsyncError();

protected:
	void AsyncHandler(sycl::exception_list errors);
	std::mutex m_AsyncMutex;
	std::string m_AsyncError;   //!< the first asynchronous error
};

//! Plain pair of floats in device memory (the layout of HIP's float2)
struct float2
{
	float x, y;
};

//! Launch geometry in HIP order: x is the fastest (contiguous) direction
struct SYCL_Dim3
{
	size_t x, y, z;
};

//! State of one grid
struct GPU_Backend_SYCL::Impl
{
	std::shared_ptr<SYCL_Context> ctx;

	SYCL_GridDim dim;
	size_t numCells;    //!< nx*ny*nz, the field buffers hold 3*numCells values

	float *volt, *curr;
	float2* energy;                      //!< per-line energy sums, see GPU_Backend_SYCL::CalcFastEnergy()
	size_t energy_count;
	std::vector<float2> energy_host;
	//! Main update coefficients: 0: full arrays vv, vi, ii, iv; 1/2: a 16/32 bit set index per node and the sets (see main_coeff())
	unsigned int coeff_mode;
	float *vv, *vi, *ii, *iv;
	void* index;
	float* coeff;

	//! The main updates cover the nodes in [main_start, main_stop), the fused UPML kernels the others (see sycl_ext_upml.cpp)
	SYCL_GridDim main_start, main_stop;
	//! UPML extensions of this grid and whether they run fused with the main updates (-1: not decided yet), see sycl_ext_upml.cpp
	std::vector<GPU_Extension*> upml;
	int upml_fused;
	//! UPML regions along z updated by the main kernels (nodes below main_start.nz / above main_stop.nz):
	//! the separate updates or the fused step (see update_fused)
	SYCL_ZSlabs zslabs;
	bool zslab_lo, zslab_hi;

	//! Fused step (see update_fused): voltages and currents of the main nodes in one pass, from the
	//! current fields (volt, curr) into the next buffers (volt_next, curr_next), swapped afterwards.
	int fused_step;                          //!< -1: not decided yet, 0: off, 1: on
	int fused_blockers;                      //!< extensions or grids that do not allow it
	float *volt_next, *curr_next;
	std::vector<size_t> volt_modified;       //!< flat indices of the voltages the extensions register as changed between the half-steps (excitation, lumped RLC)
	size_t* fixup;                           //!< main nodes whose currents are recomputed after the voltage extensions
	unsigned int fixup_count;
	bool index32;                            //!< the kernels index with 32 bit (see Init())
	//! Decide once whether the fused step is used, and prepare it; it keeps the UPML z slabs
	//! (zslab_lo, zslab_hi) in its kernel or leaves them to the region kernels (clears the flags)
	bool DecideFusedStep();

	//! Field snapshots (see GPU_Backend_SYCL::SnapshotFields()): the raw values the dumps read, copied on
	//! the device (gather_region) and downloaded on the copy queue into a host buffer
	//! while the next timesteps run
	std::vector<SYCL_GatherRegion> snap_regions[2];   //!< of the voltages, of the currents
	std::vector<void*> snap_lines;    //!< device line tables of the regions
	unsigned int snap_nv, snap_n;     //!< number of raw voltages, of all raw values
	float* snap_dev;                  //!< the raw values
	float* snap_host;                 //!< the downloaded raw values
	sycl::event snap_evaluated;       //!< snap_dev is written
	sycl::event snap_done;            //!< the download is done
	bool snap_taken;                  //!< a download from snap_dev was started
	void FreeSnapshots();

	//! Sums of weighted raw values (frequency domain dumps, see GPU_Backend_SYCL::AddFieldDFT())
	struct FieldDFT
	{
		SYCL_GatherRegion region;
		unsigned int count;   //!< number of sums per raw value
		bool currents;        //!< raw values of the currents, else of the voltages
		float2* sums;         //!< sum k of raw value e at [k*region.total + e]
	};
	//! Device copy of the line tables of \a region (one buffer, see TryAlloc()); NULL if out of memory
	void* UploadRegion(const Engine_GatherRegion& region, SYCL_GatherRegion& dev);
	std::vector<FieldDFT> field_dfts;

	Impl();
	~Impl();

	sycl::queue& Queue() const {return ctx->compute_q;}

	//! Device buffer of \a count elements, initialized with \a host or zero; freed with this Impl
	template <typename T>
	T* Alloc(size_t count, const T* host=NULL)
	{
		const size_t n = std::max(count, (size_t)1);
		T* ptr = NULL;
		SYCL_Check("malloc_device", [&] {ptr = sycl::malloc_device<T>(n, Queue());});
		if (ptr==NULL)
			throw std::runtime_error("GPU_Backend_SYCL: malloc_device failed: not enough device memory for "
			                         + std::to_string(n*sizeof(T)) + " bytes");
		m_Allocations.push_back(ptr);
		// on the work queue: in order with the kernels
		if (host && count)
			SYCL_Check("memcpy", [&] {Queue().memcpy(ptr, host, count*sizeof(T));});
		else
			SYCL_Check("memset", [&] {Queue().memset(ptr, 0, n*sizeof(T));});
		Flush();
		return ptr;
	}

	//! Uninitialized device buffer of \a count elements for an optional optimization, freed with
	//! this Impl or by Free(); NULL if the device memory is not sufficient
	template <typename T>
	T* TryAlloc(size_t count)
	{
		T* ptr = NULL;
		try
		{
			ptr = sycl::malloc_device<T>(std::max(count, (size_t)1), Queue());
		}
		catch (const sycl::exception&)
		{
			ptr = NULL;
		}
		if (ptr==NULL)
			return NULL;
		m_Allocations.push_back(ptr);
		return ptr;
	}

	//! Free a buffer of Alloc() or TryAlloc() before this Impl
	void Free(void* ptr)
	{
		if (ptr==NULL)
			return;
		m_Allocations.erase(std::remove(m_Allocations.begin(), m_Allocations.end(), ptr), m_Allocations.end());
		try
		{
			sycl::free(ptr, *ctx->context);
		}
		catch (const sycl::exception&)
		{
		}
	}

	//! Device buffer of the set indices of \a sets (16 or 32 bit)
	void* AllocIndex(const GPU_CoeffSets& sets)
	{
		if (sets.mode==1)
		{
			std::vector<unsigned short> index16(sets.index.begin(), sets.index.end());
			return Alloc<unsigned short>(index16.size(), index16.data());
		}
		return Alloc<unsigned int>(sets.index.size(), sets.index.data());
	}

	//! No UPML regions updated by the main kernels (call after changing main_start/main_stop)
	void ResetZSlabs();
	//! Nodes [B, E) of the main kernels: main_start/main_stop and the UPML regions along z they update
	void MainRange(SYCL_GridDim& B, SYCL_GridDim& E, unsigned int& threads_z);

	//! Wait until all work on the compute queue is done, throw a std::runtime_error on a device error
	void Flush();

	//! Page-lock host memory for fast transfers: nothing in SYCL, which cannot page-lock memory
	//! it did not allocate (the host mirror stays pageable, the transfers stage it)
	void PinHostMemory(void* ptr, size_t bytes) {(void)ptr; (void)bytes;}

	//! Launch geometry: one work-item per (i,j,k), i fastest (HIP order, see SYCL_Launch())
	static SYCL_Dim3 Block(size_t ni, size_t nj);
	static SYCL_Dim3 Grid(SYCL_Dim3 block, size_t ni, size_t nj, size_t nk);

protected:
	std::vector<void*> m_Allocations;
};

//! Kernel timing for development (environment variable OPENEMS_SYCL_KERNEL_TIMES=1): the
//! time of every launch, per kernel name, printed at exit. Waits for each kernel.
struct SYCL_KernelTimes
{
	static bool Enabled();
	static void Add(const char* name, double ms);
};

//! Launch \a kernel with one work-item per (i,j,k) on the compute queue of \a d; the kernel checks its bounds
/*!
  \a kernel is called with a sycl::nd_item<3>: HIP's x (i, contiguous) is dimension 2, y (j)
  dimension 1 and the grid z (k) dimension 0. So threadIdx.x is it.get_local_id(2),
  blockIdx.x*blockDim.x is it.get_group(2)*it.get_local_range(2), and blockIdx.z is it.get_group(0).
  */
template <typename Kernel>
void SYCL_Launch(GPU_Backend_SYCL::Impl* d, const char* name, size_t ni, size_t nj, size_t nk, const Kernel& kernel)
{
	if (ni==0 || nj==0 || nk==0)
		return;
	const SYCL_Dim3 block = GPU_Backend_SYCL::Impl::Block(ni, nj);
	const SYCL_Dim3 grid = GPU_Backend_SYCL::Impl::Grid(block, ni, nj, nk);
	const sycl::range<3> local(block.z, block.y, block.x);
	const sycl::range<3> global(grid.z*block.z, grid.y*block.y, grid.x*block.x);
	const bool timed = SYCL_KernelTimes::Enabled();
	std::chrono::steady_clock::time_point start;
	if (timed)
	{
		d->Flush();
		start = std::chrono::steady_clock::now();
	}
	try
	{
		d->Queue().parallel_for(sycl::nd_range<3>(global, local), kernel);
	}
	catch (const sycl::exception& e)
	{
		throw std::runtime_error(std::string("GPU_Backend_SYCL: ") + name + " failed: " + e.what());
	}
	if (timed)
	{
		d->Flush();
		SYCL_KernelTimes::Add(name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
	}
}

//! Factory of a SYCL extension: the device implementation of \a eng_ext, or NULL if \a eng_ext is not of its type
typedef GPU_Extension* (*SYCL_ExtensionFactory)(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_Excitation(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_UPML(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_Mur_ABC(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_LorentzMaterial(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_DebyeMaterial(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_LumpedRLC(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_TFSF(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_Absorbing_BC(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_SteadyState(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* SYCL_CreateExt_Cylinder(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng);

#endif // SYCL_INTERNAL_H
