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

// HIP only: shared by the HIP backend and the HIP extensions

#ifndef HIP_INTERNAL_H
#define HIP_INTERNAL_H

#include <hip/hip_runtime.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gpu_backend_hip.h"
#include "FDTD/gpu_coeff_sets.h"

//! Grid size (number of mesh lines) as passed to the kernels
struct HIP_GridDim
{
	unsigned int nx, ny, nz;
};

//! The kernels index the fields with size_t. The fused step (update_fused) is also built with
//! 32 bit indices, chosen in Init() (see GPU_Backend_HIP::Impl::index32); the helpers it uses take
//! the index type I as a template parameter, size_t by default.
template<typename T> struct HIP_NoDeduce { typedef T type; };

//! Fields and coefficients use the ArrayNIJK layout of the host:
//! index(n,x,y,z) = n*nx*ny*nz + x*ny*nz + y*nz + z
template<typename I = size_t>
__device__ __forceinline__ I nijk(const HIP_GridDim& N, unsigned int n, unsigned int x, unsigned int y, unsigned int z)
{
	return (((I)n*N.nx + x)*N.ny + y)*N.nz + z;
}

//! Main update coefficients of node i, see GPU_Backend_HIP::Impl::coeff_mode:
//! a[n*s] and b[n*s] are vv/vi (voltages, set_offset 0) or ii/iv (currents, set_offset 6)
//! of direction n. Mode 0: ca/cb are the full arrays, else ca is the table of sets.
//! The raw values of a dump on the device, see Engine_GatherRegion: value e of component n is
//! the edge (n, lines[n][0][a], lines[n][1][b], lines[n][2][c]) with e = start[n] + (a*size[n][1] + b)*size[n][2] + c
struct HIP_GatherRegion
{
	const unsigned int* lines[3][3];
	unsigned int size[3][3];
	size_t start[3];
	size_t total;
};

__device__ __forceinline__ size_t region_field_index(const HIP_GatherRegion& R, const HIP_GridDim& N, size_t e)
{
	const int n = (e>=R.start[2]) ? 2 : ((e>=R.start[1]) ? 1 : 0);
	size_t l = e - R.start[n];
	const unsigned int c = l % R.size[n][2];
	l /= R.size[n][2];
	const unsigned int b = l % R.size[n][1];
	const unsigned int a = l / R.size[n][1];
	return nijk(N, n, R.lines[n][0][a], R.lines[n][1][b], R.lines[n][2][c]);
}

template<typename I> struct HIP_MainCoeffT { const float* a; const float* b; I s; };
typedef HIP_MainCoeffT<size_t> HIP_MainCoeff;

template<typename I = size_t>
__device__ __forceinline__ HIP_MainCoeffT<I> main_coeff(const void* index, const float* ca, const float* cb, unsigned int mode,
                                                        typename HIP_NoDeduce<I>::type sn, typename HIP_NoDeduce<I>::type i,
                                                        unsigned int set_offset)
{
	HIP_MainCoeffT<I> c;
	if (mode==0)
	{
		c.a = ca + i; c.b = cb + i; c.s = sn;
		return c;
	}
	const unsigned int set = (mode==1) ? (unsigned int)((const unsigned short*)index)[i] : ((const unsigned int*)index)[i];
	c.a = ca + (size_t)12*set + set_offset; c.b = c.a + 3; c.s = 1;   // 64 bit: more than 358M sets wrap in 32
	return c;
}

//! The main voltage update of component \a n of node \a i from its old value \a old: the statements of
//! update_voltages, for the kernels of OPENEMS_HIP_MUR_FUSED (see GPU_Backend_HIP::Impl::mur_mode), which
//! must reproduce its results bit for bit. xm, ym, zm: the shifts to the previous line (0 on the first).
__device__ __forceinline__ float volt_update_node(unsigned int n, float old, const float* curr, const HIP_MainCoeff& C,
                                                  size_t sn, size_t i, size_t xm, size_t ym, size_t zm)
{
	float v;
	if (n==0)
	{
		v  = old * C.a[0];
		v += C.b[0] * (curr[2*sn+i] - curr[2*sn+i-ym] - curr[sn+i] + curr[sn+i-zm]);
	}
	else if (n==1)
	{
		v  = old * C.a[C.s];
		v += C.b[C.s] * (curr[i] - curr[i-zm] - curr[2*sn+i] + curr[2*sn+i-xm]);
	}
	else
	{
		v  = old * C.a[2*C.s];
		v += C.b[2*C.s] * (curr[sn+i] - curr[sn+i-xm] - curr[i] + curr[i-ym]);
	}
	return v;
}

//! volt_update_node() of the node (x, y, z)
__device__ __forceinline__ float volt_update_at(unsigned int n, float old, const float* curr, const void* index, const float* ca,
                                                const float* cb, unsigned int mode, const HIP_GridDim& N,
                                                unsigned int x, unsigned int y, unsigned int z)
{
	const size_t sn = (size_t)N.nx*N.ny*N.nz;
	const size_t i  = nijk(N, 0, x, y, z);
	const size_t xm = (x>0) ? (size_t)N.ny*N.nz : 0;
	const size_t ym = (y>0) ? N.nz : 0;
	const size_t zm = (z>0) ? 1 : 0;
	return volt_update_node(n, old, curr, main_coeff(index, ca, cb, mode, sn, i, 0), sn, i, xm, ym, zm);
}

//! Mur planes of OPENEMS_HIP_MUR_FUSED=1 (mur_shell), face f = 2*ny + (line>0); the
//! kernel reads them from device memory (GPU_Backend_HIP::Impl::mur_dev), the active bits are an argument
struct HIP_MurFace
{
	const float* c_nyP;    //!< coefficients of component nyP, at (i,j) = i*numLines[nyPP] + j
	const float* c_nyPP;   //!< coefficients of component nyPP
	unsigned int line, shift;
	unsigned int rank;     //!< position in the apply order of the Mur planes: the higher one is applied later
};
struct HIP_MurFaces
{
	HIP_MurFace f[6];
	unsigned int active;   //!< bit f: face f has a Mur plane that is active in this timestep
};

//! Nodes of OPENEMS_HIP_MUR_FUSED=1: update_voltages_inner updates the main nodes [B, E) inside [L, H), mur_shell the others
struct HIP_MurShell
{
	HIP_GridDim B, E, L, H;
};

//! The z faces of OPENEMS_HIP_MUR_FUSED=2: the interior of each (x and y inside [HIP_MurShell::L, H)) is updated
//! by update_voltages_inner (mur_zface_lane()), which exchanges the values of the Mur line and the shift line of a
//! z line through warp shuffles. Index 0 the z-low, 1 the z-high plane
struct HIP_MurZFaces
{
	const float* c_x[2];   //!< coefficients of component x (the plane's nyP), at x*ny + y
	const float* c_y[2];   //!< coefficients of component y (the plane's nyPP), at x*ny + y
	unsigned int line[2], shift[2];
	unsigned int active;   //!< bit f: plane f exists and is active in this timestep
};

//! Lane mask of the warp shuffles: 64 bit on AMD (the HIP warp sync builtins ask for it), 32 bit on NVIDIA
#if defined(__HIP_PLATFORM_NVIDIA__)
typedef unsigned int HIP_LaneMask;
#else
typedef unsigned long long HIP_LaneMask;
#endif

//! A Mur plane of the grid, registered by HIP_Ext_Mur_ABC
struct HIP_MurPlane
{
	int ny;
	unsigned int line, shift;
	const float *c_nyP, *c_nyPP;
	size_t seq;   //!< position of its extension in the extension order (see GPU_Backend_HIP::Impl::ext_log)
};

#include "hip_upml.hiph"

//! Throw a std::runtime_error if \a err is an error
void HIP_Check(hipError_t err, const char* what);

//! Device and work stream, shared by the backends of all grids of a simulation (see NewSubGridBackend())
struct HIP_Context
{
	int device;
	hipStream_t stream;
	hipStream_t copy_stream;   //!< downloads of field snapshots, overlapping the work (see GPU_Backend_HIP::SnapshotFields())
	std::string name;
	unsigned int warp_size;    //!< threads that run in lockstep: 32 on NVIDIA, 64 on the CDNA GPUs of AMD

	HIP_Context();
	~HIP_Context();
};

//! What a device buffer holds: the classes of OPENEMS_HIP_COMPRESSIBLE (see hip_compressible.hip)
enum HIP_AllocClass {HIP_AllocOther, HIP_AllocFields, HIP_AllocCoeff};

//! Compressible device memory (see hip_compressible.hip). The classes OPENEMS_HIP_COMPRESSIBLE
//! asks for, one bit per HIP_AllocClass (read once): unset or 0 none, 1 or all every class,
//! else a comma separated list of fields, coeff
unsigned HIP_CompressibleClasses();
//! Allocation granularity of compressible memory on \a device (queried once), 0 if unknown
size_t HIP_CompressibleGranularity(int device);
//! Compressible buffer of at least \a bytes, padded to the granularity; NULL (\a granted false)
//! if the driver hands back plain memory or anything fails, with nothing left allocated
void* HIP_CompressibleAlloc(int device, size_t bytes, bool& granted);
//! Unmap and release \a ptr if it is a buffer of HIP_CompressibleAlloc(); false otherwise
bool HIP_CompressibleFree(void* ptr);
//! share (0..1) of the complete 128-byte lines of \a data whose bytes all equal the line's first 4 bytes (host memory)
double HIP_UniformLineShare(const void* data, size_t bytes);
//! true if OPENEMS_HIP_COMPRESSIBLE_STATS is set to a non-zero value (read once)
bool HIP_CompressibleStats();

//! State of one grid
struct GPU_Backend_HIP::Impl
{
	std::shared_ptr<HIP_Context> ctx;

	HIP_GridDim dim;
	size_t numCells;    //!< nx*ny*nz, the field buffers hold 3*numCells values

	float *volt, *curr;
	float2* energy;                      //!< per-line energy sums, see GPU_Backend_HIP::CalcFastEnergy()
	size_t energy_count;
	std::vector<float2> energy_host;
	//! Main update coefficients: 0: full arrays vv, vi, ii, iv; 1/2: a 16/32 bit set index per node and the sets (see main_coeff())
	unsigned int coeff_mode;
	float *vv, *vi, *ii, *iv;
	void* index;
	float* coeff;

	//! The main updates cover the nodes in [main_start, main_stop), the fused UPML kernels the others (see hip_ext_upml.cu)
	HIP_GridDim main_start, main_stop;
	//! UPML extensions of this grid and whether they run fused with the main updates (-1: not decided yet), see hip_ext_upml.cu
	std::vector<GPU_Extension*> upml;
	int upml_fused;
	//! UPML regions along z updated by the main kernels (nodes below main_start.nz / above main_stop.nz)
	HIP_ZSlabs zslabs;
	bool zslab_lo, zslab_hi;

	//! Fused step (see update_fused): voltages and currents of the main nodes in one pass, from the
	//! current fields (volt, curr) into the next buffers (volt_next, curr_next), swapped afterwards.
	int fused_step;                          //!< -1: not decided yet, 0: off, 1: on
	int fused_blockers;                      //!< extensions or grids that do not allow it
	float *volt_next, *curr_next;
	std::vector<size_t> volt_modified;       //!< flat indices of the voltages the extensions register as changed between the half-steps (excitation, lumped RLC)
	HIP_FusedRegions fregions;              //!< the UPML regions in the kernel (see HIP_Ext_UPML::CanFuse())
	size_t* fixup;                           //!< main nodes whose currents are recomputed after the voltage extensions
	unsigned int fixup_count;
	bool index32;                            //!< the fused step indexes with 32 bit (see Init())
	//! Decide once whether the fused step is used, and prepare it
	bool DecideFusedStep();

	//! Extensions in the order the engine created them (its extension order), see CreateExtension(): the
	//! voltages they register in volt_modified are [vm_begin, vm_end), the nodes they register in
	//! ext_nodes [nodes_begin, nodes_end)
	struct ExtRecord { Engine_Extension* eng_ext; size_t vm_begin, vm_end, nodes_begin, nodes_end; };
	std::vector<ExtRecord> ext_log;
	//! Flat node indices (x*ny*nz + y*nz + z) an extension reads or writes the voltages of at its hooks, registered
	//! by its constructor (the dispersive materials and sheets), for the check of MurRefusal()
	std::vector<size_t> ext_nodes;
	//! OPENEMS_HIP_MUR_FUSED (read in Init(), default 1): 0 the three launches per Mur plane, 1 the Mur planes in
	//! the voltage update (update_voltages_inner and mur_shell, see mur_volt_component), 2 as 1 with the interior
	//! of the z faces in update_voltages_inner (mur_zface_lane())
	int mur_knob;
	//! the mode in use, -1 until DecideMurFused() (the first voltage update); the Mur hooks of a timestep
	//! that started before the decision finish it as mode 0
	int mur_mode;
	std::vector<HIP_MurPlane> mur_planes;
	HIP_MurFaces mur;                         //!< the planes of mode 1, active bits set by the Mur hooks
	HIP_MurFace* mur_dev;                     //!< device copy of mur.f (the kernel indexes it)
	HIP_MurShell mur_shell;                   //!< the nodes of the two kernels of mode 1
	size_t mur_shell_count;                   //!< number of nodes of mur_shell
	bool mur_released;                        //!< a Mur extension was deleted (host fallback): mode 0
	bool mur_zf_on;                           //!< mode 1 with the z faces of OPENEMS_HIP_MUR_FUSED=2
	HIP_MurZFaces mur_zf;                     //!< the z planes of mode 2 (active bits set at the launch)
	//! Decide once whether OPENEMS_HIP_MUR_FUSED=1 (or 2) is used, log it
	void DecideMurFused();
	//! Why mode 1 cannot be used on this grid, empty if it can
	std::string MurRefusal() const;
	//! Why the z faces of mode 2 cannot run in update_voltages_inner, empty if they can (call after MurRefusal())
	std::string MurZFaceRefusal();

	//! OPENEMS_HIP_WINDOW (read in Init(), see HIP_ParseWindowKnob()): windowed blocks of T timesteps and windows of W x planes
	struct WindowKnob
	{
		bool on;              //!< set and not 0
		bool supported;       //!< a block shape this build runs: T = 1 and W = nx (the degenerate window)
		unsigned int T, W;    //!< W: the grid's nx for "NX"
		std::string text;     //!< the value as set
	};
	WindowKnob window;
	//! the windowed blocks run (see GPU_Backend_HIP::SetWindowExtensions()), with these extensions in the engine's order
	bool window_used;
	std::vector<GPU_Extension*> window_exts;
	//! the timestep of the windowed block that runs (one int, device memory): the base the window-capable
	//! extensions read their step from (see GPU_Extension::LaunchInWindow()); allocated in Init()
	int* step_base;
	//! The main voltage / current update of the nodes [x0,x1) x [y0,y1) (all z) for the windowed blocks:
	//! the separate kernels of UpdateVoltages()/UpdateCurrents() on the main range cut to the window
	void WindowVoltages(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1);
	void WindowCurrents(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1);

	//! Field snapshots (see GPU_Backend_HIP::SnapshotFields()): the raw values the dumps read, copied on
	//! the device (gather_region) and downloaded on the copy stream into a page-locked host buffer
	//! while the next timesteps run
	std::vector<HIP_GatherRegion> snap_regions[2];   //!< of the voltages, of the currents
	std::vector<void*> snap_lines;    //!< device line tables of the regions
	unsigned int snap_nv, snap_n;     //!< number of raw voltages, of all raw values
	float* snap_dev;                  //!< the raw values
	float* snap_host;                 //!< the downloaded raw values
	hipEvent_t snap_evaluated;       //!< snap_dev is written
	hipEvent_t snap_done;            //!< the download is done
	bool snap_taken;                  //!< a download from snap_dev was started
	void FreeSnapshots();

	//! Sums of weighted raw values (frequency domain dumps, see GPU_Backend_HIP::AddFieldDFT())
	struct FieldDFT
	{
		HIP_GatherRegion region;
		unsigned int count;   //!< number of sums per raw value
		bool currents;        //!< raw values of the currents, else of the voltages
		float2* sums;         //!< sum k of raw value e at [k*region.total + e]
	};
	//! Device copy of the line tables of \a region (one buffer, see TryAlloc()); NULL if out of memory
	void* UploadRegion(const Engine_GatherRegion& region, HIP_GatherRegion& dev);
	std::vector<FieldDFT> field_dfts;

	Impl();
	~Impl();

	hipStream_t Stream() const {return ctx->stream;}

	//! Device buffer of \a count elements, initialized with \a host or zero; freed with this Impl.
	//! \a cls: compressible memory if OPENEMS_HIP_COMPRESSIBLE asks for the class (see AllocCompressible())
	template <typename T>
	T* Alloc(size_t count, const T* host=NULL, HIP_AllocClass cls=HIP_AllocOther, const char* name=NULL)
	{
		if (name && HIP_CompressibleStats())
			ReportUniformLines(name, host, count*sizeof(T));
		T* ptr = NULL;
		if (HIP_CompressibleClasses() & (1u<<cls))
			ptr = static_cast<T*>(AllocCompressible(std::max(count, (size_t)1)*sizeof(T), cls));
		if (ptr==NULL)
			HIP_Check(hipMalloc(&ptr, std::max(count, (size_t)1)*sizeof(T)), "hipMalloc");
		m_Allocations.push_back(ptr);
		// on the work stream: it does not synchronize with the legacy default stream
		if (host && count)
			HIP_Check(hipMemcpyAsync(ptr, host, count*sizeof(T), hipMemcpyHostToDevice, Stream()), "hipMemcpy");
		else
			HIP_Check(hipMemsetAsync(ptr, 0, std::max(count, (size_t)1)*sizeof(T), Stream()), "hipMemset");
		Flush();
		return ptr;
	}

	//! Uninitialized device buffer of \a count elements for an optional optimization, freed with
	//! this Impl or by Free(); NULL if the device memory is not sufficient. \a cls as for Alloc()
	template <typename T>
	T* TryAlloc(size_t count, HIP_AllocClass cls=HIP_AllocOther, const char* name=NULL)
	{
		(void)name;
		T* ptr = NULL;
		if (HIP_CompressibleClasses() & (1u<<cls))
			ptr = static_cast<T*>(AllocCompressible(std::max(count, (size_t)1)*sizeof(T), cls));
		if ((ptr==NULL) && (hipMalloc(&ptr, std::max(count, (size_t)1)*sizeof(T))!=hipSuccess))
		{
			(void)hipGetLastError();   // not sticky, but reported by the next CheckLaunch()
			return NULL;
		}
		m_Allocations.push_back(ptr);
		return ptr;
	}

	//! Free a buffer of Alloc() or TryAlloc() before this Impl
	void Free(void* ptr)
	{
		if (ptr==NULL)
			return;
		m_Allocations.erase(std::remove(m_Allocations.begin(), m_Allocations.end(), ptr), m_Allocations.end());
		if (!FreeCompressible(ptr))
			(void)hipFree(ptr);
	}

	//! Compressible buffer of \a bytes of class \a cls for Alloc()/TryAlloc() (counted for
	//! LogCompressible()); NULL if it is smaller than the granularity or than 2 MiB (these stay
	//! plain: padding, and they do not stream), or not granted
	void* AllocCompressible(size_t bytes, HIP_AllocClass cls);
	//! Release \a ptr if it is a compressible buffer of this Impl; false otherwise (then hipFree it)
	bool FreeCompressible(void* ptr);
	//! Report the compressible buffers once (verbose level 1 and up), after the last of Init() and DecideFusedStep()
	void LogCompressible();

	//! Device buffer of the set indices of \a sets (16 or 32 bit)
	void* AllocIndex(const GPU_CoeffSets& sets)
	{
		if (sets.mode==1)
		{
			std::vector<unsigned short> index16(sets.index.begin(), sets.index.end());
			return Alloc<unsigned short>(index16.size(), index16.data(), HIP_AllocCoeff, "index");
		}
		return Alloc<unsigned int>(sets.index.size(), sets.index.data(), HIP_AllocCoeff, "index");
	}

	//! No UPML regions updated by the main kernels (call after changing main_start/main_stop)
	void ResetZSlabs();
	//! Nodes [B, E) of the main kernels: main_start/main_stop and the UPML regions along z they update
	void MainRange(HIP_GridDim& B, HIP_GridDim& E, unsigned int& threads_z);

	//! Statistics (OPENEMS_HIP_COMPRESSIBLE_STATS): print the uniform line share of a buffer about to be
	//! uploaded from \a host (NULL: zero fill, 100 % by construction)
	void ReportUniformLines(const char* name, const void* host, size_t bytes);

	//! Wait until all work on the stream is done
	void Flush();

	//! Check for a kernel launch error
	void CheckLaunch(const char* kernel);

	//! Page-lock host memory (once) for fast transfers
	void PinHostMemory(void* ptr, size_t bytes);

	//! Launch geometry: one thread per (i,j,k), i fastest
	static dim3 Block(size_t ni, size_t nj);
	static dim3 Grid(dim3 block, size_t ni, size_t nj, size_t nk);

protected:
	std::vector<void*> m_Allocations;
	std::set<void*> m_Pinned;
	//! Compressible buffers among m_Allocations: class and padded size; requests not granted
	std::map<void*, std::pair<HIP_AllocClass, size_t> > m_Compressible;
	unsigned int m_CompressibleRefused;
	bool m_CompressibleLogged;
};

//! Kernel timing for development (environment variable OPENEMS_HIP_KERNEL_TIMES=1): the
//! time of every launch, per kernel name, printed at exit. Waits for each kernel.
struct HIP_KernelTimes
{
	static bool Enabled();
	static void Begin(hipStream_t stream);
	static void End(hipStream_t stream, const char* name);
};

//! Launch \a kernel with one thread per (i,j,k) on the stream of \a d; the kernel checks its bounds
template <typename Kernel, typename... Args>
void HIP_Launch(GPU_Backend_HIP::Impl* d, const char* name, Kernel kernel, size_t ni, size_t nj, size_t nk, Args... args)
{
	if (ni==0 || nj==0 || nk==0)
		return;
	dim3 block = GPU_Backend_HIP::Impl::Block(ni, nj);
	const bool timed = HIP_KernelTimes::Enabled();
	if (timed)
		HIP_KernelTimes::Begin(d->Stream());
	kernel<<<GPU_Backend_HIP::Impl::Grid(block, ni, nj, nk), block, 0, d->Stream()>>>(args...);
	d->CheckLaunch(name);
	if (timed)
		HIP_KernelTimes::End(d->Stream(), name);
}

//! Parse OPENEMS_HIP_WINDOW (\a text, NULL if unset) for a grid of \a nx x lines: unset or 0 off; "1,<W>" with
//! W = nx or the string NX the degenerate window (T = 1, one window of the whole grid); anything else is
//! on but unsupported in this build (the backend then steps one timestep at a time)
GPU_Backend_HIP::Impl::WindowKnob HIP_ParseWindowKnob(const char* text, unsigned int nx);

//! Factory of a HIP extension: the device implementation of \a eng_ext, or NULL if \a eng_ext is not of its type
typedef GPU_Extension* (*HIP_ExtensionFactory)(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_Excitation(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_UPML(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_Mur_ABC(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_LorentzMaterial(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_DebyeMaterial(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_LumpedRLC(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_TFSF(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_Absorbing_BC(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_SteadyState(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* HIP_CreateExt_Cylinder(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng);

#endif // HIP_INTERNAL_H
