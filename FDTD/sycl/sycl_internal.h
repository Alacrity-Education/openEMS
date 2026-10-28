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
#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_backend_sycl.h"
#include "FDTD/gpu_coeff_sets.h"

//! Grid size (number of mesh lines) as passed to the kernels. nzd: the values stored per (x, y) line of the
//! device layout, the z stride (OPENEMS_SYCL_ZPAD, see GPU_Backend_SYCL::Impl::dim); nz <= nzd, equal without
//! padding. nx, ny, nz are the extents (bounds, last lines). A range of nodes (B, E, ...) does not use nzd.
struct SYCL_GridDim
{
	unsigned int nx, ny, nz, nzd;
};

//! The kernels index the fields with the index type I: 32 bit while the fields fit, else size_t,
//! chosen in Init() (see GPU_Backend_SYCL::Impl::index32); the helpers take it as a template
//! parameter, size_t by default.
template<typename T> struct SYCL_NoDeduce { typedef T type; };

//! Fields and coefficients use the ArrayNIJK layout of the host with the z stride nzd (OPENEMS_SYCL_ZPAD):
//! index(n,x,y,z) = n*nx*ny*nzd + x*ny*nzd + y*nzd + z; the padded values z >= nz are zero and never read for a result
template<typename I = size_t>
inline I nijk(const SYCL_GridDim& N, unsigned int n, unsigned int x, unsigned int y, unsigned int z)
{
	return (((I)n*N.nx + x)*N.ny + y)*N.nzd + z;
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

//! Main update coefficients of node i, see GPU_Backend_SYCL::Impl::coeff_mode: a[n*s] and b[n*s]
//! are vv/vi (voltages, set_offset 0) or ii/iv (currents, set_offset 6) of direction n.
//! FACTOR false (modes 0-2): pointers, read where the kernel uses them; mode 0: into the full
//! arrays ca/cb, 1/2: into the table of sets. FACTOR true (modes 3/4): the values, reconstructed
//! by main_coeff(), s is 1. A compile time choice, so that the kernels of the default modes
//! contain no factor code: the launches pick the kernel by coeff_mode>=3.
template<typename I, bool FACTOR = false> struct SYCL_MainCoeffT { const float* a; const float* b; I s; };
template<typename I> struct SYCL_MainCoeffT<I, true> { float a[3]; float b[3]; static constexpr I s = 1; };

//! Factorized coefficients (coeff_mode 3/4, OPENEMS_SYCL_COEFF=factor): the buffer ca (= cb) holds
//!   [0]                            the first exact set (uint32 bits), then padding up to SYCL_FACTOR_HDR,
//!   [SYCL_FACTOR_HDR + 2*t*L + o]  four geometry tables t of L = nx+ny+nz values, o = x, nx+y, nx+ny+z,
//!                                  each as a high float and (at + L) the low float of the double value:
//!                                  t=0 primal edge length, 1 inverse dual edge length (voltages),
//!                                  t=2 dual edge length, 3 inverse primal edge length (currents),
//!   [SYCL_FactorSets(N)]           the sets of 12, normalized (set < exact_start) or exact.
//! A normalized set holds vv, ii as they are and vi, iv divided by the geometry of the edge,
//! len_n * inv_p * inv_q (p<q the other two directions). SYCL_FactorMul() multiplies it back,
//! in the same operations on the host (GPU_Backend_SYCL::Init checks every node) and the device.
#define SYCL_FACTOR_HDR 4
inline size_t SYCL_FactorSets(const SYCL_GridDim& N)
{
	return SYCL_FACTOR_HDR + 8*((size_t)N.nx+N.ny+N.nz);
}

//! r*(a*b*c) for the double values a, b, c given as high and low floats: the products in
//! double-float (Dekker with fma), so that the result is r times the double geometry, rounded
//! once or nearly (not the three roundings of a float product of the rounded factors)
inline float SYCL_FactorMul(float r, float ah, float al, float bh, float bl, float ch, float cl)
{
	const float p  = ah*bh;
	const float pe = sycl::fma(ah, bh, -p) + (ah*bl + al*bh);
	const float q  = p*ch;
	const float qe = sycl::fma(p, ch, -q) + (p*cl + pe*ch);
	const float t  = r*q;
	const float te = sycl::fma(r, q, -t) + r*qe;
	return t + te;
}

//! Mode 0: ca/cb are the full arrays (vv, vi or ii, iv); 1/2: ca is the table of sets and index a
//! 16/32 bit set index per node; 3/4 (FACTOR only): the same index, ca the factorized buffer (see
//! SYCL_FACTOR_HDR). N and x, y, z are read by FACTOR only.
template<typename I, bool FACTOR>
inline SYCL_MainCoeffT<I, FACTOR> main_coeff(const void* index, const float* ca, const float* cb, unsigned int mode,
                                             typename SYCL_NoDeduce<I>::type sn, typename SYCL_NoDeduce<I>::type i,
                                             unsigned int set_offset, const SYCL_GridDim& N,
                                             unsigned int x, unsigned int y, unsigned int z)
{
	SYCL_MainCoeffT<I, FACTOR> c;
	if constexpr (!FACTOR)
	{
		(void)N; (void)x; (void)y; (void)z;
		if (mode==0)
		{
			c.a = ca + i; c.b = cb + i; c.s = sn;
			return c;
		}
		const unsigned int set = (mode==1) ? (unsigned int)((const unsigned short*)index)[i] : ((const unsigned int*)index)[i];
		c.a = ca + (size_t)12*set + set_offset; c.b = c.a + 3; c.s = 1;   // 64 bit: more than 358M sets wrap in 32
		return c;
	}
	else
	{
		(void)cb; (void)sn;
		const unsigned int set = (mode&1) ? (unsigned int)((const unsigned short*)index)[i] : ((const unsigned int*)index)[i];
		const float* T = ca + SYCL_FactorSets(N) + (size_t)12*set + set_offset;
		for (unsigned int n=0; n<3; ++n)
		{
			c.a[n] = T[n];
			c.b[n] = T[3+n];
		}
		if (set < ((const unsigned int*)ca)[0])
		{
			const unsigned int L = N.nx+N.ny+N.nz;
			const float* len = ca + SYCL_FACTOR_HDR + (set_offset ? 4 : 0)*(size_t)L;   // high, low
			const float* inv = len + 2*L;
			const unsigned int ox = x, oy = N.nx+y, oz = N.nx+N.ny+z;
			const float lxh = len[ox], lxl = len[L+ox], lyh = len[oy], lyl = len[L+oy], lzh = len[oz], lzl = len[L+oz];
			const float wxh = inv[ox], wxl = inv[L+ox], wyh = inv[oy], wyl = inv[L+oy], wzh = inv[oz], wzl = inv[L+oz];
			c.b[0] = SYCL_FactorMul(c.b[0], lxh, lxl, wyh, wyl, wzh, wzl);
			c.b[1] = SYCL_FactorMul(c.b[1], lyh, lyl, wxh, wxl, wzh, wzl);
			c.b[2] = SYCL_FactorMul(c.b[2], lzh, lzl, wxh, wxl, wyh, wyl);
		}
		return c;
	}
}

//! The main voltage update of component \a n of node \a i from its old value \a old: the statements of
//! update_voltages, for the kernels of the Mur modes (see GPU_Backend_SYCL::Impl::mur_mode), which
//! must reproduce its results bit for bit. xm, ym, zm: the shifts to the previous line (0 on the first).
template<typename I, bool FACTOR>
inline float volt_update_node(unsigned int n, float old, const float* curr, const SYCL_MainCoeffT<I, FACTOR>& C,
                              I sn, I i, I xm, I ym, I zm)
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
template<typename I, bool FACTOR>
inline float volt_update_at(unsigned int n, float old, const float* curr, const void* index, const float* ca,
                            const float* cb, unsigned int mode, const SYCL_GridDim& N,
                            unsigned int x, unsigned int y, unsigned int z)
{
	const I sn = (I)N.nx*N.ny*N.nzd;
	const I i  = nijk<I>(N, 0, x, y, z);
	const I xm = (x>0) ? (I)N.ny*N.nzd : 0;
	const I ym = (y>0) ? N.nzd : 0;
	const I zm = (z>0) ? 1 : 0;
	return volt_update_node<I, FACTOR>(n, old, curr, main_coeff<I, FACTOR>(index, ca, cb, mode, sn, i, 0, N, x, y, z), sn, i, xm, ym, zm);
}

//! Mur planes of Mur mode 1 (mur_shell), face f = 2*ny + (line>0); the kernel reads them from device
//! memory (GPU_Backend_SYCL::Impl::mur_dev: a by-value array indexed at run time would be copied to
//! private memory by every work-item), the active bits are an argument
struct SYCL_MurFace
{
	const float* c_nyP;    //!< coefficients of component nyP, at (i,j) = i*numLines[nyPP] + j
	const float* c_nyPP;   //!< coefficients of component nyPP
	unsigned int line, shift;
	unsigned int rank;     //!< position in the apply order of the Mur planes: the higher one is applied later
};
struct SYCL_MurFaces
{
	SYCL_MurFace f[6];
	unsigned int active;   //!< bit f: face f has a Mur plane that is active in this timestep
};

//! Nodes of Mur mode 1: update_voltages_inner updates the main nodes [B, E) inside [L, H), mur_shell the others
struct SYCL_MurShell
{
	SYCL_GridDim B, E, L, H;
};

//! The z faces of Mur mode 2: the interior of each (x and y inside [SYCL_MurShell::L, H)) is updated by
//! update_voltages_inner (mur_zface_lanes()), which exchanges the values of the Mur line and the shift line of a
//! z line through sub-group shifts. Index 0 the z-low, 1 the z-high plane
struct SYCL_MurZFaces
{
	const float* c_x[2];   //!< coefficients of component x (the plane's nyP), at x*ny + y
	const float* c_y[2];   //!< coefficients of component y (the plane's nyPP), at x*ny + y
	unsigned int line[2], shift[2];
	unsigned int active;   //!< bit f: plane f exists and is active in this timestep
};

//! A Mur plane of the grid, registered by SYCL_Ext_Mur_ABC
struct SYCL_MurPlane
{
	int ny;
	unsigned int line, shift;
	const float *c_nyP, *c_nyPP;
	size_t seq;   //!< position of its extension in the extension order (see GPU_Backend_SYCL::Impl::ext_log)
	unsigned int start_ts;   //!< the first timestep the plane is active in (see GPU_Backend_SYCL::Impl::MurActive())
};

//! An executable SYCL command graph (sycl_ext_oneapi_graph), see GPU_Backend_SYCL::Impl::window_graphs
typedef sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable> SYCL_ExecGraph;

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
	size_t numCells;    //!< nx*ny*nz, the host count of cells
	//! nx*ny*nzd, the cells of the device layout: the field buffers hold 3*numCellsD values, the full coefficient
	//! arrays and the set index of the main updates too (OPENEMS_SYCL_ZPAD, see Init(); numCells without padding)
	size_t numCellsD;
	//! the device flat index of host flat index \a h (ArrayNIJK of the host, component included): the z lines
	//! keep their order, each stored with the stride dim.nzd
	size_t Dev(size_t h) const {return (h/dim.nz)*dim.nzd + h%dim.nz;}
	//! \a host (\a count values in the host layout, whole z lines) in the device layout, the padding zero
	template <typename T>
	std::vector<T> Padded(const T* host, size_t count) const
	{
		std::vector<T> dev(count/dim.nz*dim.nzd, T(0));
		for (size_t l=0; l<count/dim.nz; ++l)
			std::copy(host + l*dim.nz, host + (l+1)*dim.nz, dev.begin() + l*dim.nzd);
		return dev;
	}

	float *volt, *curr;
	float2* energy;                      //!< per-line energy sums, see GPU_Backend_SYCL::CalcFastEnergy()
	size_t energy_count;
	std::vector<float2> energy_host;
	//! Main update coefficients: 0: full arrays vv, vi, ii, iv; 1/2: a 16/32 bit set index per node and the sets;
	//! 3/4: a 16/32 bit set index and the factorized sets and geometry tables in coeff (see main_coeff())
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
	//! flat indices of the voltages the extensions register as changed between the half-steps
	//! (excitation, lumped RLC, Mur ABC, absorbing sheets, TF/SF, Lorentz and Debye materials)
	std::vector<size_t> volt_modified;
	//! flat indices of the currents the extensions change after the current update (TF/SF,
	//! absorbing sheets, Lorentz materials): none may be on the last mesh lines, which the fused step
	//! does not write (see DecideFusedStep())
	std::vector<size_t> curr_modified;
	size_t* fixup;                           //!< main nodes whose currents are recomputed after the voltage extensions
	unsigned int fixup_count;
	//! extensions the fused step handles with a large fix-up (Mur ABC, absorbing sheets, TF/SF,
	//! Lorentz and Debye materials): fused only with OPENEMS_SYCL_FUSED_STEP=1 (see DecideFusedStep())
	int fused_optin;
	bool index32;                            //!< the kernels index with 32 bit (see Init())
	//! Decide once whether the fused step is used, and prepare it; it keeps the UPML z slabs
	//! (zslab_lo, zslab_hi) in its kernel or leaves them to the region kernels (clears the flags)
	bool DecideFusedStep();

	//! Extensions in the order the engine created them (its extension order), see CreateExtension(): the
	//! voltages they register in volt_modified are [vm_begin, vm_end), the nodes they register in
	//! ext_nodes [nodes_begin, nodes_end)
	struct ExtRecord { Engine_Extension* eng_ext; size_t vm_begin, vm_end, nodes_begin, nodes_end; };
	std::vector<ExtRecord> ext_log;
	//! Flat node indices (x*ny*nz + y*nz + z) an extension reads or writes the voltages of at its hooks, registered
	//! by its constructor (the dispersive materials and sheets), for the check of MurRefusal()
	std::vector<size_t> ext_nodes;
	//! The Mur mode in use, -1 until DecideMurFused() (the first voltage update), which takes the highest one the grid
	//! allows: 0 the three launches per Mur plane, 1 the Mur planes in the voltage update (update_voltages_inner and
	//! mur_shell, see mur_volt_component), 2 as 1 with the interior of the z faces in update_voltages_inner
	//! (mur_zface_lanes()); the Mur hooks of a timestep that started before the decision finish it as mode 0
	int mur_mode;
	std::vector<SYCL_MurPlane> mur_planes;
	SYCL_MurFaces mur;                        //!< the planes of mode 1, active bits set by the Mur hooks
	SYCL_MurFace* mur_dev;                    //!< device copy of mur.f (the kernel indexes it)
	SYCL_MurShell mur_shell;                  //!< the nodes of the two kernels of mode 1
	size_t mur_shell_count;                   //!< number of nodes of mur_shell
	bool mur_released;                        //!< a Mur extension was deleted (host fallback): mode 0
	bool mur_zf_on;                           //!< mode 1 with the z faces of Mur mode 2
	SYCL_MurZFaces mur_zf;                    //!< the z planes of mode 2 (active bits set at the launch)
	unsigned int mur_zf_sg;                   //!< the sub-group size of update_voltages_inner in mode 2 (16 or 32)
	//! Decide once which Mur mode is used (the highest one the grid allows), log it
	void DecideMurFused();
	//! Why mode 1 cannot be used on this grid, empty if it can
	std::string MurRefusal() const;
	//! Why the z faces of mode 2 cannot run in update_voltages_inner, empty if they can (call after MurRefusal())
	std::string MurZFaceRefusal();
	//! The faces (bits as SYCL_MurFaces::active) of the Mur planes active at timestep \a ts (their start timesteps)
	unsigned int MurActive(unsigned int ts) const;
	//! The first timestep after \a ts at which the active Mur planes change, UINT_MAX if never
	unsigned int MurNextChange(unsigned int ts) const;

	//! OPENEMS_SYCL_WINDOW (read in Init(), see SYCL_ParseWindowKnob()): windowed blocks of T timesteps and windows of W x planes
	struct WindowKnob
	{
		bool on;              //!< set and not 0
		bool valid;           //!< auto, rule, or T,W with T, W > 0
		bool automatic;       //!< auto or rule: T = 8 and W from the L2 rule (see SYCL_WindowRule())
		bool tune;            //!< auto: the tuner times candidates on the first timesteps (see WindowTune()); rule: not
		unsigned int T, W;    //!< forced; W: the grid's nx for "NX"
		std::string text;     //!< the value as set
	};
	WindowKnob window;
	//! the windowed blocks run (see GPU_Backend_SYCL::SetWindowExtensions()), with these extensions in the engine's order
	bool window_used;
	std::vector<GPU_Extension*> window_exts;
	//! the timestep of the windowed block that runs (one int, device memory): the base the window-capable
	//! extensions read their step from (see GPU_Extension::LaunchInWindow()); allocated in Init()
	int* step_base;
	//! WindowPrepare() ran: 1 the schedule runs, 0 refused at the first block, -1 not yet
	int window_ready;
	//! the schedule: blocks of at most win_T timesteps, windows of win_W x planes
	unsigned int win_T, win_W;
	//! V frontiers that would split the line and the shift line of a Mur plane on an x face (see WindowFrontiers())
	std::vector<unsigned int> win_forbidden;
	size_t l2_bytes;                //!< L2 cache of the device (Init())
	size_t coeff_table_bytes;       //!< the coefficient table shared by all nodes (sets or factorized), 0 with full arrays
	//! bytes per node of the main update coefficients the kernels read besides the table: the set index, or the
	//! four full arrays (see coeff_mode); for the window rule
	size_t CoeffBytesPerNode() const {return (coeff_mode==0) ? 48 : (((coeff_mode==2) || (coeff_mode==4)) ? 4 : 2);}
	unsigned long long launches;    //!< kernel launches issued by SYCL_Launch() (recorded ones included)
	//! graphs: one per (block length, W, active Mur planes), recorded at the first block of its kind and replayed
	struct WindowGraph { std::shared_ptr<SYCL_ExecGraph> exec; unsigned long long launches; double capture_ms; };
	std::map<std::vector<unsigned int>, WindowGraph> window_graphs;
	bool window_graphs_on;          //!< not with OPENEMS_SYCL_KERNEL_TIMES, without graph support, or after a failed recording
	bool window_logged;             //!< the schedule's line is printed (after the first chunk)
	unsigned long long window_steps, window_blocks, window_chunks;
	unsigned long long window_block_launches;   //!< the launches of the last block of T steps launched without a graph
	//! where win_T, win_W come from, for the schedule's line: forced, rule, tuned, untuned
	std::string win_source;
	//! The main voltage / current update of the nodes [x0,x1) x [y0,y1) (all z) for the windowed blocks:
	//! the separate kernels of UpdateVoltages()/UpdateCurrents() on the main range cut to the window, the Mur
	//! planes of \a mur_active in the voltage update as UpdateVoltages() has them (mode 1 and 2)
	void WindowVoltages(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1, unsigned int mur_active);
	void WindowCurrents(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1);
	//! The decisions of the first timestep (UPML, Mur), the schedule's T and W; false: refused
	bool WindowPrepare();
	//! The frontiers of the window whose V frontier of step 0 is \a F, for a block of \a L timesteps
	void WindowFrontiers(unsigned int F, unsigned int L, std::vector<unsigned int>& eV, std::vector<unsigned int>& eI) const;
	//! The launches of a block of \a L timesteps from timestep \a ts with windows of \a W planes (the step base is
	//! written before); \a host: the extension whose HostStep() runs after its apply-voltage launch (L = 1, W = nx)
	void WindowBlock(unsigned int ts, unsigned int L, unsigned int W, GPU_Extension* host);
	//! n timesteps from \a start_ts as blocks (graphs when possible)
	void WindowRun(unsigned int start_ts, unsigned int n);
	//! Destroy the graphs of every W but \a keep_W (UINT_MAX: all)
	void WindowDropGraphs(unsigned int keep_W);
	//! WindowRun() with the current win_T, win_W, timed by the host clock between two waits for the queue, in ms
	double WindowTimedRun(unsigned int start_ts, unsigned int n);
	//! The tuner of auto: candidates (T, W) timed on the live grid, on real timesteps (every candidate is
	//! bit-identical, so the tune does not change the result); mcs < 0: not measured yet
	struct WindowCandidate { unsigned int T, W; double mcs; };
	std::vector<WindowCandidate> tune_cands;
	int tune_state;                 //!< 0 not tuning (forced, rule or done), 1 the candidate list, 2 the refinement
	unsigned int tune_chunks;       //!< chunks the tune has run in so far
	unsigned int tune_rule_T;       //!< T and W of the rule (SYCL_WindowRuleTW()), the schedule while nothing is measured
	unsigned int tune_rule_W;
	unsigned long long tune_steps;  //!< timesteps the candidates ran (warm-up and timed blocks)
	double tune_ms;                 //!< time of the timed blocks
	size_t tune_plane_bytes;        //!< plane bytes of the window rule (WindowPrepare())
	//! The tuner's candidate list; in WindowPrepare() for auto
	void WindowTunePrepare();
	//! Run the tuner's candidates that fit into the chunk [start_ts, start_ts+n), each one warm-up block and two
	//! timed blocks of its T; returns the timesteps used, the rest of the chunk runs with the best (T, W) so far
	unsigned int WindowTune(unsigned int start_ts, unsigned int n);
	//! The end of the tune: the best measured candidate (or the rule), logged
	void WindowTuneFinish(bool complete);

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

	//! Device buffer of the set indices of \a sets (16 or 32 bit); \a padded: the index of the main updates,
	//! one per node of the device layout (see numCellsD)
	void* AllocIndex(const GPU_CoeffSets& sets, bool padded=false)
	{
		if (sets.mode==1)
		{
			std::vector<unsigned short> index16(sets.index.begin(), sets.index.end());
			if (padded)
				index16 = Padded(index16.data(), index16.size());
			return Alloc<unsigned short>(index16.size(), index16.data());
		}
		if (padded)
		{
			const std::vector<unsigned int> index32 = Padded(sets.index.data(), sets.index.size());
			return Alloc<unsigned int>(index32.size(), index32.data());
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
	//! Work-group of the main kernels (update_voltages, update_voltages_inner, update_currents), TZ x TY work-items:
	//! OPENEMS_SYCL_SHAPE, read in Init(): shape_mode 0 Block(), 2 auto (default: timed at the first voltage update,
	//! see DecideShape(), which sets 1 and the best shape_tz x shape_ty), 1 forced
	int shape_mode;
	unsigned int shape_tz, shape_ty;
	bool shape_decided, shape_forced;
	SYCL_Dim3 MainBlock(size_t ni, size_t nj) const {return (shape_mode==1) ? SYCL_Dim3{shape_tz, shape_ty, 1} : Block(ni, nj);}
	//! Decide the work-group of the main kernels once (log it); auto: time the candidates on copies of the fields
	void DecideShape();
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
void SYCL_Launch(GPU_Backend_SYCL::Impl* d, const char* name, size_t ni, size_t nj, size_t nk, const Kernel& kernel, bool main=false)
{
	if (ni==0 || nj==0 || nk==0)
		return;
	const SYCL_Dim3 block = main ? d->MainBlock(ni, nj) : GPU_Backend_SYCL::Impl::Block(ni, nj);
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
		++d->launches;
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

//! Parse OPENEMS_SYCL_WINDOW (\a text, NULL if unset: auto): 0 off; auto (the rule, then the tuner); rule (the rule
//! alone); T,W with T, W > 0 (W may be the string NX: the grid's nx). Anything else is on but not valid (the backend
//! then steps one timestep at a time)
GPU_Backend_SYCL::Impl::WindowKnob SYCL_ParseWindowKnob(const char* text, unsigned int nx);
//! The window rule: W = (margin * L2 - table) / plane_bytes - halo, at least 1, at most nx; the coefficient table (read
//! by every window) counts when it is below a third of that budget, a larger one cannot stay in L2 anyway. halo 0
//! means the schedule's halo of T + 1 planes (the frontiers step back one plane per timestep)
unsigned int SYCL_WindowRule(size_t l2_bytes, size_t plane_bytes, unsigned int T, unsigned int nx, size_t table_bytes,
                             double margin = 0.65, unsigned int halo = 0);
//! The rule of auto and rule: SYCL_WindowRule() at 0.65 of the L2 for \a T (8); where that leaves fewer than T planes a
//! window, T steps down to 4, then to 2 (on a small L2 the halo of T = 8 alone takes most of the budget). Returns W and
//! leaves the T it settled on in \a T
unsigned int SYCL_WindowRuleTW(size_t l2_bytes, size_t plane_bytes, unsigned int& T, unsigned int nx, size_t table_bytes);

//! What the windowed schedule needs of a window-capable SYCL extension besides GPU_Extension::LaunchInWindow()
class SYCL_WindowHooks
{
public:
	virtual ~SYCL_WindowHooks() {}
	//! The decisions the extension makes in its first hook, before the first block (nothing may wait in a recording)
	virtual void WindowPrepare() {}
	//! The first timestep >= \a ts at which HostStep() must run, UINT_MAX if none: the schedule runs that timestep alone,
	//! over the whole grid, and calls HostStep() right after this extension's apply-voltage launch (where the single
	//! step does its host work)
	virtual unsigned int NextHostStep(unsigned int ts) const {(void)ts; return UINT_MAX;}
	virtual void HostStep(unsigned int ts) {(void)ts;}
};

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
