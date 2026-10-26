/*
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include <climits>
#include <cstdint>
#include <cstring>

#include "sycl_internal.h"
#include "FDTD/gpu_setup_model.h"
#include "FDTD/gpu_setup_kernels.h"

// The material pass of the operator build on the device (see GPU_Backend::SetupOperator() and
// gpu_setup_kernels.h): the algorithm of the reference backend (gpu_backend_reference.cpp,
// SetupOperator), one work-item where the reference has one loop iteration, calling the functions
// of the shared header. The arithmetic is that header compiled with the plugin's flags (no
// contraction, no fast math), in double; only the float stores round, as on the CPU.
//
// Kernels, all on the in-order compute queue, plain 1D launches (sycl::nd_range<1>, the global
// size rounded up to the work-group, every kernel checks its bound):
//   setup_columns_count  one work-item per column (x, y), col = x*ny + y: the length of the material
//                        and the edge list of the column (the filters InsideBox and MayContain of the
//                        reference loop, on the lists of the pass in their order)
//   (host)               exclusive prefix sums of the two counts: the CSR offsets
//   setup_columns_fill   the same loop again, writing the entries in list order (never sorted)
//   setup_samples        per n, one work-item per cell, z fastest: c = (x*ny + y)*nz + z, so that
//                        consecutive work-items share the column list; QuarterCell and EdgeValues,
//                        the float stores into the per-n arrays at x + nx*(y + ny*z)
//   setup_edges          per n, the same mapping: YeeCoords and FirstHit on the column's edge list,
//                        the index into edgeList (or -1) at the first-hit table's (x*ny + y)*nz + z
// The used flags are byte stores of 1 (every writer stores the same value). A NaN or Inf effective
// value records the smallest linear index ((x*ny + y)*nz + z)*3 + n with an atomic minimum, which is
// the first edge in the loop order (x, y, z, n) of the reference backend; the host recomputes that
// one edge with the header to find the parameter and fills GPU_SetupResult::nan*, from which the
// operator throws the message of the CPU path.

namespace
{

const size_t SETUP_WG = 128;   // work-group size of every setup kernel

//! The device buffers of the pass, freed when the pass ends, also on an exception
struct SetupBuffers
{
	sycl::queue& q;
	std::vector<void*> ptrs;
	size_t bytes;

	SetupBuffers(sycl::queue& queue) : q(queue), bytes(0) {}
	~SetupBuffers()
	{
		try
		{
			q.wait();
		}
		catch (...)
		{
		}
		for (void* p : ptrs)
		{
			try
			{
				sycl::free(p, q);
			}
			catch (...)
			{
			}
		}
	}

	//! Device buffer of \a count elements, a copy of \a host when given (as GPU_Backend_SYCL::Impl::Alloc)
	template <typename T>
	T* Alloc(size_t count, const T* host=NULL)
	{
		const size_t n = std::max(count, (size_t)1);
		T* ptr = NULL;
		SYCL_Check("malloc_device", [&] {ptr = sycl::malloc_device<T>(n, q);});
		if (ptr==NULL)
			throw std::runtime_error("GPU_Backend_SYCL: setup: malloc_device failed: not enough device memory for "
			                         + std::to_string(n*sizeof(T)) + " bytes");
		ptrs.push_back(ptr);
		bytes += n*sizeof(T);
		if (host && count)
			SYCL_Check("memcpy", [&] {q.memcpy(ptr, host, count*sizeof(T));});
		return ptr;
	}
};

//! One plain 1D launch of \a count work-items on \a q, timed per kernel name with OPENEMS_SYCL_KERNEL_TIMES=1
template <typename Kernel>
void SetupLaunch(GPU_Backend_SYCL::Impl* d, const char* name, size_t count, const Kernel& kernel)
{
	if (count==0)
		return;
	const size_t global = ((count + SETUP_WG - 1)/SETUP_WG)*SETUP_WG;
	const bool timed = SYCL_KernelTimes::Enabled();
	std::chrono::steady_clock::time_point start;
	if (timed)
	{
		d->Flush();
		start = std::chrono::steady_clock::now();
	}
	try
	{
		d->Queue().parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(SETUP_WG)), kernel);
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

//! The column lists of column (x, y), as the reference backend builds them: \a out (if not NULL)
//! receives the entries, \a outIdx (if not NULL) the positions in \a list; returns the count
inline int ColumnList(const GPU_SetupView& v, unsigned int x, unsigned int y, const int* list, int listLen, int* out, int* outIdx)
{
	double boundBox[6], queryBox[6];
	GPU_Setup::ColumnBox(v, x, y, boundBox);
	GPU_Setup::QueryBox(v, x, y, queryBox);
	int count = 0;
	for (int j=0; j<listLen; ++j)
		if ((GPU_Setup::InsideBox(v.primitives[list[j]], boundBox)>=0) && GPU_Setup::MayContain(v.primitives[list[j]], queryBox, v.tol))
		{
			if (out)
				out[count] = list[j];
			if (outIdx)
				outIdx[count] = j;
			++count;
		}
	return count;
}

} // namespace

bool GPU_Backend_SYCL::SetupOperator(const GPU_SetupModel& model, GPU_SetupResult& result)
{
	sycl::queue& q = d->Queue();
	const sycl::device& dev = d->ctx->device;
	if (!dev.has(sycl::aspect::fp64))
		return false;

	const unsigned int nx = model.numLines[0], ny = model.numLines[1], nz = model.numLines[2];
	const size_t cells = (size_t)nx*ny*nz;
	const size_t columns = (size_t)nx*ny;
	// the kernels index with 32 bit, and the NaN index covers 3*cells
	if ((cells==0) || (3*cells>=(size_t)INT_MAX) || (model.matList.size()>=(size_t)INT_MAX) || (model.edgeList.size()>=(size_t)INT_MAX))
		return false;

	int nStorage = 0;
	for (int k=0; k<4; ++k)
		if (result.storage[k])
			++nStorage;

	// the memory plan: the model, the counts and offsets, the result arrays (one buffer per array and
	// direction), the first-hit table (one per direction), the used flags; the column lists are added
	// below when their size is known
	const size_t cellBytes = cells*sizeof(float);
	const size_t modelBytes = ((size_t)nx+ny+nz)*sizeof(double) + model.properties.size()*sizeof(GPU_SetupProperty)
	                          + model.primitives.size()*sizeof(GPU_SetupPrimitive) + model.vertices.size()*sizeof(double)
	                          + (model.matList.size() + model.edgeList.size())*sizeof(int);
	const size_t fixedBytes = modelBytes + 4*(columns+1)*sizeof(int) + (12 + 3*(size_t)nStorage)*cellBytes + 3*cells*sizeof(int32_t)
	                          + model.primitives.size() + sizeof(int);
	if (cellBytes>d->ctx->max_mem_alloc_size)
		return false;
	// the free device memory where the driver reports it (sycl::aspect::ext_intel_free_memory), else the global memory size; 10 % margin
	unsigned long long avail = dev.get_info<sycl::info::device::global_mem_size>();
	try
	{
		if (dev.has(sycl::aspect::ext_intel_free_memory))
			avail = dev.get_info<sycl::ext::intel::info::device::free_memory>();
	}
	catch (const sycl::exception&)
	{
	}
	avail -= avail/10;
	if (fixedBytes>avail)
		return false;

	SetupBuffers buf(q);

	// the model
	GPU_SetupView v;
	for (int n=0; n<3; ++n)
	{
		v.numLines[n] = model.numLines[n];
		v.lines[n] = buf.Alloc<double>(model.lines[n].size(), model.lines[n].data());
	}
	v.gridDelta = model.gridDelta;
	v.tol = model.tol;
	for (int k=0; k<4; ++k)
		v.bg[k] = model.bg[k];
	v.properties = buf.Alloc<GPU_SetupProperty>(model.properties.size(), model.properties.data());
	v.primitives = buf.Alloc<GPU_SetupPrimitive>(model.primitives.size(), model.primitives.data());
	v.vertices = buf.Alloc<double>(model.vertices.size(), model.vertices.data());
	v.matList = buf.Alloc<int>(model.matList.size(), model.matList.data());
	v.matCount = (int)model.matList.size();
	v.edgeList = buf.Alloc<int>(model.edgeList.size(), model.edgeList.data());
	v.edgeCount = (int)model.edgeList.size();

	// (a) the column lists: counts, offsets, entries
	int* matCount = buf.Alloc<int>(columns);
	int* edgeCount = buf.Alloc<int>(columns);
	SetupLaunch(d, "setup_columns_count", columns, [=](sycl::nd_item<1> it)
	{
		const size_t col = it.get_global_id(0);
		if (col>=columns)
			return;
		const unsigned int x = (unsigned int)col/ny, y = (unsigned int)col%ny;
		matCount[col] = ColumnList(v, x, y, v.matList, v.matCount, NULL, NULL);
		edgeCount[col] = ColumnList(v, x, y, v.edgeList, v.edgeCount, NULL, NULL);
	});
	std::vector<int> hostCount[2];
	hostCount[0].resize(columns);
	hostCount[1].resize(columns);
	SYCL_Check("memcpy", [&] {q.memcpy(hostCount[0].data(), matCount, columns*sizeof(int));});
	SYCL_Check("memcpy", [&] {q.memcpy(hostCount[1].data(), edgeCount, columns*sizeof(int));});
	d->Flush();
	std::vector<int> hostOffset[2];
	size_t total[2];
	for (int l=0; l<2; ++l)
	{
		hostOffset[l].resize(columns+1);
		size_t sum = 0;
		for (size_t col=0; col<columns; ++col)
		{
			hostOffset[l][col] = (int)sum;
			sum += hostCount[l][col];
			if (sum>=(size_t)INT_MAX)
				return false;
		}
		hostOffset[l][columns] = (int)sum;
		total[l] = sum;
	}
	if (fixedBytes + (total[0] + 2*total[1])*sizeof(int)>avail)
		return false;
	const int* matOffset = buf.Alloc<int>(columns+1, hostOffset[0].data());
	const int* edgeOffset = buf.Alloc<int>(columns+1, hostOffset[1].data());
	int* matPrim = buf.Alloc<int>(total[0]);
	int* edgePrim = buf.Alloc<int>(total[1]);
	int* edgeIdx = buf.Alloc<int>(total[1]);
	SetupLaunch(d, "setup_columns_fill", columns, [=](sycl::nd_item<1> it)
	{
		const size_t col = it.get_global_id(0);
		if (col>=columns)
			return;
		const unsigned int x = (unsigned int)col/ny, y = (unsigned int)col%ny;
		ColumnList(v, x, y, v.matList, v.matCount, matPrim + matOffset[col], NULL);
		ColumnList(v, x, y, v.edgeList, v.edgeCount, edgePrim + edgeOffset[col], edgeIdx + edgeOffset[col]);
	});

	// the results
	uint8_t* used = buf.Alloc<uint8_t>(model.primitives.size());
	SYCL_Check("memset", [&] {q.memset(used, 0, std::max(model.primitives.size(), (size_t)1));});
	int* nanIndex = buf.Alloc<int>(1);
	SYCL_Check("fill", [&] {q.fill(nanIndex, (int)INT_MAX, 1);});
	float* ec[4][3];
	float* store[4][3];
	int32_t* firstHit[3];
	for (int n=0; n<3; ++n)
	{
		for (int k=0; k<4; ++k)
		{
			ec[k][n] = buf.Alloc<float>(cells);
			store[k][n] = result.storage[k] ? buf.Alloc<float>(cells) : NULL;
		}
		firstHit[n] = buf.Alloc<int32_t>(cells);
	}

	// (b) the samples and the edge values, (c) the first hits; per direction n
	for (int n=0; n<3; ++n)
	{
		float* EC0 = ec[0][n];
		float* EC1 = ec[1][n];
		float* EC2 = ec[2][n];
		float* EC3 = ec[3][n];
		float* S0 = store[0][n];
		float* S1 = store[1][n];
		float* S2 = store[2][n];
		float* S3 = store[3][n];
		SetupLaunch(d, "setup_samples", cells, [=](sycl::nd_item<1> it)
		{
			const size_t gid = it.get_global_id(0);
			if (gid>=cells)
				return;
			const unsigned int i = (unsigned int)gid;
			const unsigned int z = i%nz;
			const unsigned int col = i/nz;
			unsigned int pos[3] = {col/ny, col%ny, z};
			const int off = matOffset[col];
			double EffMat[4];
			int badK = 0;
			if (GPU_Setup::QuarterCell(v, n, pos, EffMat, matPrim + off, matOffset[col+1] - off, used, &badK)==false)
			{
				sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> a(*nanIndex);
				a.fetch_min((int)(3*i + n));
				return;
			}
			const unsigned int c = pos[0] + nx*(pos[1] + ny*z);
			if (S0) S0[c] = EffMat[0];
			if (S1) S1[c] = EffMat[1];
			if (S2) S2[c] = EffMat[2];
			if (S3) S3[c] = EffMat[3];
			double E[4];
			GPU_Setup::EdgeValues(v, n, pos, EffMat, E);
			EC0[c] = E[0];
			EC1[c] = E[1];
			EC2[c] = E[2];
			EC3[c] = E[3];
		});
		int32_t* FH = firstHit[n];
		SetupLaunch(d, "setup_edges", cells, [=](sycl::nd_item<1> it)
		{
			const size_t gid = it.get_global_id(0);
			if (gid>=cells)
				return;
			const unsigned int i = (unsigned int)gid;
			const unsigned int z = i%nz;
			const unsigned int col = i/nz;
			unsigned int pos[3] = {col/ny, col%ny, z};
			double coord[3];
			GPU_Setup::YeeCoords(v, n, pos, coord, false);
			const int off = edgeOffset[col];
			const int h = GPU_Setup::FirstHit(v, coord, edgePrim + off, edgeOffset[col+1] - off);
			int32_t hit = -1;
			if (h>=0)
			{
				hit = edgeIdx[off + h];
				used[edgePrim[off + h]] = 1;
			}
			FH[i] = hit;
		});
	}

	// the downloads, one wait
	int hostNan = INT_MAX;
	for (int n=0; n<3; ++n)
	{
		for (int k=0; k<4; ++k)
		{
			float* dst = result.EC[k] + n*cells;
			const float* src = ec[k][n];
			SYCL_Check("memcpy", [&] {q.memcpy(dst, src, cellBytes);});
			if (result.storage[k])
			{
				float* sdst = result.storage[k] + n*cells;
				const float* ssrc = store[k][n];
				SYCL_Check("memcpy", [&] {q.memcpy(sdst, ssrc, cellBytes);});
			}
		}
		int32_t* fdst = result.firstHit + n*cells;
		const int32_t* fsrc = firstHit[n];
		SYCL_Check("memcpy", [&] {q.memcpy(fdst, fsrc, cells*sizeof(int32_t));});
	}
	if (model.primitives.size())
		SYCL_Check("memcpy", [&] {q.memcpy(result.used, used, model.primitives.size());});
	SYCL_Check("memcpy", [&] {q.memcpy(&hostNan, nanIndex, sizeof(int));});
	d->Flush();

	result.nanCount = 0;
	result.nanN = result.nanK = 0;
	result.nanPos[0] = result.nanPos[1] = result.nanPos[2] = 0;
	if (hostNan!=INT_MAX)
	{
		// the first edge in the loop order (x, y, z, n): recompute it on the host with the header for the parameter index
		const unsigned int n = (unsigned int)hostNan%3;
		const unsigned int i = (unsigned int)hostNan/3;
		const unsigned int col = i/nz;
		unsigned int pos[3] = {col/ny, col%ny, i%nz};
		GPU_SetupView hv = v;
		for (int l=0; l<3; ++l)
			hv.lines[l] = model.lines[l].data();
		hv.properties = model.properties.data();
		hv.primitives = model.primitives.data();
		hv.vertices = model.vertices.data();
		hv.matList = model.matList.data();
		hv.edgeList = model.edgeList.data();
		std::vector<int> list(model.matList.size() + 1);
		const int len = ColumnList(hv, pos[0], pos[1], hv.matList, hv.matCount, list.data(), NULL);
		std::vector<uint8_t> scratch(model.primitives.size() + 1, 0);
		double EffMat[4];
		int badK = 0;
		GPU_Setup::QuarterCell(hv, n, pos, EffMat, list.data(), len, scratch.data(), &badK);
		result.nanCount = 1;
		result.nanN = n;
		result.nanK = badK;
		for (int k=0; k<3; ++k)
			result.nanPos[k] = pos[k];
	}
	return true;
}
