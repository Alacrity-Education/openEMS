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

// Compressible device memory (OPENEMS_HIP_COMPRESSIBLE): buffers allocated through the virtual
// memory API with generic compression requested. Since Ampere the L2 of an NVIDIA GPU stores
// lines of equal values in compressed form in such buffers, which saves DRAM traffic; plain
// hipMalloc memory is never compressed. Lossless and transparent to the kernels and copies.
// The AMD runtime does not honour the flag: the property read back says so, and the caller
// falls back to hipMalloc.

#include "hip/hip_runtime.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "hip_internal.h"

namespace
{
//! CU_MEM_ALLOCATION_COMP_GENERIC of the CUDA driver API, for which HIP has no name
const unsigned char HIP_COMPRESSION_GENERIC = 1;

//! A mapped compressible buffer: padded size and the handle of its physical memory
struct HIP_Mapping
{
	size_t size;
	hipMemGenericAllocationHandle_t handle;
};

//! The engine allocates and frees on its host thread, but several engines may run in one
//! process (threads of a script): the table and the granularities are guarded
std::mutex& Lock()
{
	static std::mutex m;
	return m;
}

std::map<void*, HIP_Mapping>& Mappings()
{
	static std::map<void*, HIP_Mapping> m;
	return m;
}

hipMemAllocationProp CompressibleProp(int device)
{
	hipMemAllocationProp prop;
	memset(&prop, 0, sizeof(prop));
	prop.type = hipMemAllocationTypePinned;
	prop.location.type = hipMemLocationTypeDevice;
	prop.location.id = device;
	prop.allocFlags.compressionType = HIP_COMPRESSION_GENERIC;
	return prop;
}

unsigned ParseClasses(const char* env)
{
	const unsigned all = (1u<<HIP_AllocOther) | (1u<<HIP_AllocFields) | (1u<<HIP_AllocCoeff);
	if (env==NULL)
		return 0;
	const std::string s = env;
	if (s.empty() || (s=="0"))
		return 0;
	if ((s=="1") || (s=="all"))
		return all;
	unsigned classes = 0;
	size_t start = 0;
	while (start<=s.size())
	{
		size_t end = s.find(',', start);
		if (end==std::string::npos)
			end = s.size();
		const std::string name = s.substr(start, end-start);
		start = end+1;
		if (name=="fields")
			classes |= 1u<<HIP_AllocFields;
		else if (name=="coeff")
			classes |= 1u<<HIP_AllocCoeff;
		else
			std::cerr << "GPU_Backend_HIP: OPENEMS_HIP_COMPRESSIBLE: unknown class \"" << name
			          << "\" ignored (1 or all, or a list of fields, coeff)" << std::endl;
	}
	return classes;
}
}

unsigned HIP_CompressibleClasses()
{
	static const unsigned classes = ParseClasses(getenv("OPENEMS_HIP_COMPRESSIBLE"));
	return classes;
}

size_t HIP_CompressibleGranularity(int device)
{
	static std::map<int, size_t> granularity;
	std::lock_guard<std::mutex> guard(Lock());
	std::map<int, size_t>::const_iterator it = granularity.find(device);
	if (it!=granularity.end())
		return it->second;
	const hipMemAllocationProp prop = CompressibleProp(device);
	size_t g = 0;
	if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum)!=hipSuccess)
	{
		(void)hipGetLastError();
		g = 0;
	}
	granularity[device] = g;
	return g;
}

void* HIP_CompressibleAlloc(int device, size_t bytes, bool& granted)
{
	granted = false;
	const size_t g = HIP_CompressibleGranularity(device);
	if ((g==0) || (bytes==0))
		return NULL;
	const size_t size = (bytes + g - 1)/g*g;
	const hipMemAllocationProp prop = CompressibleProp(device);
	hipMemGenericAllocationHandle_t handle;
	if (hipMemCreate(&handle, size, &prop, 0)!=hipSuccess)
	{
		(void)hipGetLastError();
		return NULL;
	}
	// the driver may hand back plain memory: then hipMalloc does as well
	hipMemAllocationProp got;
	memset(&got, 0, sizeof(got));
	if ((hipMemGetAllocationPropertiesFromHandle(&got, handle)!=hipSuccess) || (got.allocFlags.compressionType!=HIP_COMPRESSION_GENERIC))
	{
		(void)hipMemRelease(handle);
		(void)hipGetLastError();
		return NULL;
	}
	void* ptr = NULL;
	if (hipMemAddressReserve(&ptr, size, 0, NULL, 0)!=hipSuccess)
	{
		(void)hipMemRelease(handle);
		(void)hipGetLastError();
		return NULL;
	}
	if (hipMemMap(ptr, size, 0, handle, 0)!=hipSuccess)
	{
		(void)hipMemAddressFree(ptr, size);
		(void)hipMemRelease(handle);
		(void)hipGetLastError();
		return NULL;
	}
	hipMemAccessDesc access;
	memset(&access, 0, sizeof(access));
	access.location.type = hipMemLocationTypeDevice;
	access.location.id = device;
	access.flags = hipMemAccessFlagsProtReadWrite;
	if (hipMemSetAccess(ptr, size, &access, 1)!=hipSuccess)
	{
		(void)hipMemUnmap(ptr, size);
		(void)hipMemAddressFree(ptr, size);
		(void)hipMemRelease(handle);
		(void)hipGetLastError();
		return NULL;
	}
	HIP_Mapping m;
	m.size = size;
	m.handle = handle;
	{
		std::lock_guard<std::mutex> guard(Lock());
		Mappings()[ptr] = m;
	}
	granted = true;
	return ptr;
}

bool HIP_CompressibleFree(void* ptr)
{
	HIP_Mapping m;
	{
		std::lock_guard<std::mutex> guard(Lock());
		std::map<void*, HIP_Mapping>::iterator it = Mappings().find(ptr);
		if (it==Mappings().end())
			return false;
		m = it->second;
		Mappings().erase(it);
	}
	// unmap, then free the address range and release the physical memory
	(void)hipMemUnmap(ptr, m.size);
	(void)hipMemAddressFree(ptr, m.size);
	(void)hipMemRelease(m.handle);
	return true;
}

// Statistics for the compressible memory: the share of 128-byte lines of a buffer whose
// words are all equal (such lines compress with the generic compression of the GPU memory).

double HIP_UniformLineShare(const void* data, size_t bytes)
{
	const size_t lines = bytes/128;
	if (data==NULL || lines==0)
		return 0.0;
	const unsigned char* base = (const unsigned char*)data;
	size_t uniform = 0;
	for (size_t n=0; n<lines; ++n)
	{
		uint32_t w[32];
		std::memcpy(w, base + n*128, 128);
		uint32_t diff = 0;
		for (int k=1; k<32; ++k)
			diff |= w[k]^w[0];
		if (diff==0)
			++uniform;
	}
	return (double)uniform/(double)lines;
}

bool HIP_CompressibleStats()
{
	static const bool stats = []() {
		const char* env = getenv("OPENEMS_HIP_COMPRESSIBLE_STATS");
		return env && atoi(env)!=0;
	}();
	return stats;
}

void GPU_Backend_HIP::Impl::ReportUniformLines(const char* name, const void* host, size_t bytes)
{
	const size_t lines = bytes/128;
	const double share = host ? 100.0*HIP_UniformLineShare(host, bytes) : 100.0;
	std::cout << "GPU_Backend_HIP: uniform lines: " << name << " " << std::fixed << std::setprecision(1) << share
	          << " % of " << lines << " (" << std::setprecision(2) << (double)bytes/(1024.0*1024.0) << " MB)"
	          << std::defaultfloat << std::endl;
}
