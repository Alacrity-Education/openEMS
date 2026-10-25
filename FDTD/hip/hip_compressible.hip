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

#include "hip/hip_runtime.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include "hip_internal.h"

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
