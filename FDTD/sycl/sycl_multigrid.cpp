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

// Coupling of a cylindrical multi-grid level and its sub-grid on the device, see GPU_MultiGridLink.
// Stub: ported in task T16 (from FDTD/hip/hip_multigrid.hip).

#include "sycl_internal.h"

GPU_MultiGridLink* GPU_Backend_SYCL::CreateMultiGridLink(GPU_Backend* sub_grid, const GPU_MultiGridInterpolation& interpol)
{
	(void)sub_grid; (void)interpol;
	return NULL;   // ported in task T16: until then the multi-grid coupling runs on the host
}
