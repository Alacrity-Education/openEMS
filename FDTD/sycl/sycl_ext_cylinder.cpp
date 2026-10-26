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

// Closed alpha direction and r=0 axis of cylindrical meshes, see Engine_Ext_Cylinder.
// Stub: the device implementation is ported in task T16 (from FDTD/hip/hip_ext_cylinder.hip).

#include "sycl_internal.h"

GPU_Extension* SYCL_CreateExt_Cylinder(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	(void)d; (void)eng_ext; (void)eng;
	return NULL;   // ported in task T16: until then the engine runs this extension on the host
}
