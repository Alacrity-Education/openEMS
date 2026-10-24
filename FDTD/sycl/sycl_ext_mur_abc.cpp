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

// Mur absorbing boundary condition, see Engine_Ext_Mur_ABC.
// Stub: the device implementation is ported in task T9 (from FDTD/hip/hip_ext_mur_abc.hip).

#include "sycl_internal.h"

GPU_Extension* SYCL_CreateExt_Mur_ABC(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	(void)d; (void)eng_ext; (void)eng;
	return NULL;   // ported in task T9: until then the engine runs this extension on the host
}
