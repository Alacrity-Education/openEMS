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
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include "hip_internal.h"
#include "FDTD/engine.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"

// Drude/Lorentz materials and conducting sheets, see Engine_Ext_LorentzMaterial
// and Engine_Ext_Dispersive. One launch per dispersion order, one thread per mesh
// position of that order (unique within an order). The ADE state and coefficients
// are stored per order as [direction][position].
struct ADEParam { size_t count; unsigned int lorentz; size_t sn; };

// ADE update before the main update, reads the field
__device__ __forceinline__ void lorentz_pre_pos(const float* field, float* ade, float* lor, const float* c_int, const float* c_ext,
                                                const float* c_lor, const size_t* pos, const ADEParam& P, size_t i)
{
	for (unsigned int n=0; n<3; ++n)
	{
		const size_t k = n*P.count + i;
		const float f = field[n*P.sn + pos[i]];
		if (P.lorentz)
		{
			lor[k] = lor[k] + c_lor[k]*ade[k];
			ade[k] = ade[k] * c_int[k];
			ade[k] = ade[k] + c_ext[k]*(f - lor[k]);
		}
		else
		{
			ade[k] = ade[k] * c_int[k];
			ade[k] = ade[k] + c_ext[k]*f;
		}
	}
}

// subtract the ADE current from the field
__device__ __forceinline__ void dispersive_apply_pos(float* field, const float* ade, const size_t* pos, const ADEParam& P, size_t i)
{
	for (unsigned int n=0; n<3; ++n)
	{
		const size_t g = n*P.sn + pos[i];
		field[g] = field[g] - ade[n*P.count + i];
	}
}

__global__ void lorentz_pre(const float* field, float* ade, float* lor, const float* c_int, const float* c_ext,
                            const float* c_lor, const size_t* pos, ADEParam P)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (i>=P.count)
		return;
	lorentz_pre_pos(field, ade, lor, c_int, c_ext, c_lor, pos, P, i);
}

__global__ void dispersive_apply(float* field, const float* ade, const size_t* pos, ADEParam P)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (i>=P.count)
		return;
	dispersive_apply_pos(field, ade, pos, P, i);
}

// Windowed blocks (OPENEMS_HIP_WINDOW): the positions inside [x0,x1) x [y0,y1), from their device index
struct ADEWindow { unsigned int x0, x1, y0, y1; HIP_GridDim N; };

__device__ __forceinline__ bool ade_in_window(const ADEWindow& W, size_t p)
{
	const size_t line = p / W.N.nzd;   // x*ny + y
	const unsigned int x = (unsigned int)(line / W.N.ny), y = (unsigned int)(line % W.N.ny);
	return (x>=W.x0) && (x<W.x1) && (y>=W.y0) && (y<W.y1);
}

__global__ void lorentz_pre_window(const float* field, float* ade, float* lor, const float* c_int, const float* c_ext,
                                   const float* c_lor, const size_t* pos, ADEParam P, ADEWindow W)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if ((i>=P.count) || !ade_in_window(W, pos[i]))
		return;
	lorentz_pre_pos(field, ade, lor, c_int, c_ext, c_lor, pos, P, i);
}

__global__ void dispersive_apply_window(float* field, const float* ade, const size_t* pos, ADEParam P, ADEWindow W)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if ((i>=P.count) || !ade_in_window(W, pos[i]))
		return;
	dispersive_apply_pos(field, ade, pos, P, i);
}

// Conducting sheets on z planes as dense per-plane state.
// A conducting sheet has two dispersion orders on the same positions, voltages only, not
// Lorentz. When the z component has no ADE at any position (the sheet normal is z, see
// ADE_Inert()), the positions are grouped by z plane, and all planes share one xy box, the
// bounding box of the positions. Per cell of the box and plane, a mask byte and, per order o
// and in-plane component n (slot o*2+n), the state and the two coefficients, stored as
// [o*2+n][x-x0][y-y0][plane]: the planes of an (x, y) line are adjacent, as their field values
// are (z is the fastest axis of the fields), then y, then x, the order of the fields. One
// thread per cell, in this order: a warp reads and writes the state and the coefficients
// coalesced, and the field values of a few (x, y) lines, one or two sectors each, as the
// position list of the separate orders does. Mask bit o*2+n: the cell is a position and order
// o of component n is not inert; a cell without a bit is not touched. One launch per step and
// phase for all planes and both orders; each cell runs the statements of lorentz_pre and
// dispersive_apply: the state of order 0, then of order 1, from the same field value, and
// V = V - ade0, then V = V - ade1.
#define SHEET_ORDERS 2
//! a few layers of a board, not a volume
#define SHEET_MAX_PLANES 32

struct SheetArgs
{
	const unsigned int* plane_z;   //!< z line of each plane
	const unsigned char* mask;
	float* ade;
	const float *c_int, *c_ext;
	unsigned int x0, y0, ny;       //!< the box: first x and y line, number of y lines
	unsigned int planes;
	size_t cells;                  //!< box cells times planes, the stride of the slots
	size_t sn;                     //!< stride of the field components
	HIP_GridDim N;
};

// cell c of the dense blocks: m its mask, g its field index of component 0
#define SHEET_CELL \
	const unsigned int m = A.mask[c]; \
	if (m==0) \
		return; \
	const size_t xy = c / A.planes; \
	const size_t g = nijk(A.N, 0, A.x0 + (unsigned int)(xy / A.ny), A.y0 + (unsigned int)(xy % A.ny), A.plane_z[c % A.planes]);

// ADE update of both orders before the main update, reads the field (see lorentz_pre)
__device__ __forceinline__ void sheet_pre_cell(const float* field, const SheetArgs& A, size_t c)
{
	SHEET_CELL
	for (unsigned int n=0; n<2; ++n)
	{
		if ((m & ((1u|(1u<<2))<<n))==0)
			continue;
		const float f = field[n*A.sn + g];
		for (unsigned int o=0; o<SHEET_ORDERS; ++o)
		{
			if ((m & (1u<<(2*o+n)))==0)
				continue;
			const size_t k = (2*o+n)*A.cells + c;
			A.ade[k] = A.ade[k] * A.c_int[k];
			A.ade[k] = A.ade[k] + A.c_ext[k]*f;
		}
	}
}

// subtract the ADE currents of order 0, then of order 1, from the field (see dispersive_apply)
__device__ __forceinline__ void sheet_apply_cell(float* field, const SheetArgs& A, size_t c)
{
	SHEET_CELL
	for (unsigned int n=0; n<2; ++n)
	{
		if ((m & ((1u|(1u<<2))<<n))==0)
			continue;
		float v = field[n*A.sn + g];
		for (unsigned int o=0; o<SHEET_ORDERS; ++o)
			if (m & (1u<<(2*o+n)))
				v = v - A.ade[(2*o+n)*A.cells + c];
		field[n*A.sn + g] = v;
	}
}

// one thread per cell c of the dense blocks
__global__ void sheet_pre(const float* field, SheetArgs A)
{
	const size_t c = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (c<A.cells)
		sheet_pre_cell(field, A, c);
}

__global__ void sheet_apply(float* field, SheetArgs A)
{
	const size_t c = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (c<A.cells)
		sheet_apply_cell(field, A, c);
}

// Windowed blocks (OPENEMS_HIP_WINDOW): the cells [c0, c0 + count) (the x lines of the window, adjacent in this
// order), those whose y is outside [y0, y1) skipped
__device__ __forceinline__ bool sheet_window_cell(const SheetArgs& A, size_t c0, size_t count, unsigned int y0, unsigned int y1, size_t& c)
{
	const size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (t>=count)
		return false;
	c = c0 + t;
	const unsigned int y = A.y0 + (unsigned int)((c / A.planes) % A.ny);
	return (y>=y0) && (y<y1);
}

__global__ void sheet_pre_window(const float* field, SheetArgs A, size_t c0, size_t count, unsigned int y0, unsigned int y1)
{
	size_t c;
	if (sheet_window_cell(A, c0, count, y0, y1, c))
		sheet_pre_cell(field, A, c);
}

__global__ void sheet_apply_window(float* field, SheetArgs A, size_t c0, size_t count, unsigned int y0, unsigned int y1)
{
	size_t c;
	if (sheet_window_cell(A, c0, count, y0, y1, c))
		sheet_apply_cell(field, A, c);
}

//! Whether a component with these coefficients of a non-Lorentz order is inert: its correction
//! never changes the field, so it may be skipped. The state starts at +0 and stays +0: +0*c_int
//! is +0 for a finite c_int without sign bit, c_ext*V is +0 or -0 for c_ext == +0 and a finite V,
//! and +0 + (+-0) is +0 (round to nearest). V - (+0) is V for every V, -0 included. Only a
//! non-finite V (a diverged run) would differ: 0*inf is NaN, which the separate kernels carry
//! into the state and the field.
static bool ADE_Inert(float c_int, float c_ext)
{
	unsigned int bits;
	memcpy(&bits, &c_ext, sizeof(bits));
	return (bits==0) && !std::signbit(c_int) && std::isfinite(c_int);
}

class HIP_Ext_LorentzMaterial : public GPU_Extension
{
public:
	//! Tries the dense per-plane state first (see SetupSheetPlanes()); \a log: say how it went
	HIP_Ext_LorentzMaterial(GPU_Backend_HIP::Impl* impl, Operator_Ext_LorentzMaterial* op_ext, bool log);

	virtual void DoPreVoltageUpdates()
	{
		if (m_Sheet.cells)
			HIP_Launch(d, "sheet_pre", sheet_pre, m_Sheet.cells, 1, 1, (const float*)d->volt, m_Sheet);
		for (size_t o=0; o<m_Volt.size(); ++o) Pre(m_Volt[o], d->volt);
	}
	virtual void Apply2Voltages()
	{
		if (m_Sheet.cells)
			HIP_Launch(d, "sheet_apply", sheet_apply, m_Sheet.cells, 1, 1, d->volt, m_Sheet);
		for (size_t o=0; o<m_Volt.size(); ++o) Apply(m_Volt[o], d->volt);
	}
	virtual void DoPreCurrentUpdates() {for (size_t o=0; o<m_Curr.size(); ++o) Pre(m_Curr[o], d->curr);}
	virtual void Apply2Current()       {for (size_t o=0; o<m_Curr.size(); ++o) Apply(m_Curr[o], d->curr);}

	// Windowed blocks: the hooks above on the cells inside the window (each cell's state and field value only)
	virtual bool WindowCapable() const {return true;}
	virtual void LaunchInWindow(GPU_WindowPhase phase, unsigned int step_in_block, unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
	{
		UNUSED(step_in_block);
		if ((phase==GPU_WindowPreVoltage) || (phase==GPU_WindowApplyVoltage))
		{
			const bool pre = phase==GPU_WindowPreVoltage;
			if (m_Sheet.cells)
			{
				// the x lines of the box inside the window
				const unsigned int bx1 = m_Sheet.x0 + (unsigned int)(m_Sheet.cells/((size_t)m_Sheet.ny*m_Sheet.planes));
				const unsigned int a = std::max(x0, m_Sheet.x0), b = std::min(x1, bx1);
				if (a<b)
				{
					const size_t line = (size_t)m_Sheet.ny*m_Sheet.planes;
					const size_t c0 = (a - m_Sheet.x0)*line, count = (b - a)*line;
					if (pre)
						HIP_Launch(d, "sheet_pre_window", sheet_pre_window, count, 1, 1, (const float*)d->volt, m_Sheet, c0, count, y0, y1);
					else
						HIP_Launch(d, "sheet_apply_window", sheet_apply_window, count, 1, 1, d->volt, m_Sheet, c0, count, y0, y1);
				}
			}
			for (size_t o=0; o<m_Volt.size(); ++o)
				WindowOrder(m_Volt[o], pre ? (const float*)d->volt : NULL, pre ? NULL : d->volt, x0, x1, y0, y1);
		}
		else if (phase==GPU_WindowPreCurrent)
			for (size_t o=0; o<m_Curr.size(); ++o)
				WindowOrder(m_Curr[o], d->curr, NULL, x0, x1, y0, y1);
		else if (phase==GPU_WindowApplyCurrent)
			for (size_t o=0; o<m_Curr.size(); ++o)
				WindowOrder(m_Curr[o], NULL, d->curr, x0, x1, y0, y1);
	}

protected:
	//! one dispersion order of the voltages or currents
	struct Order
	{
		ADEParam param;
		size_t* pos;                //!< flat NIJK index of direction 0
		float *ade, *lor;           //!< state
		float *c_int, *c_ext, *c_lor;
		unsigned int x_min, x_max;  //!< the x lines of the positions (windowed blocks)
	};

	void Setup(std::vector<Order>& orders, size_t count, unsigned int** pos,
	           bool lorentz, FDTD_FLOAT** c_int, FDTD_FLOAT** c_ext, FDTD_FLOAT** c_lor);
	float* Coefficients(size_t count, FDTD_FLOAT** c);
	void Pre(Order& o, float* field)
	{
		HIP_Launch(d, "lorentz_pre", lorentz_pre, o.param.count, 1, 1, (const float*)field, o.ade, o.lor,
		            (const float*)o.c_int, (const float*)o.c_ext, (const float*)o.c_lor, (const size_t*)o.pos, o.param);
	}
	void Apply(Order& o, float* field)
	{
		HIP_Launch(d, "dispersive_apply", dispersive_apply, o.param.count, 1, 1, field, (const float*)o.ade, (const size_t*)o.pos, o.param);
	}
	//! Pre (\a pre_field) or Apply (\a apply_field) of order \a o on the positions inside the window
	void WindowOrder(Order& o, const float* pre_field, float* apply_field, unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
	{
		if ((x1<=o.x_min) || (x0>o.x_max))
			return;
		const ADEWindow W = {x0, x1, y0, y1, d->dim};
		if (pre_field)
			HIP_Launch(d, "lorentz_pre_window", lorentz_pre_window, o.param.count, 1, 1, pre_field, o.ade, o.lor,
			            (const float*)o.c_int, (const float*)o.c_ext, (const float*)o.c_lor, (const size_t*)o.pos, o.param, W);
		else
			HIP_Launch(d, "dispersive_apply_window", dispersive_apply_window, o.param.count, 1, 1, apply_field, (const float*)o.ade,
			            (const size_t*)o.pos, o.param, W);
	}

	//! The voltages of a conducting sheet on z planes as dense per-plane state (m_Sheet, see
	//! sheet_pre); an empty string if set up, else the reason why not (and m_Sheet.cells is 0)
	std::string SetupSheetPlanes(Operator_Ext_LorentzMaterial* op_ext, size_t& bytes);
	SheetArgs m_Sheet;

	GPU_Backend_HIP::Impl* d;
	std::vector<Order> m_Volt;
	std::vector<Order> m_Curr;
};

HIP_Ext_LorentzMaterial::HIP_Ext_LorentzMaterial(GPU_Backend_HIP::Impl* impl, Operator_Ext_LorentzMaterial* op_ext, bool log)
{
	d = impl;
	memset(&m_Sheet, 0, sizeof(m_Sheet));
	size_t bytes = 0;
	const std::string why = SetupSheetPlanes(op_ext, bytes);
	if (log && why.empty())
		std::cout << "GPU_Backend_HIP: sheet planes: " << m_Sheet.planes << " planes, " << m_Sheet.cells << " cells, "
		          << (bytes+1023)/1024 << " KB state" << std::endl;
	else if (log)
		std::cout << "GPU_Backend_HIP: sheet planes: unavailable: " << why << std::endl;
	if (m_Sheet.cells)
		return;
	for (int o=0; o<op_ext->m_Order; ++o)
	{
		const size_t count = op_ext->m_LM_Count.at(o);
		if (op_ext->m_volt_ADE_On[o])
			Setup(m_Volt, count, op_ext->m_LM_pos[o], op_ext->m_volt_Lor_ADE_On[o],
			      op_ext->v_int_ADE[o], op_ext->v_ext_ADE[o], op_ext->m_volt_Lor_ADE_On[o] ? op_ext->v_Lor_ADE[o] : NULL);
		if (op_ext->m_curr_ADE_On[o])
			Setup(m_Curr, count, op_ext->m_LM_pos[o], op_ext->m_curr_Lor_ADE_On[o],
			      op_ext->i_int_ADE[o], op_ext->i_ext_ADE[o], op_ext->m_curr_Lor_ADE_On[o] ? op_ext->i_Lor_ADE[o] : NULL);
	}
}

float* HIP_Ext_LorentzMaterial::Coefficients(size_t count, FDTD_FLOAT** c)
{
	std::vector<float> data(3*(size_t)count, 0);
	if (c)
		for (int n=0; n<3; ++n)
			for (size_t i=0; i<count; ++i)
				data[n*count + i] = c[n][i];
	return d->Alloc<float>(data.size(), data.data());
}

void HIP_Ext_LorentzMaterial::Setup(std::vector<Order>& orders, size_t count, unsigned int** pos,
                                     bool lorentz, FDTD_FLOAT** c_int, FDTD_FLOAT** c_ext, FDTD_FLOAT** c_lor)
{
	if (count==0)
		return;
	Order ord;
	ord.param.count = count;
	ord.param.lorentz = lorentz;
	ord.param.sn = d->numCellsD;

	std::vector<size_t> flat(count);
	ord.x_min = UINT_MAX;
	ord.x_max = 0;
	for (size_t i=0; i<count; ++i)
	{
		flat[i] = ((size_t)pos[0][i]*d->dim.ny + pos[1][i])*d->dim.nz + pos[2][i];
		ord.x_min = std::min(ord.x_min, pos[0][i]);
		ord.x_max = std::max(ord.x_max, pos[0][i]);
	}
	d->ext_nodes.insert(d->ext_nodes.end(), flat.begin(), flat.end());   // see GPU_Backend_HIP::Impl::MurRefusal()
	for (size_t i=0; i<count; ++i)
		flat[i] = d->Dev(flat[i]);   // the device layout (OPENEMS_HIP_ZPAD)
	ord.pos = d->Alloc<size_t>(count, flat.data());

	ord.ade = d->Alloc<float>(3*(size_t)count);
	ord.lor = d->Alloc<float>(3*(size_t)count);
	ord.c_int = Coefficients(count, c_int);
	ord.c_ext = Coefficients(count, c_ext);
	ord.c_lor = Coefficients(count, c_lor);
	orders.push_back(ord);
}

std::string HIP_Ext_LorentzMaterial::SetupSheetPlanes(Operator_Ext_LorentzMaterial* op_ext, size_t& bytes)
{
	if (!dynamic_cast<Operator_Ext_ConductingSheet*>(op_ext))
		return "not a conducting sheet";
	if (op_ext->m_Order!=SHEET_ORDERS)
		return "not two orders";
	const size_t count = op_ext->m_LM_Count.at(0);
	for (int o=0; o<SHEET_ORDERS; ++o)
	{
		if (!op_ext->m_volt_ADE_On[o] || op_ext->m_curr_ADE_On[o] || op_ext->m_volt_Lor_ADE_On[o])
			return "not voltage-only non-Lorentz orders";
		if (op_ext->m_LM_Count.at(o)!=count)
			return "orders on different positions";
		for (int n=0; n<3; ++n)
			if (memcmp(op_ext->m_LM_pos[o][n], op_ext->m_LM_pos[0][n], count*sizeof(unsigned int)))
				return "orders on different positions";
	}
	if (count==0)
		return "no positions";

	// the active bits of every position; the z component must be inert in both orders
	unsigned int** pos = op_ext->m_LM_pos[0];
	for (size_t i=0; i<count; ++i)   // see GPU_Backend_HIP::Impl::MurRefusal()
		d->ext_nodes.push_back(((size_t)pos[0][i]*d->dim.ny + pos[1][i])*d->dim.nz + pos[2][i]);
	std::vector<unsigned char> bits(count, 0);
	for (size_t i=0; i<count; ++i)
	{
		for (int o=0; o<SHEET_ORDERS; ++o)
		{
			if (!ADE_Inert(op_ext->v_int_ADE[o][2][i], op_ext->v_ext_ADE[o][2][i]))
				return "positions not on z planes (an ADE on the z component)";
			for (int n=0; n<2; ++n)
				if (!ADE_Inert(op_ext->v_int_ADE[o][n][i], op_ext->v_ext_ADE[o][n][i]))
					bits[i] |= 1u<<(2*o+n);
		}
	}

	// the planes (their index, in z order) and the box
	std::map<unsigned int, unsigned int> plane;
	unsigned int x0 = UINT_MAX, y0 = UINT_MAX, x1 = 0, y1 = 0;
	for (size_t i=0; i<count; ++i)
	{
		plane[pos[2][i]] = 0;
		x0 = std::min(x0, pos[0][i]);
		y0 = std::min(y0, pos[1][i]);
		x1 = std::max(x1, pos[0][i]+1);
		y1 = std::max(y1, pos[1][i]+1);
	}
	if (plane.size()>SHEET_MAX_PLANES)
		return "more than " + std::to_string(SHEET_MAX_PLANES) + " planes";
	std::vector<unsigned int> plane_z;
	for (std::map<unsigned int, unsigned int>::iterator it=plane.begin(); it!=plane.end(); ++it)
	{
		it->second = plane_z.size();
		plane_z.push_back(it->first);
	}
	const unsigned int planes = plane_z.size();
	const unsigned int ny = y1 - y0;
	const size_t cells = (size_t)(x1-x0)*ny*planes;

	// the dense blocks: mask, coefficients, zero state
	const size_t slots = 2*SHEET_ORDERS*cells;
	std::vector<unsigned char> mask(cells, 0);
	std::vector<unsigned char> used(cells, 0);
	std::vector<float> c_int(slots, 0), c_ext(slots, 0);
	for (size_t i=0; i<count; ++i)
	{
		const size_t c = ((size_t)(pos[0][i]-x0)*ny + (pos[1][i]-y0))*planes + plane[pos[2][i]];
		if (used[c])
			return "a position twice in an order";
		used[c] = 1;
		mask[c] = bits[i];
		for (int o=0; o<SHEET_ORDERS; ++o)
			for (int n=0; n<2; ++n)
			{
				c_int[(2*o+n)*cells + c] = op_ext->v_int_ADE[o][n][i];
				c_ext[(2*o+n)*cells + c] = op_ext->v_ext_ADE[o][n][i];
			}
	}

	SheetArgs& A = m_Sheet;
	A.plane_z = d->Alloc<unsigned int>(planes, plane_z.data());
	A.mask = d->Alloc<unsigned char>(cells, mask.data());
	A.ade = d->Alloc<float>(slots);
	A.c_int = d->Alloc<float>(slots, c_int.data());
	A.c_ext = d->Alloc<float>(slots, c_ext.data());
	A.x0 = x0;
	A.y0 = y0;
	A.ny = ny;
	A.planes = planes;
	A.cells = cells;
	A.sn = d->numCellsD;
	A.N = d->dim;
	bytes = planes*sizeof(unsigned int) + cells + 3*slots*sizeof(float);
	return std::string();
}

GPU_Extension* HIP_CreateExt_LorentzMaterial(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	UNUSED(eng);
	if (!dynamic_cast<Engine_Ext_LorentzMaterial*>(eng_ext))
		return NULL;
	Operator_Ext_LorentzMaterial* op_ext = dynamic_cast<Operator_Ext_LorentzMaterial*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	// logged: how every conducting sheet runs, and why a Drude/Lorentz material keeps its own kernels
	return new HIP_Ext_LorentzMaterial(d, op_ext, g_settings.GetVerboseLevel()>1);
}
