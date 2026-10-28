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

#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include "sycl_internal.h"
#include "FDTD/engine.h"
#include "tools/global.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"

// Drude/Lorentz materials and conducting sheets, see Engine_Ext_LorentzMaterial
// and Engine_Ext_Dispersive. The ADE state and coefficients are stored per order as
// [direction][position]. The orders of one side (voltages or currents) on the same
// positions (e.g. the two orders of a conducting sheet) run in one launch, one thread
// per position: the field is read once for all of them, and the corrections are
// subtracted one after the other in the order of the orders, as separate launches
// would do. A direction no order of a launch has a coefficient for is skipped: its
// state stays zero, so its correction would be a subtraction of zero (e.g. the
// normal direction of a conducting sheet).
#define ADE_MAX_ORDERS 4

struct ADEOrder
{
	float *ade, *lor;
	const float *c_int, *c_ext, *c_lor;
	unsigned int lorentz;
	unsigned int dirs;     //!< bit n: direction n has a nonzero coefficient
};

struct ADEGroup
{
	size_t count;
	size_t sn;
	const size_t* pos;     //!< flat NIJK index of direction 0, in memory order
	unsigned int orders;
	unsigned int dirs;     //!< union of the directions of the orders
	unsigned int x_min, x_max;   //!< the x lines of the positions (windowed blocks)
	ADEOrder o[ADE_MAX_ORDERS];
};

// ADE update before the main update, reads the field (position i of the group)
static inline void lorentz_pre_pos(const float* field, const ADEGroup& G, size_t i)
{
	const size_t p = G.pos[i];
	for (unsigned int n=0; n<3; ++n)
	{
		if (!((G.dirs>>n) & 1))
			continue;
		const size_t k = n*G.count + i;
		const float f = field[n*G.sn + p];
		#pragma unroll
		for (unsigned int q=0; q<ADE_MAX_ORDERS; ++q)   // fixed indices: the arguments stay in registers
		{
			const ADEOrder& O = G.o[q];
			if ((q>=G.orders) || !((O.dirs>>n) & 1))
				continue;
			if (O.lorentz)
			{
				O.lor[k] = O.lor[k] + O.c_lor[k]*O.ade[k];
				O.ade[k] = O.ade[k] * O.c_int[k];
				O.ade[k] = O.ade[k] + O.c_ext[k]*(f - O.lor[k]);
			}
			else
			{
				O.ade[k] = O.ade[k] * O.c_int[k];
				O.ade[k] = O.ade[k] + O.c_ext[k]*f;
			}
		}
	}
}

// subtract the ADE currents of the orders from the field (position i of the group)
static inline void dispersive_apply_pos(float* field, const ADEGroup& G, size_t i)
{
	const size_t p = G.pos[i];
	for (unsigned int n=0; n<3; ++n)
	{
		if (!((G.dirs>>n) & 1))
			continue;
		const size_t g = n*G.sn + p;
		float f = field[g];
		#pragma unroll
		for (unsigned int q=0; q<ADE_MAX_ORDERS; ++q)
			if ((q<G.orders) && ((G.o[q].dirs>>n) & 1))
				f = f - G.o[q].ade[n*G.count + i];
		field[g] = f;
	}
}

static inline void lorentz_pre(const sycl::nd_item<3>& it, const float* field, const ADEGroup& G)
{
	const size_t i = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (i<G.count)
		lorentz_pre_pos(field, G, i);
}

static inline void dispersive_apply(const sycl::nd_item<3>& it, float* field, const ADEGroup& G)
{
	const size_t i = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (i<G.count)
		dispersive_apply_pos(field, G, i);
}

// Windowed blocks (OPENEMS_SYCL_WINDOW): the positions inside [x0,x1) x [y0,y1), from their flat index
struct ADEWindow { unsigned int x0, x1, y0, y1; SYCL_GridDim N; };

inline bool ade_in_window(const ADEWindow& W, size_t p)
{
	const size_t line = p / W.N.nz;   // x*ny + y
	const unsigned int x = (unsigned int)(line / W.N.ny), y = (unsigned int)(line % W.N.ny);
	return (x>=W.x0) && (x<W.x1) && (y>=W.y0) && (y<W.y1);
}

static inline void lorentz_pre_window(const sycl::nd_item<3>& it, const float* field, const ADEGroup& G, const ADEWindow& W)
{
	const size_t i = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if ((i<G.count) && ade_in_window(W, G.pos[i]))
		lorentz_pre_pos(field, G, i);
}

static inline void dispersive_apply_window(const sycl::nd_item<3>& it, float* field, const ADEGroup& G, const ADEWindow& W)
{
	const size_t i = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if ((i<G.count) && ade_in_window(W, G.pos[i]))
		dispersive_apply_pos(field, G, i);
}

// Conducting sheets on z planes as dense per-plane state.
// A conducting sheet has two dispersion orders on the same positions, voltages only, not
// Lorentz. When the z component has no ADE at any position (the sheet normal is z, see
// ADE_Inert()), the positions are grouped by z plane, and all planes share one xy box, the
// bounding box of the positions. Per cell of the box and plane, a mask byte and, per order o
// and in-plane component n (slot o*2+n), the state and the two coefficients, stored as
// [o*2+n][x-x0][y-y0][plane]: the planes of an (x, y) line are adjacent, as their field values
// are (z is the fastest axis of the fields), then y, then x, the order of the fields. One
// work-item per cell, in this order: a sub-group reads and writes the state and the
// coefficients contiguously, and the field values of a few (x, y) lines. Mask bit o*2+n: the
// cell is a position and order o of component n is not inert; a cell without a bit is not
// touched. One launch per step and phase for all planes and both orders; each cell runs the
// statements of lorentz_pre and dispersive_apply: the state of order 0, then of order 1, from
// the same field value, and V = V - ade0, then V = V - ade1.
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
	SYCL_GridDim N;
};

// cell c of the dense blocks: m its mask, g its field index of component 0
#define SHEET_CELL \
	const unsigned int m = A.mask[c]; \
	if (m==0) \
		return; \
	const size_t xy = c / A.planes; \
	const size_t g = nijk(A.N, 0, A.x0 + (unsigned int)(xy / A.ny), A.y0 + (unsigned int)(xy % A.ny), A.plane_z[c % A.planes]);

// ADE update of both orders before the main update, reads the field (see lorentz_pre)
static inline void sheet_pre_cell(const float* field, const SheetArgs& A, size_t c)
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
static inline void sheet_apply_cell(float* field, const SheetArgs& A, size_t c)
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

// one work-item per cell c of the dense blocks
static inline void sheet_pre(const sycl::nd_item<3>& it, const float* field, const SheetArgs& A)
{
	const size_t c = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (c<A.cells)
		sheet_pre_cell(field, A, c);
}

static inline void sheet_apply(const sycl::nd_item<3>& it, float* field, const SheetArgs& A)
{
	const size_t c = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (c<A.cells)
		sheet_apply_cell(field, A, c);
}

// Windowed blocks (OPENEMS_SYCL_WINDOW): the cells [c0, c0 + count) (the x lines of the window, adjacent in this
// order), those whose y is outside [y0, y1) skipped
inline bool sheet_window_cell(const sycl::nd_item<3>& it, const SheetArgs& A, size_t c0, size_t count, unsigned int y0, unsigned int y1, size_t& c)
{
	const size_t t = (size_t)it.get_group(2)*it.get_local_range(2) + it.get_local_id(2);
	if (t>=count)
		return false;
	c = c0 + t;
	const unsigned int y = A.y0 + (unsigned int)((c / A.planes) % A.ny);
	return (y>=y0) && (y<y1);
}

//! Whether a component with these coefficients of a non-Lorentz order is inert: its correction
//! never changes the field, so it may be skipped. The state starts at +0 and stays +0: +0*c_int
//! is +0 for a finite c_int without sign bit, c_ext*V is +0 or -0 for c_ext == +0 and a finite V,
//! and +0 + (+-0) is +0 (round to nearest). V - (+0) is V for every V, -0 included. Only a
//! non-finite V (a diverged run) would differ: 0*inf is NaN, which the position list carries
//! into the state and the field.
static bool ADE_Inert(float c_int, float c_ext)
{
	unsigned int bits;
	memcpy(&bits, &c_ext, sizeof(bits));
	return (bits==0) && !std::signbit(c_int) && std::isfinite(c_int);
}

class SYCL_Ext_LorentzMaterial : public GPU_Extension
{
public:
	//! Tries the dense per-plane state first (see SetupSheetPlanes()); \a log: say how it went
	SYCL_Ext_LorentzMaterial(GPU_Backend_SYCL::Impl* impl, Operator_Ext_LorentzMaterial* op_ext, bool log);

	virtual void DoPreVoltageUpdates()
	{
		if (m_Sheet.cells)
		{
			const float* f = d->volt;
			const SheetArgs A = m_Sheet;
			SYCL_Launch(d, "sheet_pre", A.cells, 1, 1, [=](sycl::nd_item<3> it)
			{
				sheet_pre(it, f, A);
			});
		}
		for (size_t g=0; g<m_Volt.size(); ++g) Pre(m_Volt[g], d->volt);
	}
	virtual void Apply2Voltages()
	{
		if (m_Sheet.cells)
		{
			float* f = d->volt;
			const SheetArgs A = m_Sheet;
			SYCL_Launch(d, "sheet_apply", A.cells, 1, 1, [=](sycl::nd_item<3> it)
			{
				sheet_apply(it, f, A);
			});
		}
		for (size_t g=0; g<m_Volt.size(); ++g) Apply(m_Volt[g], d->volt);
	}
	virtual void DoPreCurrentUpdates() {for (size_t g=0; g<m_Curr.size(); ++g) Pre(m_Curr[g], d->curr);}
	virtual void Apply2Current()       {for (size_t g=0; g<m_Curr.size(); ++g) Apply(m_Curr[g], d->curr);}

	// Windowed blocks: the hooks above on the cells inside the window (each cell's state and field value only)
	virtual bool WindowCapable() const {return true;}
	virtual void LaunchInWindow(GPU_WindowPhase phase, unsigned int step_in_block, unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
	{
		(void)step_in_block;
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
					const SheetArgs A = m_Sheet;
					float* f = d->volt;
					if (pre)
						SYCL_Launch(d, "sheet_pre_window", count, 1, 1, [=](sycl::nd_item<3> it)
						{
							size_t c;
							if (sheet_window_cell(it, A, c0, count, y0, y1, c))
								sheet_pre_cell(f, A, c);
						});
					else
						SYCL_Launch(d, "sheet_apply_window", count, 1, 1, [=](sycl::nd_item<3> it)
						{
							size_t c;
							if (sheet_window_cell(it, A, c0, count, y0, y1, c))
								sheet_apply_cell(f, A, c);
						});
				}
			}
			for (size_t g=0; g<m_Volt.size(); ++g)
				WindowGroup(m_Volt[g], pre, d->volt, x0, x1, y0, y1);
		}
		else if (phase==GPU_WindowPreCurrent)
			for (size_t g=0; g<m_Curr.size(); ++g)
				WindowGroup(m_Curr[g], true, d->curr, x0, x1, y0, y1);
		else if (phase==GPU_WindowApplyCurrent)
			for (size_t g=0; g<m_Curr.size(); ++g)
				WindowGroup(m_Curr[g], false, d->curr, x0, x1, y0, y1);
	}

protected:
	//! Pre (\a pre) or Apply of group \a group on the positions inside the window
	void WindowGroup(const ADEGroup& group, bool pre, float* field, unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1)
	{
		if ((group.dirs==0) || (x1<=group.x_min) || (x0>group.x_max))
			return;
		const ADEWindow W = {x0, x1, y0, y1, d->dim};
		const ADEGroup G = group;
		float* f = field;
		if (pre)
			SYCL_Launch(d, "lorentz_pre_window", G.count, 1, 1, [=](sycl::nd_item<3> it)
			{
				lorentz_pre_window(it, f, G, W);
			});
		else
			SYCL_Launch(d, "dispersive_apply_window", G.count, 1, 1, [=](sycl::nd_item<3> it)
			{
				dispersive_apply_window(it, f, G, W);
			});
	}

	void Setup(std::vector<ADEGroup>& groups, std::vector<std::vector<size_t>>& group_pos, size_t count, unsigned int** pos,
	           bool lorentz, FDTD_FLOAT** c_int, FDTD_FLOAT** c_ext, FDTD_FLOAT** c_lor);
	float* Coefficients(size_t count, FDTD_FLOAT** c);
	void Pre(const ADEGroup& group, float* field)
	{
		if (group.dirs==0)
			return;
		const float* f = field;
		const ADEGroup G = group;
		SYCL_Launch(d, "lorentz_pre", G.count, 1, 1, [=](sycl::nd_item<3> it)
		{
			lorentz_pre(it, f, G);
		});
	}
	void Apply(const ADEGroup& group, float* field)
	{
		if (group.dirs==0)
			return;
		float* f = field;
		const ADEGroup G = group;
		SYCL_Launch(d, "dispersive_apply", G.count, 1, 1, [=](sycl::nd_item<3> it)
		{
			dispersive_apply(it, f, G);
		});
	}

	//! The voltages of a conducting sheet on z planes as dense per-plane state (m_Sheet, see
	//! sheet_pre); an empty string if set up, else the reason why not (and m_Sheet.cells is 0)
	std::string SetupSheetPlanes(Operator_Ext_LorentzMaterial* op_ext, size_t& bytes);
	SheetArgs m_Sheet;

	GPU_Backend_SYCL::Impl* d;
	std::vector<ADEGroup> m_Volt;
	std::vector<ADEGroup> m_Curr;
};

SYCL_Ext_LorentzMaterial::SYCL_Ext_LorentzMaterial(GPU_Backend_SYCL::Impl* impl, Operator_Ext_LorentzMaterial* op_ext, bool log)
{
	d = impl;
	memset(&m_Sheet, 0, sizeof(m_Sheet));
	size_t bytes = 0;
	const std::string why = SetupSheetPlanes(op_ext, bytes);
	if (log && why.empty())
		std::cout << "GPU_Backend_SYCL: sheet planes: " << m_Sheet.planes << " planes, " << m_Sheet.cells << " cells, "
		          << (bytes+1023)/1024 << " KB state" << std::endl;
	else if (log)
		std::cout << "GPU_Backend_SYCL: sheet planes: unavailable: " << why << std::endl;
	if (m_Sheet.cells)
		return;
	std::vector<std::vector<size_t>> volt_pos, curr_pos;
	for (int o=0; o<op_ext->m_Order; ++o)
	{
		const size_t count = op_ext->m_LM_Count.at(o);
		if (op_ext->m_volt_ADE_On[o])
			Setup(m_Volt, volt_pos, count, op_ext->m_LM_pos[o], op_ext->m_volt_Lor_ADE_On[o],
			      op_ext->v_int_ADE[o], op_ext->v_ext_ADE[o], op_ext->m_volt_Lor_ADE_On[o] ? op_ext->v_Lor_ADE[o] : NULL);
		if (op_ext->m_curr_ADE_On[o])
			Setup(m_Curr, curr_pos, count, op_ext->m_LM_pos[o], op_ext->m_curr_Lor_ADE_On[o],
			      op_ext->i_int_ADE[o], op_ext->i_ext_ADE[o], op_ext->m_curr_Lor_ADE_On[o] ? op_ext->i_Lor_ADE[o] : NULL);
	}
	// fused step (see update_fused): Apply2Voltages() changes the voltages of these nodes after the
	// voltage update, the currents next to them are recomputed; Apply2Current() changes their
	// currents after the current update
	for (size_t g=0; g<m_Volt.size(); ++g)
		for (unsigned int n=0; n<3; ++n)
			if ((m_Volt[g].dirs>>n) & 1)
				for (size_t k=0; k<volt_pos[g].size(); ++k)
					d->volt_modified.push_back(n*d->numCells + volt_pos[g][k]);
	for (size_t g=0; g<m_Curr.size(); ++g)
		if (m_Curr[g].dirs)
			for (unsigned int n=0; n<3; ++n)
				if ((m_Curr[g].dirs>>n) & 1)
					for (size_t k=0; k<curr_pos[g].size(); ++k)
						d->curr_modified.push_back(n*d->numCells + curr_pos[g][k]);
}

float* SYCL_Ext_LorentzMaterial::Coefficients(size_t count, FDTD_FLOAT** c)
{
	std::vector<float> data(3*(size_t)count, 0);
	if (c)
		for (int n=0; n<3; ++n)
			for (size_t i=0; i<count; ++i)
				data[n*count + i] = c[n][i];
	return d->Alloc<float>(data.size(), data.data());
}

void SYCL_Ext_LorentzMaterial::Setup(std::vector<ADEGroup>& groups, std::vector<std::vector<size_t>>& group_pos, size_t count, unsigned int** pos,
                                     bool lorentz, FDTD_FLOAT** c_int, FDTD_FLOAT** c_ext, FDTD_FLOAT** c_lor)
{
	if (count==0)
		return;
	std::vector<size_t> flat(count);
	for (size_t i=0; i<count; ++i)
		flat[i] = ((size_t)pos[0][i]*d->dim.ny + pos[1][i])*d->dim.nz + pos[2][i];

	ADEOrder ord;
	ord.lorentz = lorentz;
	// a direction without a coefficient keeps a zero state (ade = ade*c_int + 0, lor = lor + c_lor*0)
	ord.dirs = 0;
	for (int n=0; n<3; ++n)
		for (size_t i=0; c_ext && (i<count); ++i)
			if (c_ext[n][i]!=0)
			{
				ord.dirs |= 1u<<n;
				break;
			}
	ord.ade = d->Alloc<float>(3*(size_t)count);
	ord.lor = d->Alloc<float>(3*(size_t)count);
	ord.c_int = Coefficients(count, c_int);
	ord.c_ext = Coefficients(count, c_ext);
	ord.c_lor = Coefficients(count, c_lor);

	// the same positions as the previous order of this side: one launch for both
	if (!groups.empty() && (group_pos.back()==flat) && (groups.back().orders<ADE_MAX_ORDERS))
	{
		ADEGroup& G = groups.back();
		G.o[G.orders++] = ord;
		G.dirs |= ord.dirs;
		return;
	}
	ADEGroup G;
	G.count = count;
	G.sn = d->numCells;
	G.pos = d->Alloc<size_t>(count, flat.data());
	d->ext_nodes.insert(d->ext_nodes.end(), flat.begin(), flat.end());   // see GPU_Backend_SYCL::Impl::MurRefusal()
	G.orders = 1;
	G.dirs = ord.dirs;
	G.x_min = UINT_MAX;
	G.x_max = 0;
	for (size_t i=0; i<count; ++i)
	{
		G.x_min = std::min(G.x_min, pos[0][i]);
		G.x_max = std::max(G.x_max, pos[0][i]);
	}
	G.o[0] = ord;
	groups.push_back(G);
	group_pos.push_back(flat);
}

std::string SYCL_Ext_LorentzMaterial::SetupSheetPlanes(Operator_Ext_LorentzMaterial* op_ext, size_t& bytes)
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
	A.sn = d->numCells;
	A.N = d->dim;
	bytes = planes*sizeof(unsigned int) + cells + 3*slots*sizeof(float);

	// see GPU_Backend_SYCL::Impl::MurRefusal(); fused step (see update_fused): Apply2Voltages() changes the
	// in-plane voltages of the positions after the voltage update, the currents next to them are recomputed
	for (size_t i=0; i<count; ++i)
	{
		const size_t flat = ((size_t)pos[0][i]*d->dim.ny + pos[1][i])*d->dim.nz + pos[2][i];
		d->ext_nodes.push_back(flat);
		for (unsigned int n=0; n<2; ++n)
			if (bits[i] & ((1u|(1u<<2))<<n))
				d->volt_modified.push_back(n*d->numCells + flat);
	}
	return std::string();
}

GPU_Extension* SYCL_CreateExt_LorentzMaterial(GPU_Backend_SYCL::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	(void)eng;
	if (!dynamic_cast<Engine_Ext_LorentzMaterial*>(eng_ext))
		return NULL;
	Operator_Ext_LorentzMaterial* op_ext = dynamic_cast<Operator_Ext_LorentzMaterial*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	// logged: how every conducting sheet runs, and why a Drude/Lorentz material keeps its own kernels
	return new SYCL_Ext_LorentzMaterial(d, op_ext, g_settings.GetVerboseLevel()>1);
}
