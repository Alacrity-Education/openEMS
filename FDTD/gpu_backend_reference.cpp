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

#include <algorithm>
#include <stdexcept>

#include "gpu_backend_reference.h"
#include "gpu_setup_kernels.h"
#include "operator.h"

GPU_Backend_Reference::GPU_Backend_Reference()
{
	for (int n=0; n<3; ++n)
		numLines[n] = 0;
}

bool GPU_Backend_Reference::Init(const Operator* op)
{
	for (int n=0; n<3; ++n)
		numLines[n] = op->GetNumberOfLines(n, true);

	m_volt.Init("gpu_volt", numLines);
	m_curr.Init("gpu_curr", numLines);
	m_vv.Init("gpu_vv", numLines);
	m_vi.Init("gpu_vi", numLines);
	m_ii.Init("gpu_ii", numLines);
	m_iv.Init("gpu_iv", numLines);

	// upload the final operator coefficients, including all changes by operator extensions
	unsigned int pos[3];
	for (int n=0; n<3; ++n)
		for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
			for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
				for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
				{
					m_vv(n, pos[0], pos[1], pos[2]) = op->GetVV(n, pos[0], pos[1], pos[2]);
					m_vi(n, pos[0], pos[1], pos[2]) = op->GetVI(n, pos[0], pos[1], pos[2]);
					m_ii(n, pos[0], pos[1], pos[2]) = op->GetII(n, pos[0], pos[1], pos[2]);
					m_iv(n, pos[0], pos[1], pos[2]) = op->GetIV(n, pos[0], pos[1], pos[2]);
				}
	return true;
}

// operation by operation like Engine::UpdateVoltages(): reordering the float
// operations alone would break the bit-identical results
void GPU_Backend_Reference::UpdateVoltages()
{
	unsigned int pos[3];
	bool shift[3];

	// shift is 0 on the lower boundary, where the neighbour term cancels itself
	for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
	{
		shift[0]=pos[0];
		for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
		{
			shift[1]=pos[1];
			for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
			{
				shift[2]=pos[2];
				//for x
				m_volt(0, pos[0], pos[1], pos[2]) *= m_vv(0, pos[0], pos[1], pos[2]);
				m_volt(0, pos[0], pos[1], pos[2]) +=
				    m_vi(0, pos[0], pos[1], pos[2]) * (
				        m_curr(2, pos[0], pos[1]         , pos[2]         ) -
				        m_curr(2, pos[0], pos[1]-shift[1], pos[2]         ) -
				        m_curr(1, pos[0], pos[1]         , pos[2]         ) +
				        m_curr(1, pos[0], pos[1]         , pos[2]-shift[2])
				    );

				//for y
				m_volt(1, pos[0], pos[1], pos[2]) *= m_vv(1, pos[0], pos[1], pos[2]);
				m_volt(1, pos[0], pos[1], pos[2]) +=
				    m_vi(1, pos[0], pos[1], pos[2]) * (
				        m_curr(0, pos[0]         , pos[1], pos[2]         ) -
				        m_curr(0, pos[0]         , pos[1], pos[2]-shift[2]) -
				        m_curr(2, pos[0]         , pos[1], pos[2]         ) +
				        m_curr(2, pos[0]-shift[0], pos[1], pos[2]         )
				    );

				//for z
				m_volt(2, pos[0], pos[1], pos[2]) *= m_vv(2, pos[0], pos[1], pos[2]);
				m_volt(2, pos[0], pos[1], pos[2]) +=
				    m_vi(2, pos[0], pos[1], pos[2]) * (
				        m_curr(1, pos[0]         , pos[1]         , pos[2]) -
				        m_curr(1, pos[0]-shift[0], pos[1]         , pos[2]) -
				        m_curr(0, pos[0]         , pos[1]         , pos[2]) +
				        m_curr(0, pos[0]         , pos[1]-shift[1], pos[2])
				    );
			}
		}
	}
}

// see Engine::UpdateCurrents()
void GPU_Backend_Reference::UpdateCurrents()
{
	unsigned int pos[3];

	// the last line of each direction has no upper neighbour and stays untouched
	for (pos[0]=0; pos[0]<numLines[0]-1; ++pos[0])
	{
		for (pos[1]=0; pos[1]<numLines[1]-1; ++pos[1])
		{
			for (pos[2]=0; pos[2]<numLines[2]-1; ++pos[2])
			{
				//for x
				m_curr(0, pos[0], pos[1], pos[2]) *= m_ii(0, pos[0], pos[1], pos[2]);
				m_curr(0, pos[0], pos[1], pos[2]) +=
				    m_iv(0, pos[0], pos[1], pos[2]) * (
				        m_volt(2, pos[0], pos[1]  , pos[2]  ) -
				        m_volt(2, pos[0], pos[1]+1, pos[2]  ) -
				        m_volt(1, pos[0], pos[1]  , pos[2]  ) +
				        m_volt(1, pos[0], pos[1]  , pos[2]+1)
				    );

				//for y
				m_curr(1, pos[0], pos[1], pos[2]) *= m_ii(1, pos[0], pos[1], pos[2]);
				m_curr(1, pos[0], pos[1], pos[2]) +=
				    m_iv(1, pos[0], pos[1], pos[2]) * (
				        m_volt(0, pos[0]  , pos[1], pos[2]  ) -
				        m_volt(0, pos[0]  , pos[1], pos[2]+1) -
				        m_volt(2, pos[0]  , pos[1], pos[2]  ) +
				        m_volt(2, pos[0]+1, pos[1], pos[2]  )
				    );

				//for z
				m_curr(2, pos[0], pos[1], pos[2]) *= m_ii(2, pos[0], pos[1], pos[2]);
				m_curr(2, pos[0], pos[1], pos[2]) +=
				    m_iv(2, pos[0], pos[1], pos[2]) * (
				        m_volt(1, pos[0]  , pos[1]  , pos[2]) -
				        m_volt(1, pos[0]+1, pos[1]  , pos[2]) -
				        m_volt(0, pos[0]  , pos[1]  , pos[2]) +
				        m_volt(0, pos[0]  , pos[1]+1, pos[2])
				    );
			}
		}
	}
}

static void CopyField(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& src, ArrayLib::ArrayNIJK<FDTD_FLOAT>& dst)
{
	if (src.size()!=dst.size())
		throw std::runtime_error("GPU_Backend_Reference: host and device field size mismatch");
	std::copy_n(src.data(), src.size(), dst.data());
}

void GPU_Backend_Reference::DownloadVoltages(ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CopyField(m_volt, volt);
}

void GPU_Backend_Reference::DownloadCurrents(ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CopyField(m_curr, curr);
}

void GPU_Backend_Reference::UploadVoltages(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CopyField(volt, m_volt);
}

void GPU_Backend_Reference::UploadCurrents(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CopyField(curr, m_curr);
}

bool GPU_Backend_Reference::SetupOperator(const GPU_SetupModel& model, GPU_SetupResult& result)
{
	// the model as the raw pointers a kernel gets
	GPU_SetupView v;
	for (int n=0; n<3; ++n)
	{
		v.numLines[n] = model.numLines[n];
		v.lines[n] = model.lines[n].data();
	}
	v.gridDelta = model.gridDelta;
	v.tol = model.tol;
	for (int k=0; k<4; ++k)
		v.bg[k] = model.bg[k];
	v.properties = model.properties.data();
	v.primitives = model.primitives.data();
	v.vertices = model.vertices.data();
	v.matList = model.matList.data();
	v.matCount = (int)model.matList.size();
	v.edgeList = model.edgeList.data();
	v.edgeCount = (int)model.edgeList.size();

	const size_t nx = v.numLines[0], ny = v.numLines[1], nz = v.numLines[2];
	const size_t cells = nx*ny*nz;

	for (size_t i=0; i<model.primitives.size(); ++i)
		result.used[i] = 0;
	result.nanCount = 0;
	result.nanN = result.nanK = 0;
	result.nanPos[0] = result.nanPos[1] = result.nanPos[2] = 0;

	// one work-item per column (x, y), the loop order (x, y, z, n): the column lists, the samples and edge values, the first hits
	std::vector<int> matCol, edgeCol, edgeIdx;
	unsigned int pos[3];
	for (pos[0]=0; pos[0]<v.numLines[0]; ++pos[0])
		for (pos[1]=0; pos[1]<v.numLines[1]; ++pos[1])
		{
			// the column lists: the filter of Operator::GetPrimitivesBoundBox (operator.cpp 2123) on the lists of the pass, order preserved
			double boundBox[6];
			GPU_Setup::ColumnBox(v, pos[0], pos[1], boundBox);
			matCol.clear();
			for (int j=0; j<v.matCount; ++j)
				if (GPU_Setup::InsideBox(v.primitives[v.matList[j]], boundBox)>=0)
					matCol.push_back(v.matList[j]);
			edgeCol.clear();
			edgeIdx.clear();
			for (int j=0; j<v.edgeCount; ++j)
				if (GPU_Setup::InsideBox(v.primitives[v.edgeList[j]], boundBox)>=0)
				{
					edgeCol.push_back(v.edgeList[j]);
					edgeIdx.push_back(j);
				}

			for (pos[2]=0; pos[2]<v.numLines[2]; ++pos[2])
			{
				const size_t c = pos[0] + nx*(pos[1] + ny*(size_t)pos[2]);
				for (int n=0; n<3; ++n)
				{
					// Operator::Calc_ECPos (operator.cpp 1408)
					double EffMat[4];
					int badK = 0;
					if (GPU_Setup::QuarterCell(v, n, pos, EffMat, matCol.data(), (int)matCol.size(), result.used, &badK)==false)
					{
						result.nanCount = 1;
						result.nanN = n;
						result.nanK = badK;
						for (int k=0; k<3; ++k)
							result.nanPos[k] = pos[k];
						return true;
					}
					for (int k=0; k<4; ++k)
						if (result.storage[k])
							result.storage[k][n*cells + c] = EffMat[k];
					double EC[4];
					GPU_Setup::EdgeValues(v, n, pos, EffMat, EC);
					for (int k=0; k<4; ++k)
						result.EC[k][n*cells + c] = EC[k];
				}
				for (int n=0; n<3; ++n)
				{
					// Operator::CalcPEC_Range (operator.cpp 2441): the midpoint of the primal edge and the first primitive of the column list
					double coord[3];
					GPU_Setup::YeeCoords(v, n, pos, coord, false);
					int i = GPU_Setup::FirstHit(v, coord, edgeCol.data(), (int)edgeCol.size());
					int32_t hit = -1;
					if (i>=0)
					{
						hit = edgeIdx[i];
						result.used[edgeCol[i]] = 1;
					}
					result.firstHit[((n*nx + pos[0])*ny + pos[1])*nz + pos[2]] = hit;
				}
			}
		}
	return true;
}
