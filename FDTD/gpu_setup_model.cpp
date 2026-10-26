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

#include <limits>
#include <map>
#include <sstream>

#include "gpu_setup_model.h"
#include "operator.h"

#include "CSPrimBox.h"
#include "CSPrimPolygon.h"
#include "CSPrimLinPoly.h"
#include "CSPropMaterial.h"

namespace
{
std::string PrimitiveText(CSPrimitives* prim, const char* what)
{
	std::stringstream ss;
	ss << "primitive " << prim->GetID() << " (" << prim->GetTypeName() << ") " << what;
	return ss.str();
}
}

bool GPU_SetupModel::Qualify(const Operator* op, std::string& reason)
{
	GPU_SetupModel model;
	return model.Build(op, reason);
}

bool GPU_SetupModel::Build(const Operator* op, std::string& reason)
{
	reason.clear();
	properties.clear();
	primitives.clear();
	vertices.clear();
	matList.clear();
	edgeList.clear();
	primPtr.clear();

	if ((op->CSX==NULL) || (op->m_MeshType!=CARTESIAN))
	{
		reason = "mesh not cartesian";
		return false;
	}
	if (op->m_MatAverageMethod!=Operator::QuarterCell)
	{
		reason = "material averaging is not the quarter cell method";
		return false;
	}
	for (int n=0; n<3; ++n)
		if (op->numLines[n]<2)
		{
			reason = "mesh has fewer than two lines in a direction";
			return false;
		}

	// the two lists in the orders the CPU path searches them (BeginPassPrimitives)
	std::vector<CSPrimitives*> matPrims = op->CSX->GetAllPrimitives(true, CSProperties::MATERIAL);
	std::vector<CSPrimitives*> edgePrims = op->CSX->GetAllPrimitives(true, (CSProperties::PropertyType)(CSProperties::MATERIAL | CSProperties::METAL));
	if (edgePrims.size()>(size_t)std::numeric_limits<int32_t>::max())
	{
		reason = "too many primitives";
		return false;
	}

	std::map<CSPrimitives*, int> index;
	std::map<CSProperties*, int> propIndex;
	for (int list=0; list<2; ++list)
	{
		const std::vector<CSPrimitives*>& prims = (list==0) ? edgePrims : matPrims;
		for (size_t i=0; i<prims.size(); ++i)
		{
			CSPrimitives* prim = prims[i];
			if (index.find(prim)==index.end())
			{
				int type = -1;
				switch (prim->GetType())
				{
				case CSPrimitives::BOX:
					if (dynamic_cast<CSPrimBox*>(prim)) type = 0;
					break;
				case CSPrimitives::POLYGON:
					if (dynamic_cast<CSPrimPolygon*>(prim)) type = 1;
					break;
				case CSPrimitives::LINPOLY:
					if (dynamic_cast<CSPrimLinPoly*>(prim)) type = 2;
					break;
				default:
					break;
				}
				if (type<0)
				{
					reason = PrimitiveText(prim, "not a box or polygon");
					return false;
				}
				if (prim->HasTransform())
				{
					reason = PrimitiveText(prim, "has a transformation");
					return false;
				}
				if (prim->GetCoordInputType()!=CARTESIAN)
				{
					reason = PrimitiveText(prim, "is not in cartesian coordinates");
					return false;
				}
				if ((prim->GetCoordinateSystem()!=CARTESIAN) && (prim->GetCoordinateSystem()!=UNDEFINED_CS))
				{
					reason = PrimitiveText(prim, "has a non cartesian coordinate system");
					return false;
				}

				CSProperties* prop = prim->GetProperty();
				if (prop==NULL)
				{
					reason = PrimitiveText(prim, "has no property");
					return false;
				}
				int pi;
				std::map<CSProperties*, int>::iterator pit = propIndex.find(prop);
				if (pit==propIndex.end())
				{
					CSPropMaterial* mat = dynamic_cast<CSPropMaterial*>(prop);
					if (prop->GetType() & CSProperties::DISCRETE_MATERIAL)
					{
						reason = "material '" + prop->GetName() + "' is a discrete material";
						return false;
					}
					GPU_SetupProperty gp;
					for (int ny=0; ny<3; ++ny)
						gp.eps[ny] = gp.kappa[ny] = gp.mue[ny] = gp.sigma[ny] = 0;
					gp.isMaterial = (mat!=NULL);
					gp.isMetal = (prop->GetType()==CSProperties::METAL);
					if (mat)
					{
						if (mat->HasConstantWeights()==false)
						{
							reason = "material '" + prop->GetName() + "' has a weight function";
							return false;
						}
						// the weights are constants: the weighted getters do not read the coordinates
						const double c0[3] = {0, 0, 0};
						for (int ny=0; ny<3; ++ny)
						{
							gp.eps[ny] = mat->GetEpsilonWeighted(ny, c0);
							gp.kappa[ny] = mat->GetKappaWeighted(ny, c0);
							gp.mue[ny] = mat->GetMueWeighted(ny, c0);
							gp.sigma[ny] = mat->GetSigmaWeighted(ny, c0);
						}
					}
					pi = (int)properties.size();
					properties.push_back(gp);
					propIndex[prop] = pi;
				}
				else
					pi = pit->second;

				GPU_SetupPrimitive gp;
				gp.type = type;
				gp.property = pi;
				for (int k=0; k<6; ++k)
					gp.bbox[k] = 0;
				for (int k=0; k<3; ++k)
					gp.start[k] = gp.stop[k] = 0;
				gp.normDir = 0;
				gp.vertexOffset = 0;
				gp.vertexCount = 0;

				// can IsInsideBox() decide? a probe box that no bounding box is separated from returns 1 when it can, else 0 (read before anything updates the primitive)
				const double probe[6] = {-1e300, 1e300, -1e300, 1e300, -1e300, 1e300};
				gp.bboxFilter = (prim->IsInsideBox(probe)!=0);

				if (type==0)
				{
					CSPrimBox* box = static_cast<CSPrimBox*>(prim);
					prim->GetBoundBox(gp.bbox);
					const double* start = box->GetStartCoord()->GetCoords(prim->GetCoordinateSystem());
					const double* stop = box->GetStopCoord()->GetCoords(prim->GetCoordinateSystem());
					for (int k=0; k<3; ++k)
					{
						gp.start[k] = start[k];
						gp.stop[k] = stop[k];
					}
				}
				else
				{
					CSPrimPolygon* poly = static_cast<CSPrimPolygon*>(prim);
					prim->GetDimension(); // CSPrimPolygon::IsInside() updates a primitive without an evaluated bounding box before its first test
					prim->GetBoundBox(gp.bbox);
					gp.normDir = poly->GetNormDir();
					size_t qty = 0;
					double* coords = poly->GetAllCoords(qty, NULL);
					gp.vertexCount = (qty<2) ? 0 : qty/2;
					gp.vertexOffset = vertices.size();
					for (size_t k=0; k<2*gp.vertexCount; ++k)
						vertices.push_back(coords[k]);
					delete[] coords;
				}
				index[prim] = (int)primitives.size();
				primitives.push_back(gp);
				primPtr.push_back(prim);
			}
		}
	}
	for (size_t i=0; i<edgePrims.size(); ++i)
		edgeList.push_back(index[edgePrims[i]]);
	for (size_t i=0; i<matPrims.size(); ++i)
		matList.push_back(index[matPrims[i]]);

	for (int n=0; n<3; ++n)
	{
		numLines[n] = op->numLines[n];
		lines[n].assign(op->discLines[n], op->discLines[n] + op->numLines[n]);
	}
	gridDelta = op->gridDelta;
	tol = op->CSX->GetDrawingTolerance();
	bg[0] = op->GetBackgroundEpsR();
	bg[1] = op->GetBackgroundKappa();
	bg[2] = op->GetBackgroundMueR();
	bg[3] = op->GetBackgroundSigma();
	storage[0] = (op->m_epsR_ptr!=NULL);
	storage[1] = (op->m_kappa_ptr!=NULL);
	storage[2] = (op->m_mueR_ptr!=NULL);
	storage[3] = (op->m_sigma_ptr!=NULL);
	return true;
}
