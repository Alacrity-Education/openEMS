/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "operator_ext_lorentzmaterial.h"
#include "tools/fingerprint.h"
#include "FDTD/operator_cache.h"
#include "engine_ext_lorentzmaterial.h"
#include "operator_ext_cylinder.h"
#include "../operator_cylinder.h"

#include "CSPropLorentzMaterial.h"
#include "tools/global.h"
#include "tools/useful.h"

#include <cstdlib>
#include <thread>

namespace
{
//! The results of one x-slab of the Lorentz build (one order), merged in slab order
struct LorentzSlab
{
	std::vector<unsigned int> v_pos[3];
	std::vector<double> v_int[3];
	std::vector<double> v_ext[3];
	std::vector<double> i_int[3];
	std::vector<double> i_ext[3];
	std::vector<double> v_Lor[3];
	std::vector<double> i_Lor[3];
	bool volt_ADE = false;
	bool curr_ADE = false;
	bool volt_Lor_ADE = false;
	bool curr_Lor_ADE = false;
	std::vector<CSPrimitives*> used;
};
}

using std::cout;
using std::endl;

void Operator_Ext_LorentzMaterial::BuildSlabs(unsigned int numX, std::vector<unsigned int>& start, std::vector<unsigned int>& stop)
{
	unsigned int threads = AvailableThreads();
	const char* env = getenv("OPENEMS_SHEET_THREADS");
	if (env && (atoi(env)>0))
		threads = atoi(env);
	std::vector<unsigned int> jobs = AssignJobs2Threads(numX, threads, true);
	unsigned int line = 0;
	for (size_t n=0; n<jobs.size(); ++n)
	{
		start.push_back(line);
		line += jobs.at(n);
		stop.push_back(line-1);
	}
}

Operator_Ext_LorentzMaterial::Operator_Ext_LorentzMaterial(Operator* op) : Operator_Ext_Dispersive(op)
{
	v_int_ADE = NULL;
	v_ext_ADE = NULL;
	i_int_ADE = NULL;
	i_ext_ADE = NULL;

	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	m_curr_Lor_ADE_On = NULL;
	m_volt_Lor_ADE_On = NULL;
}

Operator_Ext_LorentzMaterial::Operator_Ext_LorentzMaterial(Operator* op, Operator_Ext_LorentzMaterial* op_ext) : Operator_Ext_Dispersive(op,op_ext)
{
	v_int_ADE = NULL;
	v_ext_ADE = NULL;
	i_int_ADE = NULL;
	i_ext_ADE = NULL;

	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	m_curr_Lor_ADE_On = NULL;
	m_volt_Lor_ADE_On = NULL;
}

Operator_Ext_LorentzMaterial::~Operator_Ext_LorentzMaterial()
{
	if ((m_curr_Lor_ADE_On==NULL) || (m_volt_Lor_ADE_On==NULL)) // all data already cleaned up or never used
		return;
	for (int i=0;i<m_Order;++i)
	{
		for (int n=0; n<3; ++n)
		{
			if (m_volt_ADE_On[i])
			{
				delete[] v_int_ADE[i][n];
				delete[] v_ext_ADE[i][n];
			}
			if (m_curr_ADE_On[i])
			{
				delete[] i_int_ADE[i][n];
				delete[] i_ext_ADE[i][n];
			}
			if (m_volt_Lor_ADE_On[i])
				delete[] v_Lor_ADE[i][n];
			if (m_curr_Lor_ADE_On[i])
				delete[] i_Lor_ADE[i][n];
		}
		if (m_volt_ADE_On[i])
		{
			delete[] v_int_ADE[i];
			delete[] v_ext_ADE[i];
		}
		if (m_curr_ADE_On[i])
		{
			delete[] i_int_ADE[i];
			delete[] i_ext_ADE[i];
		}
		if (m_volt_Lor_ADE_On[i])
			delete[] v_Lor_ADE[i];
		if (m_curr_Lor_ADE_On[i])
			delete[] i_Lor_ADE[i];
	}
	delete[] v_int_ADE;
	delete[] v_ext_ADE;
	delete[] i_int_ADE;
	delete[] i_ext_ADE;
	v_int_ADE = NULL;
	v_ext_ADE = NULL;
	i_int_ADE = NULL;
	i_ext_ADE = NULL;

	delete[] v_Lor_ADE;
	delete[] i_Lor_ADE;
	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	delete[] m_curr_Lor_ADE_On;
	delete[] m_volt_Lor_ADE_On;
	m_curr_Lor_ADE_On = NULL;
	m_volt_Lor_ADE_On = NULL;
}

Operator_Extension* Operator_Ext_LorentzMaterial::Clone(Operator* op)
{
	return new Operator_Ext_LorentzMaterial(op, this);
}

bool Operator_Ext_LorentzMaterial::BuildExtension()
{
	double dT = m_Op->GetTimestep();
	unsigned int numLines[3] = {m_Op->GetNumberOfLines(0,true),m_Op->GetNumberOfLines(1,true),m_Op->GetNumberOfLines(2,true)};

	std::vector<unsigned int> v_pos[3];

	// drude material parameter
	std::vector<double> v_int[3];
	std::vector<double> v_ext[3];
	std::vector<double> i_int[3];
	std::vector<double> i_ext[3];

	//additional Dorentz material parameter
	std::vector<double> v_Lor[3];
	std::vector<double> i_Lor[3];

	m_Order = 0;
	std::vector<CSProperties*> LD_props = m_Op->CSX->GetPropertyByType(CSProperties::LORENTZMATERIAL);
	for (size_t n=0;n<LD_props.size();++n)
	{
		CSPropLorentzMaterial* LorMat = dynamic_cast<CSPropLorentzMaterial*>(LD_props.at(n));
		if (LorMat==NULL)
			return false; //sanity check, this should not happen
		if (LorMat->GetDispersionOrder()>m_Order)
			m_Order=LorMat->GetDispersionOrder();
	}

	// The cells are built by x-slabs in parallel, one thread per slab (OPENEMS_SHEET_THREADS=<n>, default: the
	// --threads count; one slab runs inline). Each slab keeps its own lists, flags and used primitives; they are
	// merged in slab order after the join, so the lists equal the serial ones.
	std::vector<unsigned int> start, stop;
	BuildSlabs(numLines[0], start, stop);
	// A weight function is evaluated by a new CSFunctionParser per query, whose constructor calls setlocale(),
	// which is not thread safe: keep the build serial if any is given.
	for (size_t n=0;(n<LD_props.size()) && (start.size()>1);++n)
	{
		CSPropLorentzMaterial* LorMat = dynamic_cast<CSPropLorentzMaterial*>(LD_props.at(n));
		for (int order=0;order<LorMat->GetDispersionOrder();++order)
			for (int ny=0;ny<3;++ny)
				if (!LorMat->GetEpsPlasmaFreqWeightFunction(order,ny).empty() || !LorMat->GetEpsLorPoleFreqWeightFunction(order,ny).empty() ||
					!LorMat->GetEpsRelaxTimeWeightFunction(order,ny).empty() || !LorMat->GetMuePlasmaFreqWeightFunction(order,ny).empty() ||
					!LorMat->GetMueLorPoleFreqWeightFunction(order,ny).empty() || !LorMat->GetMueRelaxTimeWeightFunction(order,ny).empty())
				{
					start.resize(1);
					stop.assign(1, numLines[0]-1);
				}
	}
	if (g_settings.GetVerboseLevel()>1)
		cout << "Operator_Ext_LorentzMaterial: building with " << start.size() << " threads" << endl;

	m_LM_pos = new unsigned int**[m_Order];

	m_volt_ADE_On = new bool[m_Order];
	m_curr_ADE_On = new bool[m_Order];
	m_volt_Lor_ADE_On = new bool[m_Order];
	m_curr_Lor_ADE_On = new bool[m_Order];

	v_int_ADE = new FDTD_FLOAT**[m_Order];
	v_ext_ADE = new FDTD_FLOAT**[m_Order];
	i_int_ADE = new FDTD_FLOAT**[m_Order];
	i_ext_ADE = new FDTD_FLOAT**[m_Order];

	v_Lor_ADE = new FDTD_FLOAT**[m_Order];
	i_Lor_ADE = new FDTD_FLOAT**[m_Order];

	for (int order=0;order<m_Order;++order)
	{
		m_volt_ADE_On[order]=false;
		m_curr_ADE_On[order]=false;

		m_volt_Lor_ADE_On[order]=false;
		m_curr_Lor_ADE_On[order]=false;

		for (int n=0;n<3;++n)
		{
			v_pos[n].clear();

			v_int[n].clear();
			v_ext[n].clear();
			i_int[n].clear();
			i_ext[n].clear();

			v_Lor[n].clear();
			i_Lor[n].clear();
		}

		std::vector<LorentzSlab> slabs(start.size());
		auto build_slab = [&](size_t s)
		{
			LorentzSlab& slab = slabs[s];
			unsigned int pos[] = {0,0,0};
			double coord[3];
			CSPropLorentzMaterial* mat = NULL;
			bool b_pos_on;
			double w_plasma,t_relax;
			double L_D[3], C_D[3];
			double R_D[3], G_D[3];
			double w_Lor_Pol;
			double C_L[3];
			double L_L[3];
			for (pos[0]=start[s]; pos[0]<=stop[s]; ++pos[0])
			{
				// the voltage loop takes the first hit of the PEC pass at the same edge midpoint against the same column
				// list; with r==0 included its point at r==0 differs from the PEC pass: it queries
				bool firstHit = (m_CC_R0_included==false) && m_Op->HasFirstHit(pos[0]);
				for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
				{
					std::vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(
						pos[0], pos[1], -1,
						(CSProperties::PropertyType)(CSProperties::MATERIAL | CSProperties::METAL)
					);

					for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
					{
						unsigned int index = m_Op->MainOp->GetPos(pos[0],pos[1],pos[2]);
						//calc epsilon lorentz material
						b_pos_on = false;
						for (int n=0; n<3; ++n)
						{
							L_D[n]=0;
							R_D[n]=0;
							C_L[n]=0;
							if (m_Op->GetYeeCoords(n,pos,coord,false)==false)
								continue;
							if (m_CC_R0_included && (n==2) && (pos[0]==0))
								coord[1] = m_Op->GetDiscLine(1,0);

							if (m_Op->GetVI(n,pos[0],pos[1],pos[2])==0)
								continue;

//						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord,(CSProperties::PropertyType)(CSProperties::METAL | CSProperties::MATERIAL), true);
							CSPrimitives* found = NULL;
							CSProperties* prop = NULL;
							if (firstHit)
							{
								found = m_Op->GetFirstHit(n,pos);
								if (found)
									prop = found->GetProperty();
							}
							else
								prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrims, false, &found);
							if (found && (slab.used.empty() || (slab.used.back()!=found)))
								slab.used.push_back(found); //SetPrimitiveUsed(true) after the join

							if (prop==NULL) continue;

							if ((mat = prop->ToLorentzMaterial()))
							{
								w_plasma = mat->GetEpsPlasmaFreqWeighted(order,n,coord) * 2 * PI;
								if ((w_plasma>0) && (m_Op->EC_C[n][index]>0))
								{
									b_pos_on = true;
									slab.volt_ADE = true;
									L_D[n] = 1/(w_plasma*w_plasma*m_Op->EC_C[n][index]);
								}
								t_relax = mat->GetEpsRelaxTimeWeighted(order,n,coord);
								if ((t_relax>0) && slab.volt_ADE)
								{
									R_D[n] = L_D[n]/t_relax;
								}
								w_Lor_Pol = mat->GetEpsLorPoleFreqWeighted(order,n,coord) * 2 * PI;
								if ((w_Lor_Pol>0) && (L_D[n]>0))
								{
									slab.volt_Lor_ADE = true;
									C_L[n] = 1/(w_Lor_Pol*w_Lor_Pol*L_D[n]);
								}
							}
						}

						for (int n=0; n<3; ++n)
						{
							C_D[n]=0;
							G_D[n]=0;
							L_L[n]=0;
							if (m_Op->GetYeeCoords(n,pos,coord,true)==false)
								continue;
							if (m_Op->GetIV(n,pos[0],pos[1],pos[2])==0)
								continue;

//						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord,(CSProperties::PropertyType)(CSProperties::METAL | CSProperties::MATERIAL), true);
							CSPrimitives* found = NULL;
							CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrims, false, &found);
							if (found && (slab.used.empty() || (slab.used.back()!=found)))
								slab.used.push_back(found); //SetPrimitiveUsed(true) after the join

							if (prop==NULL) continue;

							if ((mat = prop->ToLorentzMaterial()))
							{
								w_plasma = mat->GetMuePlasmaFreqWeighted(order,n,coord) * 2 * PI;
								if ((w_plasma>0) && (m_Op->EC_L[n][index]>0))
								{
									b_pos_on = true;
									slab.curr_ADE = true;
									C_D[n] = 1/(w_plasma*w_plasma*m_Op->EC_L[n][index]);
								}
								t_relax = mat->GetMueRelaxTimeWeighted(order,n,coord);
								if ((t_relax>0) && slab.curr_ADE)
								{
									G_D[n] = C_D[n]/t_relax;
								}
								w_Lor_Pol = mat->GetMueLorPoleFreqWeighted(order,n,coord) * 2 * PI;
								if ((w_Lor_Pol>0) && (C_D[n]>0))
								{
									slab.curr_Lor_ADE = true;
									L_L[n] = 1/(w_Lor_Pol*w_Lor_Pol*C_D[n]);
								}
							}
						}

						if (b_pos_on) //this position has active drude material
						{
							for (unsigned int n=0; n<3; ++n)
							{
								slab.v_pos[n].push_back(pos[n]);
								if (L_D[n]>0)
								{
									slab.v_int[n].push_back((2.0*L_D[n]-dT*R_D[n])/(2.0*L_D[n]+dT*R_D[n]));
									// check for r==0 in clyindrical coords and get special VI coefficient
									if (m_CC_R0_included && n==2 && pos[0]==0)
										slab.v_ext[n].push_back(dT/(L_D[n]+dT*R_D[n]/2.0)*m_Op_Cyl->m_Cyl_Ext->vi_R0[pos[2]]);
									else
										slab.v_ext[n].push_back(dT/(L_D[n]+dT*R_D[n]/2.0)*m_Op->GetVI(n,pos[0],pos[1],pos[2]));
								}
								else
								{
									slab.v_int[n].push_back(1);
									slab.v_ext[n].push_back(0);
								}
								if (C_D[n]>0)
								{
									slab.i_int[n].push_back((2.0*C_D[n]-dT*G_D[n])/(2.0*C_D[n]+dT*G_D[n]));
									slab.i_ext[n].push_back(dT/(C_D[n]+dT*G_D[n]/2.0)*m_Op->GetIV(n,pos[0],pos[1],pos[2]));
								}
								else
								{
									slab.i_int[n].push_back(1);
									slab.i_ext[n].push_back(0);
								}
								if (C_L[n]>0)
									slab.v_Lor[n].push_back(dT/C_L[n]/m_Op->GetVI(n,pos[0],pos[1],pos[2]));
								else
									slab.v_Lor[n].push_back(0);
								if (L_L[n]>0)
									slab.i_Lor[n].push_back(dT/L_L[n]/m_Op->GetIV(n,pos[0],pos[1],pos[2]));
								else
									slab.i_Lor[n].push_back(0);
							}
						}
					}
				}
			}
		};
		m_Op->MainOp->SetPos(0,0,0);
		if (start.size()>1)
		{
			std::vector<std::thread> threads;
			for (size_t s=0; s<start.size(); ++s)
				threads.push_back(std::thread(build_slab, s));
			for (size_t s=0; s<threads.size(); ++s)
				threads[s].join();
		}
		else
			build_slab(0);
		m_Op->MainOp->SetPos(numLines[0]-1,numLines[1]-1,numLines[2]-1); //where the serial loop left it

		// merge in slab order
		for (size_t s=0; s<slabs.size(); ++s)
		{
			m_volt_ADE_On[order] = m_volt_ADE_On[order] || slabs[s].volt_ADE;
			m_curr_ADE_On[order] = m_curr_ADE_On[order] || slabs[s].curr_ADE;
			m_volt_Lor_ADE_On[order] = m_volt_Lor_ADE_On[order] || slabs[s].volt_Lor_ADE;
			m_curr_Lor_ADE_On[order] = m_curr_Lor_ADE_On[order] || slabs[s].curr_Lor_ADE;
			for (int n=0; n<3; ++n)
			{
				v_pos[n].insert(v_pos[n].end(), slabs[s].v_pos[n].begin(), slabs[s].v_pos[n].end());
				v_int[n].insert(v_int[n].end(), slabs[s].v_int[n].begin(), slabs[s].v_int[n].end());
				v_ext[n].insert(v_ext[n].end(), slabs[s].v_ext[n].begin(), slabs[s].v_ext[n].end());
				i_int[n].insert(i_int[n].end(), slabs[s].i_int[n].begin(), slabs[s].i_int[n].end());
				i_ext[n].insert(i_ext[n].end(), slabs[s].i_ext[n].begin(), slabs[s].i_ext[n].end());
				v_Lor[n].insert(v_Lor[n].end(), slabs[s].v_Lor[n].begin(), slabs[s].v_Lor[n].end());
				i_Lor[n].insert(i_Lor[n].end(), slabs[s].i_Lor[n].begin(), slabs[s].i_Lor[n].end());
			}
			for (size_t i=0; i<slabs[s].used.size(); ++i)
				slabs[s].used[i]->SetPrimitiveUsed(true);
		}

		//copy all vectors into the array's
		m_LM_Count.push_back(v_pos[0].size());

		m_LM_pos[order] = new unsigned int*[3];

		if (m_volt_ADE_On[order])
		{
			v_int_ADE[order] = new FDTD_FLOAT*[3];
			v_ext_ADE[order] = new FDTD_FLOAT*[3];
		}
		else
		{
			v_int_ADE[order] = NULL;
			v_ext_ADE[order] = NULL;
		}

		if (m_curr_ADE_On[order])
		{
			i_int_ADE[order] = new FDTD_FLOAT*[3];
			i_ext_ADE[order] = new FDTD_FLOAT*[3];
		}
		else
		{
			i_int_ADE[order] = NULL;
			i_ext_ADE[order] = NULL;
		}

		if (m_volt_Lor_ADE_On[order])
			v_Lor_ADE[order] = new FDTD_FLOAT*[3];
		else
			v_Lor_ADE[order] = NULL;

		if (m_curr_Lor_ADE_On[order])
			i_Lor_ADE[order] = new FDTD_FLOAT*[3];
		else
			i_Lor_ADE[order] = NULL;

		for (int n=0; n<3; ++n)
		{
			m_LM_pos[order][n] = new unsigned int[m_LM_Count.at(order)];
			for (size_t i=0; i<m_LM_Count.at(order); ++i)
				m_LM_pos[order][n][i] = v_pos[n].at(i);
			if (m_volt_ADE_On[order])
			{
				v_int_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				v_ext_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];

				for (size_t i=0; i<m_LM_Count.at(order); ++i)
				{
					v_int_ADE[order][n][i] = v_int[n].at(i);
					v_ext_ADE[order][n][i] = v_ext[n].at(i);
				}
			}
			if (m_curr_ADE_On[order])
			{
				i_int_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				i_ext_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];

				for (size_t i=0; i<m_LM_Count.at(order); ++i)
				{
					i_int_ADE[order][n][i] = i_int[n].at(i);
					i_ext_ADE[order][n][i] = i_ext[n].at(i);
				}
			}

			if (m_volt_Lor_ADE_On[order])
			{
				v_Lor_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				for (size_t i=0; i<m_LM_Count.at(order); ++i)
					v_Lor_ADE[order][n][i] = v_Lor[n].at(i);
			}
			if (m_curr_Lor_ADE_On[order])
			{
				i_Lor_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				for (size_t i=0; i<m_LM_Count.at(order); ++i)
					i_Lor_ADE[order][n][i] = i_Lor[n].at(i);
			}
		}
	}

	return true;
}

Engine_Extension* Operator_Ext_LorentzMaterial::CreateEngineExtention()
{
	Engine_Ext_LorentzMaterial* eng_ext_lor = new Engine_Ext_LorentzMaterial(this);
	return eng_ext_lor;
}

std::string Operator_Ext_LorentzMaterial::Fingerprint() const
{
	FNVHash hash;
	FingerprintBase(hash);
	for (int o=0; o<m_Order; ++o)
	{
		bool volt_on = m_volt_ADE_On && m_volt_ADE_On[o];
		bool curr_on = m_curr_ADE_On && m_curr_ADE_On[o];
		bool volt_lor = m_volt_Lor_ADE_On && m_volt_Lor_ADE_On[o];
		bool curr_lor = m_curr_Lor_ADE_On && m_curr_Lor_ADE_On[o];
		hash.AddValue((uint8_t)volt_lor);
		hash.AddValue((uint8_t)curr_lor);
		size_t count = ((size_t)o<m_LM_Count.size()) ? m_LM_Count[o] : 0;
		for (int n=0; n<3; ++n)
		{
			hash.AddArray((volt_on && v_int_ADE) ? v_int_ADE[o][n] : NULL, count);
			hash.AddArray((volt_on && v_ext_ADE) ? v_ext_ADE[o][n] : NULL, count);
			hash.AddArray((curr_on && i_int_ADE) ? i_int_ADE[o][n] : NULL, count);
			hash.AddArray((curr_on && i_ext_ADE) ? i_ext_ADE[o][n] : NULL, count);
			hash.AddArray((volt_lor && v_Lor_ADE) ? v_Lor_ADE[o][n] : NULL, count);
			hash.AddArray((curr_lor && i_Lor_ADE) ? i_Lor_ADE[o][n] : NULL, count);
		}
	}
	return hash.Hex();
}

std::string Operator_Ext_LorentzMaterial::FingerprintNote() const
{
	std::string positions;
	for (size_t o=0; o<m_LM_Count.size(); ++o)
		positions += (o ? "/" : "") + std::to_string(m_LM_Count[o]);
	return std::to_string(m_Order) + " orders, positions " + positions;
}

bool Operator_Ext_LorentzMaterial::Serialize(std::ostream& out) const
{
	SerializeBase(out);
	FDTD_FLOAT*** arrays[6] = {v_int_ADE, v_ext_ADE, i_int_ADE, i_ext_ADE, v_Lor_ADE, i_Lor_ADE};
	uint8_t has[8] = {m_volt_Lor_ADE_On!=NULL, m_curr_Lor_ADE_On!=NULL};
	for (int a=0; a<6; ++a)
		has[2+a] = arrays[a]!=NULL;
	Operator_Cache::Write(out, has, 8);
	for (int o=0; o<m_Order; ++o)
	{
		if (has[0])
			Operator_Cache::WriteValue(out, (uint8_t)m_volt_Lor_ADE_On[o]);
		if (has[1])
			Operator_Cache::WriteValue(out, (uint8_t)m_curr_Lor_ADE_On[o]);
		size_t count = ((size_t)o<m_LM_Count.size()) ? m_LM_Count[o] : 0;
		for (int a=0; a<6; ++a)
		{
			// an array of an order exists if its ADE is on
			uint8_t inner = (arrays[a]!=NULL) && (arrays[a][o]!=NULL);
			Operator_Cache::WriteValue(out, inner);
			for (int n=0; inner && (n<3); ++n)
				Operator_Cache::Write(out, arrays[a][o][n], count);
		}
	}
	return true;
}

bool Operator_Ext_LorentzMaterial::Deserialize(std::istream& in)
{
	DeserializeBase(in);
	FDTD_FLOAT**** arrays[6] = {&v_int_ADE, &v_ext_ADE, &i_int_ADE, &i_ext_ADE, &v_Lor_ADE, &i_Lor_ADE};
	uint8_t has[8];
	Operator_Cache::Read(in, has, 8);
	if (has[0])
		m_volt_Lor_ADE_On = new bool[m_Order];
	if (has[1])
		m_curr_Lor_ADE_On = new bool[m_Order];
	for (int a=0; a<6; ++a)
		*arrays[a] = has[2+a] ? new FDTD_FLOAT**[m_Order] : NULL;
	for (int o=0; o<m_Order; ++o)
	{
		if (has[0])
			m_volt_Lor_ADE_On[o] = Operator_Cache::ReadValue<uint8_t>(in)!=0;
		if (has[1])
			m_curr_Lor_ADE_On[o] = Operator_Cache::ReadValue<uint8_t>(in)!=0;
		size_t count = ((size_t)o<m_LM_Count.size()) ? m_LM_Count[o] : 0;
		for (int a=0; a<6; ++a)
		{
			uint8_t inner = Operator_Cache::ReadValue<uint8_t>(in);
			if (*arrays[a]==NULL)
			{
				if (inner)
					return false;
				continue;
			}
			(*arrays[a])[o] = inner ? new FDTD_FLOAT*[3] : NULL;
			for (int n=0; inner && (n<3); ++n)
			{
				(*arrays[a])[o][n] = new FDTD_FLOAT[count];
				Operator_Cache::Read(in, (*arrays[a])[o][n], count);
			}
		}
	}
	return true;
}

void Operator_Ext_LorentzMaterial::ShowStat(std::ostream &ostr)  const
{
	Operator_Extension::ShowStat(ostr);
	std::string On_Off[2] = {"Off", "On"};
	ostr << " Max. Dispersion Order N = " << m_Order << endl;
	for (int i=0;i<m_Order;++i)
	{
		ostr << " N=" << i << ":\t Active cells\t\t: " << 	m_LM_Count.at(i) << endl;
		ostr << " N=" << i << ":\t Voltage ADE is \t: " << On_Off[m_volt_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Voltage Lor-ADE is \t: " << On_Off[m_volt_Lor_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Current ADE is \t: " << On_Off[m_curr_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Current Lor-ADE is \t: " << On_Off[m_curr_Lor_ADE_On[i]] << endl;
	}
}
