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

#include "operator_ext_dispersive.h"
#include "tools/fingerprint.h"
#include "FDTD/operator_cache.h"

using namespace std;

Operator_Ext_Dispersive::Operator_Ext_Dispersive(Operator* op) : Operator_Extension(op)
{
	m_curr_ADE_On = NULL;
	m_volt_ADE_On = NULL;

	m_LM_pos=NULL;
	m_curr_ADE_On=NULL;
	m_volt_ADE_On=NULL;

	m_Order = 0;
}

Operator_Ext_Dispersive::Operator_Ext_Dispersive(Operator* op, Operator_Ext_Dispersive* op_ext) : Operator_Extension(op,op_ext)
{
	m_curr_ADE_On = NULL;
	m_volt_ADE_On = NULL;

	m_LM_pos=NULL;
	m_curr_ADE_On=NULL;
	m_volt_ADE_On=NULL;

	m_Order = 0;
}

Operator_Ext_Dispersive::~Operator_Ext_Dispersive()
{
	delete[] m_curr_ADE_On;
	delete[] m_volt_ADE_On;
	m_curr_ADE_On=NULL;
	m_volt_ADE_On=NULL;

	for (int n=0;n<m_Order;++n)
	{
		delete[] m_LM_pos[n][0];
		delete[] m_LM_pos[n][1];
		delete[] m_LM_pos[n][2];
		delete[] m_LM_pos[n];
	}
	delete[] m_LM_pos;
	m_LM_pos=NULL;
	m_Order=0;
	m_LM_Count.clear();
}

void Operator_Ext_Dispersive::FingerprintBase(FNVHash& hash) const
{
	hash.AddValue((int32_t)m_Order);
	hash.AddValue((uint64_t)m_LM_Count.size());
	for (size_t o=0; o<m_LM_Count.size(); ++o)
		hash.AddValue((uint64_t)m_LM_Count[o]);
	for (int o=0; o<m_Order; ++o)
	{
		hash.AddValue((uint8_t)(m_volt_ADE_On ? m_volt_ADE_On[o] : false));
		hash.AddValue((uint8_t)(m_curr_ADE_On ? m_curr_ADE_On[o] : false));
		for (int n=0; n<3; ++n)
			hash.AddArray((m_LM_pos && (size_t)o<m_LM_Count.size()) ? m_LM_pos[o][n] : NULL, ((size_t)o<m_LM_Count.size()) ? m_LM_Count[o] : 0);
	}
}

void Operator_Ext_Dispersive::SerializeBase(std::ostream& out) const
{
	Operator_Cache::WriteValue(out, (int32_t)m_Order);
	Operator_Cache::WriteValue(out, (uint64_t)m_LM_Count.size());
	for (size_t o=0; o<m_LM_Count.size(); ++o)
		Operator_Cache::WriteValue(out, (uint64_t)m_LM_Count[o]);
	uint8_t has[3] = {m_volt_ADE_On!=NULL, m_curr_ADE_On!=NULL, m_LM_pos!=NULL};
	Operator_Cache::Write(out, has, 3);
	for (int o=0; o<m_Order; ++o)
	{
		if (has[0])
			Operator_Cache::WriteValue(out, (uint8_t)m_volt_ADE_On[o]);
		if (has[1])
			Operator_Cache::WriteValue(out, (uint8_t)m_curr_ADE_On[o]);
		size_t count = ((size_t)o<m_LM_Count.size()) ? m_LM_Count[o] : 0;
		for (int n=0; has[2] && (n<3); ++n)
			Operator_Cache::Write(out, m_LM_pos[o][n], count);
	}
}

void Operator_Ext_Dispersive::DeserializeBase(std::istream& in)
{
	m_Order = Operator_Cache::ReadValue<int32_t>(in);
	uint64_t orders = Operator_Cache::ReadValue<uint64_t>(in);
	if ((m_Order<0) || (m_Order>1024) || (orders>1024))
		throw std::runtime_error("operator cache: implausible dispersion order in an extension file");
	m_LM_Count.clear();
	for (uint64_t o=0; o<orders; ++o)
		m_LM_Count.push_back((size_t)Operator_Cache::ReadValue<uint64_t>(in));
	uint8_t has[3];
	Operator_Cache::Read(in, has, 3);
	if (has[0])
		m_volt_ADE_On = new bool[m_Order];
	if (has[1])
		m_curr_ADE_On = new bool[m_Order];
	if (has[2])
		m_LM_pos = new unsigned int**[m_Order];
	for (int o=0; o<m_Order; ++o)
	{
		if (has[0])
			m_volt_ADE_On[o] = Operator_Cache::ReadValue<uint8_t>(in)!=0;
		if (has[1])
			m_curr_ADE_On[o] = Operator_Cache::ReadValue<uint8_t>(in)!=0;
		size_t count = ((size_t)o<m_LM_Count.size()) ? m_LM_Count[o] : 0;
		if (has[2])
			m_LM_pos[o] = new unsigned int*[3];
		for (int n=0; has[2] && (n<3); ++n)
		{
			m_LM_pos[o][n] = new unsigned int[count];
			Operator_Cache::Read(in, m_LM_pos[o][n], count);
		}
	}
}

void Operator_Ext_Dispersive::ShowStat(ostream &ostr)  const
{
	Operator_Extension::ShowStat(ostr);
	string On_Off[2] = {"Off", "On"};
	ostr << " Max. Dispersion Order N = " << m_Order << endl;
	for (int i=0;i<m_Order;++i)
	{
		ostr << " N=" << i << ":\t Active cells\t\t: " << 	m_LM_Count.at(i) << endl;
		ostr << " N=" << i << ":\t Voltage ADE is \t: " << On_Off[m_volt_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Current ADE is \t: " << On_Off[m_curr_ADE_On[i]] << endl;
	}
}
