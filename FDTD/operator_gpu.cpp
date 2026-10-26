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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "engine_gpu.h"
#include "gpu_backend.h"
#include "gpu_setup_model.h"
#include "operator_gpu.h"
#include "tools/fingerprint.h"
#include "tools/global.h"
#include "tools/useful.h"

#include "CSPrimitives.h"

using std::cout;
using std::endl;

Operator_GPU* Operator_GPU::New(GPU_Backend* backend)
{
	cout << "Create FDTD operator (GPU)" << endl;
	Operator_GPU* op = new Operator_GPU(backend);
	op->Init();
	return op;
}

Operator_GPU::Operator_GPU(GPU_Backend* backend) : Operator()
{
	m_Backend = backend;
	m_SetupModel = NULL;
	m_SetupVerify = false;
	m_VerifySeconds = 0;
	m_VerifyBytes = 0;
}

Operator_GPU::~Operator_GPU()
{
	FreeSetupDevice();
	delete m_Backend;
}

Engine* Operator_GPU::CreateEngine()
{
	GPU_Backend* backend = m_Backend;
	m_Backend = NULL;
	m_Engine = Engine_GPU::New(this, backend);
	// The device holds the coefficients now: free the host arrays (48 bytes per node), as
	// Operator_SSE_Compressed frees its full arrays. GetVV() and friends are invalid after this.
	delete vv_ptr; vv_ptr = NULL;
	delete vi_ptr; vi_ptr = NULL;
	delete ii_ptr; ii_ptr = NULL;
	delete iv_ptr; iv_ptr = NULL;
	return m_Engine;
}

void Operator_GPU::ThreadRanges(std::vector<unsigned int>& start, std::vector<unsigned int>& stop) const
{
	std::vector<unsigned int> jobs = AssignJobs2Threads(numLines[0], AvailableThreads(), true);
	unsigned int line = 0;
	for (size_t n=0; n<jobs.size(); ++n)
	{
		start.push_back(line);
		line += jobs.at(n);
		stop.push_back(line-1);
	}
}

unsigned int Operator_GPU::SetupThreads() const
{
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	return start.size();
}

namespace
{
//! OPENEMS_SETUP_DEVICE: 0 for "0" or "off" (the default), 1 for "auto", 2 for "1" or "on" (read once)
int SetupDeviceMode()
{
	static const int mode = []()
	{
		const char* val = getenv("OPENEMS_SETUP_DEVICE");
		if ((val==NULL) || (val[0]=='\0') || (strcmp(val,"0")==0) || (strcmp(val,"off")==0))
			return 0;
		if (strcmp(val,"auto")==0)
			return 1;
		if ((strcmp(val,"1")==0) || (strcmp(val,"on")==0))
			return 2;
		throw std::runtime_error(std::string("OPENEMS_SETUP_DEVICE: unknown value '") + val + "', use 0, auto or 1");
	}();
	return mode;
}

double SecondsSince(std::chrono::steady_clock::time_point t0)
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string ValueText(double val)
{
	char buf[96];
	snprintf(buf, sizeof(buf), "%.17g (%a)", val, val);
	return std::string(buf);
}
}

void Operator_GPU::FreeSetupDevice()
{
	delete m_SetupModel;
	m_SetupModel = NULL;
	std::vector<int32_t>().swap(m_SetupFirstHit);
	std::vector<uint8_t>().swap(m_SetupUsed);
	std::vector<uint8_t>().swap(m_SetupUsed0);
	m_SetupVerify = false;
}

bool Operator_GPU::Calc_EC()
{
	if (CSX==NULL)
	{
		std::cerr << "Operator_GPU::Calc_EC: CSX not given or invalid!!!" << std::endl;
		return false;
	}
	MainOp->SetPos(0,0,0);

	FreeSetupDevice();
	int mode = SetupDeviceMode();
	if (mode==0)
	{
		if (g_settings.GetVerboseLevel()>1)
			cout << "setup device: off (OPENEMS_SETUP_DEVICE=0)" << endl;
		return Calc_EC_Threads();
	}

	std::string reason;
	if (Calc_EC_Device(FNVHash::EnvOn("OPENEMS_SETUP_DEVICE_VERIFY"), reason))
		return true;
	if (mode==2)
		throw std::runtime_error("OPENEMS_SETUP_DEVICE=1: " + reason);
	cout << "setup device: CPU path: " << reason << endl;
	return Calc_EC_Threads();
}

bool Operator_GPU::Calc_EC_Threads()
{
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	std::vector<std::thread> threads;
	for (size_t n=0; n<start.size(); ++n)
		threads.push_back(std::thread(&Operator_GPU::Calc_EC_Range, this, start[n], stop[n]));
	for (size_t n=0; n<threads.size(); ++n)
		threads[n].join();
	return true;
}

bool Operator_GPU::Calc_EC_Device(bool verify, std::string& reason)
{
	if (m_Backend==NULL)
	{
		reason = "no backend";
		return false;
	}
	GPU_SetupModel* model = new GPU_SetupModel;
	if (model->Build(this, reason)==false)
	{
		delete model;
		return false;
	}

	const size_t nx = numLines[0], ny = numLines[1], nz = numLines[2];
	const size_t cells = nx*ny*nz;
	if (MainOp->GetSize()!=cells)
	{
		delete model;
		reason = "unexpected operator size";
		return false;
	}

	// the buffers of the result are owned here
	GPU_SetupResult result;
	std::vector<float> ec[4], storage[4];
	for (int k=0; k<4; ++k)
	{
		ec[k].resize(3*cells);
		result.EC[k] = ec[k].data();
		result.storage[k] = NULL;
		if (model->storage[k])
		{
			storage[k].resize(3*cells);
			result.storage[k] = storage[k].data();
		}
	}
	std::vector<int32_t> firstHit(3*cells);
	std::vector<uint8_t> used(model->primitives.size(), 0);
	result.firstHit = firstHit.data();
	result.used = used.data();

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	if (m_Backend->SetupOperator(*model, result)==false)
	{
		reason = "backend " + m_Backend->GetName() + " declined the setup pass (no entry, or the model or the device memory does not fit)";
		delete model;
		return false;
	}
	cout << "setup device: " << m_Backend->GetName() << ", " << model->primitives.size() << " primitives, " << (nx*ny) << " columns, pass ";
	char buf[32];
	snprintf(buf, sizeof(buf), "%.3f", SecondsSince(t0));
	cout << buf << " s" << endl;

	m_SetupModel = model;
	m_SetupVerify = verify;
	m_VerifySeconds = 0;
	m_VerifyBytes = 0;

	if (verify)
	{
		m_SetupUsed0.resize(model->primitives.size());
		for (size_t i=0; i<model->primitives.size(); ++i)
			m_SetupUsed0[i] = model->primPtr[i]->GetPrimitiveUsed() ? 1 : 0;
		if (result.nanCount)
		{
			Calc_EC_Threads(); // throws the message of the CPU path
			throw std::runtime_error("setup device: verify FAILED: the device flagged a NaN or Inf effective material parameter that the host did not");
		}
		Calc_EC_Threads();
		VerifyEC(result, cells);
	}
	else
	{
		if (result.nanCount)
		{
			std::cerr << "Operator::AverageMatQuarterCell: Error, An effective material parameter is not a valid result, this should NOT have happened..." << std::endl;
			std::cerr << result.nanN << "@" << result.nanK << " : " << result.nanPos[0] << "," << result.nanPos[1] << "," << result.nanPos[2] << std::endl;
			throw std::runtime_error("Operator: effective material parameter is NaN/Inf");
		}
		FDTD_FLOAT** dst[4] = {EC_C, EC_G, EC_L, EC_R};
		for (int k=0; k<4; ++k)
		{
			for (int n=0; n<3; ++n)
				memcpy(dst[k][n], ec[k].data() + n*cells, cells*sizeof(float));
			std::vector<float>().swap(ec[k]);
		}
		ArrayLib::ArrayNIJK<float>* store[4] = {m_epsR_ptr, m_kappa_ptr, m_mueR_ptr, m_sigma_ptr};
		for (int k=0; k<4; ++k)
		{
			if (store[k]==NULL)
				continue;
			for (int n=0; n<3; ++n)
				for (size_t z=0; z<nz; ++z)
					for (size_t y=0; y<ny; ++y)
						for (size_t x=0; x<nx; ++x)
							(*store[k])(n,x,y,z) = storage[k][n*cells + x + nx*(y + ny*z)];
			std::vector<float>().swap(storage[k]);
		}
		for (size_t i=0; i<used.size(); ++i)
			if (used[i])
				model->primPtr[i]->SetPrimitiveUsed(true);
	}
	m_SetupFirstHit.swap(firstHit);
	m_SetupUsed.swap(used);
	return true;
}

void Operator_GPU::VerifyEC(const GPU_SetupResult& result, size_t cells)
{
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	const size_t nx = numLines[0], ny = numLines[1], nz = numLines[2];
	const char* name[4] = {"EC_C", "EC_G", "EC_L", "EC_R"};
	FDTD_FLOAT** host[4] = {EC_C, EC_G, EC_L, EC_R};
	for (int k=0; k<4; ++k)
		for (int n=0; n<3; ++n)
		{
			const float* h = host[k][n];
			const float* d = result.EC[k] + n*cells;
			m_VerifyBytes += cells*sizeof(float);
			if (memcmp(h, d, cells*sizeof(float))==0)
				continue;
			for (size_t c=0; c<cells; ++c)
				if (memcmp(h+c, d+c, sizeof(float))!=0)
				{
					std::string text = std::string("setup device: verify FAILED: ") + name[k] + " n=" + std::to_string(n) + " (" + std::to_string(c%nx) + "," + std::to_string((c/nx)%ny) + "," + std::to_string(c/(nx*ny)) + ") host " + ValueText(h[c]) + " device " + ValueText(d[c]);
					cout << text << endl;
					throw std::runtime_error(text);
				}
		}
	const char* sname[4] = {"epsR", "kappa", "mueR", "sigma"};
	ArrayLib::ArrayNIJK<float>* store[4] = {m_epsR_ptr, m_kappa_ptr, m_mueR_ptr, m_sigma_ptr};
	for (int k=0; k<4; ++k)
	{
		if ((store[k]==NULL) != (result.storage[k]==NULL))
		{
			std::string text = std::string("setup device: verify FAILED: ") + sname[k] + " requested on one side only";
			cout << text << endl;
			throw std::runtime_error(text);
		}
		if (store[k]==NULL)
			continue;
		for (int n=0; n<3; ++n)
			for (size_t z=0; z<nz; ++z)
				for (size_t y=0; y<ny; ++y)
					for (size_t x=0; x<nx; ++x)
					{
						float h = (*store[k])(n,x,y,z);
						float d = result.storage[k][n*cells + x + nx*(y + ny*z)];
						if (memcmp(&h, &d, sizeof(float))!=0)
						{
							std::string text = std::string("setup device: verify FAILED: ") + sname[k] + " n=" + std::to_string(n) + " (" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ") host " + ValueText(h) + " device " + ValueText(d);
							cout << text << endl;
							throw std::runtime_error(text);
						}
					}
		m_VerifyBytes += 3*cells*sizeof(float);
	}
	m_VerifySeconds += SecondsSince(t0);
}

void Operator_GPU::VerifyPEC()
{
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	const size_t nx = numLines[0], ny = numLines[1], nz = numLines[2];
	std::string text;
	if ((m_FirstHit==NULL) || (m_FirstHit->size()!=m_SetupFirstHit.size()))
		text = "setup device: verify FAILED: first-hit table not allocated or of another size";
	else if (m_FirstHitPrims.size()!=m_SetupModel->edgeList.size())
		text = "setup device: verify FAILED: first-hit primitive list of another size";
	else
	{
		for (size_t i=0; i<m_FirstHitPrims.size(); ++i)
			if (m_FirstHitPrims[i]!=m_SetupModel->primPtr[m_SetupModel->edgeList[i]])
			{
				text = "setup device: verify FAILED: first-hit primitive list differs at entry " + std::to_string(i);
				break;
			}
	}
	if (text.empty())
	{
		const int32_t* h = m_FirstHit->data();
		const int32_t* d = m_SetupFirstHit.data();
		m_VerifyBytes += m_SetupFirstHit.size()*sizeof(int32_t);
		if (memcmp(h, d, m_SetupFirstHit.size()*sizeof(int32_t))!=0)
			for (size_t i=0; i<m_SetupFirstHit.size(); ++i)
				if (h[i]!=d[i])
				{
					text = std::string("setup device: verify FAILED: firstHit n=") + std::to_string(i/(nx*ny*nz)) + " (" + std::to_string((i/(ny*nz))%nx) + "," + std::to_string((i/nz)%ny) + "," + std::to_string(i%nz) + ") host " + std::to_string(h[i]) + " device " + std::to_string(d[i]);
					break;
				}
	}
	if (text.empty())
		for (size_t x=0; x<m_FirstHitLine.size(); ++x)
			if (m_FirstHitLine[x]==0)
			{
				text = "setup device: verify FAILED: firstHitLine x=" + std::to_string(x) + " host 1 device 0";
				break;
			}
	if (text.empty())
		for (size_t i=0; i<m_SetupModel->primitives.size(); ++i)
		{
			int host = m_SetupModel->primPtr[i]->GetPrimitiveUsed() ? 1 : 0;
			int dev = (m_SetupUsed0[i] || m_SetupUsed[i]) ? 1 : 0;
			if (host!=dev)
			{
				text = "setup device: verify FAILED: used primitive " + std::to_string(i) + " (ID " + std::to_string(m_SetupModel->primPtr[i]->GetID()) + ") host " + std::to_string(host) + " device " + std::to_string(dev);
				break;
			}
		}
	m_VerifySeconds += SecondsSince(t0);
	if (text.empty()==false)
	{
		cout << text << endl;
		throw std::runtime_error(text);
	}
	char buf[96];
	snprintf(buf, sizeof(buf), "%.1f MB in %.3f s", m_VerifyBytes/1.0e6, m_VerifySeconds);
	cout << "setup device: verify OK, compared " << buf << endl;
}

void Operator_GPU::CalcTimestep_Var3_Search(Timestep_Min& tmin)
{
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	CalcTimestep_Var3_Slabs(start, stop, tmin);
}

void Operator_GPU::Calc_ECOperator_All()
{
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	Calc_ECOperator_Slabs(start, stop);
}

bool Operator_GPU::CalcPEC()
{
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	std::vector<unsigned int> counter(3*start.size(), 0);   // three PEC counters per thread, summed up below
	std::vector<std::thread> threads;
	if (m_SetupModel && (m_SetupVerify==false))
	{
		// the device pass made the first-hit table: the PEC decision reads it, CSXCAD is not asked again
		if ((m_FirstHit==NULL) || (m_FirstHit->size()!=m_SetupFirstHit.size()) || (m_FirstHitPrims.size()!=m_SetupModel->edgeList.size()))
			throw std::runtime_error("Operator_GPU::CalcPEC: the first-hit table does not fit the device pass");
		for (size_t i=0; i<m_FirstHitPrims.size(); ++i)
			if (m_FirstHitPrims[i]!=m_SetupModel->primPtr[m_SetupModel->edgeList[i]])
				throw std::runtime_error("Operator_GPU::CalcPEC: the primitive list of the PEC pass differs from the one of the device pass");
		memcpy(m_FirstHit->data(), m_SetupFirstHit.data(), m_SetupFirstHit.size()*sizeof(int32_t));
		m_FirstHitLine.assign(numLines[0], 1);
		for (size_t n=0; n<start.size(); ++n)
			threads.push_back(std::thread(&Operator_GPU::CalcPEC_FromTable, this, start[n], stop[n], &counter[3*n]));
	}
	else
		for (size_t n=0; n<start.size(); ++n)
			threads.push_back(std::thread(&Operator_GPU::CalcPEC_Range, this, start[n], stop[n], &counter[3*n]));
	for (size_t n=0; n<threads.size(); ++n)
		threads[n].join();

	for (int n=0; n<3; ++n)
	{
		m_Nr_PEC[n] = 0;
		for (size_t t=0; t<start.size(); ++t)
			m_Nr_PEC[n] += counter[3*t+n];
	}
	CalcPEC_Curves();
	if (m_SetupModel && m_SetupVerify)
		VerifyPEC();
	FreeSetupDevice();
	return true;
}

void Operator_GPU::CalcPEC_FromTable(unsigned int xStart, unsigned int xStop, unsigned int* counter)
{
	// Operator::CalcPEC_Range (operator.cpp 2441) with the first hit read from the table
	const size_t nx = numLines[0], ny = numLines[1], nz = numLines[2];
	const int32_t* table = m_SetupFirstHit.data();
	const GPU_SetupModel& model = *m_SetupModel;
	for (unsigned int x=xStart; x<=xStop; ++x)
		for (unsigned int y=0; y<ny; ++y)
			for (unsigned int z=0; z<nz; ++z)
				for (unsigned int n=0; n<3; ++n)
				{
					int32_t idx = table[((n*nx + x)*ny + y)*nz + z];
					if (idx<0)
						continue;
					if (model.properties[model.primitives[model.edgeList[idx]].property].isMetal) //set to PEC
					{
						SetVV(n,x,y,z, 0 );
						SetVI(n,x,y,z, 0 );
						++counter[n];
					}
				}
}
