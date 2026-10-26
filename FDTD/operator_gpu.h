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

#ifndef OPERATOR_GPU_H
#define OPERATOR_GPU_H

#include "operator.h"

#include <cstdint>
#include <string>
#include <vector>

class GPU_Backend;
struct GPU_SetupModel;
struct GPU_SetupResult;

//! Operator for the GPU engine
/*!
  The operator is built on the host like the basic operator. Engine_GPU uploads
  the final coefficients to the device.
  */
class Operator_GPU : public Operator
{
public:
	//! Create a new operator for the GPU backend \a backend (see GPU_Backend::New), which it takes over and hands on to its engine
	static Operator_GPU* New(GPU_Backend* backend);
	virtual ~Operator_GPU();

	virtual Engine* CreateEngine();

protected:
	//! use New() for creating a new Operator
	Operator_GPU(GPU_Backend* backend);

	//! Calc_EC_Range() and CalcPEC_Range() over x line ranges in parallel
	/*!
	  Operator_GPU derives from the basic Operator, whose storage the host mirror
	  of Engine_GPU uses; Operator_Multithread is an Operator_SSE_Compressed and
	  brings the wrong one. Only these two build steps are worth repeating here.
	  */
	virtual bool Calc_EC();
	virtual bool CalcPEC();
	//! The CPU paths of Calc_EC and CalcPEC, the threads over x line ranges
	bool Calc_EC_Threads();
	bool CalcPEC_Threads();
	//! The material pass on the backend (OPENEMS_SETUP_DEVICE, see operator_gpu.cpp), false with the reason when the CPU path has to run
	bool Calc_EC_Device(bool verify, std::string& reason);
	//! The PEC decision from the first-hit table the device pass produced
	void CalcPEC_FromTable(unsigned int xStart, unsigned int xStop, unsigned int* counter);
	//! Compare the device result with the host arrays, see OPENEMS_SETUP_DEVICE_VERIFY
	void VerifyEC(const GPU_SetupResult& result, size_t cells);
	void VerifyPEC();
	void FreeSetupDevice();
	virtual unsigned int SetupThreads() const;
	//! the timestep search and the coefficient loop over the same x line ranges
	virtual void CalcTimestep_Var3_Search(Timestep_Min& tmin);
	virtual void Calc_ECOperator_All();

	//! x line ranges of the threads, at most one per available CPU, bounds inclusive
	void ThreadRanges(std::vector<unsigned int>& start, std::vector<unsigned int>& stop) const;

	GPU_Backend* m_Backend; //!< until CreateEngine()

	// the state of the device pass between Calc_EC and CalcPEC (the first-hit table is made with the material pass, the PEC decision needs the table allocated by CalcECOperator)
	GPU_SetupModel* m_SetupModel; //!< NULL when the CPU path runs
	std::vector<int32_t> m_SetupFirstHit; //!< the table of the device pass, n-major in the memory order of m_FirstHit
	std::vector<uint8_t> m_SetupUsed; //!< the used flags of the device pass per model primitive
	std::vector<uint8_t> m_SetupUsed0; //!< verify: the used flags of the model primitives before the CPU pass
	bool m_SetupVerify;
	double m_VerifySeconds;
	double m_VerifyBytes;
};

#endif // OPERATOR_GPU_H
