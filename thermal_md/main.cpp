/*!
 * @author    Naoyuki MORITA
 * @date      2019
 * @copyright BSD-3-Clause
 */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "sim_conf.hh"

#include "matrix.hh"
#include "gettime.hh"

namespace {

class CLenv_t {
private:
    std::vector<cl::Platform> m_platforms;
    std::vector<cl::Device> m_devices;
    std::vector<cl::Context> m_contexts;
    std::vector<cl::CommandQueue> m_command_queues;

public:
    CLenv_t( void ) {
        cl::Platform::get(&m_platforms);
        m_platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &m_devices);
        std::cout << m_devices.size() << " devices." << std::endl;
        for (int i=0; i<2; i+=1) {
            auto context = cl::Context(m_devices[i]);
            auto queue_ = cl::CommandQueue(context, m_devices[i]);
            m_contexts.push_back(std::move(context));
            m_command_queues.push_back(std::move(queue_));
        }
    }

    inline cl::Platform & platform( void ) {
        return m_platforms[0];
    }

    inline cl::Context & context(int idx = 0) {
        return m_contexts[idx];
    }

    inline cl::CommandQueue & command_queue(int idx = 0) {
        return m_command_queues[idx];
    }

    inline cl::Device & device(int idx = 0) {
        return m_devices[idx];
    }
};

void
init_state(std::vector<FLOAT_t>& enths)
{
    for (int i=0; i<(NX*NY*NZ); i+=1) {
        enths[i] = T_melt * Cv_SUS304 + H_melt + 100.0;
    }
}

inline size_t getFileSize(std::ifstream& file)
{
    file.seekg(0, std::ios::end);
    size_t ret = file.tellg();
    file.seekg(0, std::ios::beg);

    return ret;
}

inline void loadFile(std::ifstream& file, std::vector<char>& d, size_t size)
{
    d.resize(size);
    file.read(reinterpret_cast<char*>(d.data()), size);
}

cl::Program createProgram(cl::Context& context, const std::vector<cl::Device>& devices, const std::string& filename)
{
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);

    if (file.fail()) {
        throw "can not open kernel file";
    }

    size_t            filesize = getFileSize(file);
    std::vector<char> binary_data;
    loadFile(file, binary_data, filesize);

    cl::Program::Binaries binaries;
    binaries.push_back(std::make_pair(&binary_data[0], filesize));

    return cl::Program(context, devices, binaries, nullptr, nullptr);
}

cl::Program createProgram(cl::Context& context, const cl::Device& device, const std::string& filename)
{
    std::vector<cl::Device> devices { device };
    return createProgram(context, devices, filename);
}

void
output_temperature(int k, const std::vector<uint8_t>& pixels)
{
    char buf[128];
    std::ofstream file;
    std::snprintf(buf, 128, "result/%05d.pnm", k);

    file.open(buf, std::ios::out | std::ios::binary);

    file << "P6" << std::endl;
    file << NX << " " << NY << std::endl;
    file << "255" << std::endl;

    file.write((char *)&(pixels[0]), NX*NY*3);

    file.close( );
}

void calc_differential(CLenv_t & clenv, size_t num,
                       SparseMatrix_t & mtx0,
                       SparseMatrix_t & mtx1,
                       std::vector<FLOAT_t>& enths,
                       std::vector<int>& perm_fwd,
                       std::vector<int>& perm_rev,
                       int nsteps)
{
    try {
        std::vector<uint8_t> pixels(NX*NY*3);
        std::vector<cl::Kernel> kernels0;
        std::vector<cl::Kernel> kernels1;
        auto & context0 = clenv.context(0);
        auto & device0 = clenv.device(0);
        auto & queue0 = clenv.command_queue(0);
        auto & context1 = clenv.context(1);
        auto & device1 = clenv.device(1);
        auto & queue1 = clenv.command_queue(1);

        // Create Program.
        // Load compiled binary file and create cl::Program object.
        auto program0 = createProgram(context0, device0, "kernel/kernel.pz");
        auto program1 = createProgram(context1, device1, "kernel/kernel.pz");

        // Create Kernel.
        // Give kernel name without pzc_ prefix.
        {
            auto kernel0 = cl::Kernel(program0, "calcDiffuse");
            auto kernel1 = cl::Kernel(program0, "calcBoundary");
            auto kernel2 = cl::Kernel(program0, "enth2temp");
            auto kernel3 = cl::Kernel(program0, "extractrgb");
            kernels0.push_back(std::move(kernel0));
            kernels0.push_back(std::move(kernel1));
            kernels0.push_back(std::move(kernel2));
            kernels0.push_back(std::move(kernel3));
        }

        {
            auto kernel0 = cl::Kernel(program1, "calcDiffuse");
            auto kernel1 = cl::Kernel(program1, "calcBoundary");
            auto kernel2 = cl::Kernel(program1, "enth2temp");
            kernels1.push_back(std::move(kernel0));
            kernels1.push_back(std::move(kernel1));
            kernels1.push_back(std::move(kernel2));
        }

        // Get stack size modify function.
        typedef CL_API_ENTRY cl_int(CL_API_CALL * pfnPezyExtSetPerThreadStackSize)(cl_kernel kernel, size_t size);
        const auto clExtSetPerThreadStackSize = reinterpret_cast<pfnPezyExtSetPerThreadStackSize>(clGetExtensionFunctionAddress("pezy_set_per_thread_stack_size"));
        if (clExtSetPerThreadStackSize == nullptr) {
            throw "pezy_set_per_thread_stack_size not found";
        }
        size_t stack_size_per_thread = 1024;
        clExtSetPerThreadStackSize(kernels0[0](), stack_size_per_thread);
        clExtSetPerThreadStackSize(kernels1[0](), stack_size_per_thread);

        // Create Buffers.
        auto device_rowptr0  = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(int) * (num/2+1));
        auto device_rowptr1  = cl::Buffer(context1, CL_MEM_READ_WRITE, sizeof(int) * (num/2+1));
        auto device_idxs0    = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(int) * num * 7);
        auto device_idxs1    = cl::Buffer(context1, CL_MEM_READ_WRITE, sizeof(int) * num * 7);
        auto device_rows0    = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num * 7);
        auto device_rows1    = cl::Buffer(context1, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num * 7);
        auto device_enths0   = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num);
        auto device_enths1   = cl::Buffer(context1, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num);
        auto device_temps0   = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num);
        auto device_temps1   = cl::Buffer(context1, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num);
        // auto device_perm_fwd = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * num);
        auto device_perm_rev0 = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(int) * num);
        auto device_perm_rev1 = cl::Buffer(context1, CL_MEM_READ_WRITE, sizeof(int) * num);
        auto device_pixels   = cl::Buffer(context0, CL_MEM_READ_WRITE, sizeof(uint8_t) * NX * NY * 3);

        // Send src.
        queue0.enqueueWriteBuffer(device_rowptr0 , false, 0, sizeof(int) * ((num/2) + 1), &(mtx0.m_rowptr[0]));
        queue1.enqueueWriteBuffer(device_rowptr1 , false, 0, sizeof(int) * ((num/2) + 1), &(mtx1.m_rowptr[0]));
        queue0.enqueueWriteBuffer(device_idxs0   , false, 0, sizeof(int) * (num/2) * 7, &(mtx0.m_idxs[0]));
        queue1.enqueueWriteBuffer(device_idxs1   , false, 0, sizeof(int) * (num/2) * 7, &(mtx1.m_idxs[0]));
        queue0.enqueueWriteBuffer(device_rows0   , false, 0, sizeof(FLOAT_t) * (num/2) * 7, &(mtx0.m_elems[0]));
        queue1.enqueueWriteBuffer(device_rows1   , false, 0, sizeof(FLOAT_t) * (num/2) * 7, &(mtx1.m_elems[0]));
        queue0.enqueueWriteBuffer(device_enths0  , false, 0, sizeof(FLOAT_t) * num, &enths[0]);
        queue1.enqueueWriteBuffer(device_enths1  , false, 0, sizeof(FLOAT_t) * num, &enths[0]);
        // queue0.enqueueWriteBuffer(device_perm_fwd, true, 0, sizeof(int) * num, &(perm_fwd[0]));
        queue0.enqueueWriteBuffer(device_perm_rev0, true, 0, sizeof(int) * num, &(perm_rev[0]));
        queue1.enqueueWriteBuffer(device_perm_rev1, true, 0, sizeof(int) * num, &(perm_rev[0]));

        // Set kernel args.
        kernels0[0].setArg(0, num/2);
        kernels0[0].setArg(1, (size_t)0);
        kernels0[0].setArg(2, device_enths0);
        kernels0[0].setArg(3, device_temps0);
        kernels0[0].setArg(4, device_rowptr0);
        kernels0[0].setArg(5, device_idxs0);
        kernels0[0].setArg(6, device_rows0);

        kernels0[1].setArg(0, (size_t)NX*NZ);
        kernels0[1].setArg(1, device_enths0);
        kernels0[1].setArg(2, device_perm_rev0);

        kernels0[2].setArg(0, num);
        kernels0[2].setArg(1, device_temps0);
        kernels0[2].setArg(2, device_enths0);

        kernels0[3].setArg(0, (size_t)NX*NY);
        kernels0[3].setArg(1, (int)8);
        kernels0[3].setArg(2, device_pixels);
        kernels0[3].setArg(3, device_enths0);
        kernels0[3].setArg(4, device_perm_rev0);

        kernels1[0].setArg(0, num/2);
        kernels1[0].setArg(1, num/2);
        kernels1[0].setArg(2, device_enths1);
        kernels1[0].setArg(3, device_temps1);
        kernels1[0].setArg(4, device_rowptr1);
        kernels1[0].setArg(5, device_idxs1);
        kernels1[0].setArg(6, device_rows1);

        kernels1[1].setArg(0, (size_t)NX*NZ);
        kernels1[1].setArg(1, device_enths1);
        kernels1[1].setArg(2, device_perm_rev1);

        kernels1[2].setArg(0, num);
        kernels1[2].setArg(1, device_temps1);
        kernels1[2].setArg(2, device_enths1);

        // Get workitem size.
        // sc1-64: 8192  (1024 PEs * 8 threads)
        // sc2   : 15782 (1984 PEs * 8 threads)
        size_t global_work_size = 0;
        for (int i=0; i<2; i+=1) {
            auto & dev_tmp = clenv.device(i);
            std::string device_name;
            dev_tmp.getInfo(CL_DEVICE_NAME, &device_name);

            size_t global_work_size_[3] = { 0 };
            dev_tmp.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &global_work_size_);

            global_work_size = global_work_size_[0];
            if (device_name.find("PEZY-SC2") != std::string::npos) {
                global_work_size = std::min(global_work_size, (size_t)15872);
            }

            std::cout << "Use device :" << i << ": " << device_name << std::endl;
            std::cout << "workitem   :" << i << ": " << global_work_size << std::endl;
        }

        // Run device kernel.
        // Enthalpy to temperature
        queue0.enqueueNDRangeKernel(kernels0[2], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
        queue1.enqueueNDRangeKernel(kernels1[2], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
        // Convert enthalpy to RGB
        queue0.enqueueNDRangeKernel(kernels0[3], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
        // Get temperature.
        queue0.enqueueReadBuffer(device_pixels, true, 0, sizeof(uint8_t) * NX*NY*3, &pixels[0]);

        int ofs0 = mtx0.border_min;
        int len0 = mtx0.border_max - ofs0 + 1;
        int ofs1 = mtx1.border_min;
        int len1 = mtx1.border_max - ofs1 + 1;

        for (int k=0; k<nsteps; k+=1) {
            for (int i=0; i<96; i+=1) {
                cl::Event ev0;
                cl::Event ev1;
                // Calculate diffusion
                queue0.enqueueNDRangeKernel(kernels0[0], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
                queue1.enqueueNDRangeKernel(kernels1[0], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);

                // Distribute result
                queue0.enqueueReadBuffer(device_enths0, false, sizeof(FLOAT_t)*ofs1, sizeof(FLOAT_t)*len1, &(enths[ofs1]), nullptr, &ev0);
                queue1.enqueueReadBuffer(device_enths1, false, sizeof(FLOAT_t)*ofs0, sizeof(FLOAT_t)*len0, &(enths[ofs0]), nullptr, &ev1);

                ev0.wait( );
                ev1.wait( );

                queue0.enqueueWriteBuffer(device_enths0, false, sizeof(FLOAT_t)*ofs0, sizeof(FLOAT_t)*len0, &(enths[ofs0]));
                queue1.enqueueWriteBuffer(device_enths1, false, sizeof(FLOAT_t)*ofs1, sizeof(FLOAT_t)*len1, &(enths[ofs1]));

                // Calculate boundary condition
                queue0.enqueueNDRangeKernel(kernels0[1], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
                queue1.enqueueNDRangeKernel(kernels1[1], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);

                // Enthalpy to temperature
                queue0.enqueueNDRangeKernel(kernels0[2], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
                queue1.enqueueNDRangeKernel(kernels1[2], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
            }

            // Aggregate result
            queue1.enqueueReadBuffer(device_enths1, true, sizeof(FLOAT_t)*(num/2), sizeof(FLOAT_t)*(num/2), &(enths[num/2]));
            queue0.enqueueWriteBuffer(device_enths0, false, sizeof(FLOAT_t)*(num/2), sizeof(FLOAT_t)*(num/2), &(enths[num/2]));

            // Convert enthalpy to RGB
            queue0.enqueueNDRangeKernel(kernels0[3], cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
            output_temperature(k, pixels);

            // Get temperature.
            queue0.enqueueReadBuffer(device_pixels, true, 0, sizeof(uint8_t) * NX*NY*3, &pixels[0]);
        }

        // Finish all commands.
        queue0.flush();
        queue0.finish();

        output_temperature(nsteps, pixels);

    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}


}

int main(int argc, char** argv)
{
    size_t num = NX*NY*NZ;

    std::cout << "int: " << sizeof(int) << std::endl;
    std::cout << "cl_int: " << sizeof(cl_int) << std::endl;

    std::cout << "num " << num << std::endl;

    std::vector<FLOAT_t> enths(num);

    std::vector<int>     rowptr0(num/2+1);
    std::vector<int>     idxs0((num/2)*7);
    std::vector<FLOAT_t> rows0((num/2)*7);
    std::vector<int>     rowptr1(num/2+1);
    std::vector<int>     idxs1((num/2)*7);
    std::vector<FLOAT_t> rows1((num/2)*7);

    std::vector<int> perm_c(num, 0);
    std::vector<int> perm_r(num, 0);

    auto mtx0 = SparseMatrix_t(rowptr0, idxs0, rows0);
    auto mtx1 = SparseMatrix_t(rowptr1, idxs1, rows1);

    {
        std::vector<int> rowptr_(num+1);
        std::vector<int> idxs_(num*7);
        std::vector<FLOAT_t> rows_(num*7);
        auto mtx_ = SparseMatrix_t(rowptr_, idxs_, rows_);

        std::vector<int> tmp_rowptr(num+1);
        std::vector<int> tmp_idxs(num*7);
        std::vector<FLOAT_t> tmp_rows(num*7);
        auto tmp_mtx = SparseMatrix_t(tmp_rowptr, tmp_idxs, tmp_rows);

        init_differential_matrix(tmp_mtx);

        matrix_reorder_CuthillMckee(mtx_, tmp_mtx, perm_c, perm_r);

        split_matrix(mtx0, mtx1, mtx_);
        std::cout << "Border 0: " << mtx0.border_min << " - " << mtx0.border_max << std::endl;
        std::cout << "Border 1: " << mtx1.border_min << " - " << mtx1.border_max << std::endl;
    }

    std::cout.flush( );

    init_state(enths);

    double t1 = gettime( );

    // run device add
    try {
        auto clenv = CLenv_t( );

        calc_differential(clenv, num, mtx0, mtx1, enths, perm_c, perm_r, 2048);
    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }

    double t2 = gettime( );

    std::cout << "Elapsed time: " << (t2 - t1) << " sec." << std::endl;

    return 0;
}

// Local Variables:
// indent-tabs-mode: nil
// End:
