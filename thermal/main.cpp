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
    cl::Context m_context;
    cl::CommandQueue m_command_queue;

public:
    CLenv_t( void ) {
        cl::Platform::get(&m_platforms);
        m_platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &m_devices);
        m_context = cl::Context(m_devices[0]);
        m_command_queue = cl::CommandQueue(m_context, m_devices[0], 0);
    }

    inline cl::Context & context( void ) {
        return m_context;
    }

    inline cl::CommandQueue & command_queue( void ) {
        return m_command_queue;
    }

    inline cl::Platform & platform( void ) {
        return m_platforms[0];
    }

    inline cl::Device & device( void ) {
        return m_devices[0];
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

void calc_differential(CLenv_t & clenv, size_t num, SparseMatrix_t & mtx,
                       std::vector<FLOAT_t>& enths,
                       std::vector<int>& perm_fwd,
                       std::vector<int>& perm_rev,
                       int nsteps)
{
    try {
        std::vector<uint8_t> pixels(NX*NY*3);
        auto & context = clenv.context( );
        auto & device = clenv.device( );
        auto & command_queue = clenv.command_queue( );

        // Create Program.
        // Load compiled binary file and create cl::Program object.
        auto program = createProgram(clenv.context( ), clenv.device( ), "kernel/kernel.pz");

        // Create Kernel.
        // Give kernel name without pzc_ prefix.
        auto kernel0 = cl::Kernel(program, "calcDiffuse");
        auto kernel1 = cl::Kernel(program, "calcBoundary");
        auto kernel2 = cl::Kernel(program, "enth2temp");
        auto kernel4 = cl::Kernel(program, "extractrgb");

        // Get stack size modify function.
        typedef CL_API_ENTRY cl_int(CL_API_CALL * pfnPezyExtSetPerThreadStackSize)(cl_kernel kernel, size_t size);
        const auto clExtSetPerThreadStackSize = reinterpret_cast<pfnPezyExtSetPerThreadStackSize>(clGetExtensionFunctionAddress("pezy_set_per_thread_stack_size"));
        if (clExtSetPerThreadStackSize == nullptr) {
            throw "pezy_set_per_thread_stack_size not found";
        }
        size_t stack_size_per_thread = 1024;
        clExtSetPerThreadStackSize(kernel0(), stack_size_per_thread);

        // Create Buffers.
        auto device_rowptr   = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * (num+1));
        auto device_idxs     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * num * 8);
        auto device_rows     = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num * 8);
        auto device_enths    = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num);
        auto device_temps    = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(FLOAT_t) * num);
        auto device_perm_fwd = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * num);
        auto device_perm_rev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * num);
        auto device_pixels   = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * NX * NY * 3);

        // Send src.
        command_queue.enqueueWriteBuffer(device_rowptr  , true, 0, sizeof(int) * (num + 1), &(mtx.m_rowptr[0]));
        command_queue.enqueueWriteBuffer(device_idxs    , true, 0, sizeof(int) * num * 8, &(mtx.m_idxs[0]));
        command_queue.enqueueWriteBuffer(device_rows    , true, 0, sizeof(FLOAT_t) * num * 8, &(mtx.m_elems[0]));
        command_queue.enqueueWriteBuffer(device_enths   , true, 0, sizeof(FLOAT_t) * num, &enths[0]);
        command_queue.enqueueWriteBuffer(device_perm_fwd, true, 0, sizeof(int) * num, &(perm_fwd[0]));
        command_queue.enqueueWriteBuffer(device_perm_rev, true, 0, sizeof(int) * num, &(perm_rev[0]));

        // Set kernel args.
        kernel0.setArg(0, num);
        kernel0.setArg(1, device_enths);
        kernel0.setArg(2, device_temps);
        kernel0.setArg(3, device_rowptr);
        kernel0.setArg(4, device_idxs);
        kernel0.setArg(5, device_rows);
        kernel0.setArg(6, device_perm_fwd);

        kernel2.setArg(0, num);
        kernel2.setArg(1, device_temps);
        kernel2.setArg(2, device_enths);

        kernel4.setArg(0, (size_t)NX*NY);
        kernel4.setArg(1, (int)8);
        kernel4.setArg(2, device_pixels);
        kernel4.setArg(3, device_enths);
        kernel4.setArg(4, device_perm_rev);

        // Get workitem size.
        // sc1-64: 8192  (1024 PEs * 8 threads)
        // sc2   : 15782 (1984 PEs * 8 threads)
        size_t global_work_size = 0;
        {
            std::string device_name;
            device.getInfo(CL_DEVICE_NAME, &device_name);

            size_t global_work_size_[3] = { 0 };
            device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &global_work_size_);

            global_work_size = global_work_size_[0];
            if (device_name.find("PEZY-SC2") != std::string::npos) {
                global_work_size = std::min(global_work_size, (size_t)15872);
            }

            std::cout << "Use device : " << device_name << std::endl;
            std::cout << "workitem   : " << global_work_size << std::endl;
        }

        // Run device kernel.
        // Enthalpy to temperature
        command_queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
        // Convert enthalpy to RGB
        command_queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
        // Get temperature map as an image.
        command_queue.enqueueReadBuffer(device_pixels, true, 0, sizeof(uint8_t) * NX*NY*3, &pixels[0]);

        for (int k=0; k<nsteps; k+=1) {
            for (int i=0; i<96; i+=1) {
                // Calculate diffusion
                command_queue.enqueueNDRangeKernel(kernel0, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);

                // Enthalpy to temperature
                command_queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
            }

            // Convert enthalpy to RGB
            command_queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, nullptr);
            output_temperature(k, pixels);

            // Get temperature map as an image.
            command_queue.enqueueReadBuffer(device_pixels, true, 0, sizeof(uint8_t) * NX*NY*3, &pixels[0]);
        }

        // Finish all commands.
        command_queue.flush();
        command_queue.finish();

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

    std::vector<int> rowptr(num+1);
    std::vector<int> idxs(num*7);
    std::vector<FLOAT_t> rows(num*7);

    std::vector<int> perm_c(num, 0);
    std::vector<int> perm_r(num, 0);

    auto mtx = SparseMatrix_t(rowptr, idxs, rows);

    {
        std::vector<int> tmp_rowptr(num+1);
        std::vector<int> tmp_idxs(num*7);
        std::vector<FLOAT_t> tmp_rows(num*7);
        auto tmp_mtx = SparseMatrix_t(tmp_rowptr, tmp_idxs, tmp_rows);

        init_differential_matrix(tmp_mtx);

        matrix_reorder_CuthillMckee(mtx, tmp_mtx, perm_c, perm_r);
    }

    init_state(enths);

    double t1 = gettime( );

    // run device add
    try {
        auto clenv = CLenv_t( );

        calc_differential(clenv, num, mtx, enths, perm_c, perm_r, 2048);
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
