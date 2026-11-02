#pragma once

#include "litepgo/contact/broad_phase/spatial_hash.hpp"

#include "litepgo/common/build_info.hpp"
#include "litepgo/contact/ipctk_contact_energy.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <oneapi/tbb/global_control.h>

// Compiled from the example itself so geometry, material and loading cannot drift.
inline int run_contact_benchmark(const litepgo::mesh::TriangleMesh& surface,
                                 const litepgo::dof::SurfaceKinematics& kinematics,
                                 const std::shared_ptr<litepgo::energy::Energy>& elastic,
                                 const std::shared_ptr<litepgo::energy::Energy>& pulling,
                                 const litepgo::dof::DofReduction& fixed, const litepgo::Vec& initial,
                                 const litepgo::solver::NewtonOptions& options, double dhat, double kappa,
                                 const std::filesystem::path& output, int thread_count = 1, int repetitions = 5) {
    using namespace litepgo;
    if (std::string(LITEPGO_BENCHMARK_BUILD_TYPE) != "Release")
        throw std::runtime_error("Benchmark requires a Release build");
    const tbb::global_control threads(tbb::global_control::max_allowed_parallelism, thread_count);
    std::filesystem::create_directories(output);
    auto file = [&](const std::string& name) {
        std::ofstream stream(output / name);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream << std::setprecision(17);
        return stream;
    };
    auto meta = file("build.txt");
    const auto info = build_info();
    meta << "compiler=" << __VERSION__ << "\nbuild=" << LITEPGO_BENCHMARK_BUILD_TYPE << "\ntbb_threads=" << thread_count
         << "\neigen=" << info.eigen_version << "\ntbb=" << info.tbb_runtime_version
         << "\nipctk=" << info.ipc_toolkit_version << "\ndofs=" << initial.size()
         << "\nsurface_vertices=" << surface.topology.num_vertices() << "\ndhat=" << dhat << "\nkappa=" << kappa
         << "\ngradient_tolerance=" << options.gradient_tolerance << "\nmax_iterations=" << options.max_iterations
         << "\n";
#if defined(LITEPGO_HAS_ACCELERATE)
    meta << "solver=AccelerateLLT\n";
#elif defined(LITEPGO_HAS_MKL)
    meta << "solver=PardisoLLT\n";
#else
    meta << "solver=EigenLDLT\n";
#endif
    const std::shared_ptr<const contact::BroadPhase> spatial_hash = std::make_shared<contact::SpatialHash>();
    contact::IPCContactOptions native_options;
    native_options.dhat = dhat;
    contact::IPCTKContactOptions toolkit_options;
    toolkit_options.dhat = dhat;
    meta << "comparison=" << "native_vs_ipctk" << "\n";
    meta << "min_distance=" << native_options.min_distance << "\nccd_tolerance=" << native_options.ccd_tolerance
         << "\nccd_max_iterations=" << native_options.ccd_max_iterations
         << "\nhessian=Projected\nline_search_shrink=" << options.line_search_shrink
         << "\nmax_line_search_steps=" << options.max_line_search_steps << '\n';
    auto runs = file("runs.csv");
    runs << "round,backend,seconds,status,iterations,potential,elastic,pulling,contact,free_residual,active_pairs,min_"
            "distance,repeat_max_abs\n";
    auto traces = file("iterations.csv");
    traces << "round,backend,iteration,energy,gradient_norm,pattern_changed,diagonal_shifts,feasible_step,step,line_"
              "search_evaluations\n";
    std::array<Vec, 2> solutions;
    bool valid = true;
    // Round 0 warms both backends; measured rounds alternate backend order.
    for (int round = 0; round <= repetitions; ++round) {
        for (int order = 0; order < 2; ++order) {
            const int backend = (order + round) % 2;
            const std::string name = backend == 0 ? "native" : "ipctk";
            std::shared_ptr<contact::IPCContactEnergy> native;
            std::shared_ptr<contact::IPCTKContactEnergy> toolkit;
            std::shared_ptr<energy::Energy> barrier;
            if (backend == 0)
                barrier = native = std::make_shared<contact::IPCContactEnergy>(surface, native_options, spatial_hash);
            else
                barrier = toolkit = std::make_shared<contact::IPCTKContactEnergy>(surface, toolkit_options);
            auto mapped = std::make_shared<energy::AffineMappedEnergy>(barrier, kinematics.displacement_map(),
                                                                       kinematics.rest_positions());
            energy::WeightedEnergySum potential({{elastic, 1}, {pulling, 1}, {mapped, kappa}});
            linalg::AutoSparseSolver solver;
            std::vector<solver::NewtonStep> steps;
            steps.reserve(options.max_iterations + 1);
            const std::function<void(const solver::NewtonStep&)> capture = [&](const solver::NewtonStep& step) {
                steps.push_back(step);
            };
            const auto start = std::chrono::steady_clock::now();
            const auto result = solver::newton_solve(potential, fixed, initial, options, solver, capture);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const Vec positions = kinematics.positions(result.x);
            const int pairs = native ? native->num_active_pairs(positions) : toolkit->num_active_pairs(positions);
            const double distance =
                native ? native->min_active_distance(positions) : toolkit->min_active_distance(positions);
            const double repeat = round == 0 ? 0 : (result.x - solutions[backend]).cwiseAbs().maxCoeff();
            if (round == 0)
                solutions[backend] = result.x;
            valid = valid && result.status == solver::NewtonStatus::Converged &&
                    result.gradient_norm <= options.gradient_tolerance && pairs > 0 && distance > 0 && distance < dhat;
            runs << round << ',' << name << ',' << seconds << ',' << static_cast<int>(result.status) << ','
                 << result.iterations << ',' << result.energy << ',' << elastic->eval_value(result.x) << ','
                 << pulling->eval_value(result.x) << ',' << kappa * mapped->eval_value(result.x) << ','
                 << result.gradient_norm << ',' << pairs << ',' << distance << ',' << repeat << '\n';
            for (const auto& s : steps)
                traces << round << ',' << name << ',' << s.iteration << ',' << s.energy << ',' << s.gradient_norm << ','
                       << s.pattern_changed << ',' << s.diagonal_shifts << ',' << s.feasible_step << ',' << s.step
                       << ',' << s.line_search_evaluations << '\n';
            auto displacement = file(name + "_" + std::to_string(round) + "_displacement.csv");
            displacement << "dof,displacement\n";
            for (Eigen::Index i = 0; i < result.x.size(); ++i)
                displacement << i << ',' << result.x[i] << '\n';
            std::cout << name << " round=" << round << " seconds=" << seconds
                      << " status=" << static_cast<int>(result.status) << " iterations=" << result.iterations
                      << std::endl;
        }
    }
    auto comparison = file("displacement_comparison.csv");
    const Vec delta = solutions[0] - solutions[1];
    comparison << "max_abs,rms,native_norm," << "ipctk_norm\n"
               << delta.cwiseAbs().maxCoeff() << ',' << delta.norm() / std::sqrt(double(delta.size())) << ','
               << solutions[0].norm() << ',' << solutions[1].norm() << '\n';
    contact::IPCContactEnergy native(surface, native_options, spatial_hash);
    contact::IPCTKContactEnergy toolkit(surface, toolkit_options);
    auto shared = file("shared_states.csv");
    shared << "state,hessian_kind,native_energy,ipctk_energy,energy_abs,gradient_max_abs,gradient_rms,gradient_"
              "relative_l2,hessian_max_abs,hessian_rms,hessian_relative_frobenius,native_gradient_norm,native_hessian_"
              "norm,native_pairs,ipctk_pairs\n";
    const std::array<Vec, 4> states{initial, solutions[0], solutions[1], (solutions[0] + solutions[1]) / 2};
    const std::array<const char*, 4> names{"initial", "native_final", "ipctk_final", "midpoint"};
    auto ccd = file("shared_ccd.csv");
    ccd << "state,native_step,ipctk_step\n";
    const Vec direction = kinematics.positions(2 * solutions[0]) - kinematics.positions(initial);
    for (std::size_t i = 0; i < states.size(); ++i) {
        const Vec positions = kinematics.positions(states[i]);
        ccd << names[i] << ',' << native.max_feasible_step(positions, direction) << ','
            << toolkit.max_feasible_step(positions, direction) << '\n';
        native.update_pattern(positions);
        toolkit.update_pattern(positions);
        for (const auto kind : {energy::HessianKind::Exact, energy::HessianKind::Projected}) {
            Vec gn, gt;
            SpMat hn = native.hessian_pattern(), ht = toolkit.hessian_pattern();
            const double en = kappa * native.eval(positions, &gn, &hn, kind);
            const double et = kappa * toolkit.eval(positions, &gt, &ht, kind);
            gn *= kappa;
            gt *= kappa;
            hn *= kappa;
            ht *= kappa;
            const Vec gd = gn - gt;
            const SpMat hd = hn - ht;
            double maximum = 0;
            for (int j = 0; j < hd.nonZeros(); ++j)
                maximum = std::max(maximum, std::abs(hd.valuePtr()[j]));
            shared << names[i] << ',' << (kind == energy::HessianKind::Exact ? "exact" : "projected") << ',' << en
                   << ',' << et << ',' << std::abs(en - et) << ',' << gd.cwiseAbs().maxCoeff() << ','
                   << gd.norm() / std::sqrt(double(gd.size())) << ',' << gd.norm() / std::max(gn.norm(), 1e-300) << ','
                   << maximum << ',' << hd.norm() / double(hd.rows()) << ',' << hd.norm() / std::max(hn.norm(), 1e-300)
                   << ',' << gn.norm() << ',' << hn.norm() << ',' << native.num_active_pairs(positions) << ','
                   << toolkit.num_active_pairs(positions) << '\n';
        }
    }
    if (!valid)
        std::cerr << "Benchmark failed convergence/contact checks; see runs.csv\n";
    return valid ? 0 : 1;
}
