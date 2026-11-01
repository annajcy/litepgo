#include "litepgo/energy/energy.hpp"

#include "../testing/finite_difference.hpp"
#include "litepgo/energy/affine_mapped_energy.hpp"
#include "litepgo/energy/weighted_energy_sum.hpp"

#include "../testing/test_energies.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace litepgo::energy::test {

namespace {

// Breaks the contract in configurable ways.
class MisbehavingEnergy final : public litepgo::energy::FixedPatternEnergy {
  public:
    double step = 1.0;
    double returned = 0.0;
    bool reenter = false;

    MisbehavingEnergy() : FixedPatternEnergy(2, chain_pattern(2, {})) {}

  private:
    double eval_values(const litepgo::ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind) const override {
        if (reenter) {
            eval_value(x);
        }
        if (gradient != nullptr) {
            gradient->setZero();
        }
        if (hessian != nullptr) {
            zero_values(*hessian);
        }
        return returned;
    }
    double feasible_step(const litepgo::ConstVecRef&, const litepgo::ConstVecRef&) const override {
        return step;
    }
};

} // namespace

TEST(Energy, ConveniencesAgreeWithEvaluate) {
    const DoubleWellChain energy(5);
    const Vec x = Vec::LinSpaced(5, -1.2, 0.9);
    Vec gradient;
    SpMat hessian = energy.hessian_pattern();
    const double value = energy.eval(x, &gradient, &hessian, HessianKind::Projected);
    EXPECT_EQ(value, energy.eval_value(x));
    EXPECT_EQ(gradient, energy.eval_gradient(x));
    EXPECT_EQ(Eigen::MatrixXd(hessian), Eigen::MatrixXd(energy.eval_hessian(x, HessianKind::Projected)));
}

TEST(Energy, ChainDerivativesMatchFiniteDifferences) {
    const DoubleWellChain energy(6);
    expect_derivatives_match(energy, Vec::LinSpaced(6, -1.3, 1.1));
}

TEST(Energy, ProjectedHessianIsPositiveSemidefinite) {
    const DoubleWellChain energy(3);
    const Vec x = Vec::Constant(3, 0.1); // inside the double wells' concave region
    const Eigen::MatrixXd exact = Eigen::MatrixXd(energy.eval_hessian(x, HessianKind::Exact));
    const Eigen::MatrixXd projected = Eigen::MatrixXd(energy.eval_hessian(x, HessianKind::Projected));
    EXPECT_NE(exact, projected);
    EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(projected).eigenvalues().minCoeff(), -1e-12);
}

TEST(Energy, ChecksArgumentsBeforeWriting) {
    const DoubleWellChain energy(4);
    EXPECT_THROW(energy.eval_value(Vec::Zero(3)), std::invalid_argument);

    SpMat wrong = chain_pattern(4, {});
    Vec gradient = Vec::Constant(2, 7.0);
    EXPECT_THROW(energy.eval(Vec::Zero(4), &gradient, &wrong), std::invalid_argument);
    EXPECT_EQ(gradient, Vec::Constant(2, 7.0)); // untouched

    EXPECT_THROW(energy.max_feasible_step(Vec::Zero(4), Vec::Zero(3)), std::invalid_argument);
}

TEST(Energy, ResizesTheGradient) {
    const DoubleWellChain energy(4);
    Vec gradient;
    energy.eval(Vec::Zero(4), &gradient, nullptr);
    EXPECT_EQ(gradient.size(), 4);
}

TEST(Energy, DefaultFeasibleStepIsTheFullStep) {
    const DoubleWellChain energy(3);
    EXPECT_EQ(energy.max_feasible_step(Vec::Zero(3), Vec::Ones(3)), 1.0);
}

TEST(Energy, RejectsAFeasibleStepOutsideTheUnitInterval) {
    MisbehavingEnergy energy;
    energy.step = 1.5;
    EXPECT_THROW(energy.max_feasible_step(Vec::Zero(2), Vec::Ones(2)), std::logic_error);
    energy.step = std::nan("");
    EXPECT_THROW(energy.max_feasible_step(Vec::Zero(2), Vec::Ones(2)), std::logic_error);
}

TEST(Energy, RejectsOverlappingCalls) {
    MisbehavingEnergy energy;
    energy.reenter = true;
    EXPECT_THROW(energy.eval_value(Vec::Zero(2)), std::logic_error);
    energy.reenter = false;
    EXPECT_NO_THROW(energy.eval_value(Vec::Zero(2)));
}

TEST(Energy, RejectsNonFiniteValuesAsABrokenEnergy) {
    const double infinity = std::numeric_limits<double>::infinity();
    for (const double returned : {infinity, -infinity, std::nan("")}) {
        const auto energy = std::make_shared<MisbehavingEnergy>();
        energy->returned = returned;
        EXPECT_THROW(energy->eval_value(Vec::Zero(2)), std::logic_error);
        // The error surfaces through composites too, and is not mistaken for infeasibility.
        const litepgo::energy::WeightedEnergySum sum({{energy, -1.0}});
        EXPECT_THROW(sum.eval_value(Vec::Zero(2)), std::logic_error);
        const litepgo::energy::AffineMappedEnergy mapped(energy, SpMat(Eigen::MatrixXd::Identity(2, 2).sparseView()));
        EXPECT_THROW(mapped.eval_value(Vec::Zero(2)), std::logic_error);
        try {
            sum.eval_value(Vec::Zero(2));
        } catch (const std::domain_error&) {
            ADD_FAILURE() << "a non-finite value must not be reported as std::domain_error";
        } catch (const std::logic_error&) {
        }
    }
}

} // namespace litepgo::energy::test

TEST(EnergyContract, RejectsBrokenRequestedDerivativesAndRecovers) {
    using namespace litepgo;
    struct Broken final : energy::FixedPatternEnergy {
        int mode = 0;
        mutable int calls = 0;
        static SpMat pattern() {
            SpMat p(2, 2);
            p.insert(0, 0) = 0;
            return p;
        }
        Broken() : FixedPatternEnergy(2, pattern()) {}
        double eval_values(const ConstVecRef&, Vec* g, SpMat* H, energy::HessianKind) const override {
            ++calls;
            if (g) {
                g->setOnes();
                if (mode == 1)
                    g->resize(1);
                if (mode == 2)
                    (*g)[0] = std::numeric_limits<double>::quiet_NaN();
            }
            if (H) {
                H->coeffs().setOnes();
                if (mode == 3)
                    H->valuePtr()[0] = std::numeric_limits<double>::infinity();
                if (mode == 4)
                    H->coeffRef(1, 1) = 1;
            }
            return 1;
        }
    } e;
    const Vec x = Vec::Zero(2);
    for (int mode : {1, 2, 3, 4}) {
        e.mode = mode;
        EXPECT_DOUBLE_EQ(e.eval_value(x), 1);
        if (mode <= 2)
            EXPECT_THROW(e.eval_gradient(x), std::logic_error);
        else
            EXPECT_THROW(e.eval_hessian(x), std::logic_error);
        e.mode = 0;
        EXPECT_TRUE(e.eval_gradient(x).isOnes());
        EXPECT_DOUBLE_EQ(e.eval_hessian(x).coeff(0, 0), 1);
    }
    Vec bad = x;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    const int calls = e.calls;
    EXPECT_THROW(e.eval_value(bad), std::invalid_argument);
    EXPECT_THROW(e.update_pattern(bad), std::invalid_argument);
    EXPECT_THROW(e.max_feasible_step(x, bad), std::invalid_argument);
    EXPECT_EQ(e.calls, calls);
}
