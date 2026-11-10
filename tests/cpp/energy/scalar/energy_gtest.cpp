#include "../../testing/evaluate_energy.hpp"
#include "litepgo/energy/scalar/detail/evaluation_access.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/energy/scalar/quadratic_energy.hpp"

#include "../../testing/finite_difference.hpp"
#include "litepgo/energy/scalar/affine_mapped_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"

#include "../../testing/test_energies.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace litepgo::energy::test {

namespace {

// Breaks the contract in configurable ways.
class MisbehavingEnergy final : public litepgo::energy::FixedPatternEnergy {
  public:
    double step = 1.0;
    double returned = 0.0;
    mutable bool reenter = false;

    MisbehavingEnergy() : FixedPatternEnergy(chain_pattern(2, {})) {}

  private:
    double eval_values(const litepgo::ConstVecRef& x, Vec* gradient, SpMat* hessian) const override {
        if (reenter) {
            reenter = false;
            evaluate(x)->value();
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
    SpMat hessian;
    const double value = litepgo::testing::evaluate_energy(energy, x, &gradient, &hessian, true);
    EXPECT_DOUBLE_EQ(value, energy.evaluate(x)->value());
    EXPECT_EQ(gradient, energy.evaluate(x)->gradient());
    EXPECT_EQ(Eigen::MatrixXd(hessian), Eigen::MatrixXd(energy.evaluate(x)->projected_hessian()));
}

TEST(Energy, ChainDerivativesMatchFiniteDifferences) {
    const DoubleWellChain energy(6);
    expect_derivatives_match(energy, Vec::LinSpaced(6, -1.3, 1.1));
}

TEST(Energy, ProjectedHessianIsPositiveSemidefinite) {
    const DoubleWellChain energy(3);
    const Vec x = Vec::Constant(3, 0.1); // inside the double wells' concave region
    const Eigen::MatrixXd exact = Eigen::MatrixXd(energy.evaluate(x)->hessian());
    const Eigen::MatrixXd projected = Eigen::MatrixXd(energy.evaluate(x)->projected_hessian());
    EXPECT_NE(exact, projected);
    EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(projected).eigenvalues().minCoeff(), -1e-12);
}

TEST(Energy, ChecksArgumentsBeforeWriting) {
    const DoubleWellChain energy(4);
    EXPECT_THROW(energy.evaluate(Vec::Zero(3))->value(), std::invalid_argument);

    SpMat wrong = chain_pattern(4, {});
    Vec gradient = Vec::Constant(2, 7.0);
    EXPECT_THROW(litepgo::testing::evaluate_energy(energy, Vec::Zero(3), &gradient, &wrong), std::invalid_argument);
    EXPECT_EQ(gradient, Vec::Constant(2, 7.0)); // untouched

    EXPECT_THROW(energy.evaluate(Vec::Zero(4))->max_feasible_step(Vec::Zero(3)), std::invalid_argument);
}

TEST(Energy, ResizesTheGradient) {
    const DoubleWellChain energy(4);
    Vec gradient;
    litepgo::testing::evaluate_energy(energy, Vec::Zero(4), &gradient, nullptr);
    EXPECT_EQ(gradient.size(), 4);
}

TEST(Energy, DefaultFeasibleStepIsTheFullStep) {
    const DoubleWellChain energy(3);
    EXPECT_EQ(energy.evaluate(Vec::Zero(3))->max_feasible_step(Vec::Ones(3)), 1.0);
}

TEST(Energy, RejectsAFeasibleStepOutsideTheUnitInterval) {
    MisbehavingEnergy energy;
    energy.step = 1.5;
    EXPECT_THROW(energy.evaluate(Vec::Zero(2))->max_feasible_step(Vec::Ones(2)), std::logic_error);
    energy.step = std::nan("");
    EXPECT_THROW(energy.evaluate(Vec::Zero(2))->max_feasible_step(Vec::Ones(2)), std::logic_error);
}

TEST(Energy, SupportsFiniteReentry) {
    MisbehavingEnergy energy;
    energy.reenter = true;
    EXPECT_NO_THROW(energy.evaluate(Vec::Zero(2))->value());
    energy.reenter = false;
    EXPECT_NO_THROW(energy.evaluate(Vec::Zero(2))->value());
}

TEST(Energy, RejectsNonFiniteValuesAsABrokenEnergy) {
    const double infinity = std::numeric_limits<double>::infinity();
    for (const double returned : {infinity, -infinity, std::nan("")}) {
        const auto energy = std::make_shared<MisbehavingEnergy>();
        energy->returned = returned;
        EXPECT_THROW(energy->evaluate(Vec::Zero(2))->value(), std::logic_error);
        // The error surfaces through composites too, and is not mistaken for infeasibility.
        const litepgo::energy::WeightedSumEnergy sum({{energy, -1.0}});
        EXPECT_THROW(sum.evaluate(Vec::Zero(2))->value(), std::logic_error);
        const litepgo::energy::AffineMappedEnergy mapped(energy, SpMat(Eigen::MatrixXd::Identity(2, 2).sparseView()));
        EXPECT_THROW(mapped.evaluate(Vec::Zero(2))->value(), std::logic_error);
        try {
            sum.evaluate(Vec::Zero(2))->value();
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
        Broken() : FixedPatternEnergy(pattern()) {}
        double eval_values(const ConstVecRef&, Vec* g, SpMat* H) const override {
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
    for (int mode : {1, 2, 3
#ifndef NDEBUG
                     ,
                     4
#endif
         }) {
        e.mode = mode;
        EXPECT_DOUBLE_EQ(e.evaluate(x)->value(), 1);
        if (mode <= 2)
            EXPECT_THROW(e.evaluate(x)->gradient(), std::logic_error);
        else
            EXPECT_THROW(e.evaluate(x)->hessian(), std::logic_error);
        e.mode = 0;
        EXPECT_TRUE(e.evaluate(x)->gradient().isOnes());
        EXPECT_DOUBLE_EQ(e.evaluate(x)->hessian().coeff(0, 0), 1);
    }
    Vec bad = x;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    const int calls = e.calls;
    EXPECT_THROW(e.evaluate(bad)->value(), std::invalid_argument);
    EXPECT_THROW(e.evaluate(bad)->hessian(), std::invalid_argument);
    EXPECT_THROW(e.evaluate(x)->max_feasible_step(bad), std::invalid_argument);
    EXPECT_EQ(e.calls, calls);
}

TEST(EnergyEvaluation, NativeBorrowsSurviveCacheUpgradesAndOtherEvaluations) {
    using namespace litepgo;
    using Access = energy::detail::EvaluationAccess;
    Eigen::Matrix2d A;
    A << -2., 0., 0., 3.;
    energy::QuadraticEnergy model(A.sparseView());
    for (bool projected_first : {false, true}) {
        auto e = model.evaluate(Vec::Ones(2));
        const Vec& g = Access::gradient(*e);
        const double* gradient_data = g.data();
        const SpMat& first = projected_first ? Access::projected_hessian(*e) : Access::hessian(*e);
        const double* first_data = first.valuePtr();
        const SpMat& second = projected_first ? Access::hessian(*e) : Access::projected_hessian(*e);
        const double* second_data = second.valuePtr();
        model.evaluate(Vec::Constant(2, 4.))->hessian();
        EXPECT_EQ(Access::gradient(*e).data(), gradient_data);
        EXPECT_EQ(g, A * Vec::Ones(2));
        EXPECT_EQ(first.valuePtr(), first_data);
        EXPECT_EQ(second.valuePtr(), second_data);
        EXPECT_EQ(Eigen::MatrixXd(Access::hessian(*e)), A);
        EXPECT_DOUBLE_EQ(Access::projected_hessian(*e).coeff(0, 0), 0.);
        Vec owned_gradient = e->gradient();
        SpMat owned_hessian = e->hessian();
        owned_gradient.setZero();
        owned_hessian.coeffs().setZero();
        EXPECT_EQ(g, A * Vec::Ones(2));
        EXPECT_EQ(Eigen::MatrixXd(Access::hessian(*e)), A);
    }
}

#ifndef NDEBUG
TEST(EnergyEvaluation, RejectsUncompressedExactAndProjectedResultsWithoutAborting) {
    using namespace litepgo;
    class Uncompressed final : public energy::Energy {
      public:
        Uncompressed() : Energy(2) {}

      private:
        static SpMat matrix() {
            SpMat H(2, 2);
            H.insert(0, 0) = 1;
            H.insert(1, 1) = 1;
            EXPECT_FALSE(H.isCompressed());
            return H;
        }
        energy::EnergyValues eval_impl(const ConstVecRef&, bool gradient, bool hessian) const override {
            energy::EnergyValues result{0., std::nullopt, std::nullopt};
            if (gradient)
                result.gradient = Vec::Zero(2);
            if (hessian)
                result.hessian = matrix();
            return result;
        }
        energy::EnergyValues eval_projected_values(const ConstVecRef& x, energy::EnergyEvaluation&) const override {
            return eval_impl(x, true, true);
        }
    } model;
    auto state = model.evaluate(Vec::Zero(2));
    EXPECT_THROW(state->hessian(), std::logic_error);
    EXPECT_THROW(state->projected_hessian(), std::logic_error);
    EXPECT_DOUBLE_EQ(state->value(), 0.);
}
#endif

TEST(EnergyEvaluation, NestedChildrenOwnIndependentSnapshots) {
    using namespace litepgo;
    static_assert(!std::is_copy_constructible_v<energy::EnergyEvaluation>);
    static_assert(!std::is_move_constructible_v<energy::EnergyEvaluation>);
    struct Probe final : energy::Energy {
        Probe() : Energy(2) {}
        mutable std::vector<const double*> inputs;
        mutable int destroyed = 0;
        struct State final : energy::EnergyEvaluation {
            const Probe& model;
            State(const Probe& model, const ConstVecRef& x) : EnergyEvaluation(model, x), model(model) {
                EXPECT_NE(x_.data(), x.data());
            }
            ~State() override {
                EXPECT_TRUE(x_.allFinite());
                ++model.destroyed;
            }
        };
        std::shared_ptr<energy::EnergyEvaluation> make_evaluation(const ConstVecRef& x) const override {
            return std::make_shared<State>(*this, x);
        }
        energy::EnergyValues eval_impl(const ConstVecRef& x, bool gradient, bool hessian) const override {
            inputs.push_back(x.data());
            energy::EnergyValues r{.5 * x.squaredNorm(), std::nullopt, std::nullopt};
            if (gradient)
                r.gradient = x;
            if (hessian) {
                r.hessian.emplace(2, 2);
                r.hessian->setIdentity();
            }
            return r;
        }
        double feasible_step(const ConstVecRef& x, const ConstVecRef&) const override {
            inputs.push_back(x.data());
            return 1. / (1. + x.squaredNorm());
        }
    };
    const auto leaf = std::make_shared<Probe>();
    const auto pair =
        std::make_shared<energy::WeightedSumEnergy>(std::vector<energy::WeightedEnergy>{{leaf, .75}, {leaf, .25}});
    energy::WeightedSumEnergy nested({{pair, .5}, {pair, .5}});
    const Vec original = (Vec(2) << 1., 2.).finished();
    Vec x = original;
    auto a = nested.evaluate(x);
    x.setConstant(7.);
    auto b = nested.evaluate(x);
    x.resize(0);
    EXPECT_EQ(Eigen::MatrixXd(a->hessian()), Eigen::Matrix2d::Identity());
    ASSERT_EQ(leaf->inputs.size(), 4);
    const auto first_inputs = leaf->inputs;
    EXPECT_EQ(std::set<const double*>(first_inputs.begin(), first_inputs.end()).size(), 4);
    leaf->inputs.clear();
    EXPECT_EQ(b->gradient(), Vec::Constant(2, 7.));
    ASSERT_EQ(leaf->inputs.size(), 4);
    std::set<const double*> all_inputs(first_inputs.begin(), first_inputs.end());
    all_inputs.insert(leaf->inputs.begin(), leaf->inputs.end());
    EXPECT_EQ(all_inputs.size(), 8);
    b.reset();
    EXPECT_EQ(leaf->destroyed, 4);
    EXPECT_EQ(a->gradient(), original);
    EXPECT_EQ(Eigen::MatrixXd(a->projected_hessian()), Eigen::Matrix2d::Identity());
    EXPECT_DOUBLE_EQ(a->max_feasible_step(Vec::Ones(2)), 1. / 6.);
    a.reset();
    EXPECT_EQ(leaf->destroyed, 8);
    auto standalone = leaf->evaluate(Vec::Ones(2));
    EXPECT_EQ(standalone->gradient(), Vec::Ones(2));
}
