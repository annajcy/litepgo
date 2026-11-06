#pragma once
#include "litepgo/solver/static_equilibrium_response.hpp"
namespace litepgo::testing {
class FormulaForce final : public energy::FixedPatternResponse {
  public:
    using Factory = std::function<std::shared_ptr<energy::Energy>(const Vec&)>;
    using Partial = std::function<SpMat(const Vec&, const Vec&)>;
    FormulaForce(int n, Vec u, Factory factory, Partial partial)
        : FixedPatternResponse(Eigen::MatrixXd::Ones(u.size(), n).sparseView()), u_(std::move(u)),
          factory_(std::move(factory)), partial_(std::move(partial)) {}

  private:
    Vec eval_impl(const ConstVecRef& p, SpMat* J) const override {
        const Vec value = factory_(p)->eval_gradient(u_);
        if (J) {
            const SpMat source = partial_(p, u_);
            if (source.rows() != J->rows() || source.cols() != J->cols() || !source.coeffs().allFinite())
                throw std::invalid_argument("Invalid force parameter derivative");
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = source.coeff(it.row(), c);
        }
        return value;
    }
    Vec u_;
    Factory factory_;
    Partial partial_;
};
inline solver::StaticEquilibriumResponse::MakeResponse force_factory(int n, FormulaForce::Factory factory,
                                                                     FormulaForce::Partial partial) {
    return [n, factory = std::move(factory), partial = std::move(partial)](const Vec& u) {
        return std::make_shared<FormulaForce>(n, u, factory, partial);
    };
}
} // namespace litepgo::testing
