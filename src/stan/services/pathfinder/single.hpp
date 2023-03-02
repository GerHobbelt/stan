#ifndef STAN_SERVICES_PATHFINDER_SINGLE_HPP
#define STAN_SERVICES_PATHFINDER_SINGLE_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/optimization/lbfgs_update.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <boost/circular_buffer.hpp>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
#include <string>
#include <vector>
#include <atomic>

// Turns on all debugging
#define STAN_DEBUG_PATH_ALL false
// prints results of lbfgs
#define STAN_DEBUG_PATH_POST_LBFGS false || STAN_DEBUG_PATH_ALL
// prints taylor approximation values each iteration
#define STAN_DEBUG_PATH_TAYLOR_APPX false || STAN_DEBUG_PATH_ALL
// prints approximate draw information each iteration
#define STAN_DEBUG_PATH_ELBO_DRAWS false || STAN_DEBUG_PATH_ALL
// prints taylor curve test info
#define STAN_DEBUG_PATH_CURVE_CHECK false || STAN_DEBUG_PATH_ALL
// prints info used for random normal generations during each iteration
#define STAN_DEBUG_PATH_RNORM_DRAWS false || STAN_DEBUG_PATH_ALL
// prints all debug info that happens each iteration
#define STAN_DEBUG_PATH_ITERS                                      \
  STAN_DEBUG_PATH_ALL || STAN_DEBUG_PATH_POST_LBFGS                \
      || STAN_DEBUG_PATH_TAYLOR_APPX || STAN_DEBUG_PATH_ELBO_DRAWS \
      || STAN_DEBUG_PATH_CURVE_CHECK || STAN_DEBUG_PATH_RNORM_DRAWS

namespace stan {
namespace services {
namespace pathfinder {
namespace internal {

/**
 * Namespace holds debug utils only used if the `STAN_DEBUG_PATH_*` flags are on
 */
namespace debug {
template <typename T0, typename T1, typename T2, typename T3>
inline void elbo_draws(T0&& logger, T1&& taylor_approx, T2&& approx_samples,
                       T3&& lp_mat, double ELBO) {
  if (STAN_DEBUG_PATH_ELBO_DRAWS) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                 ", ", ", ", "\n", "", "", " ");
    std::stringstream debug_stream;
    debug_stream
        << "\n Rando Sums: \n"
        << approx_samples.array().square().colwise().sum().eval().format(
               CommaInitFmt)
        << "\n";
    debug_stream << "logdetcholHk: " << taylor_approx.logdetcholHk << "\n";
    debug_stream << "ELBO: " << ELBO << "\n";
    debug_stream << "repeat_draws: \n"
                 << approx_samples.transpose().eval().format(CommaInitFmt)
                 << "\n";
    debug_stream << "lp_approx: \n"
                 << lp_mat.col(1).transpose().eval().format(CommaInitFmt)
                 << "\n";
    debug_stream << "fn_call: \n"
                 << lp_mat.col(0).transpose().eval().format(CommaInitFmt)
                 << "\n";
    Eigen::MatrixXd param_vals = approx_samples;
    auto mean_vals = param_vals.rowwise().mean().eval();
    debug_stream << "Mean Values: \n"
                 << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
    debug_stream << "SD Values: \n"
                 << (((param_vals.colwise() - mean_vals)
                          .array()
                          .square()
                          .matrix()
                          .rowwise()
                          .sum()
                          .array()
                      / (param_vals.cols() - 1))
                         .sqrt())
                        .transpose()
                        .eval()
                 << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T>
inline void rnorm_draws(T0&& logger, T&& approx_samples_tmp) {
  if (STAN_DEBUG_PATH_RNORM_DRAWS) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                 ", ", ", ", "\n", "", "", " ");
    Eigen::MatrixXd param_vals = approx_samples_tmp;
    auto mean_vals = param_vals.rowwise().mean().eval();
    std::stringstream debug_stream;
    debug_stream << "Mean Values: \n"
                 << mean_vals.transpose().eval().format(CommaInitFmt) << "\n";
    debug_stream << "SD Values: \n"
                 << (((param_vals.colwise() - mean_vals)
                          .array()
                          .square()
                          .matrix()
                          .rowwise()
                          .sum()
                          .array()
                      / (param_vals.cols() - 1))
                         .sqrt())
                        .transpose()
                        .eval()
                 << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1, typename T2,
          require_all_eigen_vector_t<T1, T2>* = nullptr>
inline void print_curve(T0&& logger, T1&& Dk, T2&& thetak) {
  if (STAN_DEBUG_PATH_CURVE_CHECK) {
    std::stringstream debug_stream;
    debug_stream << "\n Check Dk: \n" << Dk.transpose() << "\n";
    debug_stream << "\n Check thetak: \n" << thetak.transpose() << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1, typename T2,
          require_all_stan_scalar_t<T1, T2>* = nullptr>
inline void print_curve(T0&& logger, T1&& Dk, T2&& thetak) {
  if (STAN_DEBUG_PATH_CURVE_CHECK) {
    std::stringstream debug_stream;
    debug_stream << "\n Check Dk: \n" << Dk << "\n";
    debug_stream << "\n Check thetak: \n" << thetak << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
inline void post_lbfgs(T0&& logger, T1&& update_best_mutex, T2&& num_parameters,
                       T3&& num_elbo_draws, T4&& alpha_mat, T5&& Ykt_diff,
                       T6&& Skt_diff) {
  if (STAN_DEBUG_PATH_POST_LBFGS) {
    std::lock_guard<std::mutex> guard(update_best_mutex);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::stringstream debug_stream;
    debug_stream << "\n num_params: " << num_parameters << "\n";
    debug_stream << "\n num_elbo_params: " << num_elbo_draws << "\n";
    debug_stream << "\n Alpha mat: "
                 << alpha_mat.transpose().eval().format(CommaInitFmt) << "\n";
    debug_stream << "\n Ykt_diff mat: "
                 << Ykt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    debug_stream << "\n Skt_diff mat: "
                 << Skt_diff.transpose().eval().format(CommaInitFmt) << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5>
inline void taylor_appx_full1(T0&& logger, T1&& alpha, T2&& ninvRST, T3&& Dk,
                              T4&& point_est, T5&& grad_est) {
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::stringstream debug_stream;
    debug_stream << "---Full---\n";

    debug_stream << "Alpha: \n" << alpha.format(CommaInitFmt) << "\n";
    debug_stream << "ninvRST: \n" << ninvRST.format(CommaInitFmt) << "\n";
    debug_stream << "Dk: \n" << Dk.format(CommaInitFmt) << "\n";
    debug_stream << "Point: \n" << point_est.format(CommaInitFmt) << "\n";
    debug_stream << "grad: \n" << grad_est.format(CommaInitFmt) << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
inline void taylor_appx_full2(T0&& logger, T1&& Hk, T2&& L_hk,
                              T3&& logdetcholHk, T4&& x_center) {
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    std::cout << "---Full---\n";
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");

    std::stringstream debug_stream;
    debug_stream << "Hk: " << Hk.format(CommaInitFmt) << "\n";
    debug_stream << "L_approx: \n" << L_hk.format(CommaInitFmt) << "\n";
    debug_stream << "logdetcholHk: \n" << logdetcholHk << "\n";
    debug_stream << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1>
inline void taylor_appx_sparse1(T0&& logger, T1&& Wkbart) {
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::stringstream debug_stream;
    debug_stream << "---Sparse---\n";
    debug_stream << "Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
    logger.info(debug_stream);
  }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9,
          typename T10, typename T11>
inline void taylor_appx_sparse2(T0&& logger, T1&& qr, T2&& alpha, T3&& Qk,
                                T4&& L_approx, T5&& logdetcholHk, T6&& Mkbar,
                                T7&& Wkbart, T8&& x_center, T9&& ninvRST,
                                T10&& ninvRSTg, T11&& Rkbar) {
  if (STAN_DEBUG_PATH_TAYLOR_APPX) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n",
                                 "", "", " ");
    std::stringstream debug_stream;
    debug_stream << "Full QR: \n" << qr.matrixQR().format(CommaInitFmt) << "\n";
    debug_stream << "Alpha: \n" << alpha.format(CommaInitFmt) << "\n";
    debug_stream << "Qk: \n" << Qk.format(CommaInitFmt) << "\n";
    debug_stream << "L_approx: \n" << L_approx.format(CommaInitFmt) << "\n";
    debug_stream << "logdetcholHk: \n" << logdetcholHk << "\n";
    debug_stream << "Mkbar: \n" << Mkbar.format(CommaInitFmt) << "\n";
    debug_stream << "Decomp Wkbar: \n" << Wkbart.format(CommaInitFmt) << "\n";
    debug_stream << "x_center: \n" << x_center.format(CommaInitFmt) << "\n";
    debug_stream << "NinvRST: " << ninvRST.format(CommaInitFmt) << "\n";
    debug_stream << "ninvRSTg: \n" << ninvRSTg.format(CommaInitFmt) << "\n";
    debug_stream << "Rkbar: " << Rkbar.format(CommaInitFmt) << "\n";
    logger.info(debug_stream);
  }
}
}  // namespace debug

// t(x) * x
template <typename T1>
inline Eigen::MatrixXd tcrossprod(T1&& x) {
  return Eigen::MatrixXd(x.rows(), x.rows())
      .setZero()
      .selfadjointView<Eigen::Lower>()
      .rankUpdate(std::forward<T1>(x));
}

template <typename EigVec, stan::require_eigen_vector_t<EigVec>* = nullptr,
          typename Logger>
inline bool check_curve(const EigVec& Yk, const EigVec& Sk, Logger&& logger) {
  auto Dk = Yk.dot(Sk);
  auto thetak = std::abs(Yk.array().square().sum() / Dk);
  debug::print_curve(logger, Dk, thetak);
  return Dk > 0 && thetak <= 1e12;
}

/**
 * eq 4.9
 * Gilbert, J.C., Lemaréchal, C. Some numerical experiments with
 * variable-storage quasi-Newton algorithms. Mathematical Programming 45,
 * 407–435 (1989). https://doi.org/10.1007/BF01589113
 * @tparam EigVec1 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @tparam EigVec2 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @tparam EigVec3 Type derived from `Eigen::DenseBase` with one column at
 * compile time
 * @param alpha_init Vector of initial values to update
 * @param Yk Vector of gradients
 * @param Sk Vector of values
 */
template <typename EigVec1, typename EigVec2, typename EigVec3>
inline auto form_diag(const EigVec1& alpha_init, const EigVec2& Yk,
                      const EigVec3& Sk) {
  double y_alpha_y = Yk.dot(alpha_init.asDiagonal() * Yk);
  double y_s = Yk.dot(Sk);
  double s_inv_alpha_s
      = Sk.dot(alpha_init.array().inverse().matrix().asDiagonal() * Sk);
  return y_s
         / (y_alpha_y / alpha_init.array() + Yk.array().square()
            - (y_alpha_y / s_inv_alpha_s)
                  * (Sk.array() / alpha_init.array()).square());
}

/**
 * Information from running the taylor approximation
 */
struct taylor_approx_t {
  Eigen::VectorXd x_center;
  double logdetcholHk;       // Log deteriminant of the cholesky
  Eigen::MatrixXd L_approx;  // Approximate choleskly
  Eigen::MatrixXd Qk;  // Q of the QR decompositon. Only used for sparse approx
  Eigen::VectorXd alpha; // diagonal of the initial inv hessian
  bool use_full;  // boolean indicationg if full or sparse approx was used.
};

/**
 * Information from calling ELBO estimation
 */
struct elbo_est_t {
  double elbo{-std::numeric_limits<double>::infinity()};
  size_t fn_calls{0};  // Number of times the log_prob function is called.
  Eigen::MatrixXd repeat_draws;
  Eigen::Array<double, -1, -1> lp_mat;
  Eigen::Array<double, -1, 1> lp_ratio;
};

/**
 * Generate approximate draws using either the full or sparse taylor
 * approximation.
 * @tparam EigMat A type inheriting from `Eigen::DenseBase` with dynamic rows
 * and columns.
 * @tparam EigVec A type inheriting from `Eigen::DenseBase` with the compile
 * time number of columns equal to 1.
 * @param u A matrix of gaussian IID samples with rows equal to the size of the
 * number of samples to be made and columns equal to the number of parameters.
 * @param taylor_approx Approximation from `taylor_approximation`.
 * @param alpha TODO: Define this
 * @return A matrix with rows equal to the number of samples and columns equal
 * to the number of parameters.
 */
template <typename EigMat, 
          require_eigen_matrix_dynamic_t<EigMat>* = nullptr>
inline Eigen::MatrixXd approximate_samples(EigMat&& u,
                                 const taylor_approx_t& taylor_approx) {
  if (taylor_approx.use_full) {
    return (taylor_approx.L_approx.transpose() * std::forward<EigMat>(u)).colwise()
           + taylor_approx.x_center;
  } else {
    return (taylor_approx.alpha.array().sqrt().matrix().asDiagonal()
            * (taylor_approx.Qk
                   * (taylor_approx.L_approx
                      - Eigen::MatrixXd::Identity(
                          taylor_approx.L_approx.rows(),
                          taylor_approx.L_approx.cols()))
                   * (taylor_approx.Qk.transpose() * u) 
               + u))
               .colwise()
           + taylor_approx.x_center;
  }
}

/**
 * Generate approximate draws using either the full or sparse taylor
 * approximation.
 * @tparam EigVec1 A type inheriting from `Eigen::DenseBase` with the compile
 * time number of columns equal to 1.
 * @tparam EigVec2 A type inheriting from `Eigen::DenseBase` with the compile
 * time number of columns equal to 1.
 * @param u A matrix of gaussian IID samples with columns equal to the size of
 * the number of samples to be made and rows equal to the number of parameters.
 * @param taylor_approx Approximation from `taylor_approximation`.
 * @return A matrix with columns equal to the number of samples and rows equal
 * to the number of parameters. Each column represents an approximate draw for
 * the set of parameters.
 */
template <typename EigVec1, typename EigVec2,
          require_eigen_vector_t<EigVec1>* = nullptr>
inline Eigen::VectorXd approximate_samples(EigVec1&& u,
                                 const taylor_approx_t& taylor_approx) {
  if (taylor_approx.use_full) {
    return (taylor_approx.L_approx.transpose() * std::forward<EigVec1>(u)) + taylor_approx.x_center;
  } else {
    return (taylor_approx.alpha.array().sqrt().matrix().asDiagonal()
            * (taylor_approx.Qk
                   * (taylor_approx.L_approx
                      - Eigen::MatrixXd::Identity(
                          taylor_approx.L_approx.rows(),
                          taylor_approx.L_approx.cols()))
                   * (taylor_approx.Qk.transpose() * u)
               + u))
           + taylor_approx.x_center;
  }
}

/**
 * Generate an Eigen matrix of from an rng generator.
 * @tparam RowsAtCompileTime The number of compile time rows for the result
 * matrix.
 * @tparam ColsAtCompileTime The number of compile time cols for the result
 * matrix.
 * @tparam Generator A functor with a valid `operator()` used to generate the
 * samples.
 * @param[in,out] variate_generator An rng generator
 * @param num_params The runtime number of parameters
 * @param num_samples The runtime number of samples.
 *
 */
template <Eigen::Index RowsAtCompileTime = -1,
          Eigen::Index ColsAtCompileTime = -1, typename Generator>
inline Eigen::Matrix<double, RowsAtCompileTime, ColsAtCompileTime>
generate_matrix(Generator&& variate_generator, const Eigen::Index num_params,
                 const Eigen::Index num_samples) {
  return Eigen::Matrix<double, RowsAtCompileTime, ColsAtCompileTime>::
      NullaryExpr(num_params, num_samples,
                  [&variate_generator]() { return variate_generator(); });
}

/**
 * Estimate the approximate draws given the taylor approximation.
 *
 * @tparam ReturnElbo If true, calculate ELBO and return it in `elbo_est_t`. If
 * `false` ELBO is set in the return as `-Infinity`
 * @tparam LPF Type of log probability functor
 * @tparam ConstrainF Type of functor for constraining parameters
 * @tparam RNG Type of random number generator
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 column at
 * compile time.
 * @tparam Logger Type of logger callback
 * @param lp_fun Functor to calculate the log density
 * @param constrain_fun A functor to transform parameters to the constrained
 * space
 * @param[in,out] rng A generator to produce standard gaussian random variables
 * @param taylor_approx The taylor approximation at this iteration of LBFGS
 * @param num_samples Number of approximate samples to generate
 * @param alpha The approximation of the diagonal hessian
 * @param iter_msg The beginning of messages that includes the iteration number
 * @param logger A callback writer for messages
 */
template <bool ReturnElbo = true, typename LPF, typename ConstrainF,
          typename RNG, typename EigVec, typename Logger>
inline elbo_est_t est_approx_draws(LPF&& lp_fun, ConstrainF&& constrain_fun,
                                   RNG&& rng,
                                   const taylor_approx_t& taylor_approx,
                                   size_t num_samples, const EigVec& alpha,
                                   const std::string& iter_msg, Logger&& logger) {
  boost::variate_generator<boost::ecuyer1988&, boost::normal_distribution<>>
      rand_unit_gaus(rng, boost::normal_distribution<>());
  const auto num_params = taylor_approx.x_center.size();
  size_t lp_fun_calls = 0;
  Eigen::MatrixXd unit_samps = generate_matrix(rand_unit_gaus, num_params, num_samples);
  Eigen::Array<double, -1, -1> lp_mat(num_samples, 2);
  lp_mat.col(0) = (-taylor_approx.logdetcholHk)
                  + -0.5
                        * (unit_samps.array().square().colwise().sum()
                           + num_params * stan::math::LOG_TWO_PI);
  Eigen::MatrixXd approx_samples = approximate_samples(std::move(unit_samps), taylor_approx);
  debug::rnorm_draws(logger, approx_samples);
  Eigen::VectorXd approx_samples_col;
  std::stringstream pathfinder_ss;
  const auto log_stream = [](auto& logger, auto& pathfinder_ss) {
    if (pathfinder_ss.str().length() > 0) {
      logger.info(pathfinder_ss);
      pathfinder_ss.str(std::string());
    }
  };
  for (Eigen::Index i = 0; i < num_samples; ++i) {
    try {
      approx_samples_col = approx_samples.col(i);
      ++lp_fun_calls;
      lp_mat.coeffRef(i, 1) = lp_fun(approx_samples_col, pathfinder_ss);
      log_stream(logger, pathfinder_ss);
    } catch (const std::exception& e) {
      lp_mat.coeffRef(i, 1) = -std::numeric_limits<double>::infinity();
      log_stream(logger, pathfinder_ss);
    }
  }
  Eigen::Array<double, -1, 1> lp_ratio = (lp_mat.col(1)) - lp_mat.col(0);
  if (ReturnElbo) {
    double ELBO = lp_ratio.mean();
    debug::elbo_draws(logger, taylor_approx, approx_samples, lp_mat, ELBO);
    return elbo_est_t{ELBO, lp_fun_calls, std::move(approx_samples),
                      std::move(lp_mat), std::move(lp_ratio)};
  } else {
    return elbo_est_t{-std::numeric_limits<double>::infinity(), lp_fun_calls,
                      std::move(approx_samples), std::move(lp_mat),
                      std::move(lp_ratio)};
  }
}

/**
 * Construct the full taylor approximation
 * @tparam GradMat Type inheriting from `Eigen::DenseBase` with compile time
 * dynamic rows and columns
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam DkVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam InvMat Type inheriting from `Eigen::DenseBase` with dynamic compile
 * time rows and columns
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam Logger Type inheriting from `stan::io::logger`
 * @param Ykt_mat Matrix of the changes to the gradient with column length of
 * history size.
 * @param alpha The diagonal of the approximate hessian
 * @param Dk vector of Columnwise products of parameter and gradients with size
 * equal to history size
 * @param ninvRST Inverse of the Rk matrix
 * @param point_est The parameters for the given iteration of LBFGS
 * @param grad_est The gradients for the given iteration of LBFGS
 * @param logger used for printing out debug values
 */
template <typename GradMat, typename AlphaVec, typename DkVec, typename InvMat,
          typename EigVec, typename Logger>
inline taylor_approx_t taylor_approximation_full(
    GradMat&& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est,
    Logger&& logger) {
  debug::taylor_appx_full1(logger, alpha, ninvRST, Dk, point_est, grad_est);
  Eigen::MatrixXd y_tcrossprod_alpha = tcrossprod(
      Ykt_mat.transpose() * alpha.array().sqrt().matrix().asDiagonal());
  /*
   * + DK.asDiagonal() cannot be done one same line
   * See https://forum.kde.org/viewtopic.php?f=74&t=136617
   */
  y_tcrossprod_alpha += Dk.asDiagonal();
  const auto dk_min_size
      = std::min(y_tcrossprod_alpha.rows(), y_tcrossprod_alpha.cols());
  Eigen::MatrixXd y_mul_alpha = Ykt_mat.transpose() * alpha.asDiagonal();
  Eigen::MatrixXd Hk
      = y_mul_alpha.transpose() * ninvRST
        + ninvRST.transpose() * (y_mul_alpha + y_tcrossprod_alpha * ninvRST);
  Hk += alpha.asDiagonal();
  Eigen::MatrixXd L_hk = Hk.llt().matrixL().transpose();
  double logdetcholHk = L_hk.diagonal().array().abs().log().sum();
  Eigen::VectorXd x_center = point_est - Hk * grad_est;
  debug::taylor_appx_full2(logger, Hk, L_hk, logdetcholHk, x_center);
  return taylor_approx_t{std::move(x_center),   logdetcholHk, std::move(L_hk),
                         Eigen::MatrixXd(0, 0), alpha,        true};
}

/**
 * Construct the sparse taylor approximation
 * @tparam GradMat Type inheriting from `Eigen::DenseBase` with compile time
 * dynamic rows and columns
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam DkVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam InvMat Type inheriting from `Eigen::DenseBase` with dynamic compile
 * time rows and columns
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam Logger Type inheriting from `stan::io::logger`
 * @param Ykt_mat Matrix of the changes to the gradient with column length of
 * history size.
 * @param alpha The diagonal of the approximate hessian
 * @param Dk vector of Columnwise products of parameter and gradients with size
 * equal to history size
 * @param ninvRST
 * @param point_est The parameters for the given iteration of LBFGS
 * @param grad_est The gradients for the given iteration of LBFGS
 * @param logger used for printing out debug values
 */
template <typename GradMat, typename AlphaVec, typename DkVec, typename InvMat,
          typename EigVec, typename Logger>
inline auto taylor_approximation_sparse(
    GradMat&& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est,
    Logger&& logger) {
  const Eigen::Index history_size = Ykt_mat.cols();
  const Eigen::Index history_size_times_2 = history_size * 2;
  const Eigen::Index num_params = alpha.size();
  Eigen::MatrixXd y_mul_sqrt_alpha
      = Ykt_mat.transpose() * alpha.array().sqrt().matrix().asDiagonal();
  Eigen::MatrixXd Wkbart(history_size_times_2, num_params);
  Wkbart.topRows(history_size) = y_mul_sqrt_alpha;
  Wkbart.bottomRows(history_size)
      = ninvRST * alpha.array().inverse().sqrt().matrix().asDiagonal();
  debug::taylor_appx_sparse1(logger, Wkbart);
  Eigen::MatrixXd Mkbar(history_size_times_2, history_size_times_2);
  Mkbar.topLeftCorner(history_size, history_size).setZero();
  Mkbar.topRightCorner(history_size, history_size)
      = Eigen::MatrixXd::Identity(history_size, history_size);
  Mkbar.bottomLeftCorner(history_size, history_size)
      = Eigen::MatrixXd::Identity(history_size, history_size);
  Eigen::MatrixXd y_tcrossprod_alpha = tcrossprod(y_mul_sqrt_alpha);
  y_tcrossprod_alpha += Dk.asDiagonal();
  Mkbar.bottomRightCorner(history_size, history_size) = y_tcrossprod_alpha;
  Wkbart.transposeInPlace();
  const auto min_size = std::min(num_params, history_size_times_2);
  // Note: This is doing the QR decomp inplace using Wkbart's memory
  Eigen::HouseholderQR<Eigen::Ref<decltype(Wkbart)>> qr(Wkbart);
  Eigen::MatrixXd Rkbar
      = qr.matrixQR().topLeftCorner(min_size, history_size_times_2);
  Rkbar.triangularView<Eigen::StrictlyLower>().setZero();
  Eigen::MatrixXd Qk
      = qr.householderQ() * Eigen::MatrixXd::Identity(num_params, min_size);
  Eigen::MatrixXd L_approx = (Rkbar * Mkbar * Rkbar.transpose()
                              + Eigen::MatrixXd::Identity(min_size, min_size))
                                 .llt()
                                 .matrixL()
                                 .transpose();
  double logdetcholHk = L_approx.diagonal().array().abs().log().sum()
                        + 0.5 * alpha.array().log().sum();
  Eigen::VectorXd ninvRSTg = ninvRST * grad_est;
  Eigen::VectorXd alpha_mul_grad = (alpha.array() * grad_est.array()).matrix();
  Eigen::VectorXd x_center
      = point_est
        - (alpha_mul_grad
           + (alpha.array() * (Ykt_mat * ninvRSTg).array()).matrix()
           + (ninvRST.transpose()
              * ((Ykt_mat.transpose() * alpha_mul_grad)
                 + y_tcrossprod_alpha * ninvRSTg)));

  debug::taylor_appx_sparse2(logger, qr, alpha, Qk, L_approx, logdetcholHk,
                             Mkbar, Wkbart, x_center, ninvRST, ninvRSTg, Rkbar);
  return taylor_approx_t{std::move(x_center), logdetcholHk, std::move(L_approx),
                         std::move(Qk),       alpha,        false};
}

/**
 * Construct the taylor approximation.
 * @tparam GradMat Type inheriting from `Eigen::DenseBase` with compile time
 * dynamic rows and columns
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam DkVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam InvMat Type inheriting from `Eigen::DenseBase` with dynamic compile
 * time rows and columns
 * @tparam EigVec Type inheriting from `Eigen::DenseBase` with 1 compile time
 * column
 * @tparam Logger Type inheriting from `stan::io::logger`
 * @param Ykt_mat Matrix of the changes to the gradient with column length of
 * history size.
 * @param alpha The diagonal of the approximate hessian
 * @param Dk vector of Columnwise products of parameter and gradients with size
 * equal to history size
 * @param ninvRST
 * @param point_est The parameters for the given iteration of LBFGS
 * @param grad_est The gradients for the given iteration of LBFGS
 * @param logger used for printing out debug values
 */
template <typename GradMat, typename AlphaVec, typename DkVec, typename InvMat,
          typename EigVec, typename Logger>
inline taylor_approx_t taylor_approximation(
    GradMat&& Ykt_mat, const AlphaVec& alpha, const DkVec& Dk,
    const InvMat& ninvRST, const EigVec& point_est, const EigVec& grad_est,
    Logger&& logger) {
  // If twice the current history size is larger than the number of params
  // use a sparse approximation
  if (2 * Ykt_mat.cols() >= Ykt_mat.rows()) {
    return taylor_approximation_full(Ykt_mat, alpha, Dk, ninvRST,
                                               point_est, grad_est, logger);
  } else {
    return taylor_approximation_sparse(Ykt_mat, alpha, Dk, ninvRST,
                                                 point_est, grad_est, logger);
  }
}

/**
 * Construct the return for directly calling single pathfinder or
 *  calling single pathfinder from multi pathfinder
 * @tparam ReturnLpSamples if `true` then this function returns the lp_ratio
 *  and samples. If false then only the return code is returned
 * @tparam EigMat A type inheriting from `Eigen::DenseBase`
 * @tparam EigVec A type inheriting from `Eigen::DenseBase` with one column
 * defined at compile time
 * @return A tuple with an error code, a vector holding the log prob ratios,
 * matrix of samples, and an unsigned integer for number of times the log prob
 * functions was called.
 */
template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio, EigMat&& samples,
                           const std::atomic<size_t>& lp_calls) {
  return std::make_tuple(return_code, std::forward<EigVec>(lp_ratio),
                         std::forward<EigMat>(samples), lp_calls.load());
}

template <bool ReturnLpSamples, typename EigMat, typename EigVec,
          std::enable_if_t<!ReturnLpSamples>* = nullptr>
inline auto ret_pathfinder(int return_code, EigVec&& lp_ratio, EigMat&& samples,
                           const std::atomic<size_t>& lp_calls) noexcept {
  return return_code;
}

/**
 * Estimate the approximate draws given the taylor approximation.
 * @tparam RNG Type of random number generator
 * @tparam LPFun Type of log probability functor
 * @tparam ConstrainFun Type of functor for constraining parameters
 * @tparam Logger Type inheriting from `stan::callbacks::logger`
 * @tparam AlphaVec Type inheriting from `Eigen::DenseBase` with 1 column at
 * compile time
 * @tparam GradBuffer Boost circular buffer with inner Eigen vector type
 * @tparam CurrentParams Type inheriting from `Eigen::DenseBase` with 1 column
 * at compile time
 * @tparam CurentGrads Type inheriting from `Eigen::DenseBase` with 1 column at
 * compile time
 * @tparam ParamMat Type inheriting from `Eigen::DenseBase` with dynamic rows
 * and columns at compile time.
 * @tparam Logger Type of logger callback
 * @param[in,out] rng A generator to produce standard gaussian random variables
 * @param alpha The approximation of the diagonal hessian
 * @param lp_fun Functor to calculate the log density
 * @param constrain_fun A functor to transform parameters to the constrained
 * space
 * @param current_params Parameters from iteration of LBFGS
 * @param current_grads Gradients from iteration of LBFGS
 * @param Ykt_mat Matrix of the last `history_size` changes in the gradient.
 * @param[in,out] Skt_mat Matrix of the last `history_size` changes in the
 * parameters. `Skt_mat` is transformed in this function and will hold inverse
 * solution of RS^T
 * @param num_elbo_draws Number of draws for the ELBO estimation
 * @param iter_msg The beginning of messages that includes the iteration number
 * @param logger A callback writer for messages
 */
template <typename RNG, typename LPFun, typename ConstrainFun,
          typename AlphaVec, typename CurrentParams, typename CurrentGrads,
          typename GradMat, typename ParamMat, typename Logger>
auto pathfinder_impl(RNG&& rng, LPFun&& lp_fun,
                     ConstrainFun&& constrain_fun, AlphaVec&& alpha,
                     CurrentParams&& current_params,
                     CurrentGrads&& current_grads, GradMat&& Ykt_mat,
                     ParamMat&& Skt_mat, std::size_t num_elbo_draws,
                     const std::string& iter_msg, Logger&& logger) {
  const auto current_history_size = Ykt_mat.cols();
  //    Ykt_h.reserve(current_history_size);
  Eigen::MatrixXd Rk
      = Eigen::MatrixXd::Zero(current_history_size, current_history_size);
  Rk.template triangularView<Eigen::Upper>() = Skt_mat.transpose() * Ykt_mat;
  /*
for (Eigen::Index s = 0; s < current_history_size; s++) {
for (Eigen::Index i = 0; i <= s; i++) {
  Rk.coeffRef(i, s) = Skt_mat.col(i).dot(Ykt_mat.col(s));
}
}
*/
  Eigen::VectorXd Dk = Rk.diagonal();
  // Unfolded algorithm in paper for inverse RST
  {
    //    Skt_mat.transposeInPlace();
    Rk.triangularView<Eigen::Upper>().solveInPlace(Skt_mat.transpose());
    // Skt_mat is now ninvRST
    Skt_mat = -Skt_mat;
  }
  internal::taylor_approx_t taylor_appx
      = internal::taylor_approximation(
          Ykt_mat, alpha, Dk, Skt_mat.transpose(), current_params,
          current_grads, logger);
  try {
    return std::make_pair(
        internal::est_approx_draws<true>(lp_fun, constrain_fun, rng,
                                         taylor_appx, num_elbo_draws, alpha,
                                         iter_msg, logger),
        taylor_appx);
  } catch (const std::exception& e) {
    logger.info(iter_msg + "ELBO estimation failed "
                + " with error: " + e.what());
    return std::make_pair(internal::elbo_est_t{}, internal::taylor_approx_t{});
  }
}
}  // namespace internal
/**
 * Run single path pathfinder with specified initializations and write results
 * to the specified callbacks and it returns a return code.
 * @tparam ReturnLpSamples if `true` single pathfinder returns the lp_ratio
 * vector and approximate samples. If `false` only gives a return code.
 * @tparam Model type of model
 * @tparam DiagnosticWriter Type inheriting from @ref stan::callbacks::writer
 * @tparam ParamWriter Type inheriting from @ref stan::callbacks::writer
 * @param[in] model defining target log density and transforms (log $p$ in
 * paper)
 * @param[in] init ($pi_0$ in paper) var context for initialization. Random
 * initial values will be generated for parameters user has not supplied.
 * @param[in] random_seed seed for the random number generator
 * @param[in] path path id to advance the pseudo random number generator
 * @param[in] init_radius A non-negative value to initialize variables uniformly
 * in (-init_radius, init_radius) if not defined in the initialization var
 * context
 * @param[in] history_size  Non-negative value for (J in paper) amount of
 * history to keep for L-BFGS
 * @param[in] init_alpha Non-negative value for line search step size for first
 * iteration
 * @param[in] tol_obj Non-negative value for convergence tolerance on absolute
 * changes in objective function value
 * @param[in] tol_rel_obj ($tau^{rel}$ in paper) Non-negative value for
 * convergence tolerance on relative changes in objective function value
 * @param[in] tol_grad Non-negative value for convergence tolerance on the norm
 * of the gradient
 * @param[in] tol_rel_grad Non-negative value for convergence tolerance on the
 * relative norm of the gradient
 * @param[in] tol_param Non-negative value for convergence tolerance changes in
 * the L1 norm of parameter values
 * @param[in] num_iterations (L in paper) Non-negative value for maximum number
 * of LBFGS iterations
 * @param[in] save_iterations indicates whether all the iterations should
 *   be saved to the parameter_writer
 * @param[in] refresh Output is written to the logger for each iteration modulo
 * the refresh value
 * @param[in,out] interrupt callback to be called every iteration
 * @param[in] num_elbo_draws (K in paper) number of MC draws to evaluate ELBO
 * @param[in] num_draws (M in paper) number of approximate posterior draws to
 * return
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] parameter_writer Writer callback for parameter values
 * @param[in,out] diagnostic_writer output for diagnostics values
 * @return If `ReturnLpSamples` is `true`, returns a tuple of the error code,
 * approximate draws, and a vector of the lp ratio. If `false`, only returns an
 * error code `error_codes::OK` if successful, `error_codes::SOFTWARE` for
 * failures
 */
template <bool ReturnLpSamples = false, class Model, typename DiagnosticWriter,
          typename ParamWriter>
inline auto pathfinder_lbfgs_single(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int path, double init_radius, int history_size, double init_alpha,
    double tol_obj, double tol_rel_obj, double tol_grad, double tol_rel_grad,
    double tol_param, int num_iterations, bool save_iterations, int refresh,
    callbacks::interrupt& interrupt, int num_elbo_draws, int num_draws,
    callbacks::logger& logger, callbacks::writer& init_writer,
    ParamWriter& parameter_writer, DiagnosticWriter& diagnostic_writer) {
  const auto start_optim_time = std::chrono::steady_clock::now();
  boost::ecuyer1988 rng
      = util::create_rng<boost::ecuyer1988>(random_seed, path);
  std::vector<int> disc_vector;
  std::vector<double> cont_vector = util::initialize<false>(
      model, init, rng, init_radius, false, logger, init_writer);
  const auto num_parameters = cont_vector.size();
  // Setup LBFGS
  std::stringstream lbfgs_ss;
  stan::optimization::LSOptions<double> ls_opts;
  ls_opts.alpha0 = init_alpha;
  stan::optimization::ConvergenceOptions<double> conv_opts;
  conv_opts.tolAbsF = tol_obj;
  conv_opts.tolRelF = tol_rel_obj;
  conv_opts.tolAbsGrad = tol_grad;
  conv_opts.tolRelGrad = tol_rel_grad;
  conv_opts.tolAbsX = tol_param;
  conv_opts.maxIts = num_iterations;
  using lbfgs_update_t
      = stan::optimization::LBFGSUpdate<double, Eigen::Dynamic>;
  lbfgs_update_t lbfgs_update(history_size);
  using Optimizer
      = stan::optimization::BFGSLineSearch<Model, lbfgs_update_t, double,
                                           Eigen::Dynamic, true>;
  Optimizer lbfgs(model, cont_vector, disc_vector, std::move(ls_opts),
                  std::move(conv_opts), std::move(lbfgs_update), &lbfgs_ss);
  const std::string path_num("Path: [" + std::to_string(path) + "] ");
  if (refresh != 0) {
    logger.info(path_num + "Initial log joint density = "
                + std::to_string(lbfgs.logp()));
  }
  std::vector<std::string> names;
  model.constrained_param_names(names, true, true);
  names.push_back("lp_approx__");
  names.push_back("lp__");
  parameter_writer(names);
  int ret = 0;
  std::vector<Eigen::VectorXd> param_vecs;
  param_vecs.reserve(num_iterations);
  std::vector<Eigen::VectorXd> grad_vecs;
  grad_vecs.reserve(num_iterations);
  Eigen::VectorXd prev_params
      = Eigen::Map<Eigen::VectorXd>(cont_vector.data(), cont_vector.size());
  Eigen::VectorXd prev_grads;
  boost::circular_buffer<Eigen::VectorXd> param_buff(history_size);
  boost::circular_buffer<Eigen::VectorXd> grad_buff(history_size);
  std::size_t current_history_size = 0;
  {
    std::vector<double> g1;
    double lp = stan::model::log_prob_grad<true, true>(model, cont_vector,
                                                       disc_vector, g1);
    prev_grads = Eigen::Map<Eigen::VectorXd>(g1.data(), num_parameters);
    if (save_iterations) {
      diagnostic_writer(std::make_tuple(
          Eigen::Map<Eigen::VectorXd>(cont_vector.data(), num_parameters)
              .eval(),
          Eigen::Map<Eigen::VectorXd>(g1.data(), num_parameters).eval()));
    }
  }
  auto constrain_fun = [&model](auto&& rng, auto&& unconstrained_draws,
                                auto&& constrained_draws) {
    model.write_array(rng, unconstrained_draws, constrained_draws);
    return constrained_draws;
  };
  auto lp_fun = [&model](auto&& u, auto&& streamer) {
    return model.template log_prob<false, true>(u, &streamer);
  };
  Eigen::VectorXd alpha = Eigen::VectorXd::Ones(num_parameters);
  Eigen::Index best_E = -1;
  internal::elbo_est_t elbo_best;
  internal::taylor_approx_t taylor_approx_best;
  std::size_t num_evals{lbfgs.grad_evals()};
  Eigen::MatrixXd Ykt_mat(num_parameters, history_size);
  Eigen::MatrixXd Skt_mat(num_parameters, history_size);
  while (ret == 0) {
    std::stringstream msg;
    interrupt();
    ret = lbfgs.step();
    double lp = lbfgs.logp();
    if (refresh > 0
        && (ret != 0 || !lbfgs.note().empty() || lbfgs.iter_num() == 0
            || ((lbfgs.iter_num() + 1) % refresh == 0))) {
      std::stringstream msg;
      msg << path_num +
          "    Iter"
          "      log prob"
          "        ||dx||"
          "      ||grad||"
          "       alpha"
          "      alpha0"
          "  # evals"
          "  Notes \n";
      msg << path_num << " " << std::setw(7) << lbfgs.iter_num() << " ";
      msg << " " << std::setw(12) << std::setprecision(6) << lp << " ";
      msg << " " << std::setw(12) << std::setprecision(6)
          << lbfgs.prev_step_size() << " ";
      msg << " " << std::setw(12) << std::setprecision(6)
          << lbfgs.curr_g().norm() << " ";
      msg << " " << std::setw(10) << std::setprecision(4) << lbfgs.alpha()
          << " ";
      msg << " " << std::setw(10) << std::setprecision(4) << lbfgs.alpha0()
          << " ";
      msg << " " << std::setw(7) << lbfgs.grad_evals() << " ";
      msg << " " << lbfgs.note() << " ";
      logger.info(msg.str());
    }

    if (lbfgs_ss.str().length() > 0) {
      logger.info(lbfgs_ss);
      lbfgs_ss.str("");
    }
    /*
     * If the retcode is -1 then linesearch failed even with a hessian reset
     * so the current vals and grads are the same as the previous iter
     * and we are exiting
     */
    if (likely(ret != -1)) {
      param_buff.push_back(lbfgs.curr_x() - prev_params);
      grad_buff.push_back(lbfgs.curr_g() - prev_grads);
      prev_params = lbfgs.curr_x();
      prev_grads = lbfgs.curr_g();
      current_history_size = std::min(current_history_size + 1,
                                      static_cast<std::size_t>(history_size));
      if (internal::check_curve(param_buff.back(), grad_buff.back(), logger)) {
        alpha = internal::form_diag(alpha, grad_buff.back(), param_buff.back());
      }
      Eigen::Map<Eigen::MatrixXd> Ykt_map(Ykt_mat.data(), num_parameters,
                                          current_history_size);
      for (Eigen::Index i = 0; i < current_history_size; ++i) {
        Ykt_map.col(i) = grad_buff[i];
      }
      Eigen::Map<Eigen::MatrixXd> Skt_map(Skt_mat.data(), num_parameters,
                                          current_history_size);
      for (Eigen::Index i = 0; i < current_history_size; ++i) {
        Skt_map.col(i) = param_buff[i];
      }
      std::string iter_msg(path_num + "Iter: ["
                           + std::to_string(lbfgs.iter_num()) + "] ");
      if (STAN_DEBUG_PATH_ITERS) {
        logger.info(iter_msg + "\n------------ Iter: "
                    + std::to_string(lbfgs.iter_num()) + "------------\n");
      }

      auto pathfinder_res = internal::pathfinder_impl(
          rng, lp_fun, constrain_fun, alpha, lbfgs.curr_x(),
          lbfgs.curr_g(), Ykt_map, Skt_map, num_elbo_draws, iter_msg, logger);
      num_evals += pathfinder_res.first.fn_calls;
      if (pathfinder_res.first.elbo > elbo_best.elbo) {
        elbo_best = std::move(pathfinder_res.first);
        taylor_approx_best = std::move(pathfinder_res.second);
        best_E = lbfgs.iter_num();
      }
      if (refresh > 0
          && (lbfgs.iter_num() == 0 || (lbfgs.iter_num() % refresh == 0))) {
        logger.info(iter_msg + ": ELBO ("
                    + std::to_string(pathfinder_res.first.elbo) + ")");
      }
    }
    if (msg.str().length() > 0) {
      logger.info(msg);
    }
    if (save_iterations) {
      diagnostic_writer(std::make_tuple(lbfgs.curr_x(), lbfgs.curr_g()));
    }
  }
  int return_code = error_codes::OK;
  if (ret >= 0) {
    logger.info("Optimization terminated normally: ");
  } else {
    logger.info("Optimization terminated with error: ");
    logger.info("  " + lbfgs.get_code_string(ret));
    if (param_vecs.size() == 1) {
      logger.info("Optimization failed to start, pathfinder cannot be run.");
      return internal::ret_pathfinder<ReturnLpSamples>(
          error_codes::SOFTWARE, Eigen::Array<double, -1, 1>(0),
          Eigen::Array<double, -1, -1>(0, 0),
          std::atomic<size_t>{num_evals + lbfgs.grad_evals()});
    } else {
      logger.info(
          "Stan will still attempt pathfinder but may fail or produce "
          "incorrect "
          "results.");
    }
    return_code = error_codes::OK;
  }
  num_evals += lbfgs.grad_evals();
  if (best_E == -1) {
    logger.info(path_num +
        "Failure: None of the LBFGS iterations completed "
        "successfully");
    return internal::ret_pathfinder<ReturnLpSamples>(
        error_codes::SOFTWARE, Eigen::Array<double, -1, 1>(0),
        Eigen::Array<double, -1, -1>(0, 0), num_evals);
  } else {
    if (refresh != 0) {
      logger.info(path_num + "Best Iter: [" + std::to_string(best_E)
                  + "] ELBO (" + std::to_string(elbo_best.elbo) + ")"
                  + " evalutions: (" + std::to_string(num_evals) + ")");
    }
  }
  Eigen::Array<double, -1, -1> constrained_draws_mat;
  Eigen::Array<double, -1, 1> lp_ratio;
  auto&& elbo_draws = elbo_best.repeat_draws;
  auto&& elbo_lp_ratio = elbo_best.lp_ratio;
  auto&& elbo_lp_mat = elbo_best.lp_mat;
  const int remaining_draws = num_draws - elbo_lp_ratio.rows();
  const Eigen::Index num_unconstrained_params = names.size() - 2;
  if (remaining_draws > 0) {
    try {
      internal::elbo_est_t est_draws = internal::est_approx_draws<false>(
          lp_fun, constrain_fun, rng, taylor_approx_best, remaining_draws,
          taylor_approx_best.alpha, path_num, logger);
      num_evals += est_draws.fn_calls;
      auto&& new_lp_ratio = est_draws.lp_ratio;
      auto&& lp_draws = est_draws.lp_mat;
      auto&& new_draws = est_draws.repeat_draws;
      lp_ratio = Eigen::Array<double, -1, 1>(new_lp_ratio.size()
                                             + elbo_lp_ratio.size());
      lp_ratio.head(elbo_lp_ratio.size()) = elbo_lp_ratio.array();
      lp_ratio.tail(new_lp_ratio.size()) = new_lp_ratio.array();
      const auto total_size = elbo_draws.cols() + new_draws.cols();
      constrained_draws_mat
          = Eigen::Array<double, -1, -1>(names.size(), total_size);
      Eigen::VectorXd unconstrained_col;
      Eigen::VectorXd approx_samples_constrained_col;
      for (Eigen::Index i = 0; i < elbo_draws.cols(); ++i) {
        unconstrained_col = elbo_draws.col(i);
        constrained_draws_mat.col(i).head(num_unconstrained_params)
            = constrain_fun(rng, unconstrained_col,
                            approx_samples_constrained_col);
        constrained_draws_mat.col(i).tail(2) = elbo_lp_mat.row(i);
      }
      for (Eigen::Index i = elbo_draws.cols(), j = 0; i < total_size;
           ++i, ++j) {
        unconstrained_col = new_draws.col(j);
        constrained_draws_mat.col(i).head(num_unconstrained_params)
            = constrain_fun(rng, unconstrained_col,
                            approx_samples_constrained_col);
        constrained_draws_mat.col(i).tail(2) = lp_draws.row(j);
      }
    } catch (const std::exception& e) {
      std::string err_msg = e.what();
      logger.info(path_num + "Final sampling approximation failed with error: "
                  + err_msg);
      logger.info(
          path_num
          + "Returning the approximate samples used for ELBO calculation: "
          + err_msg);
      constrained_draws_mat
          = Eigen::Array<double, -1, -1>(names.size(), elbo_draws.cols());
      Eigen::VectorXd approx_samples_constrained_col;
      Eigen::VectorXd unconstrained_col;
      for (Eigen::Index i = 0; i < elbo_draws.cols(); ++i) {
        unconstrained_col = elbo_draws.col(i);
        constrained_draws_mat.col(i).head(num_unconstrained_params)
            = constrain_fun(rng, unconstrained_col,
                            approx_samples_constrained_col);
        constrained_draws_mat.col(i).tail(2) = elbo_lp_mat.row(i);
      }
      lp_ratio = std::move(elbo_best.lp_ratio);
    }
  } else {
    constrained_draws_mat
        = Eigen::Array<double, -1, -1>(names.size(), elbo_draws.cols());
    Eigen::VectorXd approx_samples_constrained_col;
    Eigen::VectorXd unconstrained_col;
    for (Eigen::Index i = 0; i < elbo_draws.cols(); ++i) {
      unconstrained_col = elbo_draws.col(i);
      constrained_draws_mat.col(i).head(num_unconstrained_params)
          = constrain_fun(rng, unconstrained_col,
                          approx_samples_constrained_col);
      constrained_draws_mat.col(i).tail(2) = elbo_lp_mat.row(i);
    }
    lp_ratio = std::move(elbo_best.lp_ratio);
  }
  parameter_writer(constrained_draws_mat.matrix());
  parameter_writer();
  const auto end_optim_time = std::chrono::steady_clock::now();
  const double optim_delta_time
      = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_optim_time - start_optim_time)
            .count()
        / 1000.0;
  const auto time_header = std::string("Elapsed Time: ");
  std::string optim_time_str
      = time_header + std::to_string(optim_delta_time) + " seconds (Pathfinder)";
  parameter_writer();
  return internal::ret_pathfinder<ReturnLpSamples>(
      error_codes::OK, std::move(lp_ratio), std::move(constrained_draws_mat),
      num_evals);
}

}  // namespace pathfinder
}  // namespace services
}  // namespace stan
#endif
