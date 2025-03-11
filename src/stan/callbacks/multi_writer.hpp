#ifndef STAN_CALLBACKS_MULTI_WRITER_HPP
#define STAN_CALLBACKS_MULTI_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/functor/for_each.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace stan::callbacks {

/**
 * `multi_writer` is an layer on top of a writer class that
 *  allows for multiple output streams to be written to.
 * @tparam Writers A parameter pack of types that inherit from `writer`
 */
template <typename... Writers>
class multi_writer {
 public:
  /**
   * Constructs a multi stream writer from a parameter pack of writers.
   * @tparam Args A parameter pack of writers. Should be the same type as
   * `Writers`
   * @param[in, out] args A parameter pack of writers
   */
  template <typename... Args>
  explicit multi_writer(Args&&... args)
      : output_(std::forward<Args>(args)...) {}

  multi_writer() = default;

  /**
   * @tparam T Any type accepted by a `writer` overload
   * @param[in] x A value to write to the output streams
   */
  template <typename T>
  void operator()(T&& x) {
    stan::math::for_each([&](auto&& output) { output(x); }, output_);
  }
  /**
   * Write a comment prefix to each writer
   */
  void operator()() {
    stan::math::for_each([](auto&& output) { output(); }, output_);
  }

  /**
   * Checks if all underlying writers are nonnull.
   */
  inline bool is_nonnull() const noexcept {
    return stan::math::apply(
        [](auto&&... output) { return (output.is_nonnull() && ...); }, output_);
  }

  /**
   * Get the tuple of underlying streams
   */
  inline auto& get_stream() noexcept { return output_; }
  /**
   * Assuming all streams have the same comment prefix, return the first comment
   * prefix.
   */
  const char* comment_prefix() const noexcept {
    return std::get<0>(output_).comment_prefix();
  }

 private:
  // Output streams
  std::tuple<std::reference_wrapper<Writers>...> output_;
};

namespace internal {
template <typename T>
struct is_multi_writer : std::false_type {};

template <typename... Types>
struct is_multi_writer<multi_writer<Types...>> : std::true_type {};
}  // namespace internal

/**
 * Type trait that checks if a type is a `multi_writer`
 * @tparam T A type to check
 */
template <typename T>
struct is_multi_writer : internal::is_multi_writer<std::decay_t<T>> {};

/**
 * Helper variable template to check if a type is a `multi_writer`
 */
template <typename T>
inline constexpr bool is_multi_writer_v = is_multi_writer<T>::value;

}  // namespace stan::callbacks

#endif
