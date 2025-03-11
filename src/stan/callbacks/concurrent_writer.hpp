#ifndef STAN_CALLBACKS_CONCURRENT_WRITER_HPP
#define STAN_CALLBACKS_CONCURRENT_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <tbb/concurrent_queue.h>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace stan::callbacks {
#ifdef STAN_THREADS
/**
 * Takes a writer and makes it thread safe via multiple queues.
 * On construction, a single busy thread is spawned to write to the writer.
 * This class uses an `std::thread` instead of a tbb task graph because
 * of deadlocking issues. A deadlock can occur if TBB gives all threads to the
 * parallel for loop, and all threads hit an instance of max capacity. TBB can
 * choose to wait for a thread to finish instead of spinning up the write
 * thread. So to circumvent that issue, we use an std::thread.
 * @tparam Writer A type that inherits from `writer`
 */
template <typename Writer>
struct concurrent_writer {
  // A reference to the writer to write to
  std::reference_wrapper<Writer> writer;
  // Queue for string messages
  tbb::concurrent_bounded_queue<std::string> str_messages_{};
  // Queue for vector of strings messages
  tbb::concurrent_bounded_queue<std::vector<std::string>> vec_str_messages_{};
  // Queue for Eigen vector messages
  tbb::concurrent_bounded_queue<Eigen::RowVectorXd> eigen_messages_{};
  // Flag to stop the writing thread once all queues are empty
  bool continue_writing_{true};
  // The writing thread
  std::thread thread_;
  /**
   * Constructs a concurrent writer from a writer and spins up a thread for
   * writing.
   * @param writer A writer to write to
   */
  explicit concurrent_writer(Writer& writer) : writer(writer) {
    str_messages_.set_capacity(1000);
    vec_str_messages_.set_capacity(1000);
    eigen_messages_.set_capacity(1000);
    thread_ = std::thread([&]() {
      std::string str;
      std::vector<std::string> vec_str;
      Eigen::RowVectorXd eigen;
      while (continue_writing_
             || !(str_messages_.empty() && vec_str_messages_.empty()
                  && eigen_messages_.empty())) {
        while (str_messages_.try_pop(str)) {
          writer(str);
        }
        while (vec_str_messages_.try_pop(vec_str)) {
          writer(vec_str);
        }
        while (eigen_messages_.try_pop(eigen)) {
          writer(eigen);
        }
      }
    });
  }
  /**
   * Place a value in a queue for writing.
   * @tparam T Either an `std::vector<std::string|double>`, an Eigen vector, or
   * a string
   * @param t A value to put on a queue
   */
  template <typename T>
  void operator()(T&& t) {
    if constexpr (stan::is_std_vector<T>::value) {
      if constexpr (std::is_arithmetic_v<stan::value_type_t<T>>) {
        eigen_messages_.push(Eigen::RowVectorXd::Map(t.data(), t.size()));
      } else {
        vec_str_messages_.push(t);
      }
    } else if constexpr (std::is_same_v<T, std::string>) {
      str_messages_.push(std::forward<T>(t));
    } else if constexpr (stan::is_eigen_row_vector<T>::value
                         || stan::is_eigen_col_vector<T>::value) {
      eigen_messages_.push(std::forward<T>(t));
    } else {
      static_assert(1,
                    "Unsupported type passed to concurrent_writer. "
                    "This is an internal error. Please file an issue on the "
                    "stan github repo.");
    }
  }
  /**
   * Writes a comment prefix to the writer.
   */
  void operator()() { str_messages_.push(writer.get().comment_prefix()); }
  /**
   * Waits till all writes are finished on the thread
   */
  void wait() {
    continue_writing_ = false;
    thread_.join();
  }
};
#else
/**
 * When STAN_THREADS is not defined, the concurrent writer is just a wrapper
 */
template <typename Writer>
struct concurrent_writer {
  std::reference_wrapper<Writer> writer;
  explicit concurrent_writer(Writer& writer) : writer(writer) {}
  template <typename T>
  void operator()(T&& t) {
    writer(std::forward<T>(t));
  }
  void operator()() { writer(); }
  inline static constexpr void wait() {}
};
#endif
}  // namespace stan::callbacks
#endif
