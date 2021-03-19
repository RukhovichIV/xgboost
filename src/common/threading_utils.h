/*!
 * Copyright 2015-2019 by Contributors
 * \file common.h
 * \brief Threading utilities
 */
#ifndef XGBOOST_COMMON_THREADING_UTILS_H_
#define XGBOOST_COMMON_THREADING_UTILS_H_

#include <dmlc/common.h>
#include <vector>
#include <algorithm>
#include "xgboost/logging.h"
#include "cpuid.h"

namespace xgboost {
namespace common {

// Represent simple range of indexes [begin, end)
// Inspired by tbb::blocked_range
class Range1d {
 public:
  Range1d(size_t begin, size_t end): begin_(begin), end_(end) {
    CHECK_LT(begin, end);
  }

  size_t begin() const {  // NOLINT
    return begin_;
  }

  size_t end() const {  // NOLINT
    return end_;
  }

 private:
  size_t begin_;
  size_t end_;
};


// Split 2d space to balanced blocks
// Implementation of the class is inspired by tbb::blocked_range2d
// However, TBB provides only (n x m) 2d range (matrix) separated by blocks. Example:
// [ 1,2,3 ]
// [ 4,5,6 ]
// [ 7,8,9 ]
// But the class is able to work with different sizes in each 'row'. Example:
// [ 1,2 ]
// [ 3,4,5,6 ]
// [ 7,8,9]
// If grain_size is 2: It produces following blocks:
// [1,2], [3,4], [5,6], [7,8], [9]
// The class helps to process data in several tree nodes (non-balanced usually) in parallel
// Using nested parallelism (by nodes and by data in each node)
// it helps  to improve CPU resources utilization
class BlockedSpace2d {
 public:
  // Example of space:
  // [ 1,2 ]
  // [ 3,4,5,6 ]
  // [ 7,8,9]
  // BlockedSpace2d will create following blocks (tasks) if grain_size=2:
  // 1-block: first_dimension = 0, range of indexes in a 'row' = [0,2) (includes [1,2] values)
  // 2-block: first_dimension = 1, range of indexes in a 'row' = [0,2) (includes [3,4] values)
  // 3-block: first_dimension = 1, range of indexes in a 'row' = [2,4) (includes [5,6] values)
  // 4-block: first_dimension = 2, range of indexes in a 'row' = [0,2) (includes [7,8] values)
  // 5-block: first_dimension = 2, range of indexes in a 'row' = [2,3) (includes [9] values)
  // Arguments:
  // dim1 - size of the first dimension in the space
  // getter_size_dim2 - functor to get the second dimensions for each 'row' by row-index
  // grain_size - max size of produced blocks
  template<typename Func>
  BlockedSpace2d(size_t dim1, Func getter_size_dim2, size_t grain_size) {
    for (size_t i = 0; i < dim1; ++i) {
      const size_t size = getter_size_dim2(i);
      const size_t n_blocks = size/grain_size + !!(size % grain_size);
      for (size_t iblock = 0; iblock < n_blocks; ++iblock) {
        const size_t begin = iblock * grain_size;
        const size_t end   = std::min(begin + grain_size, size);
        AddBlock(i, begin, end);
      }
    }
  }

  // Amount of blocks(tasks) in a space
  size_t Size() const {
    return ranges_.size();
  }

  // get index of the first dimension of i-th block(task)
  size_t GetFirstDimension(size_t i) const {
    CHECK_LT(i, first_dimension_.size());
    return first_dimension_[i];
  }

  // get a range of indexes for the second dimension of i-th block(task)
  Range1d GetRange(size_t i) const {
    CHECK_LT(i, ranges_.size());
    return ranges_[i];
  }

 private:
  void AddBlock(size_t first_dimension, size_t begin, size_t end) {
    first_dimension_.push_back(first_dimension);
    ranges_.emplace_back(begin, end);
  }

  std::vector<Range1d> ranges_;
  std::vector<size_t> first_dimension_;
};


// Wrapper to implement nested parallelism with simple omp parallel for
template <typename Func>
void ParallelFor2d(const BlockedSpace2d& space, int nthreads, Func func) {
  const size_t num_blocks_in_space = space.Size();
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);

  dmlc::OMPException exc;
#pragma omp parallel num_threads(nthreads)
  {
    exc.Run([&]() {
      size_t tid = omp_get_thread_num();
      size_t chunck_size =
          num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

      size_t begin = chunck_size * tid;
      size_t end = std::min(begin + chunck_size, num_blocks_in_space);
      for (auto i = begin; i < end; i++) {
        func(space.GetFirstDimension(i), space.GetRange(i));
      }
    });
  }
  exc.Rethrow();
}

template <typename Index, typename Func>
void ParallelFor(Index size, size_t nthreads, Func fn) {
  dmlc::OMPException exc;
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (Index i = 0; i < size; ++i) {
    exc.Run(fn, i);
  }
  exc.Rethrow();
}

template <typename Index, typename Func>
void ParallelFor(Index size, Func fn) {
  ParallelFor(size, omp_get_max_threads(), fn);
}

inline uint32_t GetNumberOfPhysicalCores() {
  uint32_t req_byte = 1u;
  static CPUID cpuid(req_byte);

  uint32_t num_procs_reported = omp_get_num_procs();
  uint32_t ht_bit = 1u << 28u;
  bool has_physical_ht = cpuid.EDX() & ht_bit;
  if (!has_physical_ht) {
    return num_procs_reported;
  }

  bool ht_enabled = true;
  uint32_t one_byte_mask = (1u << 8u) - 1u;
  uint32_t number_of_unique_ids = (cpuid.EBX() >> 16u) & one_byte_mask;
  if (num_procs_reported <= number_of_unique_ids) {
    ht_enabled = false;
  }

  /*
  uint32_t cpuinfo[4];
  cpuinfo[0] = cpuid.EAX();
  cpuinfo[1] = cpuid.EBX();
  cpuinfo[2] = cpuid.ECX();
  cpuinfo[3] = cpuid.EDX();
  std::cout << "\n\n";
  for (unsigned j = 0u; j < 4u; ++j) {
    for (unsigned i = 0u; i < 32u; ++i) {
      if (i && i % 8 == 0) {
        std::cout << ' ';
      }
      std::cout << static_cast<bool>(cpuinfo[j] & (1u << i));
    }
    std::cout << '\n';
  }
  std::cout << "\nNumber of ids = " << number_of_unique_ids << "\n";
  std::cout << "Hyperthreading on CPU is " << ht_enabled << "\n";
  std::cout << (ht_enabled ? (num_procs_reported >> 1u) : num_procs_reported);
  std::cout << " threads will be reported\n";
  std::cout << "\n\n";
  */
  return (ht_enabled ? (num_procs_reported >> 1u) : num_procs_reported);
}

/* \brief Configure parallel threads.
 *
 * \param p_threads Number of threads, when it's less than or equal to 0, this function
 *        will change it to number of process on system.
 *
 * \return Global openmp max threads before configuration.
 */
inline int32_t OmpSetNumThreads(int32_t* p_threads) {
  auto& threads = *p_threads;
  int32_t nthread_original = omp_get_max_threads();
  if (threads <= 0) {
    threads = omp_get_num_procs();
  }
  omp_set_num_threads(threads);
  return nthread_original;
}
inline int32_t OmpSetNumThreadsWithoutHT(int32_t* p_threads) {
  auto& threads = *p_threads;
  int32_t nthread_original = omp_get_max_threads();
  if (threads <= 0) {
    threads = GetNumberOfPhysicalCores();
  }
  omp_set_num_threads(threads);
  return nthread_original;
}

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_THREADING_UTILS_H_
