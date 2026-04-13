#ifndef CUTIE_ORT_TYPES_H
#define CUTIE_ORT_TYPES_H

#include <string>
#include <vector>

#include "cutie/ort/core/ort_config.h"

namespace cutie
{
namespace ortcore
{

struct OrtSessionInfo
{
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_TYPES_H
