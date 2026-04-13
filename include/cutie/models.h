#ifndef CUTIE_MODELS_H
#define CUTIE_MODELS_H

/// Namespace aliases a la lite.ai.toolkit.
/// Usage: cutie::cv::segmentation::CutieProcessor

#include "cutie/core/processor.h"

namespace cutie
{
namespace cv
{
namespace segmentation
{

using CutieConfig = cutie::core::CutieConfig;
using CutieProcessor = cutie::core::CutieProcessor;
using StepOptions = cutie::core::StepOptions;

}  // namespace segmentation
}  // namespace cv
}  // namespace cutie

#endif  // CUTIE_MODELS_H
